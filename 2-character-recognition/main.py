from scipy import io
import numpy as np
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from PIL import Image

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers import Conv2D, MaxPool2D, Dropout
from keras.callbacks import Callback, TensorBoard

from deepsense import neptune


def array_2d_to_image(array, autorescale=True):
    assert array.min() >= 0
    assert len(array.shape) == 2
    if array.max() <= 1 and autorescale:
        array = 255 * array
    array = array.astype('uint8')
    return Image.fromarray(array)


class NeptuneCallback(Callback):
    def __init__(self, images_per_epoch=-1):
        self.epoch_id = 0
        self.images_per_epoch = images_per_epoch

    def on_epoch_end(self, epoch, logs={}):
        self.epoch_id += 1

        # logging numeric channels
        ctx.channel_send('Log-loss training', self.epoch_id, logs['loss'])
        ctx.channel_send('Log-loss validation', self.epoch_id, logs['val_loss'])
        ctx.channel_send('Accuracy training', self.epoch_id, logs['acc'])
        ctx.channel_send('Accuracy validation', self.epoch_id, logs['val_acc'])

        # Predict the digits for images of the test set.
        validation_predictions = model.predict_classes(X_test)
        scores = model.predict(X_test)

        # Identify the incorrectly classified images and send them to Neptune Dashboard.
        image_per_epoch = 0
        for index, (prediction, actual) in enumerate(zip(validation_predictions, Y_test.argmax(axis=1))):
            if prediction != actual:
                if image_per_epoch == self.images_per_epoch:
                    break
                image_per_epoch += 1

                ctx.channel_send('false_predictions', neptune.Image(
                    name='[{}] pred: {} true: {}'.format(self.epoch_id, letters[prediction], letters[actual]),
                    description="\n".join([
                        "{} {:5.1f}% {}".format(letters[i], 100 * score, "!!!" if i == actual else "")
                        for i, score in enumerate(scores[index])]),
                    data=array_2d_to_image(X_test[index,:,:,0])))


data = io.loadmat("/input/notMNIST_small.mat")
X = data['images']
y = data['labels']

resolution = 28  # 28x28 images, grayscale
classes = 10     # 10 letters: ABCDEFGHIJ

# transforming data for TensorFlow backend
X = np.transpose(X, (2, 0, 1))
X = X.reshape((-1, resolution, resolution, 1))
X = X.astype('float32') / 255.

# 3 -> [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.]
y = y.astype('int32')
Y = np_utils.to_categorical(y, classes)

# splitting data into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size=0.20,
    random_state=137)

letters = "ABCDEFGHIJ"

ctx = neptune.Context()

# create neural network architecture

model = Sequential()

model.add(Conv2D(16, (3, 3), activation='relu',
                 input_shape=(resolution, resolution, 1)))
# model.add(Conv2D(16, (3, 3), activation='relu'))  # uncomment!
model.add(MaxPool2D())

# model.add(Conv2D(32, (3, 3), activation='relu'))  # uncomment!
# model.add(Conv2D(32, (3, 3), activation='relu'))  # uncomment!
model.add(MaxPool2D())

model.add(Flatten())
# model.add(Dropout(0.5))  # uncomment!
model.add(Dense(classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, Y_train,
          epochs=10,
          batch_size=32,
          validation_data=(X_test, Y_test),
          verbose=2,
          callbacks=[NeptuneCallback(images_per_epoch=20)])
