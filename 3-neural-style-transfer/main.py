import argparse

import numpy as np
import tensorflow as tf
from PIL import Image

from deepsense import neptune


ctx = neptune.Context()

params_parser = argparse.ArgumentParser()
params_parser.add_argument('--path_to_model', default='/input/model.ckpt')
params_parser.add_argument('--path_to_images', default='/input/images/')
params_parser.add_argument('--content', default='cat.jpg', help='name of the content image')
params_parser.add_argument('--style', default='picasso.jpg', help='name of the style image')
params_parser.add_argument('--from_content', type=bool, default=True,
                           help='style transfer starts from the content image or from a random image')
params_parser.add_argument('--learning_rate', type=float, default=5.0)
params_parser.add_argument('--content_style_balance', type=float, default=1.0,
                           help='default balance between content and style intensity')
params_parser.add_argument('--max_img_size', type=int, default=500,
                           help='maximal allowed height and width of the image'),
params_parser.add_argument('--number_of_iterations', type=int, default=2500)

params = params_parser.parse_args()

# Definition of an action.
content_style_balance = params.content_style_balance
def _change_content_style_balance_handler(csb):
    global content_style_balance
    content_style_balance = csb
    return True
ctx.job.register_action(name='Change content/style balance', handler=_change_content_style_balance_handler)


def pil_to_arr(img):
    arr = np.array(img).astype(np.float32)[:,:,::-1]
    return np.expand_dims(arr, 0)


def arr_to_pil(arr):
    img_bgr = Image.fromarray(arr)
    b, g, r = img_bgr.split()
    return Image.merge("RGB", (r, g, b))


def read_images(path_to_content, path_to_style, max_axis_size):
    content = Image.open(path_to_content)
    content.thumbnail((max_axis_size, max_axis_size), Image.ANTIALIAS)

    style = Image.open(path_to_style)
    style = style.resize(content.size, Image.ANTIALIAS)

    return pil_to_arr(content), pil_to_arr(style)


def read_image(path, size):
    img = Image.open(path)
    img.thumbnail(size, Image.ANTIALIAS)
    img_arr = np.array(img).astype(np.float32)[:,:,::-1]
    return np.expand_dims(img_arr, 0)


# Retrieves image to the printable form.
def retrieve(img):
    img = np.squeeze(img, axis=(0,))
    img += [103.939, 116.779, 123.68]
    img = np.clip(img.round(), 0, 255).astype(np.uint8)
    return arr_to_pil(img)


# Definition of the convolutional layer.
def conv2d(signal, num_outputs, kernel_size=[3, 3], stride=1, padding='SAME', activation_fn=tf.nn.relu, scope='Conv'):
    num_inputs = int(signal.get_shape()[-1])
    wshape = [kernel_size[0], kernel_size[1], num_inputs, num_outputs]
    stddev = np.sqrt(2./(num_inputs*kernel_size[0]*kernel_size[1]))
    with tf.variable_scope(scope):
        weights = tf.Variable(tf.truncated_normal(wshape, mean=0., stddev=stddev), dtype=tf.float32, name='weights')
        bias = tf.Variable(tf.constant(0., shape=[num_outputs]), dtype=tf.float32, name='bias')
        signal = tf.nn.conv2d(signal, weights, strides=[1, stride, stride, 1], padding=padding)
        signal = tf.nn.bias_add(signal, bias)
        signal = activation_fn(signal)
    return signal


# Definition of the average-pool layer.
def avg_pool(signal, kernel_size=[2, 2], stride=2, padding='SAME', name='AvgPool'):
    signal = tf.nn.avg_pool(signal, ksize=[1, kernel_size[0], kernel_size[1], 1],
                            strides=[1, stride, stride, 1], padding=padding, name=name)
    return signal


# Definition of the Gram matrix.
def gram_matrix(signal):
    _, h, w, d = map(int, signal.get_shape())
    V = tf.reshape(signal, shape=(h*w, d))
    G = tf.matmul(V, V, transpose_a=True)
    G /= 2*h*w*d
    return G


# Gets activation of the conv4_2 layer of the VGG16 network for the content image
# and Gram matrices of conv1_1, ..., conv5_1 for the style image.
# They are used as reference points to define content and style loss functions.
def get_stats(content, style):
    tf.reset_default_graph()

    assert content.shape == style.shape
    inputs = tf.placeholder(tf.float32, shape=content.shape, name='inputs')

    signal = inputs - [103.939, 116.779, 123.68]

    conv1_1 = conv2d(signal, 64, scope='conv1_1')
    conv1_2 = conv2d(conv1_1, 64, scope='conv1_2')
    pool1 = avg_pool(conv1_2, name='pool1')

    conv2_1 = conv2d(pool1, 128, scope='conv2_1')
    conv2_2 = conv2d(conv2_1, 128, scope='conv2_2')
    pool2 = avg_pool(conv2_2, name='pool2')

    conv3_1 = conv2d(pool2, 256, scope='conv3_1')
    conv3_2 = conv2d(conv3_1, 256, scope='conv3_2')
    conv3_3 = conv2d(conv3_2, 256, scope='conv3_3')
    pool3 = avg_pool(conv3_3, name='pool3')

    conv4_1 = conv2d(pool3, 512, scope='conv4_1')
    conv4_2 = conv2d(conv4_1, 512, scope='conv4_2')
    conv4_3 = conv2d(conv4_2, 512, scope='conv4_3')
    pool4 = avg_pool(conv4_3, name='pool4')

    conv5_1 = conv2d(pool4, 512, scope='conv5_1')
    conv5_2 = conv2d(conv5_1, 512, scope='conv5_2')
    conv5_3 = conv2d(conv5_2, 512, scope='conv5_3')
    pool5 = avg_pool(conv5_3, name='pool5')

    vgg16 = tf.train.Saver()

    stats = {}

    with tf.Session() as sess:
        vgg16.restore(sess, params.path_to_model)
        stats['content_stats'] = sess.run(conv4_2, feed_dict={inputs: content})
        stats['style_stats1']  = sess.run(gram_matrix(conv1_1), feed_dict={inputs: style})
        stats['style_stats2']  = sess.run(gram_matrix(conv2_1), feed_dict={inputs: style})
        stats['style_stats3']  = sess.run(gram_matrix(conv3_1), feed_dict={inputs: style})
        stats['style_stats4']  = sess.run(gram_matrix(conv4_1), feed_dict={inputs: style})
        stats['style_stats5']  = sess.run(gram_matrix(conv5_1), feed_dict={inputs: style})

    return stats


# Finetunes the VGG16 network to have an activation on conv4_2 layer close to that for the content image
# and Gram matrices close to those for the style image.
# Also, sends current loss values and image thumbnail to Neptune.
def transfer_style(stats, img):
    tf.reset_default_graph()

    if params.from_content:
        initial_image = img
    else:
        initial_image = tf.truncated_normal(img.shape, mean=0., stddev=1, dtype=tf.float32, seed=None)
    image = tf.Variable(initial_image, name='image')

    conv1_1 = conv2d(image, 64, scope='conv1_1')
    conv1_2 = conv2d(conv1_1, 64, scope='conv1_2')
    pool1 = avg_pool(conv1_2, name='pool1')

    conv2_1 = conv2d(pool1, 128, scope='conv2_1')
    conv2_2 = conv2d(conv2_1, 128, scope='conv2_2')
    pool2 = avg_pool(conv2_2, name='pool2')

    conv3_1 = conv2d(pool2, 256, scope='conv3_1')
    conv3_2 = conv2d(conv3_1, 256, scope='conv3_2')
    conv3_3 = conv2d(conv3_2, 256, scope='conv3_3')
    pool3 = avg_pool(conv3_3, name='pool3')

    conv4_1 = conv2d(pool3, 512, scope='conv4_1')
    conv4_2 = conv2d(conv4_1, 512, scope='conv4_2')
    conv4_3 = conv2d(conv4_2, 512, scope='conv4_3')
    pool4 = avg_pool(conv4_3, name='pool4')

    conv5_1 = conv2d(pool4, 512, scope='conv5_1')
    conv5_2 = conv2d(conv5_1, 512, scope='conv5_2')
    conv5_3 = conv2d(conv5_2, 512, scope='conv5_3')
    pool5 = avg_pool(conv5_3, name='pool5')

    content_style_balance_param = tf.placeholder(dtype=tf.float32, shape=[])

    loss_content = 3e-4 * tf.reduce_sum((stats['content_stats'] - conv4_2)**2)/2

    loss_style1 = tf.reduce_sum((stats['style_stats1'] - gram_matrix(conv1_1))**2)/2
    loss_style2 = tf.reduce_sum((stats['style_stats2'] - gram_matrix(conv2_1))**2)/2
    loss_style3 = tf.reduce_sum((stats['style_stats3'] - gram_matrix(conv3_1))**2)/2
    loss_style4 = tf.reduce_sum((stats['style_stats4'] - gram_matrix(conv4_1))**2)/2
    loss_style5 = tf.reduce_sum((stats['style_stats5'] - gram_matrix(conv5_1))**2)/2

    loss_style = tf.add_n([loss_style1, loss_style2, loss_style3, loss_style4, loss_style5])/5

    loss = 2 * (content_style_balance_param * loss_content + loss_style) / (1.0 + content_style_balance_param)

    train_op = tf.train.AdamOptimizer(params.learning_rate).minimize(loss, var_list=[image])
    vgg16 = tf.train.Saver(tf.global_variables()[1:27])

    cfg = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)

    with tf.Session(config=cfg) as sess:
        sess.run(tf.global_variables_initializer())
        vgg16.restore(sess, params.path_to_model)

        for step in xrange(params.number_of_iterations):
            _, l, ls, lc = sess.run(
                [train_op, loss, loss_style, loss_content],
                feed_dict={content_style_balance_param: content_style_balance})

            if step % 10 == 0:
                ctx.job.channel_send('loss', step, l)
                ctx.job.channel_send('loss style', step, ls)
                ctx.job.channel_send('loss content', step, lc)
                ctx.job.channel_send('content/style balance', step, content_style_balance)

                retrieved = retrieve(sess.run(image))
                retrieved.thumbnail((300, 300), Image.ANTIALIAS)
                ctx.job.channel_send(
                    'image_channel', step,
                    neptune.Image(name=step, description=step, data=retrieved))

        final = retrieve(sess.run(image))
        final.save('/output/final.jpg')


def main():
    content, style = read_images(
        params.path_to_images + params.content,
        params.path_to_images + params.style,
        params.max_img_size)

    stats = get_stats(content, style)
    transfer_style(stats, content)


if __name__ == "__main__":
    main()
