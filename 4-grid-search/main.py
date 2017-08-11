from deepsense import neptune

from sklearn import datasets, ensemble
import numpy as np

ctx = neptune.Context()

diabetes = datasets.load_diabetes()
(X_train, X_test) = (diabetes.data[:-100], diabetes.data[-100:])
(y_train, y_test) = (diabetes.target[:-100], diabetes.target[-100:])

model = ensemble.RandomForestRegressor(
  n_estimators=ctx.params.n_estimators,
  max_depth=ctx.params.max_depth)

model.fit(X_train, y_train)
mse = np.mean((model.predict(X_test) - y_test) ** 2)
ctx.job.channel_send('mse', mse)
