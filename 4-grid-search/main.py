import argparse

from deepsense import neptune

from sklearn import datasets, ensemble
import numpy as np

ctx = neptune.Context()

params_parser = argparse.ArgumentParser()
params_parser.add_argument('--max_depth', type=float, default=8)
params_parser.add_argument('--n_estimators', type=float, default=4)

params = params_parser.parse_args()

diabetes = datasets.load_diabetes()
(X_train, X_test) = (diabetes.data[:-100], diabetes.data[-100:])
(y_train, y_test) = (diabetes.target[:-100], diabetes.target[-100:])

model = ensemble.RandomForestRegressor(
  n_estimators=int(params.n_estimators),
  max_depth=int(params.max_depth))

model.fit(X_train, y_train)
mse = np.mean((model.predict(X_test) - y_test) ** 2)
ctx.channel_send('mse', mse)
