from time import sleep

# function y = f(x)
def f(x):
    return x**2 - 14 * x + 7

# symbolic gradient
def df(x):
    return 2 * x - 14

# parameters
learning_rate = 0.1
n_steps = 30
x = 0.

# searching for minimum with gradient descent
for step in range(n_steps):
    x -= learning_rate * df(x)
    print("x {} y {}".format(x, f(x)))

    # and a crucial part of the SleepyGradientDescent algorithm ;)
    sleep(1.)
