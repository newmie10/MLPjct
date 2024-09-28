import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = np.array(pd.read_csv('train.csv'))

m, n = data.shape

np.random.shuffle(data)

data_dev = data[0:1000].T

Y_dev = data_dev[0]
X_dev = data_dev[1:n]

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]

def init_params():
    w1 = np.random.randn(10, 784)
    b1 = np.random.randn(10, 1)
    w2 = np.random.randn(10, 10)
    b2 = np.random.randn(10, 1)
    return w1, b1, w2, b2

# ReLU is fn where x if x > 0
def ReLU(Z):
    return np.maximum(0, Z)

def softmax(Z):
    return np.exp(Z) / np.sum(np.exp(Z))

# Forward Propogation
def forward_prop(w1, b1, w2, b2, X):
    z1 = w1.dot(X) + b1
    a1 = ReLU(z1)
    z2 = w2.dot(a1) + b2
    a2 = softmax(z1)
    return z1, a1, z2, a2

def one_hot(Y):
    one_hot_y = np.zeros((Y.size, Y.max() + 1))
    one_hot_y[np.arange(Y.size), Y] = 1
    one_hot_y = one_hot_y.T
    return one_hot_y

def deriv_ReLu(Z):
    return Z > 0

def backwards_prop(z1, a1, z2, a2, w2, X, Y):
    m = Y.size
    one_hot_y = one_hot(Y)
    dz2 = a2 - one_hot_y
    dw2 = 1 / m * dz2.dot(a1.T)
    db2 = 1 / m * np.sum(dz2, 1)
    dz1 = w2.T.dot(dz2) * deriv_ReLu(z1)
    dw1 = 1 / m * dz1.dot(X.T)
    db1 = 1 / m * np.sum(dz1, 1)
    return dw1, db1, dw2, db2

def update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha):
    w1 = w1 - alpha * dw1
    b1 = b1 - alpha * db1
    b2 = b2 - alpha * db2
    w2 = w2 - alpha * dw2
    return w1, b1, w2, b2

def get_predictions(a2):
    return np.argmax(a2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, iterations, alpha):
    w1, b1, w2, b2 = init_params()
    for i in range(iterations):
        z1, a1, z2, a2 = forward_prop(w1, b1, w2, b2, X)
        dw1, db1, dw2, db2 = backwards_prop(z1, a1, z2, a2, w2, X, Y)
        w1, b1, w2, b2 = update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha)
        if i % 10 == 0:
            print(f'Iteration: {i})')
            print(f'Accuracy: {get_accuracy(get_predictions(a2), Y)}')
    return w1, b1, w2, b2

w1, b1, w2, b2 = gradient_descent(X_train, Y_train, 100, 0.1)
