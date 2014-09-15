import numpy as np

def sigmoid(m, a = 1.0):
    return 1 / (1 + np.exp(-a * m))

def d_sigmoid(m, a = 1.0):
    s = sigmoid(m, a)
    return a * s * (1 - s)

def identity(m):
    return m

def d_identity(m):
    return np.ones(m.shape)

def tanh(m, a = 1.0):
    return np.tanh(a * m)

def d_tanh(m, a = 1.0):
    t = np.tanh(a * m)
    return a * (1 - t * t)

def softmax(m):
    s = np.exp(m)
    return s / np.sum(s)

def d_softmax(m):
    y = softmax(m)
    return y * (1 - y)