import numpy as np

def l1(m):
    return np.sum(np.abs(m.reshape((1, np.prod(m.shape)))))

def d_l1(m):
    return (m > 0) * 2 - 1.0  # sign(m) = m / abs(m)

def l2(m):
    return 0.5 * np.sum(m.reshape((1, np.prod(m.shape))) ** 2)

def d_l2(m):
    return m