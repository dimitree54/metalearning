import numpy as np


def one_hot(data, depth):
    n = data.size
    result = np.zeros([n, depth], dtype=np.float32)
    result[np.arange(n), data] = 1
    return result
