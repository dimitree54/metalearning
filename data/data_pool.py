from threading import Lock
from random import randint, random
import numpy as np
from options import TN_INPUT_SIZE, TN_OUTPUT_SIZE


class DataPool:
    def __init__(self, size, drop_rate, data_size=TN_INPUT_SIZE + TN_OUTPUT_SIZE):
        self.size = size
        self.drop_rate = drop_rate
        self.pool = np.zeros(shape=(size, data_size), dtype=np.float32)
        self.locks = [Lock()] * size

    def put(self, data):
        if random() < self.drop_rate:
            return
        ind = randint(0, self.size - 1)
        with self.locks[ind]:
            self.pool[ind] = data

    def get(self):
        ind = randint(0, self.size - 1)
        with self.locks[ind]:
            return self.pool[ind]

    def put_several(self, data_iterable):
        for data in data_iterable:
            self.put(data)

    def generator(self):
        while True:
            sample = self.get()
            yield (sample[:TN_INPUT_SIZE], sample[TN_INPUT_SIZE:])

    def save(self):
        self.pool.tofile("pool.npy")
