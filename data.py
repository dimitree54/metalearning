import numpy as np
import tensorflow as tf

input_size = 784
output_size = 10


def get_mnist_dataset(batch_size):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist
    
    x_train = np.array(x_train, np.float32)
    y_train = np.array(y_train, np.float32)

    dataset_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    dataset_train = dataset_train.batch(batch_size)

    iterator = dataset_train.make_one_shot_iterator()
    next_element = iterator.get_next()

    return next_element
