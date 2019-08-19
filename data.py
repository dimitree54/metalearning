import numpy as np
import tensorflow as tf

input_size = 784
output_size = 10


def one_hot(data, depth):
    n = data.size
    result = np.zeros([n, depth], dtype=np.float32)
    result[np.arange(n), data] = 1
    return result


def get_mnist_dataset(batch_size):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    x_train = np.array(x_train, np.float32)
    x_train = np.reshape(x_train, [-1, input_size])
    y_train = one_hot(y_train, output_size)

    dataset_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    dataset_train = dataset_train.batch(batch_size)

    return dataset_train
