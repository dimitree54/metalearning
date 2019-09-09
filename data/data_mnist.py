import numpy as np
import tensorflow as tf
from data.misc import one_hot

INPUT_SIZE = 784
OUTPUT_SIZE = 10
SHUFFLE_SEED = 0


def get_mnist_dataset(batch_size, train=True):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    num_images, h, w = x_train.shape
    assert INPUT_SIZE == h * w, "wrong INPUT_SIZE for MNIST"

    if train:
        x_train = np.array(x_train, np.float32) / 255
        x_train = np.reshape(x_train, [-1, INPUT_SIZE])
        y_train = one_hot(y_train, OUTPUT_SIZE)

        dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        dataset = dataset.shuffle(buffer_size=num_images, seed=SHUFFLE_SEED)
    else:
        x_test = np.array(x_test, np.float32) / 255
        x_test = np.reshape(x_test, [-1, INPUT_SIZE])
        y_test = one_hot(y_test, OUTPUT_SIZE)

        dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset
