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
    (x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()

    x_train = np.array(x_train, np.float32) / 255
    x_train = np.reshape(x_train, [-1, input_size])
    x_train = x_train[:1000]  # TODO debug line
    y_train = one_hot(y_train, output_size)
    y_train = y_train[:1000]  # TODO debug line

    dataset_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    dataset_train = dataset_train.batch(batch_size)

    return dataset_train


class TrackWriter:
    def __init__(self, tfrecord_path):
        self.writer = tf.data.experimental.TFRecordWriter(tfrecord_path)

    def write(self, inputs, outputs, track):
        serialized_sample = tf.train.Example(
            features=tf.train.Features(
                feature={"inputs": tf.train.Feature(float_list=tf.train.FloatList(value=inputs)),
                         "outputs": tf.train.Feature(float_list=tf.train.FloatList(value=outputs)),
                         "track": tf.train.Feature(float_list=tf.train.FloatList(value=track))}
            )
        ).SerializeToString()
        self.writer.write(serialized_sample)


def get_track_dataset(tfrecord_path, batch_size):
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.batch(batch_size)
    return dataset
