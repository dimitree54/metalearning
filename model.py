import tensorflow as tf
import numpy as np
import random

MAX_LAYER_SIZE = 128
MAX_NUM_LAYERS = 10
WEIGHTS_SIGMA = 0.1

LOCAL_INFO_SIZE = 3
GLOBAL_INFO_SIZE = 1


# net description is a list of num_units in hidden layers
def get_tn_description():
    return [128, 128, 128]


def gen_random_net_description():
    net_description = []
    num_layers = random.randint(1, MAX_NUM_LAYERS)
    for _ in num_layers:
        net_description.append(random.randint(1, MAX_LAYER_SIZE))
    return net_description


def get_weights_np_from_description(net_description, input_size, output_size):
    weights_set = []
    in_size = input_size
    for out_size in net_description:
        weights_set.append(np.random.normal(0, WEIGHTS_SIGMA, size=(in_size + 1, out_size)))
        in_size = out_size
    weights_set.append(np.random.normal(0, WEIGHTS_SIGMA, size=(in_size + 1, output_size)))
    return weights_set


def get_weights_tf_from_description(net_description, input_size, output_size):
    weights_set = []
    in_size = input_size
    for out_size in net_description:
        weights = tf.Variable(
            initial_value=tf.initializers.GlorotNormal()(shape=[in_size + 1, out_size]),
            trainable=True,
            dtype=tf.float32
        )
        weights_set.append(weights)
        in_size = out_size
    weights = tf.Variable(
        initial_value=tf.initializers.GlorotNormal()(shape=[in_size + 1, output_size]),
        trainable=True,
        dtype=tf.float32
    )
    weights_set.append(weights)
    return weights_set


def fc_layer(inputs, weights):
    """
    inputs shape is [bs, n]
    weights shape is [n + 1, m]
    """
    # appending 1 to input to emulate bias:
    biased_inputs = tf.pad(inputs, [[0, 0], [0, 1]], constant_values=1)
    outputs = tf.matmul(biased_inputs, weights)
    return tf.sigmoid(outputs)


def build_net(inputs, weights_set):
    track = []
    net_in = inputs
    for weights in weights_set:
        net_out = fc_layer(net_in, weights)
        track.append((net_in, weights, net_out))
        net_in = net_out
    return net_in, track


# net = tf.concat([local_input, global_input], axis=1)


def flatten_net_track(track):
    return None


def reconstruct_delta_weights(tn_output):
    return None


def get_global_info(loss):
    return None


def loss(outputs, targets):
    return 0
