import tensorflow as tf
import random
import numpy as np


def get_random_net_description(input_size, output_size, max_layer_size, max_num_layers, seed=None):
    """
    Generate random description. Description is a list of layer sizes (first number - input size)
    """
    net_description = [input_size]
    random.seed(seed)
    num_layers = random.randint(1, max_num_layers)
    for _ in range(num_layers):
        net_description.append(random.randint(1, max_layer_size))
    net_description.append(output_size)
    return net_description


def get_net_proto_from_description(description):
    """
    Generate net proto from description. Net proto is a list of layer's weights sizes.
    """
    net_proto = []
    in_size = description[0]
    for i in range(1, len(description)):
        out_size = description[i]
        net_proto.append((in_size + 1, out_size))  # +1 to emulate bias
        in_size = out_size
    return net_proto


def get_weights_from_proto(net_proto, sigma, seed=None):
    """
    Generate weight tensors in accordance with net description.
    """
    net_weights = []
    initializer = tf.initializers.RandomNormal(stddev=sigma, seed=seed)
    for size in net_proto:
        weights = tf.Variable(
            initial_value=initializer(shape=[size[0], size[1]]),
            trainable=True,
            dtype=tf.float32
        )
        net_weights.append(weights)
    return net_weights


def fc_layer(inputs, weights, activation=tf.sigmoid):
    outputs = tf.matmul(inputs, weights)
    return activation(outputs)


def save_net_weights(path, net_weights):
    net_weights_numpy = [net_weights_tensor.numpy() for net_weights_tensor in net_weights]
    np.savez_compressed(path, *net_weights_numpy)


def load_net_weights(path):
    net_weights_numpy = np.load(path)
    net_weights_tensor = []
    for name in net_weights_numpy:
        net_weights_tensor.append(tf.Variable(initial_value=net_weights_numpy[name]))
    return net_weights_tensor


@tf.function
def net(inputs, weights_set):
    """
    Build sequential network from weights_set
    :return: output, track
     output - is an output tensor of net
     track - is a list which stores information about internal computations:
      for each layer it stores a tuple of its (input, weights, output)
    """
    track = []
    net_in = inputs
    for weights in weights_set:
        net_in = tf.pad(net_in, [[0, 0], [0, 1]], constant_values=1)  # for bias. [bs,n]->[bs,n+1]
        net_out = fc_layer(net_in, weights)
        track.append((net_in, weights, net_out))
        net_in = net_out
    return net_in, track


def tn(track, loss, tn_weights):
    tn_inputs = construct_tn_inputs(track, loss)
    tn_outputs = [net(tn_inputs[i], tn_weights)[0] for i in range(len(tn_inputs))]
    tn_outputs = reconstruct_delta_weights(tn_outputs)
    return tn_outputs


@tf.function
def construct_tn_outputs(gradients, batch_size, gradients_batched=False):
    def construct_tn_outputs_for_layer(layer_gradients):
        extended_layer_gradients = tf.expand_dims(layer_gradients, axis=0)  # [n,m] -> [1,n,m]
        extended_layer_gradients = tf.tile(extended_layer_gradients, [batch_size, 1, 1])  # [1,n,m] -> [bs,n,m]
        return extended_layer_gradients  # [bs,n,m]

    tn_outputs = []
    for i in range(len(gradients)):
        if gradients_batched:
            tn_outputs_for_layer = gradients[i]
        else:
            tn_outputs_for_layer = construct_tn_outputs_for_layer(gradients[i])
        tn_outputs_for_layer = tf.reshape(tn_outputs_for_layer, shape=[-1])

        tn_outputs.append(tn_outputs_for_layer)

    return tn_outputs


@tf.function
def construct_tn_inputs(track, loss):
    """
    Convert sn track to tn_input by combining for each trainable weight scalar its
     input, value, output (after activation function) and loss. I.e. tn_input will have shape [big_bs, 4], where
     big_bs=bs*n*m, where bs-batch size used in sn, n-layer input size, m-layer output_size
    :param track: track is a list of tuples (inputs, weights, outputs)
     inputs has shape [bs, n]
     weights has shape [n, m]
     outputs has shape [bs, m]
    :param loss: loss is a tensor with shape [bs, 1]
    :return: tn_inputs and sizes
     tn_inputs is a list of tn_inputs tensors for each layer. For each layer it has shape [bs*n*m,4]
     sizes is a list of shapes [bs,n,m,4] for each layer. It will simplify weights reconstruction after tn.
    """
    def construct_tn_inputs_for_layer(layer_track):
        inputs, weights, outputs = layer_track
        n, m = weights.shape

        def construct_tn_inputs_for_batch_sample(sample_info):
            sample_inputs, sample_outputs, sample_loss = sample_info

            extended_sample_input = tf.expand_dims(sample_inputs, axis=1)  # [n] -> [n,1]
            extended_sample_input = tf.tile(extended_sample_input, [1, m])  # [n,1] -> [n,m]

            extended_sample_outputs = tf.expand_dims(sample_outputs, axis=0)  # [m] -> [1,m]
            extended_sample_outputs = tf.tile(extended_sample_outputs, [n, 1])  # [1,m] -> [n,m]

            extended_sample_loss = tf.expand_dims(sample_loss, axis=0)  # [] -> [1]
            extended_sample_loss = tf.expand_dims(extended_sample_loss, axis=0)  # [1] -> [1,1]
            extended_sample_loss = tf.tile(extended_sample_loss, [n, m])  # [1,1] -> [n,m]

            tn_input_for_batch_sample = tf.stack(
                [extended_sample_input, weights, extended_sample_outputs, extended_sample_loss], axis=-1)
            return tn_input_for_batch_sample  # [n,m,4]

        tn_inputs_per_batch_sample = tf.map_fn(
            lambda sample_info: construct_tn_inputs_for_batch_sample(sample_info),
            [inputs, outputs, loss], dtype=tf.float32)
        return tn_inputs_per_batch_sample  # [bs,n,m,4]

    tn_inputs = []
    for i in range(len(track)):
        tn_inputs_for_layer = construct_tn_inputs_for_layer(track[i])
        tn_inputs_for_layer = tf.reshape(tn_inputs_for_layer, shape=[-1, 4])

        tn_inputs.append(tn_inputs_for_layer)

    return tn_inputs


@tf.function
def reconstruct_delta_weights(tn_output, batch_size, sn_proto):
    """
    Reverts construct_tn_inputs operation but applied for tn_output. Also averages delta weights by batch dimension.
    :return: list of weight updated for each layer
    """
    delta_weights = []
    for i in range(len(sn_proto)):
        n, m = sn_proto[i]
        layer_delta_weights = tf.reshape(tn_output[i], [batch_size, n, m])
        layer_delta_weights = tf.reduce_mean(layer_delta_weights, axis=0)
        delta_weights.append(layer_delta_weights)
    return delta_weights


@tf.function
def get_loss(outputs, targets):
    return tf.sqrt(tf.reduce_sum(tf.square(outputs - targets), axis=1))
