import tensorflow as tf
import random

MAX_LAYER_SIZE = 128
MAX_NUM_LAYERS = 10
WEIGHTS_SIGMA = 0.1

LOCAL_INFO_SIZE = 3
GLOBAL_INFO_SIZE = 1


# net description is a list of num_units in hidden layers.
def get_tn_description():
    return [128, 128, 128]


def get_random_net_description():
    """
    Generate random description
    """
    net_description = []
    num_layers = random.randint(1, MAX_NUM_LAYERS)
    for _ in range(num_layers):
        net_description.append(random.randint(1, MAX_LAYER_SIZE))
    return net_description


def get_weights_from_description(net_description, input_size, output_size):
    """
    Generate weight tensors in accordance with net description.
     WARNING! to emulate bias each weight matrix have one extra row.
    :return: list of weight tensors
    """
    extended_net_description = net_description.copy()
    extended_net_description.append(output_size)
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
    return weights_set


def fc_layer(inputs, weights):
    outputs = tf.matmul(inputs, weights)
    return tf.sigmoid(outputs)


def build_net(inputs, weights_set):
    """
    Build sequential network from weights_se
    :return: output, track
     output - is an output tensor of net
     track - is a list which stores information about internal computations:
      for each layer it stores a tuple of its (input, weights, output)
    """
    track = []
    net_in = inputs
    for weights in weights_set:
        net_in = tf.pad(inputs, [[0, 0], [0, 1]], constant_values=1)  # for bias. [bs,n]->[bs,n+1]
        net_out = fc_layer(net_in, weights)
        track.append((net_in, weights, net_out))
        net_in = net_out
    return net_in, track


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

        def construct_tn_inputs_for_batch_sample(sample_inputs, sample_outputs, sample_loss):
            extended_sample_input = tf.expand_dims(sample_inputs, axis=1)  # [n] -> [n,1]
            extended_sample_input = tf.tile(extended_sample_input, [1, m])  # [n,1] -> [n,m]

            extended_sample_outputs = tf.expand_dims(sample_outputs, axis=0)  # [m] -> [1,m]
            extended_sample_outputs = tf.tile(extended_sample_outputs, [n, 1])  # [1,m] -> [n,m]

            extended_sample_loss = tf.expand_dims(sample_loss, axis=0)  # [1] -> [1,1]
            extended_sample_loss = tf.tile(extended_sample_loss, [n, m])  # [1,1] -> [n,m]

            tn_input_for_batch_sample = tf.stack(
                [extended_sample_input, weights, extended_sample_outputs, extended_sample_loss], axis=-1)
            return tn_input_for_batch_sample  # [n,m,4]

        tn_inputs_per_batch_sample = tf.map_fn(
            lambda s_in, s_out, s_loss: construct_tn_inputs_for_batch_sample(s_in, s_out, s_loss),
            [inputs, outputs, loss], dtype=tf.float32)
        return tn_inputs_per_batch_sample  # [bs,n,m,4]

    tn_inputs = []
    sizes = []
    for i in range(len(track)):
        tn_inputs_for_layer = construct_tn_inputs_for_layer(track[i])
        sizes.append(tn_inputs_for_layer.shape)

        # TODO check the order of reshape in both directions.
        tn_inputs_for_layer = tf.reshape(tn_inputs_for_layer, shape=[-1, 4])
        tn_inputs.append(tn_inputs_for_layer)

    return tn_inputs, sizes


def reconstruct_delta_weights(tn_output, sizes):
    """
    Reverts construct_tn_inputs operation but applied for tn_output. Also averages delta weights by batch dimension.
    :return: list of weight updated for each layer
    """
    delta_weights = []
    for layer_size in sizes:
        bs, n, m, _ = layer_size
        layer_delta_weights = tf.reshape(tn_output, [bs, n, m])
        layer_delta_weights = tf.reduce_mean(layer_delta_weights, axis=0)
        delta_weights.append(layer_delta_weights)
    return delta_weights


def get_updated_weights(weights_set, deltas_set):
    """
    returns a list of tensors with updated weights. WARNING! this function does not update weights variables, just
     returns new values. To store this values call assign_updated_weights.
    """
    updated_weights_set = []
    for i in range(len(weights_set)):
        weights = weights_set[i]
        deltas = deltas_set[i]
        updated_weights = weights + deltas
        updated_weights_set.append(updated_weights)
    return updated_weights_set


def assign_updated_weights(weights_set, new_weights_set):
    """
    returns op that updates weight variables with new values
    """
    # TODO does this function work? In tf2.0 assign returns not an op, but tensor.
    assign_ops = []
    for i in range(len(weights_set)):
        weights = weights_set[i]
        new_weights = new_weights_set[i]
        assign_op = weights.assign(new_weights)
        assign_ops.append(assign_op)
    return tf.group(assign_ops)


def get_loss(outputs, targets):
    return tf.sqrt(tf.reduce_sum(tf.square(outputs - targets), axis=1))
