from options import *
import os
import time
from model.tn_model import *
from data.data_mnist import get_mnist_dataset


@tf.function
def train_target_weights(init_sn_weights, optimizer):
    mnist_dataset = get_mnist_dataset(TARGET_WEIGHTS_GENERATION_BATCH_SIZE).repeat(TARGET_WEIGHTS_GENERATION_N_EPOCHS).\
        prefetch(tf.data.experimental.AUTOTUNE)
    for inputs, targets in mnist_dataset:
        with tf.device('/device:GPU:0'):
            with tf.GradientTape() as tape:
                outputs, sn_track = net(inputs, init_sn_weights)
                sn_loss = get_loss(outputs, targets)
            sn_gradients = tape.gradient(sn_loss, init_sn_weights)
            optimizer.apply_gradients(zip(sn_gradients, init_sn_weights))
    return init_sn_weights


def get_target_weights(sn_proto):
    if os.path.exists("target_sn_weights.npz"):
        print("Loading weights from file")
        init_sn_weights = load_net_weights("target_sn_weights.npz")
    else:
        print("Creating new target weights")
        init_sn_weights = get_weights_from_proto(sn_proto, WEIGHTS_SIGMA)
        optimizer = tf.optimizers.Adam()
        init_sn_weights = train_target_weights(init_sn_weights, optimizer)
        save_net_weights("target_sn_weights.npz", init_sn_weights)
    return init_sn_weights


@tf.function
def get_tn_data(batch_inputs, batch_targets, sn_weights, optimizer):
    with tf.GradientTape(persistent=True) as tape:
        batch_outputs, batch_sn_track = net(batch_inputs, sn_weights)
        batch_sn_loss = get_loss(batch_outputs, batch_targets)
        batch_sn_loss_separated = tf.unstack(batch_sn_loss)

        # it is not good to calc gradients inside the tape, but otherwise tf.gather can not track gradients
        batch_sn_gradients = tf.map_fn(
            lambda i: tape.gradient(tf.gather(batch_sn_loss_separated, i), sn_weights),
            tf.range(TN_DATA_GENERATION_BATCH_SIZE), back_prop=False,
            parallel_iterations=TN_DATA_GENERATION_BATCH_SIZE,
            dtype=[tf.float32] * len(sn_weights))

    batch_tn_inputs = construct_tn_inputs(batch_sn_track, batch_sn_loss)
    batch_tn_outputs = construct_tn_outputs(batch_sn_gradients, batch_size=None, gradients_batched=True)

    batch_tn_inputs = tf.concat(batch_tn_inputs, axis=0)
    batch_tn_outputs = tf.concat(batch_tn_outputs, axis=0)
    batch_tn_outputs = tf.expand_dims(batch_tn_outputs, axis=-1)

    gather_mask = tf.random.uniform(shape=[tf.shape(batch_tn_outputs)[0]]) > TN_DATA_GENERATION_DROP_RATE
    batch_tn_inputs = tf.boolean_mask(batch_tn_inputs, gather_mask, axis=0)
    batch_tn_outputs = tf.boolean_mask(batch_tn_outputs, gather_mask, axis=0)

    integrated_sn_gradients = [tf.reduce_sum(layer_gradients, axis=0) for layer_gradients in batch_sn_gradients]
    optimizer.apply_gradients(zip(integrated_sn_gradients, sn_weights))

    return batch_tn_inputs, batch_tn_outputs


def get_training_data(sn_proto, init_sn_weights):
    if os.path.exists("tn_data.npz"):
        print("Loading training data")
        data = np.load("tn_data.npz")
        tn_inputs = data["tn_inputs"]
        tn_outputs = data["tn_outputs"]
    else:
        print("Generating new training data")
        tn_inputs = np.zeros(shape=(TN_DATA_SIZE, TN_INPUT_SIZE))
        tn_outputs = np.zeros(shape=(TN_DATA_SIZE, TN_OUTPUT_SIZE))
        data_pointer = 0

        optimizer = tf.optimizers.SGD()
        mnist_dataset = get_mnist_dataset(batch_size=TN_DATA_GENERATION_BATCH_SIZE)
        data_ready = False
        epoch = 0

        sn_weights = [tf.Variable(init_sn_weights[i]) for i in range(len(init_sn_weights))]
        noise_generator = tf.initializers.RandomNormal(stddev=WEIGHTS_SIGMA)

        t1 = time.time()
        t2 = time.time()
        while not data_ready:
            print(epoch, "reset_weights, data_pointer:", data_pointer, time.time() - t2)
            t2 = time.time()
            epoch += 1

            for i in range(len(sn_weights)):
                sn_weights[i].assign(init_sn_weights[i] + noise_generator(shape=[sn_proto[i][0], sn_proto[i][1]]))

            dataset = mnist_dataset.take(TN_DATA_GENERATION_RESET_STEPS // TN_DATA_GENERATION_BATCH_SIZE)
            for (sn_inputs, sn_targets) in dataset:
                tn_inputs_candidate, tn_outputs_candidate = get_tn_data(sn_inputs, sn_targets, sn_weights, optimizer)
                tn_inputs_candidate, tn_outputs_candidate = tn_inputs_candidate.numpy(), tn_outputs_candidate.numpy()

                start_index = data_pointer
                end_index = min(TN_DATA_SIZE, start_index + len(tn_inputs_candidate))

                tn_inputs[start_index: end_index] = tn_inputs_candidate[:end_index - start_index]
                tn_outputs[start_index: end_index] = tn_outputs_candidate[:end_index - start_index]

                data_pointer = end_index

                if data_pointer == TN_DATA_SIZE:
                    data_ready = True
                    break
        print("data ready in {} hours".format((time.time() - t1) / 60 / 60))
        np.savez_compressed("tn_data.npz", tn_inputs=tn_inputs, tn_outputs=tn_outputs)

    return tn_inputs, tn_outputs
