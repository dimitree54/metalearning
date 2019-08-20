import data
import model
import tensorflow as tf
from tqdm import tqdm

N_EPOCHS = 10
BATCH_SIZE = 10
LR = 0.0001
TN_INPUT_SIZE = model.LOCAL_INFO_SIZE + model.GLOBAL_INFO_SIZE
TN_OUTPUT_SIZE = 1
SN_INPUT_SIZE = data.input_size
SN_OUTPUT_SIZE = data.output_size

dataset = data.get_mnist_dataset(BATCH_SIZE)

sn_description = model.get_random_net_description(0)
tn_description = model.get_tn_description()

tn_weights = model.get_weights_from_description(tn_description, TN_INPUT_SIZE, TN_OUTPUT_SIZE)

optimizer = tf.optimizers.SGD(LR)

summary_writer_grad = tf.summary.create_file_writer('./log/grad')
summary_writer_tn_grad = tf.summary.create_file_writer('./log/tn_grad')
summary_writer_tn_pure = tf.summary.create_file_writer('./log/tn_pure')

for epoch in range(N_EPOCHS):

    sn_weights = model.get_weights_from_description(sn_description, SN_INPUT_SIZE, SN_OUTPUT_SIZE, seed=epoch)

    for step, (inputs, targets) in tqdm(enumerate(iter(dataset)), total=60000 // BATCH_SIZE, desc="epoch {}".format(epoch)):
        with tf.GradientTape() as tape:
            sn_output, sn_track = model.net(inputs, sn_weights)
            sn_loss = model.get_loss(sn_output, targets)

        sn_gradients = tape.gradient(sn_loss, sn_weights)
        optimizer.apply_gradients(zip(sn_gradients, sn_weights))

        with tf.GradientTape() as tape:
            deltas_set = model.tn(sn_track, sn_loss, tn_weights)
            new_sn_weights = model.get_updated_weights(sn_weights, deltas_set, lr=1)
            new_sn_output, _ = model.net(inputs, new_sn_weights)
            new_sn_loss = model.get_loss(new_sn_output, targets)

        tn_gradients = tape.gradient(new_sn_loss, tn_weights)
        optimizer.apply_gradients(zip(tn_gradients, tn_weights))

        with summary_writer_grad.as_default():
            tf.summary.scalar('loss{}'.format(epoch), tf.reduce_mean(sn_loss), step=step)
        with summary_writer_tn_grad.as_default():
            tf.summary.scalar('loss{}'.format(epoch), tf.reduce_mean(new_sn_loss), step=step)
            tf.summary.scalar('mean delta', tf.reduce_mean(new_sn_output), step=step)

    sn_weights = model.get_weights_from_description(sn_description, SN_INPUT_SIZE, SN_OUTPUT_SIZE, seed=epoch)

    for step, (inputs, targets) in tqdm(enumerate(iter(dataset)), total=60000 // BATCH_SIZE, desc="epoch {}, pure".format(epoch)):
        sn_output, sn_track = model.net(inputs, sn_weights)
        sn_loss = model.get_loss(sn_output, targets)

        deltas_set = model.tn(sn_track, sn_loss, tn_weights)
        new_sn_weights = model.get_updated_weights(sn_weights, deltas_set, lr=LR)
        sn_weights = new_sn_weights  # for pure metalearning

        with summary_writer_tn_pure.as_default():
            tf.summary.scalar('loss{}'.format(epoch), tf.reduce_mean(sn_loss), step=step)

sn_weights = model.get_weights_from_description(sn_description, SN_INPUT_SIZE, SN_OUTPUT_SIZE, seed=N_EPOCHS + 1)

for step, (inputs, targets) in tqdm(enumerate(iter(dataset)), total=60000 // BATCH_SIZE, desc="final pure"):
    sn_output, sn_track = model.net(inputs, sn_weights)
    sn_loss = model.get_loss(sn_output, targets)

    deltas_set = model.tn(sn_track, sn_loss, tn_weights)
    new_sn_weights = model.get_updated_weights(sn_weights, deltas_set, lr=LR)
    sn_weights = new_sn_weights  # for pure metalearning

    with summary_writer_tn_pure.as_default():
        tf.summary.scalar('loss_final', tf.reduce_mean(sn_loss), step=step)
