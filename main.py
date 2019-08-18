import data
import model
import tensorflow as tf

N_EPOCHS = 2
BATCH_SIZE = 64
LR = 0.001
TN_INPUT_SIZE = model.LOCAL_INFO_SIZE + model.GLOBAL_INFO_SIZE
TN_OUTPUT_SIZE = 1
SN_INPUT_SIZE = data.input_size
SN_OUTPUT_SIZE = data.output_size

dataset = data.get_mnist_dataset(BATCH_SIZE)
inputs, targets = next(dataset)

sn_description = model.get_random_net_description()
tn_description = model.get_tn_description()

sn_weights = model.get_weights_from_description(sn_description, SN_INPUT_SIZE, SN_OUTPUT_SIZE)
tn_weights = model.get_weights_from_description(tn_description, TN_INPUT_SIZE, TN_OUTPUT_SIZE)

sn_output, sn_track = model.build_net(inputs, sn_weights)
sn_loss = model.get_loss(sn_output, targets)

tn_input, sizes = model.construct_tn_inputs(sn_track, sn_loss)
tn_output, _ = model.build_net(tn_input, tn_weights)
deltas_set = model.reconstruct_delta_weights(tn_output, sizes)

new_sn_weights = model.get_updated_weights(sn_weights, deltas_set)
new_sn_output, _ = model.build_net(inputs, new_sn_weights)
new_sn_loss = model.get_loss(new_sn_output, targets)

grad_sn_update_op = tf.optimizers.SGD(LR).minimize(sn_loss, var_list=sn_weights)
grad_tn_update_op = tf.optimizers.SGD(LR).minimize(new_sn_loss, var_list=tn_weights)
tn_sn_update_op = model.assign_updated_weights(sn_weights, new_sn_weights)

for epoch in range(N_EPOCHS):
    pass
