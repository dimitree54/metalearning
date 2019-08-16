import data
import model
import tensorflow as tf

BATCH_SIZE = 64
TN_INPUT_SIZE = model.LOCAL_INFO_SIZE + model.GLOBAL_INFO_SIZE
TN_OUTPUT_SIZE = 1
SN_INPUT_SIZE = data.input_size
SN_OUTPUT_SIZE = data.output_size

inputs, targets = data.get_mnist_dataset(BATCH_SIZE)

sn_description = model.gen_random_net_description()
tn_description = model.get_tn_description()

sn_weights = model.get_weights_tf_from_description(sn_description, SN_INPUT_SIZE, SN_OUTPUT_SIZE)
tn_weights = model.get_weights_tf_from_description(tn_description, TN_INPUT_SIZE, TN_OUTPUT_SIZE)

sn_output, sn_track = model.build_net(inputs, sn_weights)
sn_loss = model.loss(sn_output, targets)

