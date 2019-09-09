from model.tn_model import *
from model.regression_model import *
from plotting.train_visualization import plot_history
from metrics.regression_metrics import eval_regression, adversarial_training
from options import *

from data.tn_data_generator import get_training_data, get_target_weights
from data.data_preprocessing import *
import tensorflow as tf
import os

sn_proto = get_net_proto_from_description(SN_DESCRIPTION)

target_weights = get_target_weights(sn_proto)
inputs, targets = get_training_data(sn_proto, target_weights)

inputs_mean, inputs_std = get_mean_std(inputs)
inputs = normalize(inputs, inputs_mean, inputs_std)
targets_mean, targets_std = get_mean_std(targets)
targets = normalize(targets, targets_mean, targets_std)
(train_inputs, train_targets), (val_inputs, val_targets) = split(inputs, targets)
(val_inputs, val_targets), (test_inputs, test_targets) = split(val_inputs, val_targets)

train_dataset = tf.data.Dataset.from_tensor_slices((train_inputs, train_targets))
val_dataset = tf.data.Dataset.from_tensor_slices((val_inputs, val_targets))
test_dataset = tf.data.Dataset.from_tensor_slices((test_inputs, test_targets))

for model_name in TN_DESCRIPTIONS:
    file_name = "{}.h5".format(model_name)

    if os.path.exists(file_name):
        model = keras.models.load_model(file_name)
        print("Previously trained model loaded")
    else:
        print("Start training {} model".format(model_name))
        model = build_keras_model_from_description(TN_DESCRIPTIONS[model_name])
        history = train_model(model, train_dataset, val_dataset)
        plot_history(history)
        model.save("{}.h5".format(model_name))

    print("{} MODEL RESULTS:".format(model_name))
    eval_regression(model, test_dataset)
    adversarial_training(model, target_weights, sn_proto, inputs_mean, inputs_std, targets_mean, targets_std)
