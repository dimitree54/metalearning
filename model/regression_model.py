from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from options import *


def build_keras_model_from_description(tn_description):
    if len(tn_description) > 2:
        layers_list = [layers.Dense(tn_description[1], activation=tf.nn.relu, input_shape=[tn_description[0]])]
        for i in range(2, len(tn_description) - 1):
            layer_size = tn_description[i]
            layers_list.append(layers.Dense(layer_size, activation=tf.nn.relu))
        layers_list.append(layers.Dense(tn_description[-1]))
        optimizer = tf.optimizers.Adam()
    else:
        # linear regression case: no relu and simple optimizer
        layers_list = [layers.Dense(tn_description[1], input_shape=[tn_description[0]])]
        optimizer = tf.optimizers.SGD()

    model = keras.Sequential(layers_list)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mse'])
    return model


def train_model(model, train_dataset, val_dataset, verbose=1):
    train_dataset = train_dataset.repeat(TN_TRAINING_N_EPOCHS // TN_N_TESTS).batch(TN_BATCH_SIZE).\
        prefetch(tf.data.experimental.AUTOTUNE)
    val_dataset = val_dataset.batch(TN_BATCH_SIZE)
    history = model.fit(x=train_dataset, validation_data=val_dataset, epochs=TN_N_TESTS, verbose=verbose)
    return history
