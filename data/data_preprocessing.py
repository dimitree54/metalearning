import numpy as np


def get_mean_std(data):
    return np.mean(data, axis=0), np.std(data, axis=0)


def normalize(data, mean, std):
    return (data - mean) / std


def denormalize(data, mean, std):
    return data * std + mean


def split(inputs, targets, rate=0.9):
    concat = np.concatenate([inputs, targets], axis=1)
    np.random.shuffle(concat)
    split_ind = int(len(concat) * rate)
    training_concat, test_concat = concat[:split_ind, :], concat[split_ind:, :]
    training_inputs, training_targets = training_concat[:, :-1],  training_concat[:, -1:]
    test_inputs, test_targets = test_concat[:, :-1],  test_concat[:, -1:]
    return (training_inputs, training_targets), (test_inputs, test_targets)
