import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def plot_history(history, names_to_plot=("mse", "val_mse")):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure(figsize=(12, 8))

    plt.xlabel('epoch')
    max_value = 0
    for name in names_to_plot:
        max_value = max(max_value, max(hist[name]))
        plt.plot(hist['epoch'], hist[name], label=name)
    plt.ylim([0, max_value])
    plt.legend()
    plt.show()


def plot_dict(history, names_to_plot=("mse", "val_mse")):
    hist = pd.DataFrame(history)
    hist['epoch'] = np.arange(0, len(history[names_to_plot[0]]))

    plt.figure(figsize=(12, 8))

    plt.xlabel('epoch')
    max_value = 0
    for name in names_to_plot:
        max_value = max(max_value, max(hist[name]))
        plt.plot(hist['epoch'], hist[name], label=name)
    plt.ylim([0, max_value])
    plt.legend()
    plt.show()
