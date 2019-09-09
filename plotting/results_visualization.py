import matplotlib.pyplot as plt


def plot_output_target_diagram(targets, outputs):
    plt.scatter(targets, outputs, s=2)
    plt.xlabel('targets')
    plt.ylabel('outputs')
    plt.axis('equal')
    plt.axis('square')
    min_value = min(min(targets), min(outputs))
    max_value = max(max(targets), max(outputs))
    _ = plt.plot([min_value, max_value], [min_value, max_value], linewidth=1, color="black")
    plt.ylim([min_value, max_value])
    plt.xlim([min_value, max_value])
    plt.show()


def plot_error_distribution(targets, outputs):
    error = outputs - targets
    plt.hist(error, bins=200, log=True)
    plt.xlabel("Prediction Error [MPG]")
    _ = plt.ylabel("Count")
    plt.show()
