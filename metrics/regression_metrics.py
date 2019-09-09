import sklearn.metrics
from plotting.train_visualization import plot_dict
from plotting.results_visualization import plot_error_distribution, plot_output_target_diagram
from options import *
from data.data_mnist import get_mnist_dataset
from model.tn_model import *
from data.data_preprocessing import normalize, denormalize

from tqdm import tqdm


# @tf.function
def train_adversarial(model, init_sn_weights, sn_proto, inputs_mean, inputs_std, targets_mean, targets_std):
    inputs_mean = tf.constant(inputs_mean, tf.float32)
    inputs_std = tf.constant(inputs_std, tf.float32)
    targets_mean = tf.constant(targets_mean, tf.float32)
    targets_std = tf.constant(targets_std, tf.float32)

    optimizer = tf.optimizers.SGD()
    bs = 2
    mnist_dataset = get_mnist_dataset(batch_size=bs, train=False)

    weights_noise = get_weights_from_proto(sn_proto, sigma=WEIGHTS_SIGMA)
    grad_sn_weights = [tf.Variable(weights_noise[i] + init_sn_weights[i]) for i in range(len(weights_noise))]
    meta_sn_weights = [tf.Variable(weights_noise[i] + init_sn_weights[i]) for i in range(len(weights_noise))]

    history = {"grad_loss": [], "meta_loss": []}

    for inputs, targets in tqdm(mnist_dataset, total=10000 // bs):
        with tf.GradientTape() as tape:
            grad_outputs, _ = net(inputs, grad_sn_weights)
            grad_sn_loss = get_loss(grad_outputs, targets)

        sn_gradients = tape.gradient(grad_sn_loss, grad_sn_weights)
        optimizer.apply_gradients(zip(sn_gradients, grad_sn_weights))

        meta_outputs, sn_track = net(inputs, meta_sn_weights)
        meta_sn_loss = get_loss(meta_outputs, targets)

        tn_inputs = construct_tn_inputs(sn_track, meta_sn_loss)
        tn_inputs = [normalize(tn_input, inputs_mean, inputs_std) for tn_input in tn_inputs]
        tn_outputs = [model(tn_input) for tn_input in tn_inputs]
        tn_outputs = [denormalize(tn_output, targets_mean, targets_std) for tn_output in tn_outputs]
        meta_gradients = reconstruct_delta_weights(tn_outputs, tf.shape(inputs)[0], sn_proto)
        optimizer.apply_gradients(zip(meta_gradients, meta_sn_weights))

        history["grad_loss"].append(grad_sn_loss)
        history["meta_loss"].append(meta_sn_loss)

    return history


def adversarial_training(model, init_sn_weights, sn_proto, inputs_mean, inputs_std, targets_mean, targets_std):
    print("Start adversarial training")
    history = train_adversarial(model, init_sn_weights, sn_proto, inputs_mean, inputs_std, targets_mean, targets_std)
    print("Training loss comparison:")
    plot_dict(history, ("grad_loss", "meta_loss"))


def evaluate_dataset(model, dataset):
    outputs = []
    targets = []
    for test_input, test_target in dataset:
        test_outputs = model(test_input)
        outputs.append(test_outputs.numpy())
        targets.append(test_target.numpy())
    outputs = np.concatenate(outputs, axis=0).flatten()
    targets = np.concatenate(targets, axis=0).flatten()
    return outputs, targets


def eval_regression(model, test_dataset):
    test_dataset = test_dataset.batch(TN_BATCH_SIZE)
    outputs, targets = evaluate_dataset(model, test_dataset)

    print("r2_score:", sklearn.metrics.r2_score(targets, outputs))
    print("median_absolute_error:", sklearn.metrics.median_absolute_error(targets, outputs))
    print("mean_absolute_error:", sklearn.metrics.mean_absolute_error(targets, outputs))
    print("mean_squared_error:", sklearn.metrics.mean_squared_error(targets, outputs))
    print("max_error:", sklearn.metrics.max_error(targets, outputs))
    print("explained_variance_score:", sklearn.metrics.explained_variance_score(targets, outputs))

    print("output_target_diagram:")
    plot_output_target_diagram(targets, outputs)

    print("error_distribution:")
    plot_error_distribution(targets, outputs)
