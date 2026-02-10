"""Plotting utilities for training curves and prediction scatter plots."""
import os
import numpy as np


def plot_train_test_loss(histories: list, loss_name: str = None,
                         val_loss_name: str = None, data_unit: str = "", model_name: str = "",
                         filepath: str = None, file_name: str = "", dataset_name: str = "",
                         figsize: list = None, dpi: float = None, show_fig: bool = True):
    """Plot training curves for a list of training history dictionaries.

    Each history is a dictionary mapping metric names to lists of values per epoch,
    as returned by a typical training loop or callback recorder.

    Args:
        histories (list): List of history dicts, e.g.
            [{"loss": [...], "val_loss": [...]}, ...] from k-fold splits.
        loss_name (str or list): Which training loss/metric key(s) to plot. If None, auto-detect.
        val_loss_name (str or list): Which validation loss/metric key(s) to plot. If None, auto-detect.
        data_unit (str or list): Unit string(s) for the loss values.
        model_name (str): Name of the model for the title.
        filepath (str): Directory to save the plot. If None, plot is not saved.
        file_name (str): Base file name for saving.
        dataset_name (str): Dataset name for the title.
        figsize (list): Figure size as [width, height].
        dpi (float): Resolution of the figure.
        show_fig (bool): Whether to display the figure.

    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    import matplotlib.pyplot as plt

    if data_unit is None:
        data_unit = ""
    if loss_name is None:
        loss_name = [x for x in list(histories[0].keys()) if "val_" not in x]
    if val_loss_name is None:
        val_loss_name = [x for x in list(histories[0].keys()) if "val_" in x]
    if not isinstance(loss_name, list):
        loss_name = [loss_name]
    if not isinstance(val_loss_name, list):
        val_loss_name = [val_loss_name]
    if not isinstance(data_unit, list):
        data_unit = [data_unit]

    if len(data_unit) < len(val_loss_name):
        data_unit = data_unit + [str(data_unit[-1])] * (len(val_loss_name) - len(data_unit))

    train_loss = []
    for x in loss_name:
        loss = np.array([np.array(hist[x]) for hist in histories])
        train_loss.append(loss)
    val_loss = []
    for x in val_loss_name:
        loss = np.array([np.array(hist[x]) for hist in histories])
        val_loss.append(loss)

    if figsize is None:
        figsize = [6.4, 4.8]
    if dpi is None:
        dpi = 100.0

    fig = plt.figure(figsize=figsize, dpi=dpi)
    for i, x in enumerate(train_loss):
        vp = plt.plot(np.arange(len(np.mean(x, axis=0))), np.mean(x, axis=0),
                       alpha=0.85, label=loss_name[i])
        plt.fill_between(np.arange(len(np.mean(x, axis=0))),
                         np.mean(x, axis=0) - np.std(x, axis=0),
                         np.mean(x, axis=0) + np.std(x, axis=0),
                         color=vp[0].get_color(), alpha=0.2)
    for i, y in enumerate(val_loss):
        val_step = len(train_loss[i][0]) / len(val_loss[i][0])
        vp = plt.plot(np.arange(len(np.mean(y, axis=0))) * val_step + val_step,
                       np.mean(y, axis=0), alpha=0.85, label=val_loss_name[i])
        plt.fill_between(
            np.arange(len(np.mean(y, axis=0))) * val_step + val_step,
            np.mean(y, axis=0) - np.std(y, axis=0),
            np.mean(y, axis=0) + np.std(y, axis=0),
            color=vp[0].get_color(), alpha=0.2)
        plt.scatter(
            [len(train_loss[i][0])], [np.mean(y, axis=0)[-1]],
            label=r"{0}: {1:0.4f} $\pm$ {2:0.4f} ".format(
                val_loss_name[i], np.mean(y, axis=0)[-1],
                np.std(y, axis=0)[-1]) + data_unit[i],
            color=vp[0].get_color())

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("%s training curve for %s" % (dataset_name, model_name))
    plt.legend(loc="upper right", fontsize="small")
    if filepath is not None:
        plt.savefig(os.path.join(filepath, model_name + "_" + dataset_name + "_" + file_name))
    if show_fig:
        plt.show()
    return fig


def plot_predict_true(y_predict, y_true, data_unit: list = None, model_name: str = "",
                      filepath: str = None, file_name: str = "", dataset_name: str = "",
                      target_names: list = None, figsize: list = None, dpi: float = None,
                      show_fig: bool = False, scaled_predictions: bool = False):
    """Make a scatter plot of predicted versus actual target values.

    Args:
        y_predict (np.ndarray): Predicted values of shape (N, n_targets) or (N,).
        y_true (np.ndarray): True values of shape (N, n_targets) or (N,).
        data_unit (list or str): Unit string(s) for each target.
        model_name (str): Name of the model.
        filepath (str): Directory to save the plot. If None, plot is not saved.
        file_name (str): Base file name for saving.
        dataset_name (str): Dataset name for the title.
        target_names (list or str): Name(s) for each target.
        figsize (list): Figure size as [width, height].
        dpi (float): Resolution of the figure.
        show_fig (bool): Whether to display the figure.
        scaled_predictions (bool): Whether predictions are standardized (adds label).

    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    import matplotlib.pyplot as plt

    y_predict = np.asarray(y_predict)
    y_true = np.asarray(y_true)

    if y_predict.ndim == 1:
        y_predict = np.expand_dims(y_predict, axis=-1)
    if y_true.ndim == 1:
        y_true = np.expand_dims(y_true, axis=-1)
    num_targets = y_true.shape[1]

    if data_unit is None:
        data_unit = ""
    if isinstance(data_unit, str):
        data_unit = [data_unit] * num_targets
    if target_names is None:
        target_names = ""
    if isinstance(target_names, str):
        target_names = [target_names] * num_targets

    if figsize is None:
        figsize = [6.4, 4.8]
    if dpi is None:
        dpi = 100.0

    fig = plt.figure(figsize=figsize, dpi=dpi)
    for i in range(num_targets):
        delta_valid = y_true[:, i] - y_predict[:, i]
        mae_valid = np.mean(np.abs(delta_valid[~np.isnan(delta_valid)]))
        plt.scatter(y_predict[:, i], y_true[:, i], alpha=0.3,
                    label=target_names[i] + " MAE: {0:0.4f} ".format(mae_valid) + "[" + data_unit[i] + "]")

    valid_mask = ~np.isnan(y_true)
    min_val = float(np.amin(y_true[valid_mask]))
    max_val = float(np.amax(y_true[valid_mask]))
    plt.plot(np.arange(min_val, max_val, 0.05), np.arange(min_val, max_val, 0.05), color="red")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plot_title = "Prediction of %s for %s " % (model_name, dataset_name)
    if scaled_predictions:
        plot_title = "(SCALED!) " + plot_title
    plt.title(plot_title)
    plt.legend(loc="upper left", fontsize="x-large")
    if filepath is not None:
        plt.savefig(os.path.join(filepath, model_name + "_" + dataset_name + "_" + file_name))
    if show_fig:
        plt.show()
    return fig
