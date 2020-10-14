import numpy as np


def multiclass_log_loss(y_true, y_pred, eps=1e-15):
    "based on this post: https://www.kaggle.com/c/predict-closed-questions-on-stack-overflow/discussion/2644"


    """Multi class version of Logarithmic Loss metric.
    https://www.kaggle.com/wiki/MultiClassLogLoss

    idea from this post:
    http://www.kaggle.com/c/emc-data-science/forums/t/2149/is-anyone-noticing-difference-betwen-validation-and-leaderboard-error/12209#post12209

    Parameters
    ----------
    y_true : array, shape = [n_samples, n_classes] one hot vectors
    y_pred : array, shape = [n_samples, n_classes]

    Returns
    -------
    loss : float
    """
    predictions = np.clip(y_pred, eps, 1 - eps)

    # normalize row sums to 1
    predictions /= predictions.sum(axis=1)[:, np.newaxis]

    # actual = np.zeros(y_pred.shape)
    # rows = actual.shape[0]
    # actual[np.arange(rows), y_true.astype(int)] = 1
    n_samples = y_true.shape[0]
    vsota = np.sum(y_true * np.log(predictions))
    return (-1.0 / n_samples) * vsota
