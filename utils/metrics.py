import numpy as np

def get_accuracy(y_pred, y_target):
    return round(np.mean(y_pred == y_target), 3)


def get_confusion_matrix(y_pred, y_target):
    nr_classes = np.max(y_target) + 1
    confusion_mat = np.zeros((nr_classes, nr_classes), dtype=int)

    for k in range(y_target.shape[0]):
        confusion_mat[y_target[k], y_pred[k]] += 1

    return confusion_mat


def get_precision(confusion_mat):
    precision = np.zeros(confusion_mat.shape[0])

    for k in range(confusion_mat.shape[0]):
        total = np.sum(confusion_mat[:, k])

        if total == 0.0:
            precision[k] = 0.0
        else:
            precision[k] = confusion_mat[k, k] / total

    return np.around(precision, 3)


def get_recall(confusion_mat):
    recall = np.zeros(confusion_mat.shape[0])

    for k in range(confusion_mat.shape[0]):
        total = np.sum(confusion_mat[k, :])

        if total == 0.0:
            recall[k] = 0.0
        else:
            recall[k] = confusion_mat[k, k] / total

    return np.around(recall, 3)


def get_f_score(precision, recall):
    f_score = np.zeros(precision.shape[0])

    for k in range(precision.shape[0]):
        total = precision[k] + recall[k]

        if total == 0.0:
            f_score[k] = 0.0
        else:
            f_score[k] = (2 * precision[k] * recall[k]) / total

    return np.around(f_score, 3)


def get_confidence_interval(mean, std, sample_size):
    """
        Computes 95% confidence interval. 
        k = 1.96
    """
    confidence_interval = round(1.96 * std / np.sqrt(sample_size), 3)

    return (mean - confidence_interval, mean + confidence_interval)

