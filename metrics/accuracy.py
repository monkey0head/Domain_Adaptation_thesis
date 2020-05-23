import torch
import numpy as np


def accuracy_score(y_true: torch.Tensor, y_predict: torch.Tensor):
    """
    :param y_true: Ground truth (correct) labels
    :param y_predict: Predicted labels, as returned by a classifier.
    :return: score
    """
    correct = (y_true == y_predict).sum().item()
    return correct / np.prod(y_predict.shape)


class AccuracyScore:
    name = 'accuracy'

    def __init__(self):
        self._correct = 0
        self._total = 0

    def __call__(self, y_true: torch.Tensor, y_predict: torch.Tensor):
        self._correct += (y_true == y_predict).sum().item()
        self._total += np.prod(y_predict.shape)

    @property
    def score(self):
        try:
            return self._correct / self._total
        except ZeroDivisionError:
            return 0

    def reset(self):
        self._correct = 0
        self._total = 0


class AccuracyScoreFromLogits(AccuracyScore):
    def __call__(self, y_true: torch.Tensor, y_predict_logits: torch.Tensor):
        y_predict = y_predict_logits.max(1)[1].data
        self._correct += (y_true == y_predict).sum().item()
        self._total += np.prod(y_predict.shape)
