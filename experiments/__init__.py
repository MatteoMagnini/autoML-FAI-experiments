import math
from logging import Logger
from pathlib import Path
import numpy as np
import torch
from torch import nn
from analysis.metrics import demographic_parity, disparate_impact, equalized_odds
from _logging import INDENT, LOG_FLOAT_PRECISION
from experiments.setup import NEURONS_PER_LAYER

PATH = Path(__file__).parents[0]
CACHE_DIR_NAME = 'cache'
CACHE_PATH = PATH / CACHE_DIR_NAME
METRIC_LIST_NAMES = [
    'accuracy',
    'precision',
    'recall',
    'f1',
    'auc',
    'demographic_parity',
    'disparate_impact',
    'equalized_odds'
]
EPSILON = 1e-4


def create_cache_directory():
    if not CACHE_PATH.exists():
        CACHE_PATH.mkdir()


class PyTorchConditions:
    def __init__(self, model: torch.nn.Module, max_epochs: int, patience: int = 10):
        super().__init__()
        self.model = model
        self.patience = patience
        self.best_loss = math.inf
        self.wait = 0
        self.max_epochs = max_epochs
        self.best_weights = None

    def on_train_begin(self):
        self.wait = 0
        self.best_loss = math.inf
        self.best_weights = self.model.state_dict()

    def early_stop(self, epoch: int, loss_value: float):
        def end():
            self.model.load_state_dict(self.best_weights)
            return True

        # First condition: reached the maximum amount of epochs
        if epoch + 1 == self.max_epochs:
            return end()

        # Second condition: loss value does not improve for patience epochs
        if loss_value < self.best_loss:
            self.best_loss = loss_value
            self.wait = 0
            self.best_weights = self.model.state_dict()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                return end()

        return False


class PytorchNN(nn.Module):
    def __init__(self, inputs, hidden_layers: int = 1, neurons: int = 64):
        super(PytorchNN, self).__init__()
        layers = []

        if hidden_layers == 0:  # Logistic Regression
            layers.append(nn.Linear(inputs, 1))
            layers.append(nn.Sigmoid())
        else:
            layers.append(nn.Linear(inputs, neurons))
            layers.append(nn.ReLU())
            for _ in range(0, hidden_layers):
                layers.append(nn.Linear(neurons, neurons))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(neurons, 1))
            layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x


def evaluate_predictions(fairness_metric: str, protected: np.array, y_pred: np.array, y_true: np.array, logger: Logger) -> dict[str: float]:
    """
    Evaluate the predictions. Compute the following metrics:
    - Accuracy
    # - Precision
    # - Recall
    # - F1 score
    # - AUC
    - Demographic parity
    # - Disparate impact
    # - Equalized odds

    @param fairness_metric: fairness metric
    @param protected: protected features
    @param y_pred: predicted labels
    @param y_true: true labels
    @param logger: logger
    """
    binary_predictions = np.squeeze(np.where(y_pred >= 0.5, 1, 0))
    tp = np.sum(np.logical_and(binary_predictions == 1, y_true == 1))
    tn = np.sum(np.logical_and(binary_predictions == 0, y_true == 0))
    fp = np.sum(np.logical_and(binary_predictions == 1, y_true == 0))
    fn = np.sum(np.logical_and(binary_predictions == 0, y_true == 1))
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    # If there are more than 1 protected features, then a fake protected feature is created
    # to compute the fairness metrics
    if fairness_metric == 'demographic_parity':
        fairness_value = demographic_parity(protected, y_pred)
    elif fairness_metric == 'disparate_impact':
        fairness_value = disparate_impact(protected, y_pred)
    elif fairness_metric == 'equalized_odds':
        fairness_value = equalized_odds(protected, y_true, y_pred)
    else:
        raise ValueError(f"Unknown fairness metric: {fairness_metric}")
    logger.info(f"metrics:")
    # for metric, value in zip(METRIC_LIST_NAMES, [accuracy, precision, recall, f1_score, auc, dp, di, eo]):
    #     logger.info(f"{INDENT}{metric}: {value:.{LOG_FLOAT_PRECISION}f}")
    logger.info(f"{INDENT}accuracy: {accuracy:.{LOG_FLOAT_PRECISION}f}")
    logger.info(f"{INDENT}{fairness_metric}: {fairness_value:.{LOG_FLOAT_PRECISION}f}")
    return {
        "1 - accuracy": 1 - accuracy,
        # "1 - precision": 1 - precision,
        # "1 - recall": 1 - recall,
        # "1 - f1": 1 - f1_score,
        # "1 - auc": 1 - auc,
        fairness_metric: fairness_value
    }
