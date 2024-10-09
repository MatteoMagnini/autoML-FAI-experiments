import math
from logging import Logger
from pathlib import Path
from typing import Iterable
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tensorflow.python.keras import Input
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Model
from torch import nn
from analysis.metrics import demographic_parity, equalized_odds, disparate_impact
from _logging import INDENT, LOG_FLOAT_PRECISION
from experiments.setup import NEURONS_PER_LAYER

PATH = Path(__file__).parents[0]
CACHE_DIR_NAME = 'cache'
CACHE_PATH = PATH / CACHE_DIR_NAME
DI_METRIC_NAME = "disparate_impact"
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


class TensorflowConditions(Callback):

    def __init__(self, patience: int = 10):
        super().__init__()
        self.patience = patience
        self.best_loss = math.inf
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None

    def reset(self):
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None
        self.best_loss = math.inf

    def on_train_begin(self, logs=None):
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = self.model.get_weights()

    def on_epoch_end(self, epoch, logs=None):

        def end():
            self.stopped_epoch = epoch
            self.model.set_weights(self.best_weights)
            print(f"Stopping at epoch {epoch + 1}")
            self.model.stop_training = True
            self.reset()

        val_loss_value = logs['val_loss']
        train_loss_value = logs['loss']

        # First condition: reached the maximum amount of epochs
        if epoch + 1 == self.params['epochs']:
            end()

        # Second condition: loss value does not improve for patience epochs
        elif train_loss_value < EPSILON:
            end()
        elif val_loss_value < self.best_loss:
            self.best_loss = val_loss_value
            self.wait = 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                end()


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
    def __init__(self, n_inputs, n_layers: int = len(NEURONS_PER_LAYER) + 2, n_hidden_units: list[int] = NEURONS_PER_LAYER):
        super(PytorchNN, self).__init__()
        layers = []

        if n_layers == 1:  # Logistic Regression
            layers.append(nn.Linear(n_inputs, 1))
            layers.append(nn.Sigmoid())
        else:
            layers.append(nn.Linear(n_inputs, n_hidden_units[0]))
            layers.append(nn.ReLU())
            for i in range(1, len(n_hidden_units)):
                layers.append(nn.Linear(n_hidden_units[i - 1], n_hidden_units[i]))
                layers.append(nn.ReLU())
            layers.append(nn.Linear(n_hidden_units[-1], 1))
            layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x


def create_fully_connected_nn_tf(
        input_size: int,
        output_size: int = 1,
        neurons_per_hidden_layer: Iterable[int] = NEURONS_PER_LAYER,
        activation_function: str = 'relu',
        latest_activation_function: str = 'sigmoid'
) -> Model:
    input_layer = Input(shape=(input_size,))
    x = input_layer
    for neurons in neurons_per_hidden_layer:
        x = Dense(neurons, activation=activation_function)(x)
    output_layer = Dense(output_size, activation=latest_activation_function)(x)
    return Model(inputs=input_layer, outputs=output_layer)


def evaluate_predictions(protected: np.array, y_pred: np.array, y_true: np.array, logger: Logger) -> dict[str: float]:
    """
    Evaluate the predictions. Compute the following metrics:
    - Accuracy
    - Precision
    - Recall
    - F1 score
    - Statistical parity
    - TPR parity (equal opportunity)
    - FPR parity

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
    # precision = tp / (tp + fp) if tp + fp > 0 else 0
    # recall = tp / (tp + fn) if tp + fn > 0 else 0
    # f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    # auc = roc_auc_score(y_true, y_pred)
    dp = demographic_parity(protected, y_pred)
    # eo = equalized_odds(protected, y_true, y_pred)
    # di = disparate_impact(protected, y_pred)
    logger.info(f"metrics:")
    # for metric, value in zip(METRIC_LIST_NAMES, [accuracy, precision, recall, f1_score, auc, dp, di, eo]):
    #     logger.info(f"{INDENT}{metric}: {value:.{LOG_FLOAT_PRECISION}f}")
    logger.info(f"{INDENT}accuracy: {accuracy:.{LOG_FLOAT_PRECISION}f}")
    logger.info(f"{INDENT}demographic_parity: {dp:.{LOG_FLOAT_PRECISION}f}")
    return {
        "1 - accuracy": 1 - accuracy,
        # "1 - precision": 1 - precision,
        # "1 - recall": 1 - recall,
        # "1 - f1": 1 - f1_score,
        # "1 - auc": 1 - auc,
        "demographic_parity": dp,
        # "1 - disparate_impact": 1 - di,
        # "equalized_odds": eo
    }
