from automl.auto_pytorch import PytorchMLP
from methods.cho import train_and_predict_cho_classifier


class ChoMLP(PytorchMLP):

    def train_and_predict_classifier(self, dataset, net, metric, lambda_, n_epochs, batch_size, conditions):
        return train_and_predict_cho_classifier(dataset, net, metric, lambda_, n_epochs, batch_size, conditions)
