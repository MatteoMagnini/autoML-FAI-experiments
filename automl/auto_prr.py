from automl.auto_pytorch import PytorchMLP
from methods.prr import train_and_predict_prr_classifier


class PRRMLP(PytorchMLP):

    features_to_drop = 2

    def train_and_predict_classifier(self, dataset, net, metric, lambda_, lr, n_epochs, batch_size, conditions):
        return train_and_predict_prr_classifier(dataset, net, metric, lambda_, lr, n_epochs, batch_size, conditions)

