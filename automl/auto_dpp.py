from automl.auto_pytorch import PytorchMLP
from methods.dpp import train_and_predict_dpp_classifier


class DPPMLP(PytorchMLP):

    def train_and_predict_classifier(self, dataset, net, metric, lambda_, lr, n_epochs, batch_size, conditions):
        return train_and_predict_dpp_classifier(dataset, net, metric, lambda_, lr, n_epochs, batch_size, conditions)

