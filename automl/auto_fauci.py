from automl.auto_pytorch import PytorchMLP
from methods.fauci import train_and_predict_fauci_classifier


class FauciMLP(PytorchMLP):

    def train_and_predict_classifier(self, dataset, net, metric, lambda_, lr, n_epochs, batch_size, conditions, on_test=False, fauci_fast_mode=False):
        return train_and_predict_fauci_classifier(dataset, net, metric, lambda_, lr, n_epochs, batch_size, conditions, on_test, fauci_fast_mode)
