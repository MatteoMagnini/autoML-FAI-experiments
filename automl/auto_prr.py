from automl.auto_pytorch import PytorchMLP
from methods.prr import train_and_predict_prr_classifier


class PRRMLP(PytorchMLP):

    extra_features_to_drop = 1

    def train_and_predict_classifier(self, dataset, net, metric, lambda_, lr, n_epochs, batch_size, conditions, on_test=False, fauci_fast_mode=False):
        return train_and_predict_prr_classifier(dataset, net, metric, lambda_, lr, n_epochs, batch_size, conditions, on_test)

