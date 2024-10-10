from automl.auto_pytorch import PytorchMLP
from methods.jiang import train_and_predict_jiang_classifier


class JiangMLP(PytorchMLP):

    def train_and_predict_classifier(self, dataset, net, metric, lambda_, lr, n_epochs, batch_size, conditions):
        return train_and_predict_jiang_classifier(dataset, net, metric, lambda_, lr, n_epochs, batch_size, conditions)
