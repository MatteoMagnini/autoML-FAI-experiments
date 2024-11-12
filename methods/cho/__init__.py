from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import device as torch_device
from torch import nn, optim
from torch.backends import mps
from torch.utils.data import DataLoader
from torch import cuda
from datasets.pipelines.pytorch_data_pipeline import CustomDataset, FairnessPyTorchDataset
from experiments import PyTorchConditions


PATH = Path(__file__).parents[0]
TAU = 0.5
# Approximation of Q-function given by López-Benítez & Casadevall (2011)
# based on a second-order exponential function & Q(x) = 1- Q(-x):
A = 0.4920
B = 0.2887
C = 1.1893
H = 0.1
DELTA = 1.0


def q_function(x):
    return torch.exp(-A * x**2 - B * x - C)


def cdf_tau(y_hat, h=0.01, tau=TAU):
    m = len(y_hat)
    y_tilde = (tau - y_hat) / h
    sum_ = (
        torch.sum(q_function(y_tilde[y_tilde > 0]))
        + torch.sum(1 - q_function(torch.abs(y_tilde[y_tilde < 0])))
        + 0.5 * (len(y_tilde[y_tilde == 0]))
    )
    return sum_ / m


def huber_loss(x, delta):
    if x.abs() < delta:
        return (x**2) / 2
    return delta * (x.abs() - delta / 2)


def measures_from_y_hat(y, z, y_hat=None, threshold=0.5):
    assert isinstance(y, np.ndarray)
    assert isinstance(z, np.ndarray)
    assert y_hat is not None
    assert isinstance(y_hat, np.ndarray)

    y_tilde = None
    if y_hat is not None:
        y_tilde = (y_hat >= threshold).astype(np.float32)
    assert y_tilde.shape == y.shape and y.shape == z.shape

    # Accuracy
    acc = (y_tilde == y).astype(np.float32).mean()
    # DP
    ddp = abs(np.mean(y_tilde[z == 0]) - np.mean(y_tilde[z == 1]))
    # EO
    y_z0, y_z1 = y[z == 0], y[z == 1]
    y1_z0 = y_z0[y_z0 == 1]
    y0_z0 = y_z0[y_z0 == 0]
    y1_z1 = y_z1[y_z1 == 1]
    y0_z1 = y_z1[y_z1 == 0]

    fpr, fnr = {}, {}
    fpr[0] = (
        np.sum(y_tilde[np.logical_and(z == 0, y == 0)]) / len(y0_z0)
        if len(y0_z0) > 0
        else 0
    )
    fpr[1] = (
        np.sum(y_tilde[np.logical_and(z == 1, y == 0)]) / len(y0_z1)
        if len(y0_z1) > 0
        else 0
    )

    fnr[0] = (
        np.sum(1 - y_tilde[np.logical_and(z == 0, y == 1)]) / len(y1_z0)
        if len(y1_z0) > 0
        else 0
    )
    fnr[1] = (
        np.sum(1 - y_tilde[np.logical_and(z == 1, y == 1)]) / len(y1_z1)
        if len(y1_z1) > 0
        else 0
    )

    tpr_diff = abs((1 - fnr[0]) - (1 - fnr[1]))
    fpr_diff = abs(fpr[0] - fpr[1])
    deo = tpr_diff + fpr_diff

    data = [acc, ddp, deo]
    columns = ["acc", "DDP", "DEO"]
    return pd.DataFrame([data], columns=columns)


def train_and_predict_cho_classifier(
    dataset: FairnessPyTorchDataset,
    net: nn.Module,
    metric: str,
    lambda_: float,
    lr: float,
    n_epochs: int,
    batch_size: int,
    conditions: PyTorchConditions
):
    device = torch_device('cuda') if cuda.is_available() else torch_device('mps') if mps.is_available() else torch_device('cpu')
    # Retrieve train/test split pytorch tensors for index=split
    train_tensors, valid_tensors, test_tensors = dataset.get_dataset_in_tensor()
    X_train, Y_train, Z_train, XZ_train = train_tensors
    X_valid, Y_valid, Z_valid, XZ_valid = valid_tensors
    X_test, Y_test, Z_test, XZ_test = test_tensors

    sensitive_attrs = dataset.sensitive_attrs

    custom_dataset = CustomDataset(XZ_train, Y_train, Z_train)
    if batch_size == "full":
        batch_size_ = XZ_train.shape[0]
    elif isinstance(batch_size, int):
        batch_size_ = batch_size
    else:
        raise ValueError("batch_size must be 'full' or an integer")
    data_loader = DataLoader(custom_dataset, batch_size=batch_size_, shuffle=True)

    pi = torch.tensor(np.pi).to(device)
    phi = lambda x: torch.exp(-0.5 * x**2) / torch.sqrt(2 * pi)  # normal distribution

    loss_function = nn.BCELoss()
    costs = []
    optimizer = optim.Adam(net.parameters(), lr=lr)

    def fairness_cost(y_pred, y_b, z_b):
        if isinstance(y_pred, torch.Tensor):
            y_pred_detached = y_pred.detach()
        else:
            y_pred = torch.tensor(y_pred).to(device)
            y_pred_detached = y_pred.detach()
        # DP_Constraint
        if metric == "demographic_parity":
            Pr_Ytilde1 = cdf_tau(y_pred_detached, H, TAU)
            for z in sensitive_attrs:
                Pr_Ytilde1_Z = cdf_tau(y_pred_detached[z_b == z], H, TAU)
                m_z = z_b[z_b == z].shape[0]

                Delta_z = Pr_Ytilde1_Z - Pr_Ytilde1
                Delta_z_grad = (
                    torch.dot(
                        phi((TAU - y_pred_detached[z_b == z]) / H).view(-1),
                        y_pred[z_b == z].view(-1),
                    )
                    / H
                    / m_z
                )
                Delta_z_grad -= (
                    torch.dot(
                        phi((TAU - y_pred_detached) / H).view(-1), y_pred.view(-1)
                    )
                    / H
                    / m
                )

                if Delta_z.abs() >= DELTA:
                    if Delta_z > 0:
                        Delta_z_grad *= lambda_ * DELTA
                        return Delta_z_grad
                    else:
                        Delta_z_grad *= -lambda_ * DELTA
                        return Delta_z_grad
                else:
                    Delta_z_grad *= lambda_ * Delta_z
                    return Delta_z_grad

        # EO_Constraint
        elif metric == "equalized_odds":
            for y in [0, 1]:
                Pr_Ytilde1_Y = cdf_tau(y_pred_detached[y_b == y], H, TAU)
                m_y = y_b[y_b == y].shape[0]
                for z in sensitive_attrs:
                    Pr_Ytilde1_ZY = cdf_tau(
                        y_pred_detached[(y_b == y) & (z_b == z)], H, TAU
                    )
                    m_zy = z_b[(y_b == y) & (z_b == z)].shape[0]
                    Delta_zy = Pr_Ytilde1_ZY - Pr_Ytilde1_Y
                    Delta_zy_grad = (
                        torch.dot(
                            phi(
                                (TAU - y_pred_detached[(y_b == y) & (z_b == z)]) / H
                            ).view(-1),
                            y_pred[(y_b == y) & (z_b == z)].view(-1),
                        )
                        / H
                        / m_zy
                    )
                    Delta_zy_grad -= (
                        torch.dot(
                            phi((TAU - y_pred_detached[y_b == y]) / H).view(-1),
                            y_pred[y_b == y].view(-1),
                        )
                        / H
                        / m_y
                    )

                    if Delta_zy.abs() >= DELTA:
                        if Delta_zy > 0:
                            Delta_zy_grad *= lambda_ * DELTA
                            return Delta_zy_grad
                        else:
                            Delta_zy_grad *= lambda_ * DELTA
                            return -lambda_ * DELTA * Delta_zy_grad
                    else:
                        Delta_zy_grad *= lambda_ * Delta_zy
                        return Delta_zy_grad
        return None

    conditions.on_train_begin()
    for epoch in range(n_epochs):
        for i, (xz_batch, y_batch, z_batch) in enumerate(data_loader):
            xz_batch, y_batch, z_batch = (
                xz_batch.to(device),
                y_batch.to(device),
                z_batch.to(device),
            )
            y_hat = net(xz_batch)
            cost = 0.0
            m = z_batch.shape[0]

            # prediction loss
            p_loss = loss_function(y_hat.view_as(y_batch), y_batch)
            # originally:
            # cost = (1 - lambda_) * p_loss + fairness_cost(Yhat, y_batch, z_batch)
            cost += (1 - lambda_) * p_loss + lambda_ * fairness_cost(y_hat, y_batch, z_batch)

            optimizer.zero_grad()
            if (torch.isnan(cost)).any():
                continue
            cost.backward()
            optimizer.step()
            costs.append(cost.item())

        y_hat_valid = net(XZ_valid)
        p_loss = loss_function(y_hat_valid.squeeze(), Y_valid)
        # originally:
        # cost = (1 - lambda_) * p_loss + fairness_cost(y_hat_valid, Y_valid, Z_valid)
        cost = (1 - lambda_) * p_loss + lambda_ * fairness_cost(y_hat_valid, Y_valid, Z_valid)

        # Early stopping
        if conditions.early_stop(epoch=epoch, loss_value=cost):
            break

    y_hat_test = net(XZ_test).squeeze().detach().cpu().numpy()
    return y_hat_test
