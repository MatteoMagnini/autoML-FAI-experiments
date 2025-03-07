from pathlib import Path
import torch
from torch import nn, optim, cuda
from torch.backends import mps
from torch import device as torch_device
from torch.utils.data import DataLoader
from datasets.pipelines.pytorch_data_pipeline import FairnessPyTorchDataset, CustomDataset
from experiments import PyTorchConditions
from methods.fauci.pt_metric import discrete_demographic_parity, discrete_equalized_odds

PATH = Path(__file__).parents[0]
epsilon = 1e-5


def train_and_predict_fauci_classifier(
    dataset: FairnessPyTorchDataset,
    net: nn.Module,
    metric: str,
    lambda_: float,
    lr: float,
    n_epochs: int,
    batch_size: int,
    conditions: PyTorchConditions,
    on_test: bool = False,
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

    loss_function = nn.BCELoss()
    costs = []
    optimizer = optim.Adam(net.parameters(), lr=lr)

    def fairness_cost(y_pred, y_true, z_b):
        if isinstance(y_pred, torch.Tensor):
            y_pred_detached = y_pred.detach()
        else:
            y_pred = torch.tensor(y_pred).to(device)
            y_pred_detached = y_pred.detach()
        # DP_Constraint
        if metric == "demographic_parity":
            return discrete_demographic_parity(z_b, y_pred)
        elif metric == "equalized_odds":
            return discrete_equalized_odds(z_b, y_true, y_pred)
        else:
            raise ValueError(f"Unknown fairness metric {metric}")

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

            # prediction loss
            p_loss = loss_function(y_hat.view_as(y_batch), y_batch)
            cost += (1 - lambda_) * p_loss + lambda_ * fairness_cost(y_hat, y_batch, z_batch)

            optimizer.zero_grad()
            if (torch.isnan(cost)).any():
                continue
            cost.backward()
            optimizer.step()
            costs.append(cost.item())

        y_hat_valid = net(XZ_train)
        p_loss = loss_function(y_hat_valid.squeeze(), Y_train)
        cost = (1 - lambda_) * p_loss + lambda_ * fairness_cost(y_hat_valid, Y_train, Z_train)

        # Early stopping
        if conditions.early_stop(epoch=epoch, loss_value=cost):
            break

    if on_test:
        return net(XZ_test).squeeze().detach().cpu().numpy()
    else:
        return net(XZ_valid).squeeze().detach().cpu().numpy()
