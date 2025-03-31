import torch
import torch.nn as nn
from torch import cuda
from torch import device as torch_device
from torch.utils.data import DataLoader
from torch.backends import mps
import torch.optim as optim
from datasets.pipelines.pytorch_data_pipeline import FairnessPyTorchDataset, CustomDataset
from experiments import PyTorchConditions
from methods.df.utils import compute_batch_counts, fairness_loss


def train_and_predict_df_classifier(
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
    X_train, Y_train, Z_train, Z1_train, Z2_train, XZ_train, XZ1Z2_train = train_tensors
    X_valid, Y_valid, Z_valid, Z1_valid, Z2_valid, XZ_valid, XZ1Z2_valid = valid_tensors
    X_test, Y_test, Z_test, Z1_test, Z2_test, XZ_test, XZ1Z2_test = test_tensors

    sensitive_attrs = dataset.sensitive_attrs

    custom_dataset = CustomDataset(XZ_train, Y_train, Z_train, Z1_train, Z2_train)
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
    z = Z_train if dataset.intersectionality else torch.concat([Z1_train, Z2_train], dim=1)
    z_uniques = torch.unique(z, dim=0)

    conditions.on_train_begin()
    for epoch in range(n_epochs):
        for i, (xz_batch, y_batch, z_batch, _, _) in enumerate(data_loader):
            xz_batch, y_batch, z_batch = (
                xz_batch.to(device),
                y_batch.to(device),
                z_batch.to(device),
            )
            y_hat = net(xz_batch)
            cost = 0.0
            loss = loss_function(y_hat.squeeze(), y_batch)
            # update Count model
            count_class, count_total = compute_batch_counts(z, z_uniques, y_hat)
            # fairness constraint
            loss_df = fairness_loss(0., count_class, count_total)
            cost += (1 - lambda_) * loss + lambda_ * loss_df
            optimizer.zero_grad()
            if (torch.isnan(cost)).any():
                continue
            cost.backward()
            optimizer.step()
            costs.append(cost.item())

        y_hat_valid = net(XZ_train)
        loss = loss_function(y_hat_valid.squeeze(), Y_train)
        count_class, count_total = compute_batch_counts(Z_train, z_uniques, y_hat_valid)
        loss_df = fairness_loss(0., count_class, count_total)
        cost = (1 - lambda_) * loss + lambda_ * loss_df

        # Early stopping
        if conditions.early_stop(epoch=epoch, loss_value=cost):
            break

    if on_test:
        return net(XZ_test).squeeze().detach().cpu().numpy()
    else:
        return net(XZ_valid).squeeze().detach().cpu().numpy()
