import torch
from torch import cuda
from torch import device as torch_device
from torch.backends import mps
from torch.utils.data import DataLoader
from datasets.pipelines.pytorch_data_pipeline import FairnessPyTorchDataset, CustomDataset
from experiments import PyTorchConditions


def mutual_information(groups: list[torch.Tensor], device) -> torch.Tensor:
    group_sizes = torch.tensor([g.shape[0] for g in groups], dtype=torch.float, device=device)
    total_size = torch.sum(group_sizes)
    # Pr[y|s]
    p_ys = torch.tensor([torch.sum(g).to(device) for g in groups], device=device) / group_sizes
    # Pr[y]
    p_y = torch.sum(torch.cat(groups)).to(device) / total_size

    log_p_ys = torch.log(p_ys)
    log_1_p_ys = torch.log(1 - p_ys)
    log_p_y = torch.log(p_y)
    log_1_p_y = torch.log(1 - p_y)

    pi = torch.tensor(0.0, device=device)
    for i, group in enumerate(groups):
        pi += torch.sum(group * (log_p_ys[i] - log_p_y))
        pi += torch.sum((1 - group) * (log_1_p_ys[i] - log_1_p_y))

    return pi


def train_and_predict_prr_classifier(
        dataset: FairnessPyTorchDataset,
        net: torch.nn.Module,
        metric: str,
        lambda_: float,
        lr: float,
        n_epochs: int,
        batch_size: int,
        conditions: PyTorchConditions,
        on_test: bool = False
):
    device = torch_device('cuda') if cuda.is_available() else torch_device('mps') if mps.is_available() else torch_device('cpu')

    # Retrieve train/test split pytorch tensors for index=split
    train_tensors, valid_tensors, test_tensors = dataset.get_dataset_in_tensor()
    X_train, Y_train, Z_train, Z1_train, Z2_train, XZ_train, XZ1Z2_train = train_tensors
    X_valid, Y_valid, Z_valid, Z1_valid, Z2_valid, XZ_valid, XZ1Z2_valid = valid_tensors
    X_test, Y_test, Z_test, Z1_test, Z2_test, XZ_test, XZ1Z2_test = test_tensors

    custom_dataset = CustomDataset(XZ_train, Y_train, Z_train, Z1_train, Z2_train)
    # cast the values of Z to 0 and 1
    custom_dataset.Z = custom_dataset.Z.type(torch.int64)
    Z_valid = Z_valid.type(torch.int64)
    data_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)

    loss_function = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    conditions.on_train_begin()
    for epoch in range(n_epochs):
        for i, (xz_batch, y_batch, z_batch, _, _) in enumerate(data_loader):
            groups_x_batch = [xz_batch[z_batch == i][:, :-1].to(device) for i in range(len(torch.unique(z_batch)))]
            groups_y_batch = [y_batch[z_batch == i].to(device) for i in range(len(torch.unique(z_batch)))]
            # Remove empty groups
            groups_x_batch = [group_x_batch for group_x_batch in groups_x_batch if group_x_batch.shape[0] > 0]
            groups_y_batch = [group_y_batch for group_y_batch in groups_y_batch if group_y_batch.shape[0] > 0]
            groups_output = [net(group_x_batch).squeeze().to(device) for group_x_batch in groups_x_batch]
            optimizer.zero_grad()
            usual_loss = sum([loss_function(group_output, group_y_batch) for group_output, group_y_batch in zip(groups_output, groups_y_batch) if group_output.shape[0] > 0]).to(device)
            pi_loss = mutual_information(groups_output, device)
            loss = (1 - lambda_) * usual_loss + lambda_ * pi_loss
            loss.backward()
            optimizer.step()

        groups_x_train = [XZ_train[Z_train == i][:, :-1].to(device) for i in range(len(torch.unique(Z_train)))]
        groups_y_train = [Y_train[Z_train == i].to(device) for i in range(len(torch.unique(Z_train)))]
        groups_y_hat_valid = [net(group_x_train).squeeze().to(device) for group_x_train in groups_x_train]
        groups_p_loss = [loss_function(group_y_hat_valid.squeeze(), group_y_train) for group_y_hat_valid, group_y_train in zip(groups_y_hat_valid, groups_y_train) if group_y_train.shape[0] > 0]
        p_loss = sum(groups_p_loss)
        pi_loss = mutual_information(groups_y_hat_valid, device)
        cost = (1 - lambda_) * p_loss + lambda_ * pi_loss

        # Early stopping
        if conditions.early_stop(epoch=epoch, loss_value=cost):
            break

    if on_test:
        return net(X_test).squeeze().detach().cpu().numpy()
    else:
        return net(X_valid).squeeze().detach().cpu().numpy()
