import torch
from torch import cuda
from torch import device as torch_device
from torch.backends import mps
from torch.utils.data import DataLoader
from datasets.pipelines.pytorch_data_pipeline import FairnessPyTorchDataset, CustomDataset
from experiments import PyTorchConditions


def mutual_information(first_group: torch.tensor, second_group: torch.tensor) -> torch.tensor:
    # For the mutual information,
    # Pr[y|s] = sum{(xi,si),si=s} sigma(xi,si) / #D[xs]
    # D[xs]
    first_group_size = first_group.shape[0]
    second_group_size = second_group.shape[0]
    first_group_size_tensor = torch.tensor(first_group_size)
    second_group_size_tensor = torch.tensor(second_group_size)
    dxisi = torch.stack((second_group_size_tensor, first_group_size_tensor))
    # Pr[y|s]
    y_pred_female = torch.sum(first_group)
    y_pred_male = torch.sum(second_group)
    p_ys = torch.stack((y_pred_male, y_pred_female)) / dxisi
    # Pr[y]
    p = torch.cat((first_group, second_group), 0)
    p_y = torch.sum(p) / (first_group_size + second_group_size)
    # P(siyi)
    p_s1y1 = torch.log(p_ys[1]) - torch.log(p_y)
    p_s1y0 = torch.log(1 - p_ys[1]) - torch.log(1 - p_y)
    p_s0y1 = torch.log(p_ys[0]) - torch.log(p_y)
    p_s0y0 = torch.log(1 - p_ys[0]) - torch.log(1 - p_y)
    # PI
    pi_s1y1 = first_group * p_s1y1
    pi_s1y0 = (1 - first_group) * p_s1y0
    pi_s0y1 = second_group * p_s0y1
    pi_s0y0 = (1 - second_group) * p_s0y0
    pi = torch.sum(pi_s1y1) + torch.sum(pi_s1y0) + torch.sum(pi_s0y1) + torch.sum(pi_s0y0)
    return pi


def train_and_predict_prr_classifier(
        dataset: FairnessPyTorchDataset,
        net: torch.nn.Module,
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

    custom_dataset = CustomDataset(XZ_train, Y_train, Z_train)
    # cast the values of Z to 0 and 1
    custom_dataset.Z = custom_dataset.Z.type(torch.int64)
    Z_valid = Z_valid.type(torch.int64)
    data_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)

    loss_function = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    conditions.on_train_begin()
    for epoch in range(n_epochs):
        for i, (xz_batch, y_batch, z_batch) in enumerate(data_loader):
            # Split the datasets into two different groups w.r.t. the sensitive attribute
            # Here we assume that the sensitive attribute is binary!
            group1_x_batch = xz_batch[z_batch == 0][:, :-1].to(device)
            group1_y_batch = y_batch[z_batch == 0].to(device)
            group2_x_batch = xz_batch[z_batch == 1][:, :-1].to(device)
            group2_y_batch = y_batch[z_batch == 1].to(device)

            group1_output = net(group1_x_batch).squeeze()
            group2_output = net(group2_x_batch).squeeze()

            optimizer.zero_grad()
            usual_loss = loss_function(group1_output, group1_y_batch) + loss_function(group2_output, group2_y_batch)
            pi_loss = mutual_information(group1_output, group2_output)
            loss = (1 - lambda_) * usual_loss + lambda_ * pi_loss
            loss.backward()
            optimizer.step()

        # Split the datasets into two different groups w.r.t. the sensitive attribute
        # Here we assume that the sensitive attribute is binary!
        group1_x_valid = X_valid[Z_valid == 0].to(device)
        group1_y_valid = Y_valid[Z_valid == 0].to(device)
        group2_x_valid = X_valid[Z_valid == 1].to(device)
        group2_y_valid = Y_valid[Z_valid == 1].to(device)

        group1_y_hat_valid = net(group1_x_valid).squeeze()
        group2_y_hat_valid = net(group2_x_valid).squeeze()
        group1_p_loss = loss_function(group1_y_hat_valid.squeeze(), group1_y_valid)
        group2_p_loss = loss_function(group2_y_hat_valid.squeeze(), group2_y_valid)
        p_loss = group1_p_loss + group2_p_loss
        pi_loss = mutual_information(group1_y_hat_valid, group2_y_hat_valid)
        cost = (1 - lambda_) * p_loss + lambda_ * pi_loss

        # Early stopping
        if conditions.early_stop(epoch=epoch, loss_value=cost):
            break

    y_hat_test = net(X_test).squeeze().detach().cpu().numpy()
    return y_hat_test
