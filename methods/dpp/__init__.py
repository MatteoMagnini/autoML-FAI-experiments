import numpy as np
import torch
import torch.nn as nn
from torch import device as torch_device, optim
from torch.utils.data import DataLoader
import sys
from torch import cuda
from datasets.pipelines.pytorch_data_pipeline import FairnessPyTorchDataset, CustomDataset
from experiments import PyTorchConditions


def balance_ppv(eta_base, eta, y_base, y, t_base, t_min, t_max):
    if (eta_base >= t_base).mean() == 0:
        s = 1
    else:
        s = np.mean(y_base[eta_base >= t_base])

    for i in range(20):
        t = (t_min + t_max) / 2
        if (eta >= t).mean() == 0:
            sc = 1
        else:
            sc = np.mean(y[eta >= t])
        if sc > s:
            t_max = t
        else:
            t_min = t
    return (t_max + t_min) / 2


def threshold_pp(eta1, eta0, y1, y0):
    t_max1 = np.max(eta1)
    t_max0 = np.max(eta0)
    datasize = len(eta1) + len(eta0)
    if (eta1 >= 0.5).mean() == 0:
        s1 = 1
    else:
        s1 = np.mean(y1[eta1 >= 0.5])

    if (eta0 >= 0.5).mean() == 0:
        s0 = 1
    else:
        s0 = np.mean(y0[eta0 >= 0.5])

    if s1 > s0:
        t1max = 0.5
        t1min = balance_ppv(eta0, eta1, y0, y1, 0.5, 0.001, 0.5)
        t0min = 0.5
        t0max = balance_ppv(eta1, eta0, y1, y0, 0.5, 0.5, t_max0)
        t1set = np.arange(t1min, t1max, 0.001)
        lent = len(t1set)
        t0set = [balance_ppv(eta1, eta0, y1, y0, t1, t0min, t0max) for t1 in t1set]
        acc_set = [(((eta1 >= t1set[s]) == y1).sum() + ((eta0 >= t0set[s]) == y0).sum()) / datasize for s in range(lent)]
        acc_set = np.array(acc_set)
        index = np.argmax(acc_set)
        t1star = t1set[index]
        t0star = t0set[index]
    else:
        t1min = 0.5
        t1max = balance_ppv(eta0, eta1, y0, y1, 0.5, 0.5, t_max1)
        t0min = balance_ppv(eta1, eta0, y1, y0, 0.5, 0, 0.5)
        t0max = 0.5
        t1set = np.arange(t1min, t1max, 0.001)
        lent = len(t1set)
        t0set = [balance_ppv(eta1, eta0, y1, y0, t1, t0min, t0max, ) for t1 in t1set]
        acc_set = [(((eta1 >= t1set[s]) == y1).sum() + ((eta0 >= t0set[s]) == y0).sum()) / datasize for s in range(lent)]
        t0set = np.array(t0set)
        acc_set = np.array(acc_set)
        index = np.argmax(acc_set)
        t1star = t1set[index]
        t0star = t0set[index]

    return [t1star, t0star]


def train_and_predict_dpp_classifier(
        dataset: FairnessPyTorchDataset,
        net: nn.Module,
        metric: str,
        lambda_: float,
        n_epochs: int,
        batch_size: int,
        conditions: PyTorchConditions
):
    device = torch_device('cuda:1') if cuda.is_available() else torch_device('cpu')

    # Retrieve train/test split pytorch tensors for index=split
    train_tensors, valid_tensors, test_tensors = dataset.get_dataset_in_tensor()
    X_train, Y_train, Z_train, XZ_train = train_tensors
    X_valid, Y_valid, Z_valid, XZ_valid = valid_tensors
    X_test, Y_test, Z_test, XZ_test = test_tensors

    # Z_valid and Z_test cast to 1 or 0 (we assume binary sensitive attribute)
    Z_valid = (Z_valid > 0).float()
    Z_test = (Z_test > 0).float()

    Z_train_np = Z_train.detach().cpu().numpy()
    Z_list = sorted(list(set(Z_train_np)))
    for z in Z_list:
        if (Z_train_np == z).sum() == 0:
            print('At least one sensitive group has no data point')
            sys.exit()

    XZ_val_att1, Y_val_att1 = XZ_valid[Z_valid == 1], Y_valid[Z_valid == 1]
    XZ_val_att0, Y_val_att0 = XZ_valid[Z_valid == 0], Y_valid[Z_valid == 0]
    XZ_test_att1, Y_test_att1 = XZ_test[Z_test == 1], Y_test[Z_test == 1]
    XZ_test_att0, Y_test_att0 = XZ_test[Z_test == 0], Y_test[Z_test == 0]

    Y_val_att1_np = Y_val_att1.clone().cpu().detach().numpy()
    Y_val_att0_np = Y_val_att0.clone().cpu().detach().numpy()
    y_val_np = Y_valid.clone().cpu().detach().numpy()

    custom_dataset = CustomDataset(XZ_train, Y_train, Z_train)
    data_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)

    loss_function = nn.BCELoss()
    optimizer = optim.Adam(net.parameters())

    costs = []
    total_train_step = 0

    conditions.on_train_begin()
    for epoch in range(n_epochs):
        net.train()

        for i, (x, y, _) in enumerate(data_loader):
            x, y = x.to(device), y.to(device)
            y_hat = net(x)
            loss = loss_function(y_hat.squeeze(), y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            costs.append(loss.item())

            total_train_step += 1
            if (i + 1) % 10 == 0 or (i + 1) == batch_size:
                print('Epoch [{}/{}], Batch [{}/{}], Cost: {:.4f}'.format(epoch + 1, n_epochs, i + 1, len(data_loader), loss.item()), end='\r')

        # choose the model with the best performance on validation set
        net.eval()
        with torch.no_grad():

            output_val = net(XZ_valid).squeeze().detach().cpu().numpy()
            cost = loss_function(torch.tensor(output_val), Y_valid).item()
            y_tilde_val = (output_val >= 0.5).astype(np.float32)
            accuracy = (y_tilde_val == y_val_np).astype(np.float32).mean()

            if epoch == 0:
                accuracy_max = accuracy
                best_net_acc_stat_dict = net.state_dict()

            if accuracy > accuracy_max:
                accuracy_max = accuracy
                best_net_acc_stat_dict = net.state_dict()

        # Early stopping
        if conditions.early_stop(epoch=epoch, loss_value=cost):
            break

    # Calculate thresholds for fair Bayes-optimal Classifier
    net.load_state_dict(best_net_acc_stat_dict)

    eta1_val = net(XZ_val_att1).squeeze().detach().cpu().numpy()
    eta0_val = net(XZ_val_att0).squeeze().detach().cpu().numpy()
    eta1_test = net(XZ_test_att1).squeeze().detach().cpu().numpy()
    eta0_test = net(XZ_test_att0).squeeze().detach().cpu().numpy()
    # df_test_pp = pd.DataFrame()

    [t1_pp, t0_pp] = threshold_pp(eta1_val, eta0_val, Y_val_att1_np, Y_val_att0_np)

    y_hat_test = (eta1_test >= t1_pp).astype(np.float32)
    y_hat_test = np.concatenate((y_hat_test, (eta0_test >= t0_pp).astype(np.float32)))
    # sort to preserve the order of the original dataset
    y_hat_test = y_hat_test[np.argsort(Z_test.clone().cpu().detach().numpy())]
    return y_hat_test

    # acc_pp = (((eta1_test >= t1_pp) == Y_test_att1_np).sum() + (
    #         (eta0_test >= t0_pp) == Y_test_att0_np).sum()) / test_size
    # if (eta1_test >= t1_pp).mean() == 0:
    #     ppv1_fair = 1
    # else:
    #     ppv1_fair = np.mean(Y_test_att1_np[eta1_test >= t1_pp])
    # if (eta0_test >= t0_pp).mean() == 0:
    #     ppv0_fair = 1
    # else:
    #     ppv0_fair = np.mean(Y_test_att0_np[eta0_test >= t0_pp])
    #
    # yyhateq11 = (Y_test_att1_np[eta1_test >= t1_pp]).sum() + (Y_test_att0_np[eta0_test >= t0_pp]).sum()
    # yhateq1 = (eta1_test >= t1_pp).sum() + (eta0_test >= t0_pp).sum()
    # dpp = abs(ppv1_fair - ppv0_fair)
    #
    # if (eta0_test > 0.5).mean() == 0:
    #     ppv0 = 1
    # else:
    #     ppv0 = np.mean(Y_test_att0_np[eta0_test >= 0.5])
    # data = [t1_pp, t0_pp, acc_pp, dpp]
    # columns = ['PP_t1', 'PP_t0', 'acc_PP', 'DPP']
    # df_test_temp = pd.DataFrame([data], columns=columns)
    # df_test_pp = df_test_pp.append(df_test_temp)
    #
    # return df_test_pp
