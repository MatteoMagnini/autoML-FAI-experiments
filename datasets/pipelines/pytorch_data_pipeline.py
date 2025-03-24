import numpy as np
import pandas as pd
import torch
from torch import cuda
from torch.backends import mps
from sklearn.preprocessing import StandardScaler


def arrays_to_tensor(X, Y, Z, Z1, Z2, XZ, XZ1Z2, device):
    return (
        torch.FloatTensor(X).to(device),
        torch.FloatTensor(Y).to(device),
        torch.FloatTensor(Z).to(device),
        torch.FloatTensor(Z1).to(device),
        torch.FloatTensor(Z2).to(device),
        torch.FloatTensor(XZ).to(device),
        torch.FloatTensor(XZ1Z2).to(device),
    )


class CustomDataset:
    def __init__(self, X, Y, Z, Z1 = None, Z2 = None):
        self.X = X
        self.Y = Y
        self.Z = Z
        self.Z1 = Z1
        self.Z2 = Z2

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, index):
        x, y, z, z1, z2 = self.X[index], self.Y[index], self.Z[index], self.Z1[index], self.Z2[index]
        return x, y, z, z1, z2


class FairnessPyTorchDataset:
    def __init__(
        self,
        train: pd.DataFrame,
        val: pd.DataFrame,
        test: pd.DataFrame,
        device=cuda.device('cuda') if cuda.is_available() else torch.device('mps') if mps.is_available() else torch.device('cpu')
    ):
        self.device = device
        self.intersectionality = False
        self.sensitive_attrs = None
        self.Z_train = None
        self.Z_val = None
        self.Z_test = None
        self.Z1_train = None
        self.Z1_val = None
        self.Z1_test = None
        self.Z2_train = None
        self.Z2_val = None
        self.Z2_test = None
        self.XZ_train = None
        self.XZ_val = None
        self.XZ_test = None
        self.XZ1Z2_train = None
        self.XZ1Z2_val = None
        self.XZ1Z2_test = None
        self.X_train = train.iloc[:, :-1]
        self.X_val = val.iloc[:, :-1]
        self.X_test = test.iloc[:, :-1]
        self.Y_train = train.iloc[:, -1]
        self.Y_val = val.iloc[:, -1]
        self.Y_test = test.iloc[:, -1]
        self.mapping = {}

    def prepare_ndarray(self, idx: int or list[int] = 0):
        def _(x, y) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray):

            if isinstance(idx, int):
                z = x.iloc[:, idx].to_numpy(dtype=np.float64)
                z1, z2, xz1z2 = None, None, None
            else:
                z1 = x.iloc[:, idx[0]].to_numpy(dtype=np.float64)
                z2 = x.iloc[:, idx[1]].to_numpy(dtype=np.float64)
                # Merge the two sensitive attributes values into one single column
                # Use z1z2_mapping to create the new values
                z = np.array([z1z2_mapping[f"{z1}_{z2}"] for z1, z2 in zip(z1, z2)])

            x = x.drop(x.columns[idx], axis=1).to_numpy(dtype=np.float64)
            xz = np.concatenate([x, z.reshape(-1, 1)], axis=1)
            xz1z2 = np.concatenate([x, z1.reshape(-1, 1), z2.reshape(-1, 1)], axis=1)
            y = y.to_numpy(dtype=np.float64)
            return x, y, z, z1, z2, xz, xz1z2

        # Create mapping if intersectionality is used (i.e. idx is a list)
        if isinstance(idx, list):
            self.intersectionality = True
            all_data = pd.concat([self.X_train, self.X_val, self.X_test], axis=0)
            z1_unique = np.unique(all_data.iloc[:, idx[0]])
            z2_unique = np.unique(all_data.iloc[:, idx[1]])
            z1_mapping = {z1: i for i, z1 in enumerate(z1_unique)}
            z2_mapping = {z2: i for i, z2 in enumerate(z2_unique)}
            # Create the values for the new sensitive attribute
            # There must be a unique numeric value for each combination of the two sensitive attributes (Cartesian product)
            # New value = z1_mapping[z1] * len(z2_unique) + z2_mapping[z2]
            z1z2_fake_values = np.array([z1_mapping[z1] * len(z2_unique) + z2_mapping[z2] for z1 in z1_unique for z2 in z2_unique])
            # Scale the values and create the mapping that can be used with z1 and z2 values
            scaler = StandardScaler()
            z1z2_fake_values = scaler.fit_transform(z1z2_fake_values.reshape(-1, 1)).reshape(-1)
            z1z2_mapping = {f"{z1}_{z2}": z1z2_fake_values[i*len(z2_unique) + j] for i, z1 in enumerate(z1_unique) for j, z2 in enumerate(z2_unique)}
        else:
            z1z2_mapping = {}
            self.intersectionality = False

        self.mapping = z1z2_mapping
        self.X_train, self.Y_train, self.Z_train, self.Z1_train, self.Z2_train, self.XZ_train, self.XZ1Z2_train = _(self.X_train, self.Y_train)
        self.X_val, self.Y_val, self.Z_val, self.Z1_val, self.Z2_val, self.XZ_val, self.XZ1Z2_val = _(self.X_val, self.Y_val)
        self.X_test, self.Y_test, self.Z_test, self.Z1_test, self.Z2_test, self.XZ_test, self.XZ1Z2_test = _(self.X_test, self.Y_test)
        self.sensitive_attrs = sorted(list(set(self.Z_train)))  # This is used only by CHO method

        # Scale
        scaler_XZ = StandardScaler()
        self.XZ_train = scaler_XZ.fit_transform(self.XZ_train)
        self.XZ_val = scaler_XZ.transform(self.XZ_val)
        self.XZ_test = scaler_XZ.transform(self.XZ_test)

        scaler_XZ1Z2 = StandardScaler()
        self.XZ1Z2_train = scaler_XZ1Z2.fit_transform(self.XZ1Z2_train)
        self.XZ1Z2_val = scaler_XZ1Z2.transform(self.XZ1Z2_val)
        self.XZ1Z2_test = scaler_XZ1Z2.transform(self.XZ1Z2_test)

        scaler_X = StandardScaler()
        self.X_train = scaler_X.fit_transform(self.X_train)
        self.X_val = scaler_X.transform(self.X_val)
        self.X_test = scaler_X.transform(self.X_test)

    def get_dataset_in_ndarray(self):
        return (
            (self.X_train, self.Y_train, self.Z_train, self.Z1_train, self.Z2_train, self.XZ_train, self.XZ1Z2_train),
            (self.X_val, self.Y_val, self.Z_val, self.Z1_val, self.Z2_val, self.XZ_val, self.XZ1Z2_val),
            (self.X_test, self.Y_test, self.Z_test, self.Z1_test, self.Z2_test, self.XZ_test, self.XZ1Z2_test),
        )

    def get_dataset_in_tensor(self):
        x_train, y_train, z_train, z1_train, z2_train, xz_train, xz1z2_train = arrays_to_tensor(
            self.X_train,
            self.Y_train,
            self.Z_train,
            self.Z1_train,
            self.Z2_train,
            self.XZ_train,
            self.XZ1Z2_train,
            self.device
        )
        x_val, y_val, z_val, z1_val, z2_val, xz_val, xz1z2_val= arrays_to_tensor(
            self.X_val,
            self.Y_val,
            self.Z_val,
            self.Z1_val,
            self.Z2_val,
            self.XZ_val,
            self.XZ1Z2_val,
            self.device
        )
        x_test, y_test, z_test, z1_test, z2_test, xz_test, xz1z2_test = arrays_to_tensor(
            self.X_test,
            self.Y_test,
            self.Z_test,
            self.Z1_test,
            self.Z2_test,
            self.XZ_test,
            self.XZ1Z2_test,
            self.device
        )
        return (
            (x_train, y_train, z_train, z1_train, z2_train, xz_train, xz1z2_train),
            (x_val, y_val, z_val, z1_val, z2_val, xz_val, xz1z2_val),
            (x_test, y_test, z_test, z1_test, z2_test, xz_test, xz1z2_test),
        )
