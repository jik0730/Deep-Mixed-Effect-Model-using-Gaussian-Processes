import os

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

PHYSIO_DIR = 'data/physionet_split'
INTERVAL = 3
BORDER = 5
INPUT_DIM = 6


class PhysioNetDataset(Dataset):
    """
    PhysioNet time series dataset.
    """
    def __init__(self, split, var_str, data_dir=None):
        if data_dir == None:
            data_dir = PHYSIO_DIR
        self.var_str = var_str
        self.data_dir = os.path.join(data_dir, split)
        self.X, self.Y = self.load_data(self.data_dir)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def load_data(self, data_dir):
        X_scaler, y_scaler = 500., 50.
        X_list, y_list = [], []
        self.filepaths = []
        for filename in os.listdir(data_dir):
            file_path = os.path.join(data_dir, filename)
            self.filepaths.append(file_path)
            data = []
            with open(file_path, 'r') as f:
                count = 0
                ts_list, ys_list = [], []
                for i, line in enumerate(f):
                    line_split = line.split(',')
                    if line_split[1] == self.var_str:
                        t = self.time2sec(line_split[0])
                        y = float(line_split[2])
                        ts_list.append(t)
                        ys_list.append(y)
                        if count < INTERVAL + BORDER - 1:
                            pass
                        else:
                            t1, t2, t3 = ts_list[-BORDER - 2], ts_list[
                                -BORDER - 1], ts_list[-BORDER]
                            y1, y2, y3 = ys_list[-BORDER - 2], ys_list[
                                -BORDER - 1], ys_list[-BORDER]
                            data.append([t1, t2, t3, y1, y2, y3, y])
                        count += 1
            data = np.array(data).astype(np.float32)
            X_data = data[:, 0:INPUT_DIM].reshape(-1, INPUT_DIM) / X_scaler
            y_data = data[:, INPUT_DIM].reshape(-1) / y_scaler
            X_list.append(X_data)
            y_list.append(y_data)
        return X_list, y_list

    def time2sec(self, time_str):
        time_str_split = time_str.split(':')
        minute = int(time_str_split[0])
        sec = int(time_str_split[1])
        return 60 * minute + sec

    def get_filepath(self, idx):
        return self.filepaths[idx]


def fetch_dataloaders_PhysioNet(splits, var_str, data_dir=None):
    """
    Args:
        splits (list): A list of strings containing train/val/test.
        var_str (string): A variable in log.
        data_dir (string): Path to the dataset.
    """
    dataloaders = {}
    for split in splits:
        dataset = PhysioNetDataset(split, var_str, data_dir=data_dir)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        dataloaders[split] = dataloader
    return dataloaders


def test():
    dataset = PhysioNetDataset('val', 'HR')
    dataloader = DataLoader(dataset, shuffle=True)
    for X, y in dataloader:
        print(X.shape)
        print(y.shape)
        print(X.dtype)
        print(y.dtype)
        break


if __name__ == '__main__':
    test()
