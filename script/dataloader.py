import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch


def load_adj(dataset_name):
    dataset_path = os.path.join('./data', dataset_name)
    adj = sp.load_npz(os.path.join(dataset_path, 'adj.npz'))
    adj = adj.tocsc()

    if dataset_name == 'metr-la':
        n_vertex = 207
    elif dataset_name == 'pems-bay':
        n_vertex = 325
    elif dataset_name == 'pemsd7-m':
        n_vertex = 228
    else:
        raise ValueError(f'ERROR: unknown dataset {dataset_name}.')

    return adj, n_vertex


def load_data(dataset_name, len_train, len_val):
    dataset_path = os.path.join('./data', dataset_name)
    vel = pd.read_csv(os.path.join(dataset_path, 'vel.csv'))

    train = vel[: len_train]
    val = vel[len_train: len_train + len_val]
    test = vel[len_train + len_val:]
    return train, val, test


def data_transform(data, n_his, n_pred, device, start_idx=0, n_tod=288, n_dow=7, start_dow=0):
    """Create sliding-window samples with their time-of-day / day-of-week indices.

    Args:
        data: numpy array [num_timesteps, n_vertex]
        n_his: number of historical steps (input window)
        n_pred: prediction horizon (steps ahead)
        device: torch device
        start_idx: global row offset (so val/test get correct absolute positions)
        n_tod: time-of-day slots per day (288 for 5-min intervals)
        n_dow: day-of-week slots (7 for all days, 5 for weekdays only)
        start_dow: day-of-week index of the first row in the ENTIRE dataset

    Returns:
        x:   [num_samples, 1, n_his, n_vertex]  -- velocity input
        tod: [num_samples, n_his]               -- time-of-day indices
        dow: [num_samples, n_his]               -- day-of-week indices
        y:   [num_samples, n_vertex]            -- prediction target
    """
    n_vertex = data.shape[1]
    len_record = len(data)
    num = len_record - n_his - n_pred

    x = np.zeros([num, 1, n_his, n_vertex])
    y = np.zeros([num, n_vertex])
    tod = np.zeros([num, n_his], dtype=np.int64)
    dow = np.zeros([num, n_his], dtype=np.int64)

    for i in range(num):
        head = i
        tail = i + n_his
        x[i, :, :, :] = data[head: tail].reshape(1, n_his, n_vertex)
        y[i] = data[tail + n_pred - 1]

        # Global step indices -> temporal features
        global_steps = np.arange(start_idx + head, start_idx + tail)
        tod[i] = global_steps % n_tod
        dow[i] = (global_steps // n_tod + start_dow) % n_dow

    return (torch.Tensor(x).to(device),
            torch.LongTensor(tod).to(device),
            torch.LongTensor(dow).to(device),
            torch.Tensor(y).to(device))
