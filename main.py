"""Train and evaluate STGCN with daily and weekly Time2Vec encoding.

The final recipe uses MSE, AdamW, and best-validation checkpoint selection.
See README.md for the configuration and reported research results.
"""

import argparse
import gc
import math
import os
import random
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils as utils
import tqdm
from sklearn import preprocessing

from script import dataloader, utility, earlystopping
from model import models


# Calendar fields: slots per day, days per week, and initial day (Monday=0).
DATASET_CFG = {
    'pemsd7-m': dict(n_tod=288, n_dow=5, start_dow=1),  # weekdays only, starts Tuesday
    'metr-la':  dict(n_tod=288, n_dow=7, start_dow=3),  # all days, starts Thursday
    'pems-bay': dict(n_tod=288, n_dow=7, start_dow=6),  # all days, starts Sunday
}


def set_env(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_parameters(arg_list=None):
    parser = argparse.ArgumentParser(description='Finalized STGCN + Time2Vec')
    parser.add_argument('--enable_cuda', type=bool, default=True, help='use CUDA if available')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--dataset', type=str, default='pemsd7-m', choices=list(DATASET_CFG.keys()))

    parser.add_argument('--n_his', type=int, default=12, help='input window length (steps)')
    parser.add_argument('--n_pred', type=int, default=3, help='forecast horizon (steps): 3=15min, 6=30min, 9=45min')
    parser.add_argument('--time_intvl', type=int, default=5, help='minutes per step')

    parser.add_argument('--Kt', type=int, default=3, help='temporal kernel size')
    parser.add_argument('--Ks', type=int, default=3, choices=[2, 3], help='Chebyshev order')
    parser.add_argument('--stblock_num', type=int, default=2, help='number of ST-Conv blocks')
    parser.add_argument('--act_func', type=str, default='glu', choices=['glu', 'gtu'])
    parser.add_argument('--gso_type', type=str, default='sym_norm_lap',
                        choices=['sym_norm_lap', 'rw_norm_lap', 'sym_renorm_adj', 'rw_renorm_adj'])
    parser.add_argument('--enable_bias', type=bool, default=True)
    parser.add_argument('--droprate', type=float, default=0.5)

    parser.add_argument('--temporal_emb', action=argparse.BooleanOptionalAction, default=True,
                        help='enable Time2Vec temporal encoding (use --no-temporal_emb for the baseline)')
    parser.add_argument('--d_t2v', type=int, default=6,
                        help='Time2Vec dim per scale; # sine waves = d_t2v - 1 (plus 1 linear trend)')
    parser.add_argument('--d_emb', type=int, default=16, help='projected temporal embedding width')

    parser.add_argument('--lr', type=float, default=0.004, help='base learning rate')
    parser.add_argument('--weight_decay_rate', type=float, default=0.001, help='AdamW weight decay (L2)')
    parser.add_argument('--freq_lr_mult', type=float, default=10.0,
                        help='LR multiplier for Time2Vec frequency params (log_scale)')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=10, help='early-stopping patience')
    parser.add_argument('--t_max', type=int, default=60, help='cosine annealing period (epochs)')
    parser.add_argument('--warmup_epochs', type=int, default=2, help='linear LR warmup epochs')
    parser.add_argument('--max_grad_norm', type=float, default=5.0, help='gradient clipping max-norm (0 = off)')

    parser.add_argument('--save_path', type=str, default=None, help='checkpoint filename (without .pt)')
    args = parser.parse_args(arg_list)
    print('Training configs: {}'.format(args))

    set_env(args.seed)

    if args.enable_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.empty_cache()
    else:
        device = torch.device('cpu')
        gc.collect()

    Ko = args.n_his - (args.Kt - 1) * 2 * args.stblock_num

    # ST-Conv blocks use a 64 -> 16 -> 64 bottleneck.
    blocks = []
    if args.temporal_emb:
        blocks.append([1 + args.d_emb])  # velocity (1) + temporal embedding channels
    else:
        blocks.append([1])
    for _ in range(args.stblock_num):
        blocks.append([64, 16, 64])
    if Ko == 0:
        blocks.append([128])
    elif Ko > 0:
        blocks.append([128, 128])
    blocks.append([1])

    cfg = DATASET_CFG[args.dataset]
    args.n_tod = cfg['n_tod']
    args.n_dow = cfg['n_dow']
    args.start_dow = cfg['start_dow']

    return args, device, blocks


def data_preparate(args, device):
    adj, n_vertex = dataloader.load_adj(args.dataset)
    gso = utility.calc_gso(adj, args.gso_type)
    gso = utility.calc_chebynet_gso(gso)
    gso = gso.toarray().astype(np.float32)
    args.gso = torch.from_numpy(gso).to(device)

    dataset_path = os.path.join('./data', args.dataset)
    data_col = pd.read_csv(os.path.join(dataset_path, 'vel.csv')).shape[0]
    # split rate train:val:test = 70:15:15
    val_and_test_rate = 0.15
    len_val = int(math.floor(data_col * val_and_test_rate))
    len_test = int(math.floor(data_col * val_and_test_rate))
    len_train = int(data_col - len_val - len_test)

    train, val_, test_ = dataloader.load_data(args.dataset, len_train, len_val)
    zscore = preprocessing.StandardScaler()
    train = zscore.fit_transform(train)
    val_ = zscore.transform(val_)
    test_ = zscore.transform(test_)

    x_train, tod_train, dow_train, y_train = dataloader.data_transform(
        train, args.n_his, args.n_pred, device,
        start_idx=0, n_tod=args.n_tod, n_dow=args.n_dow, start_dow=args.start_dow)
    x_val, tod_val, dow_val, y_val = dataloader.data_transform(
        val_, args.n_his, args.n_pred, device,
        start_idx=len_train, n_tod=args.n_tod, n_dow=args.n_dow, start_dow=args.start_dow)
    x_test, tod_test, dow_test, y_test = dataloader.data_transform(
        test_, args.n_his, args.n_pred, device,
        start_idx=len_train + len_val, n_tod=args.n_tod, n_dow=args.n_dow, start_dow=args.start_dow)

    train_data = utils.data.TensorDataset(x_train, tod_train, dow_train, y_train)
    train_iter = utils.data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)
    val_data = utils.data.TensorDataset(x_val, tod_val, dow_val, y_val)
    val_iter = utils.data.DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=False)
    test_data = utils.data.TensorDataset(x_test, tod_test, dow_test, y_test)
    test_iter = utils.data.DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False)

    return n_vertex, zscore, train_iter, val_iter, test_iter


def report_params(model):
    total = sum(p.numel() for p in model.parameters())
    print('\n--- Parameter report ---')
    print(f'Total model params: {total:,}')
    te = getattr(model, 'temporal_emb', None)
    if te is not None:
        te_total = sum(p.numel() for p in te.parameters())
        print(f'Temporal encoding: {te_total:,} params ({100 * te_total / total:.1f}% of model)')
    print('------------------------\n')


def prepare_model(args, blocks, n_vertex, device):
    loss = nn.MSELoss()  # L2 loss, faithful to the original STGCN paper

    if args.save_path is not None:
        save_path = args.save_path if args.save_path.endswith('.pt') else args.save_path + '.pt'
    else:
        suffix = '_temporal' if args.temporal_emb else '_baseline'
        save_path = f'STGCN_{args.dataset}{suffix}.pt'
    args._save_path = save_path

    es = earlystopping.EarlyStopping(delta=0.0, patience=args.patience, verbose=True, path=save_path)

    model = models.STGCNChebGraphConv(args, blocks, n_vertex).to(device)
    report_params(model)

    # Frequencies use a higher LR; frequencies and phases are not weight-decayed.
    freq_p, no_decay_p, decay_p = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if 'log_scale' in name:
            freq_p.append(p)
        elif 'phase' in name:
            no_decay_p.append(p)
        else:
            decay_p.append(p)

    param_groups = [
        {'params': decay_p,    'weight_decay': args.weight_decay_rate, 'lr': args.lr},
        {'params': no_decay_p, 'weight_decay': 0.0,                    'lr': args.lr},
        {'params': freq_p,     'weight_decay': 0.0,                    'lr': args.lr * args.freq_lr_mult},
    ]
    optimizer = optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)

    if args.warmup_epochs > 0:
        warmup = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=args.warmup_epochs)
        cosine = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, args.t_max - args.warmup_epochs))
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[args.warmup_epochs])
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.t_max))

    return loss, es, model, optimizer, scheduler


@torch.no_grad()
def val(model, loss, val_iter):
    model.eval()
    l_sum, n = 0.0, 0
    for x, tod, dow, y in val_iter:
        y_pred = model(x, tod, dow).view(len(x), -1)
        l = loss(y_pred, y)
        l_sum += l.item() * y.shape[0]
        n += y.shape[0]
    return torch.tensor(l_sum / n)


def train(args, model, loss, optimizer, scheduler, es, train_iter, val_iter):
    history = []
    for epoch in range(args.epochs):
        l_sum, n = 0.0, 0
        model.train()
        for x, tod, dow, y in tqdm.tqdm(train_iter):
            optimizer.zero_grad()
            y_pred = model(x, tod, dow).view(len(x), -1)
            l = loss(y_pred, y)
            l.backward()
            if args.max_grad_norm and args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
            optimizer.step()
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]

        val_loss = val(model, loss, val_iter)
        scheduler.step()

        gpu_mem = torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0
        train_loss_avg = l_sum / n
        print('Epoch: {:03d} | Lr: {:.6f} | Train loss: {:.6f} | Val loss: {:.6f} | GPU: {:.1f} MiB'.format(
            epoch + 1, optimizer.param_groups[0]['lr'], train_loss_avg, val_loss, gpu_mem))
        history.append({'epoch': epoch + 1, 'train_loss': train_loss_avg, 'val_loss': val_loss.item()})

        es(val_loss, model)
        if es.early_stop:
            print('Early stopping')
            break

    history_path = args._save_path.replace('.pt', '_history.csv')
    pd.DataFrame(history).to_csv(history_path, index=False)
    print(f'Training history saved to {history_path}')
    return history


@torch.no_grad()
def test(zscore, loss, model, test_iter, args):
    model.load_state_dict(torch.load(args._save_path))
    model.eval()
    test_MSE = utility.evaluate_model(model, loss, test_iter)
    test_MAE, test_RMSE, test_WMAPE, test_MAPE = utility.evaluate_metric(model, test_iter, zscore)
    print(f'Dataset {args.dataset} | Test loss {test_MSE:.6f} | MAE {test_MAE:.6f} | '
          f'RMSE {test_RMSE:.6f} | WMAPE {test_WMAPE:.8f} | MAPE {test_MAPE:.6f}')
    return test_MSE, test_MAE, test_RMSE, test_WMAPE, test_MAPE


if __name__ == '__main__':
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)

    args, device, blocks = get_parameters()
    n_vertex, zscore, train_iter, val_iter, test_iter = data_preparate(args, device)
    loss, es, model, optimizer, scheduler = prepare_model(args, blocks, n_vertex, device)
    train(args, model, loss, optimizer, scheduler, es, train_iter, val_iter)
    test(zscore, loss, model, test_iter, args)
