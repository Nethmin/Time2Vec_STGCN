"""Chebyshev STGCN with Time2Vec features concatenated to the speed input."""

import torch
import torch.nn as nn

from model import layers


class STGCNChebGraphConv(nn.Module):
    def __init__(self, args, blocks, n_vertex):
        super(STGCNChebGraphConv, self).__init__()

        if getattr(args, 'temporal_emb', False):
            self.temporal_emb = layers.TemporalEncoding(
                d_t2v=args.d_t2v, d_emb=args.d_emb,
                n_tod=args.n_tod, n_dow=args.n_dow,
            )
        else:
            self.temporal_emb = None

        modules = []
        for l in range(len(blocks) - 3):
            modules.append(layers.STConvBlock(
                args.Kt, args.Ks, n_vertex, blocks[l][-1], blocks[l + 1],
                args.act_func, args.gso, args.enable_bias, args.droprate))
        self.st_blocks = nn.Sequential(*modules)

        Ko = args.n_his - (len(blocks) - 3) * 2 * (args.Kt - 1)
        self.Ko = Ko
        if self.Ko > 1:
            self.output = layers.OutputBlock(
                Ko, blocks[-3][-1], blocks[-2], blocks[-1][0], n_vertex,
                args.act_func, args.enable_bias, args.droprate)
        elif self.Ko == 0:
            self.fc1 = nn.Linear(in_features=blocks[-3][-1], out_features=blocks[-2][0], bias=args.enable_bias)
            self.fc2 = nn.Linear(in_features=blocks[-2][0], out_features=blocks[-1][0], bias=args.enable_bias)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(p=args.droprate)

    def forward(self, x, tod_idx=None, dow_idx=None):
        if self.temporal_emb is not None and tod_idx is not None:
            h_temp = self.temporal_emb(tod_idx, dow_idx)      # [B, T, d_emb]
            h_temp = h_temp.permute(0, 2, 1).unsqueeze(-1)    # [B, d_emb, T, 1]
            h_temp = h_temp.expand(-1, -1, -1, x.size(3))     # [B, d_emb, T, N]
            x = torch.cat([x, h_temp], dim=1)                 # [B, 1+d_emb, T, N]

        x = self.st_blocks(x)
        if self.Ko > 1:
            x = self.output(x)
        elif self.Ko == 0:
            x = self.fc1(x.permute(0, 2, 3, 1))
            x = self.relu(x)
            x = self.fc2(x).permute(0, 3, 1, 2)

        return x
