"""Network building blocks for the finalized STGCN + Time2Vec temporal model.

This is the cleaned-up replication of the thesis's final model. Only the
components that the final model actually uses are kept:

  * STGCN backbone: Align -> TemporalConvLayer (GLU) -> ChebGraphConv -> ...
  * Temporal encoding (the thesis contribution): a learnable-frequency
    Time2Vec (`ReparamTime2Vec`) at the daily and weekly scales, projected to
    `d_emb` channels and concatenated to the velocity input.

Everything from the research codebase that is NOT part of the final model
(NMF adaptive adjacency + graph fusion, discrete ToD/DoW lookup embeddings,
gated fusion, the plain non-reparameterized Time2Vec, the GraphConv/GCN
variant) has been removed for clarity.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class Align(nn.Module):
    """Match channel dimensions via a 1x1 conv (down) or zero-padding (up)."""

    def __init__(self, c_in, c_out):
        super(Align, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.align_conv = nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=(1, 1))

    def forward(self, x):
        if self.c_in > self.c_out:
            x = self.align_conv(x)
        elif self.c_in < self.c_out:
            batch_size, _, timestep, n_vertex = x.shape
            x = torch.cat([x, torch.zeros([batch_size, self.c_out - self.c_in, timestep, n_vertex]).to(x)], dim=1)
        return x


class CausalConv2d(nn.Conv2d):
    """2D convolution with left/causal padding along the time axis."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, enable_padding=False, dilation=1, groups=1, bias=True):
        kernel_size = nn.modules.utils._pair(kernel_size)
        stride = nn.modules.utils._pair(stride)
        dilation = nn.modules.utils._pair(dilation)
        if enable_padding == True:
            self.__padding = [int((kernel_size[i] - 1) * dilation[i]) for i in range(len(kernel_size))]
        else:
            self.__padding = 0
        self.left_padding = nn.modules.utils._pair(self.__padding)
        super(CausalConv2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=0, dilation=dilation, groups=groups, bias=bias)

    def forward(self, input):
        if self.__padding != 0:
            input = F.pad(input, (self.left_padding[1], 0, self.left_padding[0], 0))
        return super(CausalConv2d, self).forward(input)


class TemporalConvLayer(nn.Module):
    """Gated temporal convolution (GLU / GTU).

           |--------------------------------| * residual connection *
           |                                |
           |    |--->--- causalconv2d ----- + -------|
    -------|----|                                   (x) ------>
                |--->--- causalconv2d --- sigmoid ---|
    """

    def __init__(self, Kt, c_in, c_out, n_vertex, act_func):
        super(TemporalConvLayer, self).__init__()
        self.Kt = Kt
        self.c_in = c_in
        self.c_out = c_out
        self.n_vertex = n_vertex
        self.align = Align(c_in, c_out)
        self.causal_conv = CausalConv2d(in_channels=c_in, out_channels=2 * c_out, kernel_size=(Kt, 1), enable_padding=False, dilation=1)
        self.act_func = act_func

    def forward(self, x):
        x_in = self.align(x)[:, :, self.Kt - 1:, :]
        x_causal_conv = self.causal_conv(x)
        x_p = x_causal_conv[:, : self.c_out, :, :]
        x_q = x_causal_conv[:, -self.c_out:, :, :]

        if self.act_func == 'glu':
            # (x_p + x_in) (x) sigmoid(x_q)   -- Gated Linear Unit (Dauphin et al., 2017)
            x = torch.mul((x_p + x_in), torch.sigmoid(x_q))
        elif self.act_func == 'gtu':
            # tanh(x_p + x_in) (x) sigmoid(x_q)
            x = torch.mul(torch.tanh(x_p + x_in), torch.sigmoid(x_q))
        else:
            raise NotImplementedError(f'ERROR: activation function {self.act_func} is not implemented.')

        return x


class ChebGraphConv(nn.Module):
    """Chebyshev-polynomial graph convolution (ChebyNet) over a fixed GSO."""

    def __init__(self, c_in, c_out, Ks, gso, bias):
        super(ChebGraphConv, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.Ks = Ks
        self.gso = gso
        self.weight = nn.Parameter(torch.FloatTensor(Ks, c_in, c_out))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(c_out))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        x = torch.permute(x, (0, 2, 3, 1))

        if self.Ks - 1 < 0:
            raise ValueError(f'ERROR: Ks must be a positive integer, received {self.Ks}.')
        elif self.Ks - 1 == 0:
            x_list = [x]
        elif self.Ks - 1 == 1:
            x_0 = x
            x_1 = torch.einsum('hi,btij->bthj', self.gso, x)
            x_list = [x_0, x_1]
        else:  # Ks - 1 >= 2
            x_0 = x
            x_1 = torch.einsum('hi,btij->bthj', self.gso, x)
            x_list = [x_0, x_1]
            for k in range(2, self.Ks):
                x_list.append(torch.einsum('hi,btij->bthj', 2 * self.gso, x_list[k - 1]) - x_list[k - 2])

        x = torch.stack(x_list, dim=2)
        cheb_graph_conv = torch.einsum('btkhi,kij->bthj', x, self.weight)
        if self.bias is not None:
            cheb_graph_conv = torch.add(cheb_graph_conv, self.bias)

        return cheb_graph_conv


class GraphConvLayer(nn.Module):
    """Align to `c_out`, apply ChebGraphConv, add residual."""

    def __init__(self, c_in, c_out, Ks, gso, bias):
        super(GraphConvLayer, self).__init__()
        self.align = Align(c_in, c_out)
        self.cheb_graph_conv = ChebGraphConv(c_out, c_out, Ks, gso, bias)

    def forward(self, x):
        x_gc_in = self.align(x)
        x_gc = self.cheb_graph_conv(x_gc_in)
        x_gc = x_gc.permute(0, 3, 1, 2)
        return torch.add(x_gc, x_gc_in)


class STConvBlock(nn.Module):
    """Spatio-temporal block: T -> G -> T -> LayerNorm -> Dropout (TGTND)."""

    def __init__(self, Kt, Ks, n_vertex, last_block_channel, channels, act_func, gso, bias, droprate):
        super(STConvBlock, self).__init__()
        self.tmp_conv1 = TemporalConvLayer(Kt, last_block_channel, channels[0], n_vertex, act_func)
        self.graph_conv = GraphConvLayer(channels[0], channels[1], Ks, gso, bias)
        self.tmp_conv2 = TemporalConvLayer(Kt, channels[1], channels[2], n_vertex, act_func)
        self.tc2_ln = nn.LayerNorm([n_vertex, channels[2]], eps=1e-12)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=droprate)

    def forward(self, x):
        x = self.tmp_conv1(x)
        x = self.graph_conv(x)
        x = self.relu(x)
        x = self.tmp_conv2(x)
        x = self.tc2_ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        x = self.dropout(x)
        return x


class OutputBlock(nn.Module):
    """Output block: T -> LayerNorm -> FC -> FC (TNFF)."""

    def __init__(self, Ko, last_block_channel, channels, end_channel, n_vertex, act_func, bias, droprate):
        super(OutputBlock, self).__init__()
        self.tmp_conv1 = TemporalConvLayer(Ko, last_block_channel, channels[0], n_vertex, act_func)
        self.fc1 = nn.Linear(in_features=channels[0], out_features=channels[1], bias=bias)
        self.fc2 = nn.Linear(in_features=channels[1], out_features=end_channel, bias=bias)
        self.tc1_ln = nn.LayerNorm([n_vertex, channels[0]], eps=1e-12)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=droprate)

    def forward(self, x):
        x = self.tmp_conv1(x)
        x = self.tc1_ln(x.permute(0, 2, 3, 1))
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x).permute(0, 3, 1, 2)
        return x


class ReparamTime2Vec(nn.Module):
    """Reparameterized, learnable-frequency Time2Vec (the thesis contribution).

    Maps a normalized scalar time tau in [0, 1) to `out_features` features:
        component 0      = linear trend (w0 * tau + b0)
        components 1..k  = sin(tau * omega_i + phase_i),  k = out_features - 1

    Frequencies are reparameterized as  omega_i = omega_base_i * exp(log_scale_i),
    where omega_base is FIXED at integer harmonics 1..k cycles per period
    (2*pi*[1..k]). With log_scale initialized to 0, the sine "waves" start exactly
    at integer harmonics and the network learns multiplicative deviations from
    them. `log_scale` (frequency) and `phase` are the only periodic parameters.

    The number of sine waves is k = out_features - 1; `out_features` is exposed
    as `--d_t2v` (Time2Vec dimension per scale).
    """

    def __init__(self, out_features=6):
        super(ReparamTime2Vec, self).__init__()
        self.out_features = out_features
        self.linear = nn.Linear(1, 1)
        k = out_features - 1
        self.k = k
        if k > 0:
            self.register_buffer('omega_base', 2 * math.pi * torch.arange(1, k + 1).float())
            self.log_scale = nn.Parameter(torch.zeros(k))
            self.phase = nn.Parameter(torch.zeros(k))

    def omega(self):
        return self.omega_base * torch.exp(self.log_scale)

    def forward(self, tau):
        linear_out = self.linear(tau)
        if self.k > 0:
            periodic = torch.sin(tau * self.omega() + self.phase)
            return torch.cat([linear_out, periodic], dim=-1)
        return linear_out


class TemporalEncoding(nn.Module):
    """Time2Vec temporal encoding at daily + weekly scales, projected to d_emb.

    Output is [B, T, d_emb], identical across all N nodes (time is global), and
    is concatenated onto the velocity channel before the ST backbone.
    """

    def __init__(self, d_t2v=6, d_emb=16, n_tod=288, n_dow=5):
        super(TemporalEncoding, self).__init__()
        self.n_tod = n_tod
        self.n_dow = n_dow
        self.t2v_daily = ReparamTime2Vec(out_features=d_t2v)
        self.t2v_weekly = ReparamTime2Vec(out_features=d_t2v)
        self.projection = nn.Linear(2 * d_t2v, d_emb)

    def forward(self, tod_idx, dow_idx):
        # tau_daily in [0, 1): normalized position within the day
        tau_daily = (tod_idx.float() / self.n_tod).unsqueeze(-1)                       # [B, T, 1]
        # tau_weekly in [0, 1): normalized position within the week
        tau_weekly = ((dow_idx.float() * self.n_tod + tod_idx.float())
                      / (self.n_dow * self.n_tod)).unsqueeze(-1)                       # [B, T, 1]
        e_d = self.t2v_daily(tau_daily)                                                # [B, T, d_t2v]
        e_w = self.t2v_weekly(tau_weekly)                                              # [B, T, d_t2v]
        h = torch.cat([e_d, e_w], dim=-1)                                              # [B, T, 2*d_t2v]
        return self.projection(h)                                                      # [B, T, d_emb]
