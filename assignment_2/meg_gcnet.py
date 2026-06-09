import torch
import torch.nn as nn
import math
import logging

from base_model import BaseModel


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, mean=0, std=math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(unit_tcn, self).__init__()

        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
        )
        self.bn = nn.BatchNorm2d(out_channels)

        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        return self.bn(self.conv(x))


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes):
        super(unit_gcn, self).__init__()

        self.num_nodes = num_nodes

        self.conv = nn.Conv2d(in_channels, out_channels, 1)

        # B is the learnable adjacency matrix (inter-node connections).
        # A is the fixed identity (self-loops), stored as a buffer so it
        # moves to the correct device automatically with the model.
        self.B = nn.Parameter(torch.zeros(num_nodes, num_nodes) + 1e-6)
        self.register_buffer('A', torch.eye(num_nodes))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.down = lambda x: x

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)
        conv_branch_init(self.conv, 1)


    def forward(self, x):
        N, C, T, V = x.size()

        f_in = x.contiguous().view(N, C * T, V)

        # Build adjacency matrix: learnable connections + fixed self-loops
        adj_mat = self.B + self.A

        # Min-max normalisation
        adj_mat = (adj_mat - adj_mat.min()) / (adj_mat.max() - adj_mat.min() + 1e-8)

        # Symmetric degree normalisation: D^{-1/2} A D^{-1/2}
        # Done with element-wise broadcasting instead of constructing a full
        # V×V diagonal matrix and calling torch.inverse (which was O(V³) and
        # the main bottleneck when scaling from V=7 to V=248).
        deg = adj_mat.sum(dim=1)                              # (V,)
        d_inv_sqrt = torch.pow(deg.clamp(min=1e-8), -0.5)    # (V,)
        adj_mat_norm = (
            d_inv_sqrt.unsqueeze(1) * adj_mat * d_inv_sqrt.unsqueeze(0)
        )                                                     # (V, V)

        y = self.conv(torch.matmul(f_in, adj_mat_norm).view(N, C, T, V))
        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)

        return y
    
        

class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes, stride=1, residual=True, dropout=0.0, temporal_kernel_size=3):
        super(TCN_GCN_unit, self).__init__()

        self.gcn1 = unit_gcn(in_channels, out_channels, num_nodes)
        self.tcn1 = unit_tcn(out_channels, out_channels, kernel_size=temporal_kernel_size, stride=stride)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.gcn1(x) + self.residual(x)
        x = self.tcn1(x)
        x = self.drop(self.relu(x))
        return x


class MEGGCNet(BaseModel):
    def __init__(
        self,
        logger: logging.Logger,
        num_nodes: int = 248,
        in_channels: int = 1,
        num_classes: int = 4,
        dropout: float = 0.0,
        temporal_kernel_size: int = 3,
        num_blocks: int = 3,
    ):
        super(MEGGCNet, self).__init__(logger)


        self.data_bn = nn.BatchNorm1d(in_channels * num_nodes)

        # Channel progression: 16, 32, 64, 128, ... — doubles with each block.
        # conv_reduce_dim always readss from the final block's output channels.
        # Edit: changed it to 8, 16, 32
        channels = [8 * (2 ** i) for i in range(num_blocks)]

        block_kwargs = dict(
            num_nodes=num_nodes,
            dropout=dropout,
            temporal_kernel_size=temporal_kernel_size,
        )
        self.blocks = nn.ModuleList([
            TCN_GCN_unit(in_ch, out_ch, **block_kwargs)
            for in_ch, out_ch in zip([in_channels] + channels[:-1], channels)
        ])

        # Temporal average pooling applied between blocks to halve T after each
        # block. This reduces the cost of subsequent GCN matmuls, which scale
        # with C*T (e.g. T=512 → 256 → 128 across the three default blocks).
        # Requires window_size to be divisible by 2^(num_blocks-1).
        # Edit: changed to 4
        self.temporal_pool = nn.AvgPool2d(kernel_size=(4, 1), stride=(4, 1))

        self.conv_reduce_dim = unit_tcn(channels[-1], 4, kernel_size=1, stride=1)

        # Classification head: global average pool collapses (T, V) to a
        # scalar per channel, giving a 4-dim vector; FC maps to class logits.
        # This replaces the original flatten + large FC regression head, which
        # would have been 4 * T * 248 input features — enormous for T=512+.
        self.drop = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        self.fc = nn.Linear(4, num_classes)

        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_classes))
        bn_init(self.data_bn, 1)

    def forward(self, x):
        # Accept both (N, 248, T) and (N, 1, 248, T) from the pipeline
        if x.dim() == 3:
            x = x.unsqueeze(1)                       # (N, 1, 248, T)
 
        # Pipeline delivers (N, C, V, T); the model works in (N, C, T, V)
        x = x.permute(0, 1, 3, 2).contiguous()      # (N, C, T, V)
 
        N, C, T, V = x.size()
 
        # Input batch normalisation over the (C * V) feature dimension
        x = x.permute(0, 3, 1, 2).contiguous().view(N, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, V, C, T).permute(0, 2, 3, 1).contiguous()   # (N, C, T, V)
 
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i < len(self.blocks) - 1:
                x = self.temporal_pool(x)  # (N, C, T/2, V) after each non-final block
 
        x = self.conv_reduce_dim(x)                  # (N,   4, T, V)
 
        x = x.mean(dim=[2, 3])                       # (N, 4)
        x = self.drop(x)
        x = self.fc(x)                               # (N, num_classes)
 
        return x