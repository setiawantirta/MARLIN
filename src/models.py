"""Five graph architectures sharing one masked multi-task classification head.

The methodological modification of this study is `MaskedMultiTaskHead` + `masked_bce`:
a single shared graph encoder feeds six task-specific linear classifiers, and the loss
excludes unmeasured (NaN) labels from the gradient entirely, so every compound
contributes to exactly the tasks for which it has an experimental label.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (GINEConv, GATv2Conv, AttentiveFP, GPSConv,
                                global_mean_pool, global_max_pool, global_add_pool)
from feat import ATOM_DIM, BOND_DIM, RWSE_K, TASKS

N_TASKS = len(TASKS)


# --------------------------------------------------------------------------- head
class MaskedMultiTaskHead(nn.Module):
    """Shared trunk -> six independent task logits."""

    def __init__(self, in_dim, hidden=256, n_tasks=N_TASKS, dropout=0.2):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                   nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.ReLU(),
                                   nn.Dropout(dropout))
        self.heads = nn.ModuleList([nn.Linear(hidden, 1) for _ in range(n_tasks)])

    def forward(self, h):
        z = self.trunk(h)
        return torch.cat([head(z) for head in self.heads], dim=1)


def masked_bce(logits, y, mask, pos_weight, sample_weight=None):
    """Binary cross-entropy over observed labels only.

    logits, y, mask : (B, T).  mask==1 where a label was measured.
    pos_weight      : (T,) positive-class weight per task from training counts.
    sample_weight   : (B,) optional per-compound weight (used by the np_weight arm).
    Returns the mean loss over observed (compound, task) pairs.
    """
    pw = pos_weight.unsqueeze(0).expand_as(y)
    w = torch.where(y > 0.5, pw, torch.ones_like(pw))
    loss = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    loss = loss * w * mask
    if sample_weight is not None:
        loss = loss * sample_weight.unsqueeze(1)
    denom = (w * mask).sum().clamp(min=1.0)
    if sample_weight is not None:
        denom = (w * mask * sample_weight.unsqueeze(1)).sum().clamp(min=1.0)
    return loss.sum() / denom


# ------------------------------------------------------------------- architectures
class GINNet(nn.Module):
    """Graph Isomorphism Network with edge features (GINE)."""

    def __init__(self, hidden=128, layers=3, dropout=0.2, head_hidden=128):
        super().__init__()
        self.lin0 = nn.Linear(ATOM_DIM, hidden)
        self.elin = nn.Linear(BOND_DIM, hidden)
        self.convs, self.bns = nn.ModuleList(), nn.ModuleList()
        for _ in range(layers):
            mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
            self.convs.append(GINEConv(mlp, edge_dim=hidden))
            self.bns.append(nn.BatchNorm1d(hidden))
        self.dropout = dropout
        self.head = MaskedMultiTaskHead(2 * hidden, head_hidden, dropout=dropout)

    def embed(self, d):
        h = self.lin0(d.x)
        e = self.elin(d.edge_attr)
        for conv, bn in zip(self.convs, self.bns):
            h = F.relu(bn(conv(h, d.edge_index, e)))
            h = F.dropout(h, self.dropout, self.training)
        return torch.cat([global_mean_pool(h, d.batch), global_max_pool(h, d.batch)], dim=1)

    def forward(self, d):
        return self.head(self.embed(d))


class GATNet(nn.Module):
    """Graph Attention Network v2 with edge features."""

    def __init__(self, hidden=128, layers=3, heads=4, dropout=0.2, head_hidden=128):
        super().__init__()
        self.lin0 = nn.Linear(ATOM_DIM, hidden)
        self.convs, self.bns = nn.ModuleList(), nn.ModuleList()
        for _ in range(layers):
            self.convs.append(GATv2Conv(hidden, hidden // heads, heads=heads,
                                        edge_dim=BOND_DIM, dropout=dropout))
            self.bns.append(nn.BatchNorm1d(hidden))
        self.dropout = dropout
        self.head = MaskedMultiTaskHead(2 * hidden, head_hidden, dropout=dropout)

    def embed(self, d):
        h = self.lin0(d.x)
        for conv, bn in zip(self.convs, self.bns):
            h = F.elu(bn(conv(h, d.edge_index, d.edge_attr)))
            h = F.dropout(h, self.dropout, self.training)
        return torch.cat([global_mean_pool(h, d.batch), global_max_pool(h, d.batch)], dim=1)

    def forward(self, d):
        return self.head(self.embed(d))


class AttentiveFPNet(nn.Module):
    """AttentiveFP (Xiong et al. 2020) encoder + shared multi-task head."""

    def __init__(self, hidden=96, layers=2, timesteps=1, dropout=0.2, head_hidden=128):
        super().__init__()
        self.enc = AttentiveFP(in_channels=ATOM_DIM, hidden_channels=hidden,
                               out_channels=hidden, edge_dim=BOND_DIM,
                               num_layers=layers, num_timesteps=timesteps, dropout=dropout)
        self.head = MaskedMultiTaskHead(hidden, head_hidden, dropout=dropout)

    def embed(self, d):
        return self.enc(d.x, d.edge_index, d.edge_attr, d.batch)

    def forward(self, d):
        return self.head(self.embed(d))


class DMPNNNet(nn.Module):
    """Directed message-passing neural network (Chemprop D-MPNN) re-implemented on the
    shared featurisation: messages live on directed bonds, h_vw^(t+1) = tau(h_vw^0 +
    W_h * sum_{k in N(v)\\{w}} h_kv^(t)), then atom aggregation and graph readout."""

    def __init__(self, hidden=128, depth=3, dropout=0.2, head_hidden=128):
        super().__init__()
        self.W_i = nn.Linear(ATOM_DIM + BOND_DIM, hidden, bias=False)
        self.W_h = nn.Linear(hidden, hidden, bias=False)
        self.W_o = nn.Linear(ATOM_DIM + hidden, hidden)
        self.depth, self.dropout = depth, dropout
        self.head = MaskedMultiTaskHead(2 * hidden, head_hidden, dropout=dropout)

    def embed(self, d):
        src, dst = d.edge_index[0], d.edge_index[1]
        n_atoms, n_edges = d.x.size(0), src.size(0)
        h0 = F.relu(self.W_i(torch.cat([d.x[src], d.edge_attr], dim=1)))
        h = h0
        rev = d.rev
        for _ in range(self.depth - 1):
            agg = torch.zeros(n_atoms, h.size(1), device=h.device, dtype=h.dtype)
            agg.index_add_(0, dst, h)                 # sum of incoming messages at each atom
            msg = agg[src] - h[rev]                   # exclude the reverse edge
            h = F.relu(h0 + self.W_h(msg))
            h = F.dropout(h, self.dropout, self.training)
        agg = torch.zeros(n_atoms, h.size(1), device=h.device, dtype=h.dtype)
        agg.index_add_(0, dst, h)
        ha = F.relu(self.W_o(torch.cat([d.x, agg], dim=1)))
        ha = F.dropout(ha, self.dropout, self.training)
        return torch.cat([global_mean_pool(ha, d.batch), global_max_pool(ha, d.batch)], dim=1)

    def forward(self, d):
        return self.head(self.embed(d))


class GraphGPSNet(nn.Module):
    """GraphGPS (Rampasek et al. 2022): per layer a local GINE message-passing branch plus a
    global multi-head self-attention branch; random-walk structural encoding (RWSE, k=16)
    is embedded and added to the input node representation."""

    def __init__(self, hidden=64, layers=2, heads=4, dropout=0.2, head_hidden=128):
        super().__init__()
        self.lin0 = nn.Linear(ATOM_DIM, hidden)
        self.pe_norm = nn.BatchNorm1d(RWSE_K)
        self.pe_lin = nn.Linear(RWSE_K, hidden)
        self.elin = nn.Linear(BOND_DIM, hidden)
        self.convs = nn.ModuleList()
        for _ in range(layers):
            mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
            local = GINEConv(mlp, edge_dim=hidden)
            self.convs.append(GPSConv(hidden, local, heads=heads, dropout=dropout,
                                      attn_type="multihead"))
        self.head = MaskedMultiTaskHead(2 * hidden, head_hidden, dropout=dropout)

    def embed(self, d):
        h = self.lin0(d.x) + self.pe_lin(self.pe_norm(d.pe))
        e = self.elin(d.edge_attr)
        for conv in self.convs:
            h = conv(h, d.edge_index, d.batch, edge_attr=e)
        return torch.cat([global_mean_pool(h, d.batch), global_max_pool(h, d.batch)], dim=1)

    def forward(self, d):
        return self.head(self.embed(d))


ARCHS = {"GIN": GINNet, "GAT": GATNet, "AttentiveFP": AttentiveFPNet,
         "DMPNN": DMPNNNet, "GraphGPS": GraphGPSNet}
NEEDS_PE = {"GraphGPS"}


def build(arch, **kw):
    return ARCHS[arch](**kw)
