"""Shared judge classifier — ONE classifier that every counterfactual method targets.

This file is deliberately self-contained and depends only on `torch`, `numpy` and
`pandas`. It contains NO torch_geometric imports, because it has to run unchanged in
four different environments:

    GDCE            python 3.9   torch 1.11.0  PyG 2.3.1
    C2Explainer     python 3.10  torch 2.0.1   PyG 2.3.1
    COMBINEX/UCX    python 3.11  torch 2.5.0   PyG 2.6.1
    gnn_cff (CF2)   python 3.11  torch 2.5.0   PyG 2.6.1

Message passing is implemented with `index_add_`, so nothing depends on PyG internals
that changed between 2.3.1 and 2.6.1. Weights travel as a `.npz` + spec JSON rather
than a pickled `torch.save`, which would not load across those torch versions.

Copy this file into each baseline repo and use `load_judge(...)`.

Conventions baked in (so no caller can get them wrong):
  * Self-loops are normalised INSIDE the model: any self-loop in the input is
    stripped first. The GCN variant then adds exactly one per node; the GINE variant
    adds none (the (1+eps)*W x_i term already carries self information).
  * Mean pooling for both variants. Sum pooling would let a method inflate logits by
    adding nodes — and D4/GDCE do change node counts.
  * Class indices are the raw CSV `y` field: 0/1 exactly as GDCE preprocessed them.
    The judge is never trained on another classifier's predictions.

The file also carries the ORIGINAL classifiers of the four baselines (C2PaperGCN,
CF2GCNGraph, D4MutagGCN, UCXGINENet; spec "conv" = c2gcn / cf2gcn / d4gcn / ucxgine)
for the original-classifier ablation. They keep their own self-loop and pooling
conventions; load_judge dispatches on the spec, so every caller works unchanged.
"""

from __future__ import annotations

import ast
import json
import os
from types import SimpleNamespace
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------

class SharedJudge(nn.Module):
    """GCN- or GINE-style graph classifier, implemented without torch_geometric.

    forward(x, edge_index, batch, edge_attr=None, edge_weights=None)

    Differentiable in `x`, `edge_weights` and `edge_attr` — all three are needed by
    the counterfactual methods that optimise against this classifier:
      * CF2 and C2Explainer learn a soft edge mask  -> edge_weights
      * UCExplainer perturbs node and bond features -> x, edge_attr
      * GDCE and D4 need gradients w.r.t. node features for guidance
    """

    def __init__(
        self,
        num_features: int,
        num_classes: int = 2,
        hidden: int = 128,
        num_layers: int = 3,
        edge_dim: int = 0,
        conv: str = "gcn",
        dropout: float = 0.5,
        batchnorm: bool = False,
    ) -> None:
        super().__init__()
        if conv not in ("gcn", "gine"):
            raise ValueError(f"conv must be 'gcn' or 'gine', got {conv!r}")
        if conv == "gine" and edge_dim <= 0:
            raise ValueError("conv='gine' requires edge_dim > 0")

        self.num_features = int(num_features)
        self.num_classes = int(num_classes)
        self.hidden = int(hidden)
        self.num_layers = int(num_layers)
        self.edge_dim = int(edge_dim)
        self.conv = conv
        self.dropout = float(dropout)
        self.batchnorm = bool(batchnorm)

        dims = [self.num_features] + [self.hidden] * self.num_layers

        if conv == "gcn":
            self.lins = nn.ModuleList(
                [nn.Linear(dims[i], dims[i + 1]) for i in range(self.num_layers)]
            )
        else:
            # One message MLP per layer over [x_j || edge_attr], plus a linear
            # projection of the centre node (the GINE "(1+eps) * W x_i" term).
            self.msg_mlps = nn.ModuleList()
            self.self_lins = nn.ModuleList()
            for i in range(self.num_layers):
                self.msg_mlps.append(
                    nn.Sequential(
                        nn.Linear(dims[i] + self.edge_dim, dims[i + 1]),
                        nn.ReLU(),
                        nn.Linear(dims[i + 1], dims[i + 1]),
                    )
                )
                self.self_lins.append(nn.Linear(dims[i], dims[i + 1]))
            self.eps = nn.Parameter(torch.zeros(self.num_layers))

        # One DISTINCT BatchNorm per layer — never [BatchNorm(h)] * n, which would
        # share running stats across layers and collapse in eval mode.
        # In eval mode BN uses fixed running statistics, so the judge stays
        # deterministic and differentiable for the methods optimising against it.
        self.bns = (
            nn.ModuleList([nn.BatchNorm1d(self.hidden) for _ in range(self.num_layers)])
            if self.batchnorm else None
        )

        self.head = nn.Linear(self.hidden, self.num_classes)

        # Several repos introspect `clf.hparams.<field>` because that is what the
        # Lightning classifiers they normally load expose. Mirror it so the judge
        # is a drop-in there too. Plain attribute -> not part of state_dict.
        self.hparams = SimpleNamespace(
            in_dim=self.num_features,
            hidden_dim=self.hidden,
            out_dim=self.num_classes,
            edge_dim=self.edge_dim,
            num_layers=self.num_layers,
            conv=self.conv,
        )

    # -- helpers ---------------------------------------------------------

    @staticmethod
    def _strip_self_loops(edge_index, edge_weights, edge_attr):
        keep = edge_index[0] != edge_index[1]
        if bool(keep.all()):
            return edge_index, edge_weights, edge_attr
        edge_index = edge_index[:, keep]
        if edge_weights is not None:
            edge_weights = edge_weights[keep]
        if edge_attr is not None:
            edge_attr = edge_attr[keep]
        return edge_index, edge_weights, edge_attr

    def _propagate_gcn(self, x, edge_index, edge_weights, n_nodes, layer):
        """Symmetric-normalised propagation with explicit self-loops."""
        device = x.device
        src, dst = edge_index[0], edge_index[1]
        w = (
            torch.ones(src.shape[0], device=device, dtype=x.dtype)
            if edge_weights is None
            else edge_weights.to(x.dtype)
        )

        # add exactly one self-loop per node, weight 1
        loop = torch.arange(n_nodes, device=device)
        src = torch.cat([src, loop])
        dst = torch.cat([dst, loop])
        w = torch.cat([w, torch.ones(n_nodes, device=device, dtype=x.dtype)])

        deg = torch.zeros(n_nodes, device=device, dtype=x.dtype).index_add_(0, dst, w)
        dinv = deg.clamp(min=1e-12).pow(-0.5)
        norm = dinv[src] * w * dinv[dst]

        h = self.lins[layer](x)
        msg = h[src] * norm.unsqueeze(-1)
        out = torch.zeros(n_nodes, h.shape[1], device=device, dtype=h.dtype)
        return out.index_add_(0, dst, msg)

    def _propagate_gine(self, x, edge_index, edge_attr, edge_weights, n_nodes, layer):
        """Sum-aggregated MLP messages over [x_j || edge_attr]."""
        device = x.device
        src, dst = edge_index[0], edge_index[1]

        if edge_attr is None:
            edge_attr = torch.zeros(
                src.shape[0], self.edge_dim, device=device, dtype=x.dtype
            )
        msg_in = torch.cat([x[src], edge_attr.to(x.dtype)], dim=1)
        msg = self.msg_mlps[layer](msg_in)
        if edge_weights is not None:
            msg = msg * edge_weights.to(x.dtype).unsqueeze(-1)

        aggr = torch.zeros(n_nodes, msg.shape[1], device=device, dtype=msg.dtype)
        aggr.index_add_(0, dst, msg)
        return (1.0 + self.eps[layer]) * self.self_lins[layer](x) + aggr

    def _embed(self, x, edge_index, batch, edge_attr=None, edge_weights=None):
        n_nodes = x.shape[0]
        edge_index, edge_weights, edge_attr = self._strip_self_loops(
            edge_index, edge_weights, edge_attr
        )
        for layer in range(self.num_layers):
            if self.conv == "gcn":
                x = self._propagate_gcn(x, edge_index, edge_weights, n_nodes, layer)
            else:
                x = self._propagate_gine(
                    x, edge_index, edge_attr, edge_weights, n_nodes, layer
                )
            if self.bns is not None:
                x = self.bns[layer](x)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
        return self._mean_pool(x, batch)

    @staticmethod
    def _mean_pool(x, batch):
        if batch is None:
            return x.mean(dim=0, keepdim=True)
        n_graphs = int(batch.max().item()) + 1
        out = torch.zeros(n_graphs, x.shape[1], device=x.device, dtype=x.dtype)
        out.index_add_(0, batch, x)
        cnt = torch.zeros(n_graphs, device=x.device, dtype=x.dtype).index_add_(
            0, batch, torch.ones(x.shape[0], device=x.device, dtype=x.dtype)
        )
        return out / cnt.clamp(min=1).unsqueeze(-1)

    # -- public API ------------------------------------------------------

    def forward(self, x, edge_index, batch=None, edge_attr=None, edge_weights=None):
        """Returns raw logits (NOT log-softmax) of shape [n_graphs, num_classes]."""
        return self.head(self._embed(x, edge_index, batch, edge_attr, edge_weights))

    def get_embedding_repr(
        self, x, edge_index, batch=None, edge_attr=None, edge_weights=None
    ):
        """Graph-level embedding before the classification head (UCExplainer uses it)."""
        return self._embed(x, edge_index, batch, edge_attr, edge_weights)

    def forward_with_edge_weight(self, x, edge_index, edge_weight, batch_idx=None):
        """Alias matching the signature CF2 (gnn_cff) expects."""
        return self.forward(x, edge_index, batch_idx, edge_weights=edge_weight)


# ---------------------------------------------------------------------------
# the ORIGINAL classifiers of the compared methods, re-implemented without PyG
# ---------------------------------------------------------------------------
#
# Each class reproduces the forward pass of the classifier a baseline ships in
# its own code (architecture, normalisation, pooling, head), so that every method
# can be evaluated against its own classifier while GDCE-I and the control-judge
# re-scoring use the identical implementation. Same call signature as SharedJudge:
#     forward(x, edge_index, batch=None, edge_attr=None, edge_weights=None) -> logits
# Self-loops in the input are always stripped first; each architecture then adds
# exactly the self-loops its original layer type adds.

def _graph_count(batch, n_nodes, device):
    if batch is None:
        return torch.zeros(n_nodes, dtype=torch.long, device=device), 1
    return batch, int(batch.max().item()) + 1


def _with_unit_self_loops(edge_index, edge_weights, n_nodes, dtype):
    """Append one self-loop of weight 1 per node (the PyG GCNConv convention)."""
    device = edge_index.device
    w = (torch.ones(edge_index.shape[1], device=device, dtype=dtype)
         if edge_weights is None else edge_weights.to(dtype))
    loop = torch.arange(n_nodes, device=device)
    ei = torch.cat([edge_index, torch.stack([loop, loop])], dim=1)
    return ei, torch.cat([w, torch.ones(n_nodes, device=device, dtype=dtype)])


def _sum_aggregate(h, edge_index, w, n_nodes):
    """out_i = sum over edges (j -> i) of w_ji * h_j."""
    src, dst = edge_index[0], edge_index[1]
    msg = h[src] if w is None else h[src] * w.unsqueeze(-1)
    out = torch.zeros(n_nodes, h.shape[1], device=h.device, dtype=h.dtype)
    return out.index_add_(0, dst, msg)


def _gcn_normalised(edge_index, w, n_nodes, dtype):
    """Symmetric GCN normalisation D^-1/2 A D^-1/2 with degrees taken on the target node."""
    src, dst = edge_index[0], edge_index[1]
    deg = torch.zeros(n_nodes, device=w.device, dtype=dtype).index_add_(0, dst, w)
    dinv = deg.pow(-0.5)
    dinv = torch.where(torch.isinf(dinv), torch.zeros_like(dinv), dinv)
    return dinv[src] * w * dinv[dst]


def _mean_pool(x, batch, n_graphs):
    out = torch.zeros(n_graphs, x.shape[1], device=x.device, dtype=x.dtype).index_add_(0, batch, x)
    cnt = torch.zeros(n_graphs, device=x.device, dtype=x.dtype).index_add_(
        0, batch, torch.ones(x.shape[0], device=x.device, dtype=x.dtype))
    return out / cnt.clamp(min=1).unsqueeze(-1)


def _add_pool(x, batch, n_graphs):
    return torch.zeros(n_graphs, x.shape[1], device=x.device, dtype=x.dtype).index_add_(0, batch, x)


def _max_pool(x, batch, n_graphs):
    """global_max_pool for nodes stored graph by graph (PyG / collate order); torch >= 1.11."""
    counts = torch.bincount(batch, minlength=n_graphs)
    starts = torch.cumsum(counts, 0) - counts
    pos = torch.arange(x.shape[0], device=x.device) - starts[batch]
    dense = x.new_full((n_graphs, max(int(counts.max().item()), 1), x.shape[1]), float("-inf"))
    dense[batch, pos] = x
    return dense.max(dim=1).values


def _glorot(lins):
    """Glorot-uniform init of the convolution weights, as PyG GCNConv and DGL GraphConv do;
    their biases start at zero, which the bias parameters below already do."""
    for lin in lins:
        nn.init.xavier_uniform_(lin.weight)


class _OriginalJudge(nn.Module):
    """Common attributes so that save_judge / load_judge and every repo's adapter
    treat these classifiers exactly like SharedJudge."""

    conv = "original"
    pooling = ""
    self_loops = ""

    def __init__(self, num_features, num_classes=2, hidden=128, num_layers=3,
                 edge_dim=0, dropout=0.0):
        super().__init__()
        self.num_features = int(num_features)
        self.num_classes = int(num_classes)
        self.hidden = int(hidden)
        self.num_layers = int(num_layers)
        self.edge_dim = int(edge_dim)
        self.dropout = float(dropout)
        self.batchnorm = False
        self.hparams = SimpleNamespace(
            in_dim=self.num_features, hidden_dim=self.hidden, out_dim=self.num_classes,
            edge_dim=self.edge_dim, num_layers=self.num_layers, conv=self.conv,
        )

    def forward(self, x, edge_index, batch=None, edge_attr=None, edge_weights=None):
        return self.head(self.get_embedding_repr(x, edge_index, batch, edge_attr, edge_weights))

    def forward_with_edge_weight(self, x, edge_index, edge_weight, batch_idx=None):
        return self.forward(x, edge_index, batch_idx, edge_weights=edge_weight)


class C2PaperGCN(_OriginalJudge):
    """C2Explainer's MUTAG classifier (exp/exp1_3.ipynb): 3 x GCNConv(normalize=False)
    [in -> 20 -> 20 -> 20], ReLU after the first two only, global_max_pool, dropout,
    Linear(20 -> 2). normalize=False adds no self-loops and C2's MUTAG.pickle has none,
    so a layer is h_i = sum_j w_ji x_j W + b over the real bonds only."""

    conv = "c2gcn"
    pooling = "max"
    self_loops = "stripped on input, none added, no normalisation"

    def __init__(self, num_features, num_classes=2, hidden=20, num_layers=3,
                 edge_dim=0, dropout=0.0):
        super().__init__(num_features, num_classes, hidden, num_layers, 0, dropout)
        dims = [self.num_features] + [self.hidden] * self.num_layers
        self.lins = nn.ModuleList([nn.Linear(dims[i], dims[i + 1], bias=False)
                                   for i in range(self.num_layers)])
        self.biases = nn.ParameterList([nn.Parameter(torch.zeros(dims[i + 1]))
                                        for i in range(self.num_layers)])
        self.lin = nn.Linear(self.hidden, self.num_classes)
        _glorot(self.lins)

    def get_embedding_repr(self, x, edge_index, batch=None, edge_attr=None, edge_weights=None):
        n = x.shape[0]
        batch, n_graphs = _graph_count(batch, n, x.device)
        edge_index, edge_weights, _ = SharedJudge._strip_self_loops(edge_index, edge_weights, None)
        w = None if edge_weights is None else edge_weights.to(x.dtype)
        for k in range(self.num_layers):
            x = _sum_aggregate(self.lins[k](x), edge_index, w, n) + self.biases[k]
            if k < self.num_layers - 1:
                x = F.relu(x)
        return _max_pool(x, batch, n_graphs)

    def head(self, g):
        return self.lin(F.dropout(g, self.dropout, training=self.training))


class CF2GCNGraph(_OriginalJudge):
    """CF^2's graph classifier (gnn_cff/models/gcn.py:GCNGraph, DGL): 3 x GraphConv(128)
    with ReLU, mean pooling, Linear 128 -> 16 -> 8 -> 1, sigmoid. CF^2 runs GraphConv
    (norm='both') on the COMPLETE graph of each molecule with the 0/1 adjacency as edge
    weight, so the degree normaliser is n (nodes of that graph) on both sides: a layer is
    h_i = (1/n) * sum_j A_ij x_j W + b, without self-loops. The single sigmoid output is
    returned as the logit pair [0, z]: softmax([0, z]) = [1 - sigmoid(z), sigmoid(z)]."""

    conv = "cf2gcn"
    pooling = "mean"
    self_loops = "stripped on input, none added; messages scaled by 1/n_graph (DGL complete-graph norm)"

    def __init__(self, num_features, num_classes=2, hidden=128, num_layers=3,
                 edge_dim=0, dropout=0.0):
        super().__init__(num_features, num_classes, hidden, num_layers, 0, dropout)
        assert self.num_classes == 2, "the CF^2 classifier has a single sigmoid output"
        dims = [self.num_features] + [self.hidden] * self.num_layers
        self.lins = nn.ModuleList([nn.Linear(dims[i], dims[i + 1], bias=False)
                                   for i in range(self.num_layers)])
        self.biases = nn.ParameterList([nn.Parameter(torch.zeros(dims[i + 1]))
                                        for i in range(self.num_layers)])
        self.dense1 = nn.Linear(self.hidden, 16)
        self.dense2 = nn.Linear(16, 8)
        self.dense3 = nn.Linear(8, 1)
        _glorot(self.lins)

    def get_embedding_repr(self, x, edge_index, batch=None, edge_attr=None, edge_weights=None):
        n = x.shape[0]
        batch, n_graphs = _graph_count(batch, n, x.device)
        edge_index, edge_weights, _ = SharedJudge._strip_self_loops(edge_index, edge_weights, None)
        w = (torch.ones(edge_index.shape[1], device=x.device, dtype=x.dtype)
             if edge_weights is None else edge_weights.to(x.dtype))
        sizes = torch.bincount(batch, minlength=n_graphs).to(x.dtype)
        inv_n = (1.0 / sizes.clamp(min=1))[batch].unsqueeze(-1)
        for k in range(self.num_layers):
            x = F.relu(_sum_aggregate(self.lins[k](x), edge_index, w, n) * inv_n + self.biases[k])
        return _mean_pool(x, batch, n_graphs)

    def head(self, g):
        z = self.dense3(F.relu(self.dense2(F.relu(self.dense1(g)))))
        return torch.cat([torch.zeros_like(z), z], dim=-1)


class D4MutagGCN(_OriginalJudge):
    """D4Explainer's Mutag classifier (gnns/mutag_gnn.py:Mutag_GCN as trained by its
    script, num_unit=2): 2 x GCNConv(128) (symmetric normalisation, unit self-loops)
    with ReLU, global_mean_pool, Linear(128,128) -> ReLU -> Dropout(0.5) -> Linear(128,2)."""

    conv = "d4gcn"
    pooling = "mean"
    self_loops = "stripped on input, one unit self-loop per node, symmetric GCN normalisation"

    def __init__(self, num_features, num_classes=2, hidden=128, num_layers=2,
                 edge_dim=0, dropout=0.5):
        super().__init__(num_features, num_classes, hidden, num_layers, 0, dropout)
        dims = [self.num_features] + [self.hidden] * self.num_layers
        self.lins = nn.ModuleList([nn.Linear(dims[i], dims[i + 1], bias=False)
                                   for i in range(self.num_layers)])
        self.biases = nn.ParameterList([nn.Parameter(torch.zeros(dims[i + 1]))
                                        for i in range(self.num_layers)])
        self.ffn1 = nn.Linear(self.hidden, self.hidden)
        self.ffn2 = nn.Linear(self.hidden, self.num_classes)
        _glorot(self.lins)

    def get_embedding_repr(self, x, edge_index, batch=None, edge_attr=None, edge_weights=None):
        n = x.shape[0]
        batch, n_graphs = _graph_count(batch, n, x.device)
        edge_index, edge_weights, _ = SharedJudge._strip_self_loops(edge_index, edge_weights, None)
        ei, w = _with_unit_self_loops(edge_index, edge_weights, n, x.dtype)
        norm = _gcn_normalised(ei, w, n, x.dtype)
        for k in range(self.num_layers):
            x = F.relu(_sum_aggregate(self.lins[k](x), ei, norm, n) + self.biases[k])
        return _mean_pool(x, batch, n_graphs)

    def head(self, g):
        h = F.dropout(F.relu(self.ffn1(g)), self.dropout, training=self.training)
        return self.ffn2(h)


class UCXGINENet(_OriginalJudge):
    """UCExplainer / COMBINEX graph oracle (src/oracles/models/models.py:GINENet_G):
    layer 0 is a GINE layer, (1 + eps) * Lin(x_i) + sum_j MLP([x_j || e_ji]) * w_ji with
    eps = 0 and no self-loops; layers 1.. are GCNConv(128) (symmetric normalisation,
    unit self-loops); ReLU + dropout(0.5) after every layer, global_add_pool, Linear(128, 2).
    The original returns log_softmax; we return the logits it is computed from."""

    conv = "ucxgine"
    pooling = "add"
    self_loops = "stripped on input; GINE layer none, GCN layers one unit self-loop per node"

    def __init__(self, num_features, num_classes=2, hidden=128, num_layers=3,
                 edge_dim=5, dropout=0.5):
        super().__init__(num_features, num_classes, hidden, num_layers, edge_dim, dropout)
        self.gine_mlp = nn.Sequential(
            nn.Linear(self.num_features + max(self.edge_dim, 0), self.hidden),
            nn.ReLU(),
            nn.Linear(self.hidden, self.hidden),
        )
        self.gine_lin = nn.Linear(self.num_features, self.hidden)
        self.lins = nn.ModuleList([nn.Linear(self.hidden, self.hidden, bias=False)
                                   for _ in range(self.num_layers - 1)])
        self.biases = nn.ParameterList([nn.Parameter(torch.zeros(self.hidden))
                                        for _ in range(self.num_layers - 1)])
        self.output_layer = nn.Linear(self.hidden, self.num_classes)
        _glorot(self.lins)

    def get_embedding_repr(self, x, edge_index, batch=None, edge_attr=None, edge_weights=None):
        n = x.shape[0]
        batch, n_graphs = _graph_count(batch, n, x.device)
        edge_index, edge_weights, edge_attr = SharedJudge._strip_self_loops(
            edge_index, edge_weights, edge_attr)
        src, dst = edge_index[0], edge_index[1]
        msg_in = x[src]
        if self.edge_dim > 0:
            if edge_attr is None:
                edge_attr = torch.zeros(src.shape[0], self.edge_dim, device=x.device, dtype=x.dtype)
            msg_in = torch.cat([msg_in, edge_attr.to(x.dtype)], dim=1)
        msg = self.gine_mlp(msg_in)
        if edge_weights is not None:
            msg = msg * edge_weights.to(x.dtype).unsqueeze(-1)
        aggr = torch.zeros(n, self.hidden, device=x.device, dtype=msg.dtype).index_add_(0, dst, msg)
        x = F.dropout(F.relu(self.gine_lin(x) + aggr), self.dropout, training=self.training)
        ei, w = _with_unit_self_loops(edge_index, edge_weights, n, x.dtype)
        norm = _gcn_normalised(ei, w, n, x.dtype)
        for k in range(self.num_layers - 1):
            x = _sum_aggregate(self.lins[k](x), ei, norm, n) + self.biases[k]
            x = F.dropout(F.relu(x), self.dropout, training=self.training)
        return _add_pool(x, batch, n_graphs)

    def head(self, g):
        return self.output_layer(g)


ORIGINAL_JUDGES = {cls.conv: cls for cls in (C2PaperGCN, CF2GCNGraph, D4MutagGCN, UCXGINENet)}


# ---------------------------------------------------------------------------
# portable save / load  (npz + json, never torch.save)
# ---------------------------------------------------------------------------

def save_judge(model: SharedJudge, path_prefix: str, extra_spec: Optional[dict] = None):
    """Write <prefix>.npz (weights) and <prefix>.json (spec)."""
    os.makedirs(os.path.dirname(os.path.abspath(path_prefix)), exist_ok=True)
    arrays = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    np.savez(path_prefix + ".npz", **arrays)

    spec = {
        "num_features": model.num_features,
        "num_classes": model.num_classes,
        "hidden": model.hidden,
        "num_layers": model.num_layers,
        "edge_dim": model.edge_dim,
        "conv": model.conv,
        "dropout": model.dropout,
        "batchnorm": model.batchnorm,
        "pooling": getattr(model, "pooling", "mean"),
        "self_loops": getattr(model, "self_loops",
                              "stripped on input; gcn adds one per node, gine adds none"),
        "logits": "raw (apply softmax/argmax yourself; NOT log_softmax)",
        "label_source": "raw CSV y field (0/1 as preprocessed by GDCE)",
    }
    if extra_spec:
        spec.update(extra_spec)
    with open(path_prefix + ".json", "w") as f:
        json.dump(spec, f, indent=2)
    return path_prefix + ".npz", path_prefix + ".json"


def load_judge(path_prefix: str, device: str = "cpu") -> SharedJudge:
    """Rebuild the judge from <prefix>.npz + <prefix>.json. Works on any torch >= 1.11."""
    with open(path_prefix + ".json") as f:
        spec = json.load(f)
    if spec["conv"] in ORIGINAL_JUDGES:
        model = ORIGINAL_JUDGES[spec["conv"]](
            num_features=spec["num_features"],
            num_classes=spec["num_classes"],
            hidden=spec["hidden"],
            num_layers=spec["num_layers"],
            edge_dim=spec["edge_dim"],
            dropout=spec["dropout"],
        )
    else:
        model = SharedJudge(
            num_features=spec["num_features"],
            num_classes=spec["num_classes"],
            hidden=spec["hidden"],
            num_layers=spec["num_layers"],
            edge_dim=spec["edge_dim"],
            conv=spec["conv"],
            dropout=spec["dropout"],
            batchnorm=spec.get("batchnorm", False),
        )
    z = np.load(path_prefix + ".npz")
    state = {k: torch.from_numpy(z[k]) for k in z.files}
    model.load_state_dict(state)
    model.to(device).eval()
    return model


class SharedJudgeBatchAdapter(nn.Module):
    """Wraps a SharedJudge so it can be called as `model(batch)` on a PyG Batch.

    Several repos (GDCE among them) hand their classifier a whole PyG Data/Batch
    object instead of unpacked tensors. This adapter bridges that without pulling
    torch_geometric into this file — it only reads attributes off the object.

    Edge attributes are passed only when the judge actually consumes them
    (edge_dim > 0), so the same adapter serves the GCN and GINE judges.
    """

    def __init__(self, judge: "SharedJudge") -> None:
        super().__init__()
        self.judge = judge
        # Mirrors the Lightning classifiers GDCE otherwise loads, so callers that
        # introspect `hparams['out_dim']` keep working.
        self.hparams = {"out_dim": judge.num_classes}

    def forward(self, batch):
        edge_attr = getattr(batch, "edge_attr", None)
        if self.judge.edge_dim == 0:
            edge_attr = None
        elif edge_attr is not None:
            edge_attr = edge_attr.float()
        batch_vec = getattr(batch, "batch", None)
        return self.judge(
            batch.x.float(), batch.edge_index, batch_vec, edge_attr=edge_attr
        )


# ---------------------------------------------------------------------------
# minimal CSV reader (no torch_geometric)
# ---------------------------------------------------------------------------

NUM_EDGE_TYPES = {"mutagenicity_mol": 5, "benzene": 5, "nci1": 0,
                  # non-molecular datasets: plain binary adjacency
                  "proteins": 0, "twitter": 0}


def load_split(root: str, dataset: str, split: str, max_nodes: int = 50):
    """Read <root>/<dataset>/raw/<split>.csv into a list of graph dicts.

    Mirrors GDCE's preprocessing exactly: one-hot node types over the shared
    node_type_map, one-hot bond types where the dataset has them.
    """
    import pandas as pd

    raw = os.path.join(root, dataset, "raw")
    n_node_types = int(len(np.load(os.path.join(raw, "node_type_map.npy"))))
    n_edge_types = NUM_EDGE_TYPES[dataset]

    df = pd.read_csv(os.path.join(raw, f"{split}.csv"))
    graphs = []
    for _, row in df.iterrows():
        adj_flat = np.fromstring(str(row["adjacency_matrix"]).strip("[]"), sep=",")
        n = int(round(np.sqrt(len(adj_flat))))
        if n * n != len(adj_flat) or n > max_nodes:
            continue
        adj = adj_flat.reshape(n, n)
        ei = np.array(np.nonzero(adj > 0))
        node_types = np.array(
            ast.literal_eval(str(row["node_types"])), dtype=np.int64
        )
        if node_types.shape[0] != n:
            continue

        g = {
            "x": np.eye(n_node_types, dtype=np.float32)[node_types],
            "edge_index": ei.astype(np.int64),
            "y": int(row["y"]),
            "n": n,
        }
        if n_edge_types > 0:
            et = np.fromstring(
                str(row["edge_type_matrix"]).strip("[]"), sep=","
            ).astype(np.int64).reshape(n, n)
            vals = et[ei[0], ei[1]] if ei.size else np.zeros(0, dtype=np.int64)
            g["edge_attr"] = np.eye(n_edge_types, dtype=np.float32)[vals]
        graphs.append(g)
    return graphs, n_node_types, n_edge_types


def collate(graphs, device="cpu"):
    """Batch a list of graph dicts into flat tensors (PyG-style, done by hand)."""
    xs, eis, eas, ys, batch = [], [], [], [], []
    offset = 0
    for i, g in enumerate(graphs):
        xs.append(torch.from_numpy(g["x"]))
        eis.append(torch.from_numpy(g["edge_index"]) + offset)
        if "edge_attr" in g:
            eas.append(torch.from_numpy(g["edge_attr"]))
        ys.append(g["y"])
        batch.append(torch.full((g["n"],), i, dtype=torch.long))
        offset += g["n"]

    return (
        torch.cat(xs).to(device),
        torch.cat(eis, dim=1).to(device) if eis else torch.zeros(2, 0, dtype=torch.long, device=device),
        torch.cat(batch).to(device),
        torch.cat(eas).to(device) if eas else None,
        torch.tensor(ys, dtype=torch.long, device=device),
    )
