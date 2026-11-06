"""Paper-style GCN (C2 exp1_3 architecture) as a reusable, dataset-agnostic module.

Mirrors the MUTAG paper classifier — 3x edge_weight-aware GCNConv with
normalize=False, global_max_pool, Linear head — but with a CONFIGURABLE input
dimension, so the SAME architecture can be trained/evaluated on other datasets
(e.g. GDCE Mutagenicity) for the "same classifier, different dataset" comparison.

GCNConv(normalize=False) adds NO self-loops internally, so the classifier only
sees self-loops that are present in edge_index. Datasets differ here:
  - MUTAG.pickle has NO self-loops             -> self_loops='none'
  - GDCE *_mol / nci1 set adj diagonal = 1.0   -> self_loops='full'
The wrapper canonicalizes accordingly so the original graph and C2's
self-loop-stripped counterfactual reach the classifier in the SAME
representation it was trained on (avoids the phantom 0-edit flips).
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear
from torch_geometric.nn import global_max_pool
from torch_geometric.utils import add_self_loops, remove_self_loops

from cf_explainer.gcn_conv import GCNConv


class PaperGCN(nn.Module):
    """Exact C2 exp1_3 GCN, with configurable in_dim / hidden / classes."""

    def __init__(self, in_dim: int, nhid: int = 20, nout: int = 20,
                 num_classes: int = 2, dropout: float = 0.0, batchnorm: bool = False):
        super().__init__()
        self.conv1 = GCNConv(in_dim, nhid, normalize=False)
        self.conv2 = GCNConv(nhid, nhid, normalize=False)
        self.conv3 = GCNConv(nhid, nout, normalize=False)
        self.lin = Linear(nout, num_classes)
        self.dropout = dropout
        # Opt-in BatchNorm (default off = exact paper exp1_3 arch). Distinct module
        # per conv — never a shared instance (that corrupts eval-mode running stats).
        self.use_bn = bool(batchnorm)
        if self.use_bn:
            self.bn1 = nn.BatchNorm1d(nhid)
            self.bn2 = nn.BatchNorm1d(nhid)
            self.bn3 = nn.BatchNorm1d(nout)

    def forward(self, x, edge_index, batch, edge_weight=None):
        if self.use_bn:
            x = self.bn1(self.conv1(x, edge_index, edge_weight)).relu()
            x = self.bn2(self.conv2(x, edge_index, edge_weight)).relu()
            x = self.bn3(self.conv3(x, edge_index, edge_weight))
        else:
            x = self.conv1(x, edge_index, edge_weight).relu()
            x = self.conv2(x, edge_index, edge_weight).relu()
            x = self.conv3(x, edge_index, edge_weight)
        x = global_max_pool(x, batch)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin(x)


class PaperGCNWrapperC2(nn.Module):
    """C2Explainer/PyG-Explainer-compatible wrapper around PaperGCN.

    forward(x, edge_index, batch=None, edge_weight=None). `self_loops` controls
    the diagonal canonicalization (see module docstring):
      - 'none' : pass edge_index through unchanged (MUTAG paper GCN).
      - 'full' : strip then re-add a full self-loop diagonal (GDCE *_mol / nci1
                 classifiers, which were trained with adj diagonal = 1.0).
    """

    def __init__(self, ckpt_path: str, in_dim: int, nhid: int = 20, nout: int = 20,
                 num_classes: int = 2, self_loops: str = 'none', batchnorm: bool = False):
        super().__init__()
        assert self_loops in ('none', 'full'), self_loops
        self.self_loops = self_loops
        self.clf = PaperGCN(in_dim, nhid=nhid, nout=nout,
                            num_classes=num_classes, dropout=0.0, batchnorm=batchnorm)
        sd = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        self.clf.load_state_dict(sd)
        self.clf.eval()
        for p in self.clf.parameters():
            p.requires_grad_(False)

    def forward(self, x: Tensor, edge_index: Tensor,
                batch: Optional[Tensor] = None,
                edge_weight: Optional[Tensor] = None) -> Tensor:
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        if self.self_loops == 'full':
            edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
            edge_index, edge_weight = add_self_loops(
                edge_index, edge_weight, fill_value=1.0, num_nodes=x.size(0))
        return self.clf(x.float(), edge_index, batch, edge_weight=edge_weight)
