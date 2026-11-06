"""C2Explainer wrapper around the shared judge classifier.

C2 calls its oracle as `model(x, edge_index, batch, edge_weight=None)` — note the
singular `edge_weight` — and expects raw logits back. SharedJudge uses
`edge_weights`, so this thin wrapper bridges the naming and keeps everything else
(differentiability w.r.t. x and the edge mask) intact.

Unlike MutagGINEWrapperC2 this exposes no `set_orig_graph`: the shared GCN judge
consumes binary adjacency only, so there are no bond types to look up or default.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
from torch import Tensor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shared_judge import load_judge  # noqa: E402


class SharedJudgeWrapperC2(nn.Module):
    def __init__(self, judge_prefix: str) -> None:
        super().__init__()
        self.judge = load_judge(judge_prefix, device="cpu")
        self.judge.eval()
        for p in self.judge.parameters():
            p.requires_grad_(False)
        # C2 reads this to decide how to build edge features; the GCN judge has none.
        self.edge_dim = self.judge.edge_dim or None

    def forward(self, x: Tensor, edge_index: Tensor, batch: Tensor = None,
                edge_weight: Tensor = None, edge_attr: Tensor = None) -> Tensor:
        # C2 pre-classifies single graphs as model(x, edge_index) with no batch
        # vector; SharedJudge treats batch=None as "one graph".
        if self.judge.edge_dim == 0:
            edge_attr = None
        elif edge_attr is not None:
            edge_attr = edge_attr.float()
        return self.judge(
            x.float(), edge_index, batch,
            edge_attr=edge_attr, edge_weights=edge_weight,
        )

    @staticmethod
    def is_judge_prefix(path) -> bool:
        p = str(path)
        return os.path.exists(p + ".json") and os.path.exists(p + ".npz")
