"""Adapter exposing a shared judge classifier through D4Explainer's GNN interface.

D4 does not call its oracle via `forward`; it expects `get_pred` and
`get_pred_explain` returning `(softmax_probs, logits)`, with `get_pred_explain`
taking a raw edge mask that the model itself squashes through a sigmoid. This
wrapper reproduces that contract exactly on top of a SharedJudge, so D4 can be
trained against the same classifier as every other method in the comparison.

EPS matches the value hard-coded in every gnns/*.py of this repo.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

# shared_judge.py is copied into this repo next to this file.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shared_judge import SharedJudge, load_judge  # noqa: E402

EPS = 1


class D4SharedJudgeGNN(nn.Module):
    """Wraps a SharedJudge in the interface D4Explainer's training loop expects."""

    def __init__(self, judge: SharedJudge) -> None:
        super().__init__()
        self.judge = judge
        self.softmax = nn.Softmax(dim=1)

    # -- helpers ---------------------------------------------------------

    def _edge_attr_or_none(self, edge_attr):
        """The GCN judge ignores bond types; only hand them over if it uses them."""
        if self.judge.edge_dim == 0 or edge_attr is None:
            return None
        return edge_attr.float()

    # -- D4 oracle interface --------------------------------------------

    def get_graph_rep(self, x, edge_index, batch, edge_attr=None):
        return self.judge.get_embedding_repr(
            x.float(), edge_index, batch,
            edge_attr=self._edge_attr_or_none(edge_attr),
        )

    def get_pred(self, x, edge_index, batch, edge_attr=None):
        logits = self.judge(
            x.float(), edge_index, batch,
            edge_attr=self._edge_attr_or_none(edge_attr),
        )
        return self.softmax(logits), logits

    def get_pred_explain(self, x, edge_index, edge_mask, batch, edge_attr=None):
        # D4 hands over an unsquashed mask; the oracle applies the sigmoid. Keeping
        # this identical to gnns/*.py matters — it is inside the gradient path that
        # trains the explainer.
        edge_weights = (edge_mask * EPS).sigmoid()
        logits = self.judge(
            x.float(), edge_index, batch,
            edge_attr=self._edge_attr_or_none(edge_attr),
            edge_weights=edge_weights,
        )
        return self.softmax(logits), logits

    def forward(self, x, edge_index, batch, edge_attr=None):
        return self.get_pred(x, edge_index, batch, edge_attr)[1]


def is_shared_judge_path(path: str) -> bool:
    """A shared judge is addressed by prefix: <prefix>.npz + <prefix>.json."""
    return os.path.exists(str(path) + ".json") and os.path.exists(str(path) + ".npz")


def load_shared_judge_gnn(prefix: str, device) -> D4SharedJudgeGNN:
    model = D4SharedJudgeGNN(load_judge(prefix, device="cpu"))
    return model.to(device).eval()
