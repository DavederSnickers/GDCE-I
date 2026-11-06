"""The shared judge, wearing the interface XPlore expects of a GRETEL v2 oracle.

XPlore differentiates through the classifier: every iteration it feeds the
COMPLETE n x n edge list with a soft-masked weight per entry and backpropagates
the loss into that mask. So this wrapper cannot be the usual predict-only oracle
-- it has to expose the judge itself, under the exact call signature XPlore uses:

    oracle.model(x, edge_index, edge_weights, batch)

The judge's own signature is `(x, edge_index, batch, edge_attr, edge_weights)`,
hence the adapter module below. Two details it also has to absorb:

  * XPlore carries every tensor in float64; the judge's weights are float32.
    The cast lives inside the adapter so gradients still reach `edge_weights`.
  * XPlore hands over an edge list that includes the diagonal. The judge strips
    self-loops itself and re-adds exactly one per node, so nothing is needed
    here -- but it does mean a self-loop XPlore "creates" is invisible to the
    classifier, and `xplore_eval` drops the diagonal before scoring the result.

Zero-weight edges are genuinely absent as far as the GCN is concerned: the
symmetric normalisation builds its degrees from the same weights, so a weight of
0 removes the edge from both the message and the denominator.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class _JudgeAdapter(nn.Module):
    """Reorders arguments and fixes dtypes; holds no parameters of its own."""

    def __init__(self, judge):
        super().__init__()
        self.judge = judge

    @property
    def dtype(self):
        return next(self.judge.parameters()).dtype

    def forward(self, x, edge_index, edge_weights=None, batch=None,
                edge_attr=None):
        dt = self.dtype
        x = x.to(dt)
        if edge_weights is not None:
            edge_weights = edge_weights.to(dt)
        if edge_attr is not None:
            edge_attr = edge_attr.to(dt)
        edge_index = edge_index.long()
        return self.judge(x, edge_index, batch, edge_attr=edge_attr,
                          edge_weights=edge_weights)


class XPloreJudgeOracle:
    """Duck-typed stand-in for GRETEL v2's `Oracle`.

    XPlore touches only `.model`, `.device`, `.predict()` and `._call_counter`
    (the `.T` / `.dataset` / `.get_noise_schedule` attributes it also names live
    exclusively in the diffusion branch, which this port never enters).
    """

    def __init__(self, judge_prefix, device=None):
        from shared_judge import load_judge   # GDCE's src/shared_judge, on sys.path

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.judge = load_judge(str(judge_prefix), device=self.device)
        self.judge.eval()
        for p in self.judge.parameters():
            p.requires_grad_(False)
        self.model = _JudgeAdapter(self.judge).to(self.device)
        self.model.eval()
        self._call_counter = 0

    # -- what XPlore calls ------------------------------------------------

    def predict(self, instance):
        """Class index of `instance` as a 0-dim tensor (XPlore clones it)."""
        self._call_counter += 1
        return torch.argmax(self.logits(instance), dim=-1).squeeze()

    # -- our own helpers --------------------------------------------------

    def logits(self, instance):
        x, ei, ew = self._tensors(instance)
        with torch.no_grad():
            return self.model(x, ei, ew, self._batch(x)).squeeze()

    def predict_int(self, instance) -> int:
        return int(self.predict(instance).item())

    def _batch(self, x):
        return torch.zeros(x.shape[0], dtype=torch.long, device=self.device)

    def _tensors(self, instance):
        """(x, edge_index, edge_weights) for a GRETEL v2 GraphInstance.

        The dense adjacency is the source of truth; self-loops are dropped here
        so a diagonal entry XPlore may have switched on never reaches the judge
        under a different convention than the one it was trained with.
        """
        adj = np.asarray(instance.data)
        adj = adj - np.diag(np.diag(adj))
        src, dst = np.nonzero(adj)
        x = torch.as_tensor(np.asarray(instance.node_features),
                            dtype=torch.float, device=self.device)
        if len(src) == 0:
            ei = torch.empty((2, 0), dtype=torch.long, device=self.device)
            ew = torch.empty((0,), dtype=torch.float, device=self.device)
        else:
            ei = torch.as_tensor(np.stack([src, dst]), dtype=torch.long,
                                 device=self.device)
            ew = torch.as_tensor(adj[src, dst], dtype=torch.float,
                                 device=self.device)
        return x, ei, ew

    def probabilities(self, instance):
        return F.softmax(self.logits(instance), dim=-1).cpu().numpy()
