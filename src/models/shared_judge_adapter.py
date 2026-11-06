"""Adapter: load a SharedJudge export inside the GDCE pipeline.

The baselines consume the judge as a portable export (`<prefix>.npz` plus
`<prefix>.json`) of the SharedJudge class. The GDCE pipeline, in contrast, loads
Lightning checkpoints. Without this adapter, `label_source: 'classifier'` would
condition the diffusion model on a DIFFERENT classifier from the one the
baselines attack -- the comparison would then no longer be a shared-classifier
comparison but a comparison of method-oracle pairs.

The architecture is rebuilt here rather than imported, so that the GDCE pipeline
does not depend on a baseline checkout. It has to match shared_judge/shared_judge.py
exactly: GCN propagation with added self-loops, symmetric normalisation,
optional BatchNorm, ReLU, mean pooling, raw logit output.
"""
import json
import os

import numpy as np
import torch
import torch.nn as nn


class _HParams(dict):
    """A dict that additionally allows attribute access.

    The GDCE pipeline queries `hparams` in two different styles: `'out_dim' in
    clf.hparams` and `clf.hparams.get('in_dim')` (dict style, as Lightning
    offers) alongside `clf.hparams.in_dim` (attribute style). Both have to work,
    or the adapter fails depending on the caller.
    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class SharedJudgeAdapter(nn.Module):
    """Load a SharedJudge export and expose the GDCE interface.

    It is called with a PyG Data object (`forward(batch)`), exactly as
    NCI1GCNLightning is, so that the adapter drops in wherever the pipeline
    expects a Lightning classifier.
    """

    def __init__(self, spec: dict, weights: dict):
        super().__init__()
        if spec.get("conv", "gcn") != "gcn":
            raise ValueError(f"only conv='gcn' is supported, not {spec.get('conv')!r}")
        self.num_features = int(spec["num_features"])
        self.num_classes = int(spec["num_classes"])
        self.hidden = int(spec["hidden"])
        self.num_layers = int(spec["num_layers"])
        self.batchnorm = bool(spec.get("batchnorm", False))
        self.spec = spec

        dims = [self.num_features] + [self.hidden] * self.num_layers
        self.lins = nn.ModuleList(
            nn.Linear(dims[i], dims[i + 1]) for i in range(self.num_layers))
        if self.batchnorm:
            self.bns = nn.ModuleList(nn.BatchNorm1d(self.hidden)
                                     for _ in range(self.num_layers))
        self.head = nn.Linear(self.hidden, self.num_classes)

        # Callers in the GDCE pipeline introspect `clf.hparams`, because the
        # Lightning classifiers it otherwise loads offer it: `_apply_relabeling`
        # derives the number of guidance classes from it, `accuracy_test_nci1`
        # the expected one-hot dimension.
        self.hparams = _HParams(out_dim=self.num_classes, in_dim=self.num_features,
                                hidden_dim=self.hidden, num_layers=self.num_layers)

        self._load(weights)
        self.eval()

    def _load(self, w):
        """Map the export's state dict onto this module's parameter names."""
        sd = {}
        for k, v in w.items():
            sd[k] = torch.as_tensor(v)
        missing = self.load_state_dict(sd, strict=False)
        if missing.missing_keys:
            raise RuntimeError(
                f"SharedJudge export does not match the adapter. Missing: "
                f"{missing.missing_keys[:6]} | unexpected: {missing.unexpected_keys[:6]}")

    @staticmethod
    def _norm_adj(edge_index, n):
        """Add self-loops and normalise symmetrically (the GCN convention)."""
        device = edge_index.device
        mask = edge_index[0] != edge_index[1]          # strip any self-loop already present
        ei = edge_index[:, mask]
        loop = torch.arange(n, device=device)
        ei = torch.cat([ei, torch.stack([loop, loop])], dim=1)
        deg = torch.zeros(n, device=device).scatter_add_(
            0, ei[0], torch.ones(ei.shape[1], device=device))
        dinv = deg.clamp(min=1).pow(-0.5)
        w = dinv[ei[0]] * dinv[ei[1]]
        return ei, w

    def forward(self, batch):
        x, edge_index = batch.x, batch.edge_index
        b = getattr(batch, "batch", None)
        if b is None:
            b = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        ei, w = self._norm_adj(edge_index, x.shape[0])

        for i in range(self.num_layers):
            h = self.lins[i](x)
            agg = torch.zeros_like(h).index_add_(0, ei[1], h[ei[0]] * w.unsqueeze(-1))
            x = agg
            if self.batchnorm:
                x = self.bns[i](x)
            x = torch.relu(x)

        n_graphs = int(b.max()) + 1
        pooled = torch.zeros(n_graphs, x.shape[1], device=x.device)
        pooled.index_add_(0, b, x)
        counts = torch.zeros(n_graphs, device=x.device).index_add_(
            0, b, torch.ones_like(b, dtype=torch.float))
        return self.head(pooled / counts.clamp(min=1).unsqueeze(-1))


def load_shared_judge(path_prefix: str) -> SharedJudgeAdapter:
    """Load `<prefix>.json` + `<prefix>.npz`. The extension may be included."""
    for ext in (".npz", ".json"):
        if path_prefix.endswith(ext):
            path_prefix = path_prefix[: -len(ext)]
    with open(path_prefix + ".json") as f:
        spec = json.load(f)
    weights = dict(np.load(path_prefix + ".npz"))
    model = SharedJudgeAdapter(spec, weights)
    print(f"[SharedJudge] loaded: {os.path.basename(path_prefix)} "
          f"({spec.get('dataset')}, hidden={spec['hidden']}, layers={spec['num_layers']}, "
          f"test_acc={spec.get('test_acc')})")
    return model
