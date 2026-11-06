"""Wraps GDCE's MutagGINLightning (the actual NCI1_nofreq classifier) for C2Explainer.

The GDCE checkpoint at ckpt/nci1_nofreq/epoch=124.ckpt is in fact a
MutagGINLightning instance (plain GINConv, no edge features). C2Explainer
injects the learned mask as edge_weight; we route it through GINConvW so
that messages get multiplied by the (relaxed/binarized) mask.

Forward signature: (x, edge_index, batch=None, edge_weight=None)
"""

from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import global_add_pool
from torch_geometric.utils import add_self_loops, remove_self_loops

from .gnn_layers import upgrade_gin_layers


class NCI1GINWrapperC2(nn.Module):
    """C2Explainer-compatible wrapper around GDCE's MutagGINLightning."""

    def __init__(self, ckpt_path: str):
        super().__init__()
        from src.models.Mutag_GIN import MutagGINLightning  # GDCE module
        self.clf = MutagGINLightning.load_from_checkpoint(ckpt_path, map_location='cpu')
        self.clf.eval()
        for p in self.clf.parameters():
            p.requires_grad_(False)

        upgrade_gin_layers(self.clf.convs)

        self.in_dim = int(self.clf.hparams.get('in_dim', 36))

    def forward(self, x: Tensor, edge_index: Tensor,
                batch: Optional[Tensor] = None,
                edge_weight: Optional[Tensor] = None) -> Tensor:
        device = x.device
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=device)

        # NCI1 was built with a FULL self-loop diagonal (spectre_dataset.py sets
        # adj diagonal = 1.0), and the classifier was trained on that. C2's
        # candidate/sort logic strips self-loops, so without re-adding them the
        # original and the counterfactual graph reach the classifier in different
        # representations -> phantom flips at 0 edits. Canonicalize here so the
        # classifier always sees its trained representation (full diagonal),
        # regardless of whether the incoming edge_index has self-loops or not.
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        edge_index, edge_weight = add_self_loops(
            edge_index, edge_weight, fill_value=1.0, num_nodes=x.size(0))

        h = x.float()
        for conv in self.clf.convs:
            h = conv(h, edge_index, edge_weight=edge_weight)

        h = global_add_pool(h, batch)
        return self.clf.classifier(h)
