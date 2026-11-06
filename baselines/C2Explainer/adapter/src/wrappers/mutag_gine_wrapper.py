"""Wraps GDCE's MutagGINELightning for use with C2Explainer.

C2Explainer (in add-mode) modifies the edge set: it deletes some original
edges and adds new ones. Bond types live on the original edges only; new
edges get a default bond type (single bond, class index 1 in the 5-class
one-hot encoding 0=no-bond, 1=single, 2=double, 3=triple, 4=aromatic).

The wrapper exposes a forward(x, edge_index, batch=None, edge_weight=None)
compatible with C2Explainer and PyG's Explainer with task_level='graph'.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import global_add_pool

from .gnn_layers import upgrade_gine_layers


DEFAULT_BOND_CLASS = 1  # single bond (one-hot index in 5-class encoding)


class MutagGINEWrapperC2(nn.Module):
    """C2Explainer-compatible wrapper around MutagGINELightning."""

    def __init__(self, ckpt_path: str, default_bond_class: int = DEFAULT_BOND_CLASS):
        super().__init__()
        from src.models.Mutag_GINE import MutagGINELightning  # GDCE module
        self.clf = MutagGINELightning.load_from_checkpoint(ckpt_path, map_location='cpu')
        self.clf.eval()
        for p in self.clf.parameters():
            p.requires_grad_(False)

        upgrade_gine_layers(self.clf.convs)

        self.edge_dim = int(self.clf.hparams.get('edge_dim', 5))
        self.default_bond_class = int(default_bond_class)

        # Populated per graph via set_orig_graph(). Stores a dense (N, N)
        # bond-class lookup: 0 = no original edge.
        self._lookup: Optional[Tensor] = None
        self._n_nodes: int = 0

    # --------------------------------------------------------------- helpers

    def set_orig_graph(self, edge_index: Tensor, edge_attr: Tensor, n_nodes: int) -> None:
        """Cache a dense bond-class lookup for the current original graph."""
        device = next(self.parameters()).device
        lookup = torch.zeros((n_nodes, n_nodes), dtype=torch.long, device=device)
        if edge_index.numel() > 0:
            ei = edge_index.to(device)
            ea = edge_attr.to(device)
            bonds = ea.argmax(dim=-1) if ea.dim() > 1 else ea.long()
            lookup[ei[0], ei[1]] = bonds
        self._lookup = lookup
        self._n_nodes = int(n_nodes)

    def _edge_attr_for(self, edge_index: Tensor) -> Tensor:
        """One-hot edge_attr for each edge in `edge_index`. Edges absent from
        the cached original graph get the default bond class (single bond).

        In graph-classification mode C2Explainer appends padding nodes to the
        graph (indices >= original N) and connects them with candidate edges.
        Those nodes are outside the cached (N, N) lookup, so any edge touching
        one is treated as a newly added edge and gets the default bond."""
        assert self._lookup is not None, "Call set_orig_graph() before forward."
        ei = edge_index
        n = self._n_nodes
        bonds = torch.full((ei.size(1),), self.default_bond_class,
                           dtype=torch.long, device=ei.device)
        in_range = (ei[0] < n) & (ei[1] < n)                 # both endpoints original
        if in_range.any():
            looked = self._lookup[ei[0][in_range], ei[1][in_range]]   # 0 = no orig edge
            looked = torch.where(looked == 0,
                                 torch.full_like(looked, self.default_bond_class), looked)
            bonds[in_range] = looked
        return F.one_hot(bonds, num_classes=self.edge_dim).float()

    # ----------------------------------------------------------------- forward

    def forward(self, x: Tensor, edge_index: Tensor,
                batch: Optional[Tensor] = None,
                edge_weight: Optional[Tensor] = None) -> Tensor:
        device = x.device
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=device)

        edge_attr = self._edge_attr_for(edge_index)

        h = x.float()
        for conv in self.clf.convs:
            h = conv(h, edge_index, edge_attr=edge_attr, edge_weight=edge_weight)

        h = global_add_pool(h, batch)
        h = self.clf.post_pool(h)
        return self.clf.classifier(h)
