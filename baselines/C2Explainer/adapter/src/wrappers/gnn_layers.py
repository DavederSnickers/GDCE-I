"""edge_weight-aware subclasses of GINConv and GINEConv.

C2Explainer passes the learned mask as `edge_weight` directly to the model
(see cf_explainer/c2_explainer.py::_train). Standard PyG GINConv/GINEConv
do not accept an edge_weight argument. These subclasses preserve the parent
parameters (so a checkpoint trained with the parent class loads unchanged),
extend forward() to accept edge_weight, and multiply messages by it.
"""

from typing import Optional

import torch
from torch import Tensor
from torch_geometric.nn import GINConv, GINEConv


class GINConvW(GINConv):
    """GINConv that scales messages by an optional edge_weight."""

    def forward(self, x, edge_index, edge_weight: Optional[Tensor] = None, size=None):
        if isinstance(x, Tensor):
            x = (x, x)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=size)
        x_r = x[1]
        if x_r is not None:
            out = out + (1 + self.eps) * x_r
        return self.nn(out)

    def message(self, x_j: Tensor, edge_weight: Optional[Tensor] = None) -> Tensor:
        return x_j if edge_weight is None else x_j * edge_weight.view(-1, 1)


class GINEConvW(GINEConv):
    """GINEConv that scales messages by an optional edge_weight."""

    def forward(self, x, edge_index, edge_attr: Optional[Tensor] = None,
                edge_weight: Optional[Tensor] = None, size=None):
        if isinstance(x, Tensor):
            x = (x, x)
        if self.lin is not None:
            edge_attr = self.lin(edge_attr)
        if x[0].size(-1) != edge_attr.size(-1):
            raise ValueError(
                f"Node feature dim ({x[0].size(-1)}) and edge feature dim "
                f"({edge_attr.size(-1)}) do not match"
            )
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr,
                             edge_weight=edge_weight, size=size)
        x_r = x[1]
        if x_r is not None:
            out = out + (1 + self.eps) * x_r
        return self.nn(out)

    def message(self, x_j: Tensor, edge_attr: Tensor,
                edge_weight: Optional[Tensor] = None) -> Tensor:
        msg = (x_j + edge_attr).relu()
        return msg if edge_weight is None else msg * edge_weight.view(-1, 1)


def upgrade_gine_layers(module_list) -> None:
    """In-place: rebind every GINEConv in `module_list` to GINEConvW.

    Subclass shares the parent's __init__ signature & state, so we just swap
    the class pointer — no parameter copy needed.
    """
    for conv in module_list:
        if isinstance(conv, GINEConv) and not isinstance(conv, GINEConvW):
            conv.__class__ = GINEConvW


def upgrade_gin_layers(module_list) -> None:
    """In-place: rebind every GINConv in `module_list` to GINConvW."""
    for conv in module_list:
        if isinstance(conv, GINConv) and not isinstance(conv, GINConvW):
            conv.__class__ = GINConvW
