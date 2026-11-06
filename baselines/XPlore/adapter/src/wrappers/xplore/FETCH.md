# Four files of this package are not in this repository

XPlore is a control experiment in Section S3. Its algorithm is the authors' code
and carries **no licence**, and the two instance classes it builds its
counterfactual in come from GRETEL **v2**, which is not the GRETEL commit this
port pins. Neither may be redistributed here, so this directory ships only what
is ours — `framework.py` and `__init__.py` — plus this recipe.

| file | where it comes from |
|---|---|
| `XPlore.py` | the authors' file, `https://anonymous.4open.science/r/XPlore` (Beyond Edge Deletion, KDD'26) |
| `XPlore_signfix.py` | a one-token variant of `XPlore.py`, see below |
| `graph_instance.py` | GRETEL v2, `src/dataset/instances/graph.py` (`GraphInstance`) |
| `data_instance.py` | GRETEL v2, `src/dataset/instances/base.py` (`DataInstance`) |

## 1. `XPlore.py`

Fetch the authors' `XPlore.py` and replace **its import header only** — every
line from the first `import` down to, but not including, `class XPlore(Explainer):`
— with the block below. Nothing else in the file is touched; the algorithm stays
byte-identical to the authors'.

```python
import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from src.wrappers.xplore.graph_instance import GraphInstance
from src.wrappers.xplore.framework import (Explainer, retake_oracle,
                                           BaseManipulator, NodeCentrality,
                                           torch_diffusion_namespace)
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
import os

# XPlore imports `src.oracle.nn.torch_diffusion` for exactly one purpose: the
# `isinstance(self.oracle, ...OracleTorch)` test that switches it into GRETEL's
# diffusion-oracle branch. Our oracle is the shared judge, so that test must
# come out False. Binding the name to a sentinel namespace keeps the check --
# and every other line of the algorithm -- exactly as the authors wrote it,
# without pulling GRETEL v2's Trainable/Context machinery into the port.
src = torch_diffusion_namespace()
```

What that replaces: the original header imports GRETEL v2's `Explainer`,
`retake_oracle`, `BaseManipulator`, `NodeCentrality` and
`src.oracle.nn.torch_diffusion`. `framework.py` in this directory supplies those
four symbols directly — its docstring says what each one does and what is kept —
because GRETEL v2's `Explainer` pulls in `Configurable` → `Context` →
`flufl.lock` + `jsonc_parser` and the whole JSONC factory machinery, none of
which is part of the algorithm and none of which survives contact with the
pinned GRETEL v1 checkout.

## 2. `XPlore_signfix.py`

Copy the patched `XPlore.py` to `XPlore_signfix.py` and change exactly one
character, the sign of the node-feature update:

```python
self.P_node_hat -= self.alpha * self.P_node_hat.grad     # authors'
self.P_node_hat += self.alpha * self.P_node_hat.grad     # here
```

Why it exists: the two perturbation matrices in the authors' loop move in
opposite directions on the same objective,

```
total_loss  = CE(logits, f_v) + beta * D_edges
P_hat.data += alpha * vel                 -> ascent  -> away from f_v
P_node_hat -= alpha * P_node_hat.grad     -> descent -> back toward f_v
```

so the node-feature channel undoes what the edge channel achieves. The variant
tests whether that is why XPlore++ collapses. Nothing else differs.

## 3. `graph_instance.py` and `data_instance.py`

Take `GraphInstance` and `DataInstance` from GRETEL v2 and put them here as
`graph_instance.py` and `data_instance.py`, with their imports pointed at each
other (`from src.wrappers.xplore.data_instance import DataInstance`). They are
plain data containers: an id, a label, a numpy adjacency, node and edge
features, and a `deepcopy`-based `copy()`. XPlore constructs its counterfactual
as one of these, which is the only reason the port needs them.
