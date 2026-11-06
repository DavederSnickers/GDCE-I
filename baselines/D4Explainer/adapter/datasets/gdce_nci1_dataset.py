"""GDCE-preprocessed NCI1 (freq, 10 atom types) loader for D4Explainer.

Reads GDCE's ``data/nci1/raw/{train,val,test}.csv`` (produced by
``GDCE/src/datasets/spectre_dataset.py`` for dataset name ``nci1`` = Huang et al.
atom-type frequency filter >50 -> **10** atom types, graphs <=50 nodes). This is
the *freq* variant chosen for the paper's NCI1 comparison, so D4's counterfactuals
land on the exact same graphs / test split as GDCE-I's ``inversion_nci1.py`` and
the other baselines.

Each graph becomes a PyG ``Data`` with:
  x          : [N, 10] one-hot atom types (remapped 0..9)
  edge_index : [2, E]  undirected COO, **self-loops stripped**
  y          : [1]     long class label (0=Inactive, 1=Active)

CSV schema: ``graph_id, y, adjacency_matrix, node_types``
  * adjacency_matrix: flattened n*n binary, **includes self-loops** (diagonal=1)
  * node_types:       Python-list literal of remapped 0-indexed atom-type indices

Place the CSVs at ``<root>/raw/{train,val,test}.csv`` (copy them from GDCE's
``data/nci1/raw/``). D4's own oracle (gnns/nci1_gnn.py) is trained on this
dataset -- per the baseline protocol each method uses its own paper's oracle.
"""

import ast
import os.path as osp

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, InMemoryDataset

# freq variant: Huang atom-type freq>50 filter yields exactly 10 atom types.
NUM_NODE_TYPES = 10

_FILE_MAP = {'training': 'train.csv', 'evaluation': 'val.csv', 'testing': 'test.csv'}


class GDCENci1Dataset(InMemoryDataset):
    """GDCE freq-NCI1 CSVs (10 atom types) wrapped for D4Explainer."""

    splits = list(_FILE_MAP.keys())

    def __init__(self, root, mode='testing', transform=None,
                 pre_transform=None, pre_filter=None):
        assert mode in self.splits, f"mode must be one of {self.splits}"
        self.mode = mode
        self._csv_file = _FILE_MAP[mode]
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [self._csv_file]

    @property
    def processed_file_names(self):
        return [f'gdce_nci1_{self.mode}.pt']

    def download(self):
        src = osp.join(self.raw_dir, self._csv_file)
        if not osp.exists(src):
            raise FileNotFoundError(
                f"Expected GDCE freq-NCI1 CSV at: {src}\n"
                f"Copy GDCE's data/nci1/raw/{self._csv_file} there and re-run."
            )

    def process(self):
        df = pd.read_csv(osp.join(self.raw_dir, self._csv_file))
        data_list = []
        for _, row in df.iterrows():
            node_types = ast.literal_eval(row['node_types'])
            n = len(node_types)
            adj = np.asarray(ast.literal_eval(row['adjacency_matrix']),
                             dtype=np.float32).reshape(n, n)
            np.fill_diagonal(adj, 0.0)  # strip GDCE's self-loops
            src, dst = np.nonzero(adj > 0.5)  # symmetric -> both directions kept
            edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long)
            nt = torch.tensor(node_types, dtype=torch.long)
            x = F.one_hot(nt, num_classes=NUM_NODE_TYPES).float()
            y = torch.tensor([int(row['y'])], dtype=torch.long)
            data_list.append(Data(x=x, edge_index=edge_index, y=y))

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        torch.save(self.collate(data_list), self.processed_paths[0])
