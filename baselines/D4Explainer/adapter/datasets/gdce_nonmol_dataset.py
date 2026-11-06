"""GDCE-preprocessed non-molecular datasets (PROTEINS, TWITTER) for D4Explainer.

Same CSV contract as ``gdce_nci1_dataset.py`` -- ``graph_id, y, adjacency_matrix,
node_types`` with a flattened n*n adjacency that still carries GDCE's self-loops
on the diagonal -- so one loader covers both. The only per-dataset difference is
the size of the node-type vocabulary:

  proteins : 3   (helix / sheet / turn)
  twitter  : 238 (words kept after the 500-occurrence frequency filter)

Neither dataset has edge types, which is exactly the representation D4's
generator works in, so no bond handling is needed here.

Place the CSVs at ``<root>/raw/{train,val,test}.csv`` (copy them from GDCE's
``data/proteins/raw/`` resp. ``data/twitter/raw/``).
"""

import ast
import os.path as osp

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, InMemoryDataset

# Node-type vocabulary sizes, matching GDCE's node_type_map.npy and the shared
# judges' num_features. A mismatch here silently shifts every one-hot column, so
# these are asserted against the CSVs in process().
NUM_NODE_TYPES = {
    'proteins': 3,
    'twitter': 238,
}

_FILE_MAP = {'training': 'train.csv', 'evaluation': 'val.csv', 'testing': 'test.csv'}


class GDCENonMolDataset(InMemoryDataset):
    """GDCE CSVs for a non-molecular, binary-adjacency dataset."""

    splits = list(_FILE_MAP.keys())

    def __init__(self, root, name, mode='testing', transform=None,
                 pre_transform=None, pre_filter=None):
        assert mode in self.splits, f"mode must be one of {self.splits}"
        assert name in NUM_NODE_TYPES, f"name must be one of {list(NUM_NODE_TYPES)}"
        self.name = name
        self.mode = mode
        self._csv_file = _FILE_MAP[mode]
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [self._csv_file]

    @property
    def processed_file_names(self):
        return [f'gdce_{self.name}_{self.mode}.pt']

    def download(self):
        src = osp.join(self.raw_dir, self._csv_file)
        if not osp.exists(src):
            raise FileNotFoundError(
                f"Expected GDCE {self.name} CSV at: {src}\n"
                f"Copy GDCE's data/{self.name}/raw/{self._csv_file} there and re-run."
            )

    def process(self):
        n_types = NUM_NODE_TYPES[self.name]
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
            if int(nt.max()) >= n_types:
                raise ValueError(
                    f"{self.name}: node type {int(nt.max())} exceeds the declared "
                    f"vocabulary size {n_types} -- NUM_NODE_TYPES is out of sync "
                    f"with the CSVs."
                )
            x = F.one_hot(nt, num_classes=n_types).float()
            y = torch.tensor([int(row['y'])], dtype=torch.long)
            data_list.append(Data(x=x, edge_index=edge_index, y=y))

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        torch.save(self.collate(data_list), self.processed_paths[0])


class GDCEProteinsDataset(GDCENonMolDataset):
    def __init__(self, root, mode='testing', **kw):
        super().__init__(root, name='proteins', mode=mode, **kw)


class GDCETwitterDataset(GDCENonMolDataset):
    def __init__(self, root, mode='testing', **kw):
        super().__init__(root, name='twitter', mode=mode, **kw)
