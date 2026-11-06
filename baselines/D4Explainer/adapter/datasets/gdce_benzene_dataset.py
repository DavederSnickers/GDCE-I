"""GDCE Benzene CSVs wrapped for D4Explainer.

Benzene (Sanchez-Lengeling et al., NeurIPS 2020; repackaged in GraphXAI) as
preprocessed by GDCE. Label: does the molecule contain a benzene ring.

CSV schema: ``graph_id, y, adjacency_matrix, edge_type_matrix, node_types,
gt_ring_nodes``
  * adjacency_matrix: flattened n*n binary. **No self-loops** -- unlike the
    nci1 CSVs, benzene follows the molecular convention and leaves the diagonal
    at 0, so nothing has to be stripped here.
  * edge_type_matrix:  flattened n*n, entries in {0..4}. Ignored: D4Explainer's
    GNN takes no edge features, matching how it consumes mutagenicity_mol.
  * node_types:        Python-list literal of remapped 0-indexed atom types
  * gt_ring_nodes:     ground-truth ring membership mask; unused here.

8 atom types after GDCE's Huang frequency filter: C, N, O, S, F, Cl, Br, I.
"""
import ast
import os.path as osp

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, InMemoryDataset

NUM_NODE_TYPES = 8

_FILE_MAP = {'training': 'train.csv', 'evaluation': 'val.csv', 'testing': 'test.csv'}


class GDCEBenzeneDataset(InMemoryDataset):
    """GDCE Benzene CSVs (8 atom types) wrapped for D4Explainer."""

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
        return [f'gdce_benzene_{self.mode}.pt']

    def download(self):
        src = osp.join(self.raw_dir, self._csv_file)
        if not osp.exists(src):
            raise FileNotFoundError(
                f"Expected GDCE Benzene CSV at: {src}\n"
                f"Copy GDCE's data/benzene/raw/{self._csv_file} there and re-run."
            )

    def process(self):
        df = pd.read_csv(osp.join(self.raw_dir, self._csv_file))
        data_list = []
        for _, row in df.iterrows():
            node_types = ast.literal_eval(row['node_types'])
            n = len(node_types)
            adj = np.asarray(ast.literal_eval(row['adjacency_matrix']),
                             dtype=np.float32).reshape(n, n)
            # No fill_diagonal here on purpose: benzene stores no self-loops.
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
