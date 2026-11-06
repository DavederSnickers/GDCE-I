"""GDCE CSV-backed graph dataset for COMBINEX.

Reads the preprocessed CSVs produced by GDCE's `SpectreGraphDataset.download()`
(`mutagenicity_mol`, `nci1_nofreq`) and exposes them as a single PyG
`InMemoryDataset` with `.train_mask` / `.test_mask` bool tensors over the
concatenated train+val+test set.

CSV schema (replicated from GDCE/src/datasets/spectre_dataset.py):

- nci1_nofreq:
    graph_id, y, adjacency_matrix, node_types
    * adjacency_matrix: flattened n*n binary, **includes self-loops** (diagonal=1)
    * node_types: Python-list literal of remapped 0-indexed atom-type indices

- mutagenicity_mol:
    graph_id, y, adjacency_matrix, edge_type_matrix, node_types
    * adjacency_matrix: flattened n*n binary, **no self-loops**
    * edge_type_matrix: flattened n*n int matrix, entries in {0..4}
      (0 = no bond, 1 single, 2 double, 3 triple, 4 aromatic)
    * node_types: Python-list literal of remapped 0-indexed atom-type indices

Both datasets are pre-filtered to `num_nodes <= 50` by GDCE. We re-apply the
filter at load time as a safety net (configurable via `max_nodes`).
"""

from __future__ import annotations

import ast
import os.path as osp
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, InMemoryDataset


_SUPPORTED = ("mutagenicity_mol", "nci1_nofreq", "nci1", "benzene",
              "proteins", "twitter")
# proteins: 3 secondary-structure labels. twitter: word vocabulary after the
# frequency filter at 500 occurrences (238 types). Both are read from
# node_type_map.npy at load time; these are only the fallbacks.
_NUM_NODE_TYPES_FALLBACK = {"mutagenicity_mol": 13, "nci1_nofreq": 37, "nci1": 10,
                            "benzene": 8, "proteins": 3, "twitter": 238}
# benzene carries the same 5-class bond encoding as mutagenicity_mol;
# the non-molecular datasets are plain binary adjacency like nci1.
_NUM_EDGE_TYPES = {"mutagenicity_mol": 5, "nci1_nofreq": 0, "nci1": 0, "benzene": 5,
                   "proteins": 0, "twitter": 0}


class GDCECsvGraphDataset(InMemoryDataset):
    """One InMemoryDataset spanning the concatenated train+val+test CSVs.

    The split assignment is recovered via `self.train_mask` / `self.test_mask`
    (bool tensors of length `len(self)`). Val rows are merged into the train
    split because COMBINEX's GraphTrainer only consumes train+test.
    """

    def __init__(
        self,
        root: str,
        dataset_name: str,
        max_nodes: int = 50,
        num_node_types: Optional[int] = None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        if dataset_name not in _SUPPORTED:
            raise ValueError(
                f"dataset_name must be one of {_SUPPORTED}, got {dataset_name!r}"
            )
        self.dataset_name = dataset_name
        self.max_nodes = int(max_nodes)
        self._num_node_types_override = num_node_types
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        # Masks are saved alongside the collated data.
        masks = torch.load(self.processed_paths[1], weights_only=False)
        self.train_mask: torch.Tensor = masks["train_mask"]
        self.test_mask: torch.Tensor = masks["test_mask"]
        self.num_node_types: int = int(masks["num_node_types"])
        self.num_edge_types: int = int(masks["num_edge_types"])

    # ---------- PyG plumbing ----------

    @property
    def raw_file_names(self):
        return ["train.csv", "val.csv", "test.csv"]

    @property
    def processed_file_names(self):
        suffix = f"_max{self.max_nodes}"
        return [
            f"combinex_{self.dataset_name}{suffix}.pt",
            f"combinex_{self.dataset_name}{suffix}_masks.pt",
        ]

    def download(self):
        # CSVs are produced upstream by GDCE; we never download anything.
        missing = [p for p in self.raw_paths if not osp.exists(p)]
        if missing:
            raise FileNotFoundError(
                f"Missing raw CSVs for {self.dataset_name}: {missing}. "
                "These must be produced by GDCE/src/datasets/spectre_dataset.py "
                "and live under <root>/raw/."
            )

    def process(self):
        num_node_types = self._resolve_num_node_types()
        num_edge_types = _NUM_EDGE_TYPES[self.dataset_name]

        data_list = []
        split_of_row: list[str] = []

        for split, csv_name in zip(("train", "val", "test"), self.raw_file_names):
            csv_path = osp.join(self.raw_dir, csv_name)
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                data = self._row_to_data(row, num_node_types, num_edge_types)
                if data is None:
                    continue
                data_list.append(data)
                split_of_row.append(split)

        train_mask = torch.tensor(
            [s in ("train", "val") for s in split_of_row], dtype=torch.bool
        )
        test_mask = torch.tensor(
            [s == "test" for s in split_of_row], dtype=torch.bool
        )

        collated, slices = self.collate(data_list)
        torch.save((collated, slices), self.processed_paths[0])
        torch.save(
            {
                "train_mask": train_mask,
                "test_mask": test_mask,
                "num_node_types": num_node_types,
                "num_edge_types": num_edge_types,
            },
            self.processed_paths[1],
        )

    # ---------- helpers ----------

    def _resolve_num_node_types(self) -> int:
        if self._num_node_types_override is not None:
            return int(self._num_node_types_override)
        map_path = osp.join(self.raw_dir, "node_type_map.npy")
        if osp.exists(map_path):
            return int(len(np.load(map_path)))
        return _NUM_NODE_TYPES_FALLBACK[self.dataset_name]

    def _row_to_data(
        self, row, num_node_types: int, num_edge_types: int
    ) -> Optional[Data]:
        adj_flat = np.fromstring(row["adjacency_matrix"].strip("[]"), sep=",")
        n = int(round(np.sqrt(len(adj_flat))))
        if n * n != len(adj_flat):
            raise ValueError(
                f"adjacency_matrix is not square (len={len(adj_flat)}) "
                f"for graph_id={row.get('graph_id')}"
            )
        if n > self.max_nodes:
            return None

        adj = torch.from_numpy(adj_flat.reshape(n, n)).to(torch.float32)
        edge_index = (adj > 0).nonzero(as_tuple=False).t().contiguous()

        node_type_indices = ast.literal_eval(row["node_types"])
        node_types_tensor = torch.tensor(node_type_indices, dtype=torch.long)
        if node_types_tensor.shape[0] != n:
            raise ValueError(
                f"node_types length {node_types_tensor.shape[0]} != n={n} "
                f"for graph_id={row.get('graph_id')}"
            )
        x = F.one_hot(node_types_tensor, num_classes=num_node_types).float()

        y_val = int(row["y"]) if "y" in row else 0
        y = torch.tensor([y_val], dtype=torch.long)

        # Gate on the declared edge-type count, not the dataset name: benzene
        # carries the same 5-class bond encoding as mutagenicity_mol, and
        # keying off the name silently dropped its edge_attr.
        if num_edge_types > 0:
            et_flat = np.fromstring(
                row["edge_type_matrix"].strip("[]"), sep=","
            ).astype(np.int64)
            et = torch.from_numpy(et_flat.reshape(n, n))
            if edge_index.numel() > 0:
                et_vals = et[edge_index[0], edge_index[1]]
                edge_attr = F.one_hot(et_vals, num_classes=num_edge_types).float()
            else:
                edge_attr = torch.zeros((0, num_edge_types), dtype=torch.float32)
            return Data(
                x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, num_nodes=n
            )

        # nci1 / nci1_nofreq: adjacency includes self-loops; no edge features.
        return Data(x=x, edge_index=edge_index, y=y, num_nodes=n)
