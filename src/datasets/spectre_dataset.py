import os
import pathlib
import pandas as pd
from typing import Any, Sequence
import os.path as osp
import numpy as np
import networkx as nx

import torch
from torch.utils.data import random_split
import torch_geometric.utils
from torch_geometric.data import Data, InMemoryDataset, download_url
from tqdm import tqdm
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from src.datasets.guided_data import GuidedInMemoryDataset

from src.datasets.abstract_dataset_1 import AbstractDataModule, AbstractDatasetInfos


class RemoveYTransform:
    def __call__(self, data):
        data.guidance = torch.zeros((1, 0), dtype=torch.float)
        return data


def files_exist(files) -> bool:
    # NOTE: We return `False` in case `files` is empty, leading to a
    # re-processing of files on every instantiation.
    return len(files) != 0 and all([osp.exists(f) for f in files])


def to_list(value: Any) -> Sequence:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    else:
        return [value]


class SpectreGraphDataset(InMemoryDataset):
    def __init__(self, dataset_name, split, root, transform=None, pre_transform=None, pre_filter=None):
        self.split = split
        if self.split == 'train':
            self.file_idx = 0
        elif self.split == 'val':
            self.file_idx = 1
        else:
            self.file_idx = 2
        self.dataset_name = dataset_name
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[self.file_idx])

    @property
    def split_file_name(self):
        return ['train.csv', 'val.csv', 'test.csv']

    @property
    def split_paths(self):
        r"""The absolute filepaths that must be present in order to skip
        splitting."""
        files = to_list(self.split_file_name)
        return [osp.join(self.raw_dir, f) for f in files]

    @property
    def raw_file_names(self):
        return ['train.csv', 'val.csv', 'test.csv']

    @property
    def processed_file_names(self):
            return ['train.pt', 'val.pt', 'test.pt']

    def download(self):

        if self.dataset_name == 'planar':
            raw_url = 'data/planar_8/raw/planar_david_8.pt'
            file_path = raw_url
            dataset = torch.load(file_path)
            adjs = dataset['adjs']

            # Automatically determine number of graphs
            self.num_graphs = len(adjs)
            print(f"Total number of graphs: {self.num_graphs}")

            # Shuffle the adjacency matrices
            shuffled_indices = torch.randperm(self.num_graphs)
            adjs = [adjs[i] for i in shuffled_indices]

            # Split the data into train (64%), val (16%), and test (20%)
            test_len = int(round(self.num_graphs * 0.2))
            train_len = int(round((self.num_graphs - test_len) * 0.8))
            val_len = self.num_graphs - train_len - test_len

            train_data = adjs[:train_len]
            val_data = adjs[train_len:train_len + val_len]
            test_data = adjs[train_len + val_len:]

            print(f'Dataset sizes: train {len(train_data)}, val {len(val_data)}, test {len(test_data)}')

            # Function to convert adjacency matrices to CSV-friendly format
            def save_to_csv(data_list, file_path):
                rows = []
                for idx, adj in enumerate(data_list):
                    adj_flat = adj.cpu().numpy().flatten()  # Flatten the adjacency matrix
                    row = {'graph_id': idx, 'adjacency_matrix': adj_flat.tolist()}
                    rows.append(row)

                df = pd.DataFrame(rows)
                df.to_csv(file_path, index=False)
                print(f"Saved {len(data_list)} graphs to {file_path}")

                # Save datasets to CSV files

            save_to_csv(train_data, os.path.join(self.raw_dir, 'train.csv'))
            save_to_csv(val_data, os.path.join(self.raw_dir, 'val.csv'))
            save_to_csv(test_data, os.path.join(self.raw_dir, 'test.csv'))
            return

        elif self.dataset_name == 'nci1':
            from torch_geometric.datasets import TUDataset
            from torch_geometric.utils import to_scipy_sparse_matrix
            from collections import Counter

            max_num_nodes = 50
            freq_threshold = 50  # Huang et al.: keep atom types with global count > 50

            tu_dataset = TUDataset(root=osp.join(self.raw_dir, 'TU'), name='NCI1')

            # --- Pass 1: count atom-type frequencies across the FULL dataset ---
            type_counter = Counter()
            for graph in tu_dataset:
                type_counter.update(graph.x.argmax(dim=1).cpu().numpy().tolist())

            kept_types = sorted(t for t, c in type_counter.items() if c > freq_threshold)
            old_to_new = {old: new for new, old in enumerate(kept_types)}
            num_node_types = len(kept_types)

            np.save(osp.join(self.raw_dir, 'node_type_map.npy'), np.array(kept_types))
            print(f'[NCI1] Atom-type freq filter (>{freq_threshold}): '
                  f'kept {num_node_types}/{len(type_counter)} types → {kept_types}')

            # --- Pass 2: collect graphs that pass both filters ---
            kept_types_set = set(kept_types)
            adj_all, label_all, node_types_all = [], [], []
            n_skipped_size, n_skipped_rare = 0, 0

            for graph in tu_dataset:
                n = graph.num_nodes
                if n > max_num_nodes:
                    n_skipped_size += 1
                    continue
                types = graph.x.argmax(dim=1).cpu().numpy().tolist()
                if not all(t in kept_types_set for t in types):
                    n_skipped_rare += 1
                    continue

                adj = to_scipy_sparse_matrix(graph.edge_index, num_nodes=n).toarray()
                np.fill_diagonal(adj, 1.0)
                adj_all.append(adj)
                label_all.append(int(graph.y.item()))
                node_types_all.append([old_to_new[t] for t in types])

            num_graphs = len(adj_all)
            print(f'[NCI1] Graphs kept: {num_graphs}  '
                  f'(skipped: {n_skipped_size} too large, {n_skipped_rare} rare atoms)')

            # Shuffle and split 60/20/20
            perm = np.random.permutation(num_graphs)
            adj_all = [adj_all[i] for i in perm]
            label_all = [label_all[i] for i in perm]
            node_types_all = [node_types_all[i] for i in perm]

            train_len = int(round(num_graphs * 0.6))
            val_len = int(round(num_graphs * 0.2))

            def save_to_csv_nci1(adj_list, label_list, nt_list, file_path):
                rows = []
                for idx, (adj, y, nt) in enumerate(zip(adj_list, label_list, nt_list)):
                    rows.append({
                        'graph_id': idx,
                        'y': int(y),
                        'adjacency_matrix': adj.flatten().tolist(),
                        'node_types': nt
                    })
                pd.DataFrame(rows).to_csv(file_path, index=False)

            save_to_csv_nci1(adj_all[:train_len], label_all[:train_len],
                             node_types_all[:train_len],
                             osp.join(self.raw_dir, 'train.csv'))
            save_to_csv_nci1(adj_all[train_len:train_len + val_len],
                             label_all[train_len:train_len + val_len],
                             node_types_all[train_len:train_len + val_len],
                             osp.join(self.raw_dir, 'val.csv'))
            save_to_csv_nci1(adj_all[train_len + val_len:],
                             label_all[train_len + val_len:],
                             node_types_all[train_len + val_len:],
                             osp.join(self.raw_dir, 'test.csv'))
            print(f'[NCI1] Splits: train {train_len}, val {val_len}, '
                  f'test {num_graphs - train_len - val_len}')
            return

        elif self.dataset_name == 'nci1_nofreq':
            from torch_geometric.datasets import TUDataset
            from torch_geometric.utils import to_scipy_sparse_matrix
            from collections import Counter

            max_num_nodes = 50  # only size filter; no frequency filtering

            tu_dataset = TUDataset(root=osp.join(self.raw_dir, 'TU'), name='NCI1')

            # --- Label balance before any filtering ---
            all_labels = [int(g.y.item()) for g in tu_dataset]
            n_total = len(all_labels)
            n_active = sum(all_labels)
            n_inactive = n_total - n_active
            print(f'[NCI1-nofreq] Full dataset: {n_total} graphs  |  '
                  f'Inactive (0): {n_inactive} ({100*n_inactive/n_total:.1f}%)  |  '
                  f'Active (1): {n_active} ({100*n_active/n_total:.1f}%)')

            # --- Single pass: keep graphs with <= max_num_nodes only ---
            adj_all, label_all, node_types_all = [], [], []
            type_counts = Counter()
            n_skipped_size = 0

            for graph in tu_dataset:
                if graph.num_nodes > max_num_nodes:
                    n_skipped_size += 1
                    continue
                types = graph.x.argmax(dim=1).cpu().numpy().tolist()
                type_counts.update(types)
                n = graph.num_nodes
                adj = to_scipy_sparse_matrix(graph.edge_index, num_nodes=n).toarray()
                np.fill_diagonal(adj, 1.0)
                adj_all.append(adj)
                label_all.append(int(graph.y.item()))
                node_types_all.append(types)

            kept_types = sorted(type_counts.keys())
            old_to_new = {old: new for new, old in enumerate(kept_types)}
            num_node_types = len(kept_types)

            # Remap node type indices to contiguous 0..num_node_types-1
            node_types_all = [[old_to_new[t] for t in nt] for nt in node_types_all]

            # Compute marginals over kept graphs
            total_nodes = sum(type_counts.values())
            marginals = np.array([type_counts[t] / total_nodes for t in kept_types], dtype=np.float32)

            np.save(osp.join(self.raw_dir, 'node_type_map.npy'), np.array(kept_types))
            np.save(osp.join(self.raw_dir, 'node_type_marginals.npy'), marginals)
            print(f'[NCI1-nofreq] No freq filter: {num_node_types} atom types → {kept_types}')
            print(f'[NCI1-nofreq] Node-type marginals: {np.round(marginals, 4).tolist()}')

            # Label balance after size filter
            kept_labels = label_all
            n_kept = len(kept_labels)
            n_act = sum(kept_labels)
            n_inact = n_kept - n_act
            print(f'[NCI1-nofreq] After size filter: {n_kept} graphs  |  '
                  f'Inactive (0): {n_inact} ({100*n_inact/n_kept:.1f}%)  |  '
                  f'Active (1): {n_act} ({100*n_act/n_kept:.1f}%)  |  '
                  f'(skipped: {n_skipped_size} too large)')

            num_graphs = len(adj_all)
            perm = np.random.permutation(num_graphs)
            adj_all = [adj_all[i] for i in perm]
            label_all = [label_all[i] for i in perm]
            node_types_all = [node_types_all[i] for i in perm]

            train_len = int(round(num_graphs * 0.6))
            val_len = int(round(num_graphs * 0.2))

            def save_to_csv_nci1_nofreq(adj_list, label_list, nt_list, file_path):
                rows = []
                for idx, (adj, y, nt) in enumerate(zip(adj_list, label_list, nt_list)):
                    rows.append({
                        'graph_id': idx,
                        'y': int(y),
                        'adjacency_matrix': adj.flatten().tolist(),
                        'node_types': nt
                    })
                pd.DataFrame(rows).to_csv(file_path, index=False)

            save_to_csv_nci1_nofreq(adj_all[:train_len], label_all[:train_len],
                                    node_types_all[:train_len],
                                    osp.join(self.raw_dir, 'train.csv'))
            save_to_csv_nci1_nofreq(adj_all[train_len:train_len + val_len],
                                    label_all[train_len:train_len + val_len],
                                    node_types_all[train_len:train_len + val_len],
                                    osp.join(self.raw_dir, 'val.csv'))
            save_to_csv_nci1_nofreq(adj_all[train_len + val_len:],
                                    label_all[train_len + val_len:],
                                    node_types_all[train_len + val_len:],
                                    osp.join(self.raw_dir, 'test.csv'))
            print(f'[NCI1-nofreq] Splits: train {train_len}, val {val_len}, '
                  f'test {num_graphs - train_len - val_len}')
            return

        elif self.dataset_name == 'aids':
            from torch_geometric.datasets import TUDataset
            from torch_geometric.utils import to_scipy_sparse_matrix
            from collections import Counter

            max_num_nodes = 50  # size filter only; no frequency filtering

            tu_dataset = TUDataset(root=osp.join(self.raw_dir, 'TU'), name='AIDS')

            # --- Label balance before any filtering ---
            all_labels = [int(g.y.item()) for g in tu_dataset]
            n_total = len(all_labels)
            n_active = sum(all_labels)
            n_inactive = n_total - n_active
            print(f'[AIDS] Full dataset: {n_total} graphs  |  '
                  f'Inactive (0): {n_inactive} ({100*n_inactive/n_total:.1f}%)  |  '
                  f'Active (1): {n_active} ({100*n_active/n_total:.1f}%)')

            # --- Single pass: keep graphs with <= max_num_nodes only ---
            adj_all, label_all, node_types_all = [], [], []
            type_counts = Counter()
            n_skipped_size = 0

            for graph in tu_dataset:
                if graph.num_nodes > max_num_nodes:
                    n_skipped_size += 1
                    continue
                types = graph.x.argmax(dim=1).cpu().numpy().tolist()
                type_counts.update(types)
                n = graph.num_nodes
                adj = to_scipy_sparse_matrix(graph.edge_index, num_nodes=n).toarray()
                np.fill_diagonal(adj, 1.0)
                adj_all.append(adj)
                label_all.append(int(graph.y.item()))
                node_types_all.append(types)

            kept_types = sorted(type_counts.keys())
            old_to_new = {old: new for new, old in enumerate(kept_types)}
            num_node_types = len(kept_types)

            # Remap node type indices to contiguous 0..num_node_types-1
            node_types_all = [[old_to_new[t] for t in nt] for nt in node_types_all]

            # Compute marginals over kept graphs
            total_nodes = sum(type_counts.values())
            marginals = np.array([type_counts[t] / total_nodes for t in kept_types], dtype=np.float32)

            np.save(osp.join(self.raw_dir, 'node_type_map.npy'), np.array(kept_types))
            np.save(osp.join(self.raw_dir, 'node_type_marginals.npy'), marginals)
            print(f'[AIDS] No freq filter: {num_node_types} atom types → {kept_types}')
            print(f'[AIDS] Node-type marginals: {np.round(marginals, 4).tolist()}')

            # Label balance after size filter
            n_kept = len(label_all)
            n_act = sum(label_all)
            n_inact = n_kept - n_act
            print(f'[AIDS] After size filter: {n_kept} graphs  |  '
                  f'Inactive (0): {n_inact} ({100*n_inact/n_kept:.1f}%)  |  '
                  f'Active (1): {n_act} ({100*n_act/n_kept:.1f}%)  |  '
                  f'(skipped: {n_skipped_size} too large)')

            num_graphs = len(adj_all)
            perm = np.random.permutation(num_graphs)
            adj_all = [adj_all[i] for i in perm]
            label_all = [label_all[i] for i in perm]
            node_types_all = [node_types_all[i] for i in perm]

            train_len = int(round(num_graphs * 0.6))
            val_len = int(round(num_graphs * 0.2))

            def save_to_csv_aids(adj_list, label_list, nt_list, file_path):
                rows = []
                for idx, (adj, y, nt) in enumerate(zip(adj_list, label_list, nt_list)):
                    rows.append({
                        'graph_id': idx,
                        'y': int(y),
                        'adjacency_matrix': adj.flatten().tolist(),
                        'node_types': nt
                    })
                pd.DataFrame(rows).to_csv(file_path, index=False)

            save_to_csv_aids(adj_all[:train_len], label_all[:train_len],
                             node_types_all[:train_len],
                             osp.join(self.raw_dir, 'train.csv'))
            save_to_csv_aids(adj_all[train_len:train_len + val_len],
                             label_all[train_len:train_len + val_len],
                             node_types_all[train_len:train_len + val_len],
                             osp.join(self.raw_dir, 'val.csv'))
            save_to_csv_aids(adj_all[train_len + val_len:],
                             label_all[train_len + val_len:],
                             node_types_all[train_len + val_len:],
                             osp.join(self.raw_dir, 'test.csv'))
            print(f'[AIDS] Splits: train {train_len}, val {val_len}, '
                  f'test {num_graphs - train_len - val_len}')
            return

        elif self.dataset_name == 'ba2motifs':
            # BA-2Motifs (Luo et al., PGExplainer): 1000 synthetic graphs, 25 nodes each,
            # Barabasi-Albert base + either a house motif (class 0) or a 5-cycle (class 1).
            # BALANCED 50/50 by construction -> used as the control condition against the
            # severely imbalanced AIDS. Nodes carry no meaningful features in the original
            # benchmark, so we keep a SINGLE categorical node type (same convention as
            # planar_8, whose node_types is also a single class).
            from torch_geometric.datasets import BA2MotifDataset
            from torch_geometric.utils import to_scipy_sparse_matrix
            from collections import Counter

            ds = BA2MotifDataset(root=osp.join(self.raw_dir, 'BA2Motif'))
            adj_all, label_all, node_types_all = [], [], []
            n_edges_tot, n_pairs_tot = 0, 0
            for graph in ds:
                n = graph.num_nodes
                adj = to_scipy_sparse_matrix(graph.edge_index, num_nodes=n).toarray()
                adj = (adj > 0).astype(np.float32)
                np.fill_diagonal(adj, 1.0)          # same self-loop convention as nci1/aids
                adj_all.append(adj)
                label_all.append(int(graph.y.view(-1)[0].item()))
                node_types_all.append([0] * n)      # single node type
                off = adj.copy(); np.fill_diagonal(off, 0.0)
                n_edges_tot += int(off.sum() / 2)
                n_pairs_tot += n * (n - 1) // 2

            num_graphs = len(adj_all)
            counts = Counter(label_all)
            sizes = Counter(len(nt) for nt in node_types_all)
            edge_frac = n_edges_tot / n_pairs_tot if n_pairs_tot else 0.0
            print(f'[BA2Motifs] {num_graphs} graphs | class balance: '
                  f'{ {c: counts[c] for c in sorted(counts)} }')
            print(f'[BA2Motifs] node counts: { {k: sizes[k] for k in sorted(sizes)} }')
            print(f'[BA2Motifs] edge marginal (off-diagonal): '
                  f'[{1-edge_frac:.4f}, {edge_frac:.4f}]  <- put into SpectreDatasetInfos')

            np.save(osp.join(self.raw_dir, 'node_type_map.npy'), np.array([0]))
            np.save(osp.join(self.raw_dir, 'node_type_marginals.npy'),
                    np.array([1.0], dtype=np.float32))

            perm = np.random.permutation(num_graphs)
            adj_all = [adj_all[i] for i in perm]
            label_all = [label_all[i] for i in perm]
            node_types_all = [node_types_all[i] for i in perm]
            train_len = int(round(num_graphs * 0.6))
            val_len = int(round(num_graphs * 0.2))

            def save_to_csv_ba2(adj_list, label_list, nt_list, file_path):
                rows = []
                for idx, (adj, y, nt) in enumerate(zip(adj_list, label_list, nt_list)):
                    rows.append({'graph_id': idx, 'y': int(y),
                                 'adjacency_matrix': adj.flatten().tolist(),
                                 'node_types': nt})
                pd.DataFrame(rows).to_csv(file_path, index=False)

            save_to_csv_ba2(adj_all[:train_len], label_all[:train_len],
                            node_types_all[:train_len], osp.join(self.raw_dir, 'train.csv'))
            save_to_csv_ba2(adj_all[train_len:train_len + val_len],
                            label_all[train_len:train_len + val_len],
                            node_types_all[train_len:train_len + val_len],
                            osp.join(self.raw_dir, 'val.csv'))
            save_to_csv_ba2(adj_all[train_len + val_len:], label_all[train_len + val_len:],
                            node_types_all[train_len + val_len:],
                            osp.join(self.raw_dir, 'test.csv'))
            print(f'[BA2Motifs] Splits: train {train_len}, val {val_len}, '
                  f'test {num_graphs - train_len - val_len}')
            return

        elif self.dataset_name in ('proteins', 'imdb_binary', 'twitter'):
            # Non-molecular TUDatasets with a binary adjacency and no bond types.
            # Same representation as nci1_nofreq/aids, so process() reuses that path.
            #
            # PROTEINS carries three categorical node labels (helix / sheet / turn).
            # IMDB-BINARY carries no node or edge features at all; every node gets a
            # single constant type and only the topology is modelled. Degree buckets
            # are deliberately NOT used as node types: the degree is a deterministic
            # function of the topology, so editing node types and edges independently
            # would produce graphs whose node type contradicts their own structure.
            from torch_geometric.datasets import TUDataset
            from torch_geometric.utils import to_scipy_sparse_matrix
            from collections import Counter

            # min_type_count follows Huang et al.: node types occurring at most this
            # often globally are dropped, and every graph containing one is removed.
            # TWITTER carries a 1323-word vocabulary, which makes the (bs, n, dx, dx)
            # tensor in compute_batched_over0_posterior_distribution 11.7 GB at
            # batch 128. Threshold 500 leaves 238 types (360 MB) over 65031 graphs,
            # keeps 78.9% of node instances and the class balance at 0.52/0.48, and
            # is the largest threshold that preserves the original size range
            # (n_max 13; from 750 upwards the filter starts removing the biggest
            # graphs entirely). Measured with src/analyze_freq_filter.py.
            spec = {
                'proteins':    dict(tu_name='PROTEINS',    max_num_nodes=50, min_type_count=0),
                'imdb_binary': dict(tu_name='IMDB-BINARY', max_num_nodes=50, min_type_count=0),
                'twitter':     dict(tu_name='TWITTER-Real-Graph-Partial',
                                    max_num_nodes=50, min_type_count=500),
            }[self.dataset_name]
            tag = self.dataset_name.upper()

            tu_dataset = TUDataset(root=osp.join(self.raw_dir, 'TU'), name=spec['tu_name'])

            all_labels = [int(g.y.item()) for g in tu_dataset]
            n_total = len(all_labels)
            print(f'[{tag}] Full dataset: {n_total} graphs  |  '
                  f'class balance: {np.bincount(all_labels).tolist()}')

            adj_all, label_all, node_types_all = [], [], []
            type_counts = Counter()
            n_skipped_size = 0

            for graph in tu_dataset:
                if graph.num_nodes > spec['max_num_nodes']:
                    n_skipped_size += 1
                    continue
                n = graph.num_nodes
                if getattr(graph, 'x', None) is None:
                    types = [0] * n                      # featureless -> one constant type
                else:
                    types = graph.x.argmax(dim=1).cpu().numpy().tolist()
                type_counts.update(types)
                adj = to_scipy_sparse_matrix(graph.edge_index, num_nodes=n).toarray()
                adj = (adj > 0).astype(np.float64)       # TU graphs are unweighted
                np.fill_diagonal(adj, 1.0)               # same convention as aids/nci1
                adj_all.append(adj)
                label_all.append(int(graph.y.item()))
                node_types_all.append(types)

            if spec['min_type_count'] > 0:
                rare = {t for t, c in type_counts.items() if c <= spec['min_type_count']}
                keep = [i for i, nt in enumerate(node_types_all)
                        if not rare.intersection(nt)]
                n_before = len(node_types_all)
                adj_all = [adj_all[i] for i in keep]
                label_all = [label_all[i] for i in keep]
                node_types_all = [node_types_all[i] for i in keep]
                # Recount on the surviving graphs so the marginals match the data.
                type_counts = Counter()
                for nt in node_types_all:
                    type_counts.update(nt)
                print(f'[{tag}] Frequency filter (<= {spec["min_type_count"]} occurrences): '
                      f'{len(rare)} rare types dropped, {n_before - len(keep)} of '
                      f'{n_before} graphs removed for containing one')

            kept_types = sorted(type_counts.keys())
            old_to_new = {old: new for new, old in enumerate(kept_types)}
            node_types_all = [[old_to_new[t] for t in nt] for nt in node_types_all]
            num_node_types = len(kept_types)

            total_nodes = sum(type_counts.values())
            node_marginals = np.array([type_counts[t] / total_nodes for t in kept_types],
                                      dtype=np.float32)
            np.save(osp.join(self.raw_dir, 'node_type_map.npy'), np.array(kept_types))
            np.save(osp.join(self.raw_dir, 'node_type_marginals.npy'), node_marginals)
            print(f'[{tag}] {num_node_types} node type(s) -> {kept_types}')
            print(f'[{tag}] Node-type marginals: {np.round(node_marginals, 4).tolist()}')

            # Edge marginals over all n^2 entries, matching the benzene convention.
            total_entries = sum(a.shape[0] ** 2 for a in adj_all)
            n_edge_entries = float(sum(a.sum() for a in adj_all))
            edge_marginals = np.array(
                [(total_entries - n_edge_entries) / total_entries,
                 n_edge_entries / total_entries], dtype=np.float32)
            np.save(osp.join(self.raw_dir, 'edge_type_marginals.npy'), edge_marginals)
            print(f'[{tag}] Edge marginals (0=no-edge, 1=edge): '
                  f'{np.round(edge_marginals, 4).tolist()}')

            max_n = max(a.shape[0] for a in adj_all)
            n_nodes_hist = np.zeros(max_n + 1)
            for a in adj_all:
                n_nodes_hist[a.shape[0]] += 1
            np.save(osp.join(self.raw_dir, 'n_nodes_dist.npy'),
                    n_nodes_hist / n_nodes_hist.sum())

            n_kept = len(label_all)
            print(f'[{tag}] After size filter (<= {spec["max_num_nodes"]} nodes): '
                  f'{n_kept} graphs, balance {np.bincount(label_all).tolist()}, '
                  f'max {max_n} nodes  (skipped {n_skipped_size} too large)')

            num_graphs = len(adj_all)
            perm = np.random.permutation(num_graphs)
            adj_all = [adj_all[i] for i in perm]
            label_all = [label_all[i] for i in perm]
            node_types_all = [node_types_all[i] for i in perm]

            train_len = int(round(num_graphs * 0.6))
            val_len = int(round(num_graphs * 0.2))

            def save_to_csv_tu(adj_list, label_list, nt_list, file_path):
                rows = []
                for idx, (adj, y, nt) in enumerate(zip(adj_list, label_list, nt_list)):
                    rows.append({
                        'graph_id': idx,
                        'y': int(y),
                        'adjacency_matrix': adj.flatten().tolist(),
                        'node_types': nt,
                    })
                pd.DataFrame(rows).to_csv(file_path, index=False)

            save_to_csv_tu(adj_all[:train_len], label_all[:train_len],
                           node_types_all[:train_len],
                           osp.join(self.raw_dir, 'train.csv'))
            save_to_csv_tu(adj_all[train_len:train_len + val_len],
                           label_all[train_len:train_len + val_len],
                           node_types_all[train_len:train_len + val_len],
                           osp.join(self.raw_dir, 'val.csv'))
            save_to_csv_tu(adj_all[train_len + val_len:], label_all[train_len + val_len:],
                           node_types_all[train_len + val_len:],
                           osp.join(self.raw_dir, 'test.csv'))
            print(f'[{tag}] Splits: train {train_len}, val {val_len}, '
                  f'test {num_graphs - train_len - val_len}')
            return

        elif self.dataset_name == 'mutagenicity_mol':
            from torch_geometric.datasets import TUDataset
            from torch_geometric.utils import to_scipy_sparse_matrix
            from collections import Counter

            max_num_nodes = 50

            tu_dataset = TUDataset(root=osp.join(self.raw_dir, 'TU'), name='Mutagenicity')

            # --- Label balance before any filtering ---
            all_labels = [int(g.y.item()) for g in tu_dataset]
            n_total = len(all_labels)
            n_mut = sum(all_labels)
            n_non = n_total - n_mut
            print(f'[Mutagenicity_mol] Full dataset: {n_total} graphs  |  '
                  f'Non-mutagenic (0): {n_non} ({100*n_non/n_total:.1f}%)  |  '
                  f'Mutagenic (1): {n_mut} ({100*n_mut/n_total:.1f}%)')

            # --- Single pass: keep graphs with <= max_num_nodes nodes ---
            valid_graphs = []  # list of (graph, types_list)
            type_counter = Counter()
            n_skipped_size = 0

            for graph in tu_dataset:
                if graph.num_nodes > max_num_nodes:
                    n_skipped_size += 1
                    continue
                types = graph.x.argmax(dim=1).cpu().numpy().tolist()
                type_counter.update(types)
                valid_graphs.append((graph, types))

            # Build contiguous remapping
            kept_types = sorted(type_counter.keys())
            old_to_new = {old: new for new, old in enumerate(kept_types)}
            num_node_types = len(kept_types)

            np.save(osp.join(self.raw_dir, 'node_type_map.npy'), np.array(kept_types))
            print(f'[Mutagenicity_mol] Atom types: {num_node_types} → {kept_types}')
            print(f'[Mutagenicity_mol] Graphs kept: {len(valid_graphs)}  '
                  f'(skipped: {n_skipped_size} too large)')

            # ----------------------------------------------------------------
            # Save atom_decoder / atom_encoder for SMILES reconstruction.
            # Mutagenicity TUDataset node-label ordering (from
            # Mutagenicity_label_readme.txt — this is the AUTHORITATIVE legend):
            #   0=C, 1=O, 2=Cl, 3=H, 4=N, 5=F, 6=Br, 7=S, 8=P,
            #   9=I, 10=Na, 11=K, 12=Li, 13=Ca
            # VERIFY on first run: check printed atom_decoder vs. kept_types.
            # If something looks wrong, inspect:
            #   raw/TU/Mutagenicity/raw/node_labels.txt
            # Bond type encoding in edge_type_matrix:
            #   0=no bond, 1=single, 2=double, 3=triple, 4=aromatic
            # ----------------------------------------------------------------
            import json
            # Correct mapping from Mutagenicity_label_readme.txt:
            #   0=C, 1=O, 2=Cl, 3=H, 4=N, 5=F, 6=Br, 7=S, 8=P,
            #   9=I, 10=Na, 11=K, 12=Li, 13=Ca
            _MUTAG_IDX_TO_SYMBOL = {
                0: 'C', 1: 'O', 2: 'Cl', 3: 'H', 4: 'N',
                5: 'F', 6: 'Br', 7: 'S', 8: 'P', 9: 'I',
                10: 'Na', 11: 'K', 12: 'Li', 13: 'Ca',
            }
            # atom_decoder: remapped index (str) → element symbol
            atom_decoder = {
                str(new_idx): _MUTAG_IDX_TO_SYMBOL.get(old_idx, f'UNK{old_idx}')
                for old_idx, new_idx in old_to_new.items()
            }
            # atom_encoder: element symbol → remapped index
            atom_encoder = {sym: int(idx) for idx, sym in atom_decoder.items()}
            with open(osp.join(self.raw_dir, 'atom_decoder.json'), 'w') as f:
                json.dump(atom_decoder, f, indent=2)
            with open(osp.join(self.raw_dir, 'atom_encoder.json'), 'w') as f:
                json.dump(atom_encoder, f, indent=2)
            print(f'[Mutagenicity_mol] atom_decoder (remapped_idx → symbol): {atom_decoder}')
            print(f'[Mutagenicity_mol] Bond encoding: 0=no bond, 1=single, 2=double, 3=triple, 4=aromatic')

            # Compute and save node-type marginals
            type_counts = np.zeros(num_node_types)
            for _, types in valid_graphs:
                for t in types:
                    type_counts[old_to_new[t]] += 1
            node_marginals = type_counts / type_counts.sum()
            np.save(osp.join(self.raw_dir, 'node_type_marginals.npy'), node_marginals)
            print(f'[Mutagenicity_mol] Node-type marginals: {np.round(node_marginals, 4).tolist()}')

            # Build data arrays: adjacency_matrix (binary) + edge_type_matrix (0-4)
            adj_all, edge_type_all, label_all, node_types_all = [], [], [], []
            for graph, types in valid_graphs:
                n = graph.num_nodes

                # Binary adjacency (no self-loops — molecular convention)
                adj = to_scipy_sparse_matrix(graph.edge_index, num_nodes=n).toarray()
                # Do NOT fill diagonal (no self-loops)
                adj_all.append(adj)

                # Bond types from TUDataset edge_attr (shape [E, 4] one-hot → 1-indexed)
                et_mat = np.zeros((n, n), dtype=np.int64)
                if graph.edge_attr is not None and graph.edge_attr.numel() > 0:
                    bond_types = graph.edge_attr.argmax(dim=1).cpu().numpy()
                    src = graph.edge_index[0].cpu().numpy()
                    dst = graph.edge_index[1].cpu().numpy()
                    for s, d, bt in zip(src, dst, bond_types):
                        et_mat[s, d] = int(bt) + 1  # 0 reserved for no-bond
                edge_type_all.append(et_mat)

                label_all.append(int(graph.y.item()))
                node_types_all.append([old_to_new[t] for t in types])

            # Compute edge-type marginals over all N×N matrix entries (5 classes)
            total_entries = sum(et.shape[0] ** 2 for et in edge_type_all)
            edge_counts = np.zeros(5)
            for et in edge_type_all:
                for k in range(5):
                    edge_counts[k] += (et == k).sum()
            edge_marginals = edge_counts / total_entries
            np.save(osp.join(self.raw_dir, 'edge_type_marginals.npy'), edge_marginals)
            print(f'[Mutagenicity_mol] Edge-type marginals (0=no-bond, 1-4=bond): '
                  f'{np.round(edge_marginals, 4).tolist()}')

            num_graphs = len(adj_all)

            # Shuffle and split 60/20/20
            perm = np.random.permutation(num_graphs)
            adj_all = [adj_all[i] for i in perm]
            edge_type_all = [edge_type_all[i] for i in perm]
            label_all = [label_all[i] for i in perm]
            node_types_all = [node_types_all[i] for i in perm]

            train_len = int(round(num_graphs * 0.6))
            val_len = int(round(num_graphs * 0.2))

            def save_to_csv_mutag_mol(adj_list, et_list, label_list, nt_list, file_path):
                rows = []
                for idx, (adj, et, y, nt) in enumerate(zip(adj_list, et_list, label_list, nt_list)):
                    rows.append({
                        'graph_id': idx,
                        'y': int(y),
                        'adjacency_matrix': adj.flatten().tolist(),
                        'edge_type_matrix': et.flatten().tolist(),
                        'node_types': nt
                    })
                pd.DataFrame(rows).to_csv(file_path, index=False)

            save_to_csv_mutag_mol(adj_all[:train_len], edge_type_all[:train_len],
                                  label_all[:train_len], node_types_all[:train_len],
                                  osp.join(self.raw_dir, 'train.csv'))
            save_to_csv_mutag_mol(adj_all[train_len:train_len + val_len],
                                  edge_type_all[train_len:train_len + val_len],
                                  label_all[train_len:train_len + val_len],
                                  node_types_all[train_len:train_len + val_len],
                                  osp.join(self.raw_dir, 'val.csv'))
            save_to_csv_mutag_mol(adj_all[train_len + val_len:],
                                  edge_type_all[train_len + val_len:],
                                  label_all[train_len + val_len:],
                                  node_types_all[train_len + val_len:],
                                  osp.join(self.raw_dir, 'test.csv'))
            print(f'[Mutagenicity_mol] Splits: train {train_len}, val {val_len}, '
                  f'test {num_graphs - train_len - val_len}')
            return

        elif self.dataset_name == 'benzene':
            # Benzene (Sanchez-Lengeling et al., NeurIPS 2020; repackaged in GraphXAI).
            # Binary label = molecule contains a benzene ring. 12000 graphs, 50/50
            # balanced, <=25 nodes. Ground-truth explanations (which atoms form the
            # ring) ship with the data and are kept for later faithfulness analysis.
            #
            # Source file is benzene.npz (parsed like graphxai's
            # extract_google_datasets.load_graphs). Bond types are one-hot over
            # [single, double, triple, aromatic] and are shifted by +1 here so that
            # 0 stays reserved for "no bond" -- same 5-class encoding as
            # mutagenicity_mol, which keeps the GINE classifier and the SMILES
            # reconstruction reusable.
            from collections import Counter
            import json

            npz_path = osp.join(self.raw_dir, 'benzene.npz')
            if not osp.exists(npz_path):
                raise FileNotFoundError(
                    f'{npz_path} not found. Download it from GraphXAI:\n'
                    '  https://raw.githubusercontent.com/mims-harvard/GraphXAI/main/'
                    'graphxai/datasets/real_world/benzene/benzene.npz')

            # Index -> element, from graphxai's benzene.py ATOM_TYPES
            _BENZENE_IDX_TO_SYMBOL = {
                0: 'C', 1: 'N', 2: 'O', 3: 'S', 4: 'F', 5: 'P', 6: 'Cl',
                7: 'Br', 8: 'Na', 9: 'Ca', 10: 'I', 11: 'B', 12: 'H', 13: '*',
            }
            min_type_count = 50   # Huang et al. frequency filter, as for nci1/mutagenicity

            data = np.load(npz_path, allow_pickle=True)
            att, X, y = data['attr'], data['X'][0], data['y']
            labels_raw = [int(y[i][0]) for i in range(y.shape[0])]

            n_total = len(labels_raw)
            n_pos = sum(labels_raw)
            print(f'[Benzene] Full dataset: {n_total} graphs  |  '
                  f'No ring (0): {n_total - n_pos} ({100*(n_total-n_pos)/n_total:.1f}%)  |  '
                  f'Benzene ring (1): {n_pos} ({100*n_pos/n_total:.1f}%)')

            # --- Pass 1: global atom-type frequencies ---
            type_counter = Counter()
            for i in range(len(X)):
                type_counter.update(X[i]['nodes'].argmax(axis=1).tolist())

            rare = {t for t, c in type_counter.items() if c <= min_type_count}
            kept_types = sorted(t for t in type_counter if t not in rare)
            old_to_new = {old: new for new, old in enumerate(kept_types)}
            num_node_types = len(kept_types)
            print(f'[Benzene] Atom types: {len(type_counter)} -> {num_node_types} '
                  f'(dropped rare: {sorted(rare)})')

            # --- Pass 2: build graphs, skipping those with rare atoms ---
            adj_all, edge_type_all, label_all, node_types_all, gt_nodes_all = [], [], [], [], []
            n_skipped_rare = 0

            for i in range(len(X)):
                types = X[i]['nodes'].argmax(axis=1)
                if any(int(t) in rare for t in types):
                    n_skipped_rare += 1
                    continue

                n = X[i]['nodes'].shape[0]
                src = np.asarray(X[i]['senders'], dtype=int)
                dst = np.asarray(X[i]['receivers'], dtype=int)
                bond = np.asarray(X[i]['edges']).argmax(axis=1)

                # Binary adjacency + bond types. No self-loops (molecular
                # convention, same as mutagenicity_mol).
                adj = np.zeros((n, n), dtype=np.float64)
                et_mat = np.zeros((n, n), dtype=np.int64)
                for s, d, bt in zip(src, dst, bond):
                    if s == d:
                        continue
                    adj[s, d] = 1.0
                    et_mat[s, d] = int(bt) + 1     # 0 reserved for no-bond

                # Symmetrize: the source stores each bond once per direction, but
                # guard against one-sided entries so adjacency and edge types agree.
                adj = np.maximum(adj, adj.T)
                et_mat = np.maximum(et_mat, et_mat.T)

                adj_all.append(adj)
                edge_type_all.append(et_mat)
                label_all.append(labels_raw[i])
                node_types_all.append([old_to_new[int(t)] for t in types])

                node_imp = np.asarray(att[i][0]['nodes'])
                gt_nodes_all.append((node_imp.sum(axis=1) > 0).astype(int).tolist())

            n_kept = len(adj_all)
            n_pos_kept = sum(label_all)
            print(f'[Benzene] Graphs kept: {n_kept}  (skipped {n_skipped_rare} with rare atoms)  |  '
                  f'No ring: {n_kept - n_pos_kept} ({100*(n_kept-n_pos_kept)/n_kept:.1f}%)  |  '
                  f'Ring: {n_pos_kept} ({100*n_pos_kept/n_kept:.1f}%)')

            # --- atom_decoder / atom_encoder for SMILES reconstruction ---
            atom_decoder = {
                str(new_idx): _BENZENE_IDX_TO_SYMBOL.get(int(old_idx), f'UNK{old_idx}')
                for new_idx, old_idx in enumerate(kept_types)
            }
            atom_encoder = {sym: int(idx) for idx, sym in atom_decoder.items()}
            with open(osp.join(self.raw_dir, 'atom_decoder.json'), 'w') as f:
                json.dump(atom_decoder, f, indent=2)
            with open(osp.join(self.raw_dir, 'atom_encoder.json'), 'w') as f:
                json.dump(atom_encoder, f, indent=2)
            print(f'[Benzene] atom_decoder (remapped_idx -> symbol): {atom_decoder}')
            print('[Benzene] Bond encoding: 0=no bond, 1=single, 2=double, 3=triple, 4=aromatic')

            # --- Marginals; saved so SpectreDatasetInfos never needs placeholders ---
            np.save(osp.join(self.raw_dir, 'node_type_map.npy'), np.array(kept_types))

            type_counts = np.zeros(num_node_types)
            for nt in node_types_all:
                for t in nt:
                    type_counts[t] += 1
            node_marginals = type_counts / type_counts.sum()
            np.save(osp.join(self.raw_dir, 'node_type_marginals.npy'), node_marginals)
            print(f'[Benzene] Node-type marginals: {np.round(node_marginals, 4).tolist()}')

            total_entries = sum(et.shape[0] ** 2 for et in edge_type_all)
            edge_counts = np.zeros(5)
            for et in edge_type_all:
                for k in range(5):
                    edge_counts[k] += (et == k).sum()
            edge_marginals = edge_counts / total_entries
            np.save(osp.join(self.raw_dir, 'edge_type_marginals.npy'), edge_marginals)
            print(f'[Benzene] Edge-type marginals (0=no-bond, 1-4=bond): '
                  f'{np.round(edge_marginals, 4).tolist()}')

            max_n = max(a.shape[0] for a in adj_all)
            n_nodes_hist = np.zeros(max_n + 1)
            for a in adj_all:
                n_nodes_hist[a.shape[0]] += 1
            n_nodes_dist = n_nodes_hist / n_nodes_hist.sum()
            np.save(osp.join(self.raw_dir, 'n_nodes_dist.npy'), n_nodes_dist)
            print(f'[Benzene] Graph sizes: max {max_n} nodes')

            # --- Shuffle and split 60/20/20 ---
            num_graphs = len(adj_all)
            perm = np.random.permutation(num_graphs)
            adj_all = [adj_all[i] for i in perm]
            edge_type_all = [edge_type_all[i] for i in perm]
            label_all = [label_all[i] for i in perm]
            node_types_all = [node_types_all[i] for i in perm]
            gt_nodes_all = [gt_nodes_all[i] for i in perm]

            train_len = int(round(num_graphs * 0.6))
            val_len = int(round(num_graphs * 0.2))

            def save_to_csv_benzene(adj_list, et_list, label_list, nt_list, gt_list, file_path):
                rows = []
                for idx, (adj, et, y_, nt, gt) in enumerate(
                        zip(adj_list, et_list, label_list, nt_list, gt_list)):
                    rows.append({
                        'graph_id': idx,
                        'y': int(y_),
                        'adjacency_matrix': adj.flatten().tolist(),
                        'edge_type_matrix': et.flatten().tolist(),
                        'node_types': nt,
                        'gt_ring_nodes': gt,
                    })
                pd.DataFrame(rows).to_csv(file_path, index=False)

            save_to_csv_benzene(adj_all[:train_len], edge_type_all[:train_len],
                                label_all[:train_len], node_types_all[:train_len],
                                gt_nodes_all[:train_len],
                                osp.join(self.raw_dir, 'train.csv'))
            save_to_csv_benzene(adj_all[train_len:train_len + val_len],
                                edge_type_all[train_len:train_len + val_len],
                                label_all[train_len:train_len + val_len],
                                node_types_all[train_len:train_len + val_len],
                                gt_nodes_all[train_len:train_len + val_len],
                                osp.join(self.raw_dir, 'val.csv'))
            save_to_csv_benzene(adj_all[train_len + val_len:],
                                edge_type_all[train_len + val_len:],
                                label_all[train_len + val_len:],
                                node_types_all[train_len + val_len:],
                                gt_nodes_all[train_len + val_len:],
                                osp.join(self.raw_dir, 'test.csv'))
            print(f'[Benzene] Splits: train {train_len}, val {val_len}, '
                  f'test {num_graphs - train_len - val_len}')
            return

        elif self.dataset_name == 'spmotif':
            # SPMotif (DIR, Wu et al. ICLR 2022): one base graph (tree/ladder/wheel)
            # plus one 5-node motif; the motif alone determines the label.
            #
            # Two deliberate deviations from the published 3-class setup:
            #   1. bias = 1/3, i.e. the base is drawn UNIFORMLY and carries no label
            #      information. The higher-bias variants (0.5/0.7/0.9) exist to study
            #      OOD generalisation; for us that spurious base-label correlation is
            #      exactly the AIDS failure mode and must be switched off.
            #   2. Binarised to cycle (C5, 5 edges) vs crane (K_{2,3}, 6 edges).
            #      Measured edit distances between the motifs: cycle->house = 1 edge,
            #      house->crane = 2, cycle->crane = 3. The widest pair carries ~4.5% of
            #      the edge-slot budget, above Benzene's 3.66%, whereas cycle-vs-house
            #      would sit at ~1.5% and risk the BA-2Motifs collapse.
            #
            # Nodes carry no features in SPMotif (ConstFeatureGen), so there is a
            # single node type -- as for ba2motifs.
            import sys as _sys

            # download() runs on SpectreGraphDataset, which has no cfg -- so these are
            # constants here, like max_num_nodes for aids / min_type_count for benzene.
            gen_dir = 'spmotif_gen'
            n_per_class = 3000
            bias = 1.0 / 3.0
            base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
            gen_path = gen_dir if osp.isabs(gen_dir) else osp.join(base_path, gen_dir)
            if not osp.isdir(gen_path):
                raise FileNotFoundError(
                    f'SPMotif generator not found at {gen_path}. Fetch BA3_loc.py, '
                    'featgen.py and synthetic_structsim.py from '
                    'https://github.com/Wuyxin/DIR-GNN/tree/main/spmotif_gen')
            _sys.path.insert(0, gen_path)

            import matplotlib
            matplotlib.use('Agg')          # headless: BA3_loc imports pyplot
            import synthetic_structsim
            from BA3_loc import perturb

            np.random.seed(0)

            def _base_and_width():
                k = np.random.choice([1, 2, 3], p=[bias, (1 - bias) / 2, (1 - bias) / 2])
                if k == 1:
                    return 'tree', int(np.random.choice(range(3)))
                if k == 2:
                    return 'ladder', int(np.random.choice(range(8, 12)))
                return 'wheel', int(np.random.choice(range(15, 20)))

            def _make(shape, basis_type, width_basis):
                G, role_id, _ = synthetic_structsim.build_graph(
                    width_basis, basis_type, [[shape]], start=0, rdm_basis_plugins=True)
                return perturb([G], 0.00, id=role_id)[0], role_id

            # label 0 = cycle (C5), label 1 = crane (K_{2,3})
            shapes = ['dircycle', 'crane']
            adj_all, label_all, node_types_all, motif_all = [], [], [], []
            for label, shape in enumerate(shapes):
                for _ in range(n_per_class):
                    basis, width = _base_and_width()
                    G, role_id = _make(shape, basis, width)
                    n = G.number_of_nodes()
                    adj = nx.to_numpy_array(G, nodelist=sorted(G.nodes()))
                    adj = (adj > 0).astype(np.float64)
                    np.fill_diagonal(adj, 1.0)      # same self-loop convention as nci1
                    adj_all.append(adj)
                    label_all.append(label)
                    node_types_all.append([0] * n)  # single node type
                    motif_all.append((np.asarray(role_id) > 0).astype(int).tolist())

            n_tot = len(adj_all)
            sizes = np.array([a.shape[0] for a in adj_all])
            print(f'[SPMotif] bias={bias:.3f}  {n_tot} graphs  |  '
                  f'cycle: {label_all.count(0)}  crane: {label_all.count(1)}')
            print(f'[SPMotif] Nodes min/median/max: {sizes.min()} / '
                  f'{int(np.median(sizes))} / {sizes.max()}')

            np.save(osp.join(self.raw_dir, 'node_type_map.npy'), np.array([0]))
            np.save(osp.join(self.raw_dir, 'node_type_marginals.npy'),
                    np.array([1.0], dtype=np.float32))

            # Edge marginals over the dense N*N entries (diagonal included, as stored)
            tot = sum(a.shape[0] ** 2 for a in adj_all)
            n_pos = sum(float(a.sum()) for a in adj_all)
            edge_marg = np.array([1.0 - n_pos / tot, n_pos / tot], dtype=np.float32)
            np.save(osp.join(self.raw_dir, 'edge_type_marginals.npy'), edge_marg)
            print(f'[SPMotif] Edge marginals: {np.round(edge_marg, 4).tolist()}')

            max_n = int(sizes.max())
            hist = np.zeros(max_n + 1)
            for a in adj_all:
                hist[a.shape[0]] += 1
            np.save(osp.join(self.raw_dir, 'n_nodes_dist.npy'), hist / hist.sum())

            perm = np.random.permutation(n_tot)
            adj_all = [adj_all[i] for i in perm]
            label_all = [label_all[i] for i in perm]
            node_types_all = [node_types_all[i] for i in perm]
            motif_all = [motif_all[i] for i in perm]

            train_len = int(round(n_tot * 0.6))
            val_len = int(round(n_tot * 0.2))

            def save_to_csv_spmotif(a_l, y_l, nt_l, mt_l, file_path):
                rows = []
                for idx, (adj, y_, nt, mt) in enumerate(zip(a_l, y_l, nt_l, mt_l)):
                    rows.append({
                        'graph_id': idx,
                        'y': int(y_),
                        'adjacency_matrix': adj.flatten().tolist(),
                        'node_types': nt,
                        'gt_motif_nodes': mt,
                    })
                pd.DataFrame(rows).to_csv(file_path, index=False)

            save_to_csv_spmotif(adj_all[:train_len], label_all[:train_len],
                                node_types_all[:train_len], motif_all[:train_len],
                                osp.join(self.raw_dir, 'train.csv'))
            save_to_csv_spmotif(adj_all[train_len:train_len + val_len],
                                label_all[train_len:train_len + val_len],
                                node_types_all[train_len:train_len + val_len],
                                motif_all[train_len:train_len + val_len],
                                osp.join(self.raw_dir, 'val.csv'))
            save_to_csv_spmotif(adj_all[train_len + val_len:], label_all[train_len + val_len:],
                                node_types_all[train_len + val_len:],
                                motif_all[train_len + val_len:],
                                osp.join(self.raw_dir, 'test.csv'))
            print(f'[SPMotif] Splits: train {train_len}, val {val_len}, '
                  f'test {n_tot - train_len - val_len}')
            return

        else:
            raise ValueError(f'Unknown dataset {self.dataset_name}')


    def process(self):
        # Load the dataset from the corresponding CSV file
        csv_file = self.split_paths[self.file_idx]
        df = pd.read_csv(csv_file)

        # Determine num_node_types from the saved freq-filter mapping file (mutagenicity / nci1)
        _mutag_mol_num_node_types = None
        if self.dataset_name == 'mutagenicity_mol':
            map_path = osp.join(self.raw_dir, 'node_type_map.npy')
            if osp.exists(map_path):
                _mutag_mol_num_node_types = int(len(np.load(map_path)))
            else:
                _mutag_mol_num_node_types = 14  # fallback

        _spmotif_num_node_types = None
        if self.dataset_name == 'spmotif':
            _spmotif_num_node_types = 1   # SPMotif nodes carry no features

        _benzene_num_node_types = None
        if self.dataset_name == 'benzene':
            map_path = osp.join(self.raw_dir, 'node_type_map.npy')
            if osp.exists(map_path):
                _benzene_num_node_types = int(len(np.load(map_path)))
            else:
                _benzene_num_node_types = 14  # fallback: all benzene atom feature dims

        _nci1_num_node_types = None
        if self.dataset_name == 'nci1':
            map_path = osp.join(self.raw_dir, 'node_type_map.npy')
            if osp.exists(map_path):
                _nci1_num_node_types = int(len(np.load(map_path)))
            else:
                _nci1_num_node_types = 10  # fallback: freq>50 filter yields 10 types

        if self.dataset_name == 'nci1_nofreq':
            map_path = osp.join(self.raw_dir, 'node_type_map.npy')
            if osp.exists(map_path):
                _nci1_num_node_types = int(len(np.load(map_path)))
            else:
                _nci1_num_node_types = 37  # fallback: all NCI1 atom types

        _aids_num_node_types = None
        if self.dataset_name == 'aids':
            map_path = osp.join(self.raw_dir, 'node_type_map.npy')
            if osp.exists(map_path):
                _aids_num_node_types = int(len(np.load(map_path)))
            else:
                _aids_num_node_types = 38  # fallback: all AIDS atom feature dims

        _tu_num_node_types = None
        if self.dataset_name in ('proteins', 'imdb_binary', 'twitter'):
            map_path = osp.join(self.raw_dir, 'node_type_map.npy')
            # PROTEINS: 3 secondary-structure labels. IMDB-BINARY: a single constant type.
            _tu_num_node_types = (int(len(np.load(map_path))) if osp.exists(map_path)
                                  else (3 if self.dataset_name == 'proteins' else 1))

        _ba2_num_node_types = None
        if self.dataset_name == 'ba2motifs':
            map_path = osp.join(self.raw_dir, 'node_type_map.npy')
            if osp.exists(map_path):
                _ba2_num_node_types = int(len(np.load(map_path)))
            else:
                _ba2_num_node_types = 1  # BA-2Motifs carries a single node type

        data_list = []
        i = 0
        for _, row in tqdm(df.iterrows()):
            # Convert the adjacency matrix string back to a numpy array
            adj_flat = np.fromstring(row['adjacency_matrix'].strip('[]'), sep=',')
            adj_size = int(np.sqrt(len(adj_flat)))
            adj_matrix = adj_flat.reshape((adj_size, adj_size))
            adj_tensor = torch.tensor(adj_matrix, dtype=torch.float32)

            # Convert dense adjacency matrix to edge_index (sparse representation)
            edge_index = (adj_tensor > 0).nonzero(as_tuple=False).t()

            x = torch.ones(adj_size, 1, dtype=torch.float)

            edge_attr = torch.zeros(edge_index.shape[-1], 2, dtype=torch.float32)
            edge_attr[:, 1] = 1

            y = torch.zeros((1, 0), dtype=torch.float)

            if self.dataset_name == 'planar':
                total_edge_entries = edge_index.shape[1]
                num_edges = total_edge_entries // 2
                normalized_edges = (num_edges - 7) / (18 - 7)
                guidance = torch.tensor([[normalized_edges]], dtype=torch.float32)

            elif self.dataset_name in ('nci1', 'nci1_nofreq'):
                label = int(row['y']) if 'y' in row else 0
                guidance = F.one_hot(torch.tensor(label), num_classes=2).float().unsqueeze(0)
                import ast
                node_type_indices = ast.literal_eval(row['node_types'])
                node_types_tensor = torch.tensor(node_type_indices, dtype=torch.long)
                x = F.one_hot(node_types_tensor, num_classes=_nci1_num_node_types).float()

            elif self.dataset_name == 'aids':
                label = int(row['y']) if 'y' in row else 0
                guidance = F.one_hot(torch.tensor(label), num_classes=2).float().unsqueeze(0)
                import ast
                node_type_indices = ast.literal_eval(row['node_types'])
                node_types_tensor = torch.tensor(node_type_indices, dtype=torch.long)
                x = F.one_hot(node_types_tensor, num_classes=_aids_num_node_types).float()

            elif self.dataset_name == 'ba2motifs':
                label = int(row['y']) if 'y' in row else 0
                guidance = F.one_hot(torch.tensor(label), num_classes=2).float().unsqueeze(0)
                import ast
                node_type_indices = ast.literal_eval(row['node_types'])
                node_types_tensor = torch.tensor(node_type_indices, dtype=torch.long)
                x = F.one_hot(node_types_tensor, num_classes=_ba2_num_node_types).float()

            elif self.dataset_name in ('proteins', 'imdb_binary', 'twitter'):
                label = int(row['y']) if 'y' in row else 0
                guidance = F.one_hot(torch.tensor(label), num_classes=2).float().unsqueeze(0)
                import ast
                node_type_indices = ast.literal_eval(row['node_types'])
                node_types_tensor = torch.tensor(node_type_indices, dtype=torch.long)
                x = F.one_hot(node_types_tensor, num_classes=_tu_num_node_types).float()

            elif self.dataset_name == 'mutagenicity_mol':
                import ast
                label = int(row['y']) if 'y' in row else 0
                guidance = F.one_hot(torch.tensor(label), num_classes=2).float().unsqueeze(0)
                # Node features
                node_type_indices = ast.literal_eval(row['node_types'])
                node_types_tensor = torch.tensor(node_type_indices, dtype=torch.long)
                x = F.one_hot(node_types_tensor, num_classes=_mutag_mol_num_node_types).float()
                # Override edge_attr: 5-class one-hot bond types from edge_type_matrix
                et_flat = np.fromstring(row['edge_type_matrix'].strip('[]'), sep=',').astype(int)
                n_et = int(np.sqrt(len(et_flat)))
                et_matrix = torch.tensor(et_flat.reshape(n_et, n_et), dtype=torch.long)
                # edge_index already set from adjacency_matrix above. NOTE: the
                # adjacency diagonal was set to 1.0 during dataset construction
                # (np.fill_diagonal(adj, 1.0)), so edge_index DOES contain a
                # self-loop on every node. Downstream consumers must handle the
                # diagonal consistently (see NCI1GINWrapperC2.forward).
                if edge_index.numel() > 0:
                    et_vals = et_matrix[edge_index[0], edge_index[1]]
                    edge_attr = F.one_hot(et_vals, num_classes=5).float()
                else:
                    edge_attr = torch.zeros((0, 5), dtype=torch.float32)

            elif self.dataset_name == 'spmotif':
                import ast
                label = int(row['y']) if 'y' in row else 0
                guidance = F.one_hot(torch.tensor(label), num_classes=2).float().unsqueeze(0)
                node_type_indices = ast.literal_eval(row['node_types'])
                node_types_tensor = torch.tensor(node_type_indices, dtype=torch.long)
                x = F.one_hot(node_types_tensor, num_classes=_spmotif_num_node_types).float()

            elif self.dataset_name == 'benzene':
                import ast
                label = int(row['y']) if 'y' in row else 0
                guidance = F.one_hot(torch.tensor(label), num_classes=2).float().unsqueeze(0)
                node_type_indices = ast.literal_eval(row['node_types'])
                node_types_tensor = torch.tensor(node_type_indices, dtype=torch.long)
                x = F.one_hot(node_types_tensor, num_classes=_benzene_num_node_types).float()
                # 5-class one-hot bond types, as for mutagenicity_mol. The benzene
                # adjacency carries NO self-loops, so edge_index has none either.
                et_flat = np.fromstring(row['edge_type_matrix'].strip('[]'), sep=',').astype(int)
                n_et = int(np.sqrt(len(et_flat)))
                et_matrix = torch.tensor(et_flat.reshape(n_et, n_et), dtype=torch.long)
                if edge_index.numel() > 0:
                    et_vals = et_matrix[edge_index[0], edge_index[1]]
                    edge_attr = F.one_hot(et_vals, num_classes=5).float()
                else:
                    edge_attr = torch.zeros((0, 5), dtype=torch.float32)

            else:
                raise ValueError(F'Unkown dataset')

            # Create a Data object for PyTorch Geometric
            #data = Data(x=X, edge_index=edge_index, edge_attr=edge_attr, y=y, idx=row['graph_id'])
            data = GuidedInMemoryDataset(x=x, edge_index=edge_index, edge_attr=edge_attr,
                                         y=y, idx=i, guidance=guidance)

            data_list.append(data)
            i +=1

        # Batch and save the processed data
        batched_data = self.collate(data_list[:len(data_list)])
        torch.save(batched_data, self.processed_paths[self.file_idx])


class SpectreGraphDataModule(AbstractDataModule):
    def __init__(self, cfg, regressor: bool = False, n_graphs=200):
        super().__init__(cfg)
        self.cfg = cfg
        self.n_graphs = n_graphs
        self.regressor = regressor
        self.prepare_data()
        self.inner = self.train_dataloader()
        target = getattr(cfg.guidance, 'guidance_target', None)
        transform = None

    def _make_weighted_sampler(self, dataset):
        """Return a WeightedRandomSampler that balances classes in the training split."""
        from torch.utils.data import WeightedRandomSampler
        guidance = dataset.data.guidance          # shape [N, num_classes], one-hot
        labels = guidance.argmax(dim=-1)          # [N]
        class_counts = torch.bincount(labels)
        # weight per class = 1 / count  (so rare class gets upsampled)
        class_weights = 1.0 / class_counts.float()
        sample_weights = class_weights[labels]
        print(f"[WeightedSampler] class counts: {class_counts.tolist()}, "
              f"class weights: {class_weights.tolist()}")
        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

    def _relabel_with_classifier(self, datasets):
        """Replace guidance labels with predictions from a pretrained classifier."""
        dataset_name = self.cfg.dataset.name

        # Shared judge takes precedence: one classifier that every counterfactual
        # method in the comparison targets, so validity numbers are comparable
        # across methods instead of each method answering to its own oracle.
        shared_judge_path = getattr(self.cfg.guidance, 'shared_judge_path', None)
        if shared_judge_path:
            import sys as _sys, os as _os
            _sys.path.insert(
                0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                 '..', 'shared_judge')
            )
            from shared_judge import load_judge, SharedJudgeBatchAdapter
            print(f"[Relabel] using SHARED JUDGE at {shared_judge_path}")
            classifier = SharedJudgeBatchAdapter(load_judge(shared_judge_path,
                                                            device='cpu'))
            return self._apply_relabeling(datasets, classifier)

        classifier_path = self.cfg.guidance.graph_classifier_path
        if classifier_path is None:
            raise ValueError(
                "label_source is 'classifier' but guidance.graph_classifier_path is not set. "
                "Please provide the path to a pretrained classifier checkpoint."
            )

        # A shared-judge export instead of a Lightning checkpoint. The judge ships
        # as `<prefix>.npz` + `<prefix>.json` (portable across the four baseline
        # environments), so `graph_classifier_path` may point at either file or at
        # the bare prefix. Anything that is not a `.ckpt` is treated as an export,
        # which leaves the Lightning branches below untouched.
        if not str(classifier_path).endswith('.ckpt'):
            from src.models.shared_judge_adapter import load_shared_judge
            print(f"[Relabel] graph_classifier_path is a SHARED JUDGE export: {classifier_path}")
            classifier = load_shared_judge(str(classifier_path))
            return self._apply_relabeling(datasets, classifier)

        # Select the right classifier class based on dataset
        if dataset_name == 'nci1':
            from src.models.NCI1_GCN import NCI1GCNLightning
            classifier = NCI1GCNLightning.load_from_checkpoint(classifier_path, map_location='cpu')
        elif dataset_name == 'nci1_nofreq':
            # Auto-detect GIN vs GCN from checkpoint state_dict keys
            _ckpt = torch.load(classifier_path, map_location='cpu')
            _keys = set(_ckpt.get('state_dict', {}).keys())
            if any('.nn.' in k for k in _keys):
                from src.models.Mutag_GIN import MutagGINLightning
                classifier = MutagGINLightning.load_from_checkpoint(classifier_path, map_location='cpu')
            else:
                from src.models.NCI1_GCN import NCI1GCNLightning
                classifier = NCI1GCNLightning.load_from_checkpoint(classifier_path, map_location='cpu')
        elif dataset_name == 'aids':
            from src.models.Mutag_GIN import MutagGINLightning
            classifier = MutagGINLightning.load_from_checkpoint(classifier_path, map_location='cpu')
        elif dataset_name in ('spmotif', 'proteins', 'imdb_binary', 'twitter'):
            # Same LEConv judge as NCI1; trainGCN.py builds these with NCI1GCNLightning.
            from src.models.NCI1_GCN import NCI1GCNLightning
            classifier = NCI1GCNLightning.load_from_checkpoint(classifier_path, map_location='cpu')
        elif dataset_name == 'ba2motifs':
            # BA-2Motifs labels are exact ground truth, so label_source='original' is
            # preferred; this branch only matters if relabeling is explicitly requested.
            from src.models.NCI1_GCN import NCI1GCNLightning
            classifier = NCI1GCNLightning.load_from_checkpoint(classifier_path, map_location='cpu')
        elif dataset_name in ('mutagenicity_mol', 'benzene'):
            # GINEConv classifier with 5-class edge features
            from src.models.Mutag_GINE import MutagGINELightning
            classifier = MutagGINELightning.load_from_checkpoint(classifier_path, map_location='cpu')
        else:
            from src.models.GCN import GCNLightning
            classifier = GCNLightning.load_from_checkpoint(classifier_path, map_location='cpu')

        return self._apply_relabeling(datasets, classifier)

    def _apply_relabeling(self, datasets, classifier):
        """Run `classifier` over every split and overwrite the guidance labels."""
        # Try to use GPU if available, but fallback to CPU if it's busy/unavailable
        device = torch.device('cpu')
        if torch.cuda.is_available():
            try:
                # Test if GPU is actually usable
                torch.zeros(1).to('cuda')
                device = torch.device('cuda')
            except RuntimeError:
                print("WARNING: CUDA is available but usable (possibly busy). Falling back to CPU for classifier.")
                device = torch.device('cpu')

        classifier = classifier.to(device)
        classifier.eval()

        # Determine number of classes from classifier output dim
        if hasattr(classifier, 'hparams') and 'out_dim' in classifier.hparams:
            num_classes = classifier.hparams['out_dim']
        else:
            num_classes = 2

        for split_name, dataset in datasets.items():
            original_guidance = dataset.data.guidance.clone()
            num_graphs = len(dataset)

            # Run batch inference
            loader = DataLoader(dataset, batch_size=256, shuffle=False)
            all_preds = []

            with torch.no_grad():
                for batch in loader:
                    batch = batch.to(device)
                    logits = classifier(batch)
                    preds = logits.argmax(dim=-1)
                    all_preds.append(preds.cpu())

            all_preds = torch.cat(all_preds, dim=0)

            # Convert predictions to guidance format
            new_guidance = F.one_hot(all_preds, num_classes=num_classes).float()
            dataset.data.guidance = new_guidance

            # Print agreement stats
            original_labels = original_guidance.argmax(dim=-1).view(-1)
            new_labels = all_preds.view(-1)
            agreement = (original_labels == new_labels).sum().item()
            changed = num_graphs - agreement
            print(f"[Relabel] {split_name}: {agreement}/{num_graphs} agree with original "
                  f"({changed} changed, {100 * changed / num_graphs:.1f}%)")

    def prepare_data(self) -> None:
        self.datadir = self.cfg.dataset.datadir
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, self.datadir)
        datasets = {'train': SpectreGraphDataset(dataset_name=self.cfg.dataset.name,
                                                 split='train', root=root_path, transform=None ),
                    'val': SpectreGraphDataset(dataset_name=self.cfg.dataset.name,
                                               split='val', root=root_path, transform=None ),
                    'test': SpectreGraphDataset(dataset_name=self.cfg.dataset.name,
                                                split='test', root=root_path, transform=None )}

        label_source = getattr(self.cfg.guidance, 'label_source', 'original')
        if label_source == 'classifier':
            self._relabel_with_classifier(datasets)

        samplers = {}
        if getattr(self.cfg.dataset, 'weighted_sampling', False):
            samplers['train'] = self._make_weighted_sampler(datasets['train'])

        super().prepare_data(datasets, samplers=samplers)


class SpectreDatasetInfos(AbstractDatasetInfos):
    def __init__(self, dataset_name):
        self.name = 'nx_graphs'

        # Accept Hydra config / dict-like objects; keep original for extra fields
        dataset_cfg = dataset_name if not isinstance(dataset_name, str) else None
        if not isinstance(dataset_name, str):
            dataset_name = getattr(dataset_name, "name", None) or dataset_name.get("name")


        if dataset_name == 'planar_8':
            self.n_nodes = torch.Tensor([0, 0, 0, 0, 0, 0,
                                             0, 0, 1])
            self.node_types = torch.tensor([1])
            self.edge_types = torch.tensor([0.6045, 0.3955])
        elif dataset_name == 'ba2motifs':
            # BA-2Motifs: every graph has exactly 25 nodes -> delta distribution.
            # Single node type (like planar_8). Edge marginal is printed by the
            # preprocessing step ("edge marginal (off-diagonal)") -- keep in sync.
            _n = torch.zeros(26)
            _n[25] = 1.0
            self.n_nodes = _n
            self.node_types = torch.tensor([1])
            self.edge_types = torch.tensor([0.9151, 0.0849])  # measured by preprocessing
        elif dataset_name == 'nci1':
            # NCI1 graphs filtered to max 50 nodes with freq>50 atom type filtering (Huang et al.)
            # n_nodes: original 41-element distribution extended with zeros for indices 41-50
            self.n_nodes = torch.Tensor([0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0024,
        0.0022, 0.0038, 0.0063, 0.0106, 0.0122, 0.0125, 0.0177, 0.0250, 0.0326,
        0.0337, 0.0451, 0.0468, 0.0468, 0.0446, 0.0514, 0.0479, 0.0435, 0.0438,
        0.0402, 0.0457, 0.0337, 0.0321, 0.0337, 0.0283, 0.0283, 0.0269, 0.0261,
        0.0196, 0.0231, 0.0215, 0.0201, 0.0201, 0.0103, 0.0120, 0.0109, 0.0054,
        0.0068, 0.0071, 0.0054, 0.0049, 0.0046, 0.0044])
            # ~9 atom types surviving freq>50 filter; approximate marginals (renormalized)
            self.node_types = torch.Tensor([1.3843e-01, 8.8773e-02, 7.3797e-01, 1.1672e-02, 1.1591e-02, 9.8951e-04,
        6.9770e-03, 7.9767e-04, 2.1709e-03, 6.2602e-04])
            self.edge_types = torch.tensor([0.779, 0.221])  # sparse graphs

        elif dataset_name == 'nci1_nofreq':
            # NCI1 graphs filtered to max 50 nodes only (no frequency filtering)
            # n_nodes distribution is the same as nci1 (same size filter)
            self.n_nodes = torch.Tensor([0.0000, 0.0000, 0.0000, 0.0003, 0.0000, 0.0003, 0.0000, 0.0000, 0.0026,
        0.0024, 0.0037, 0.0063, 0.0105, 0.0124, 0.0129, 0.0187, 0.0250, 0.0318,
        0.0331, 0.0442, 0.0460, 0.0471, 0.0439, 0.0508, 0.0476, 0.0439, 0.0439,
        0.0400, 0.0452, 0.0337, 0.0321, 0.0326, 0.0284, 0.0284, 0.0274, 0.0263,
        0.0192, 0.0237, 0.0210, 0.0197, 0.0200, 0.0108, 0.0116, 0.0129, 0.0055,
        0.0068, 0.0082, 0.0053, 0.0047, 0.0047, 0.0045])
            # Load node-type marginals saved by download(); fall back to uniform over 37 types.
            self.node_types = torch.Tensor([1.3899e-01, 8.8119e-02, 7.3650e-01, 1.1694e-02, 1.1713e-02, 1.0826e-03,
        6.8466e-03, 1.5605e-04, 7.7049e-04, 1.2679e-04, 5.8518e-05, 5.8518e-05,
        1.9506e-05, 2.3115e-03, 2.9259e-05, 1.9506e-05, 6.8271e-05, 2.9259e-05,
        1.9506e-05, 3.9012e-05, 6.1444e-04, 2.9259e-04, 9.7531e-06, 3.9012e-05,
        9.7531e-05, 1.0728e-04, 1.9506e-05, 7.8024e-05, 1.9506e-05, 9.7531e-06,
        9.7531e-06, 9.7531e-06, 9.7531e-06, 9.7531e-06, 9.7531e-06, 9.7531e-06]) # uniform fallback over all NCI1 types
            self.edge_types = torch.tensor([0.78, 0.22])  # same edge density as nci1

        elif dataset_name == 'mutagenicity_mol':
            # Same graph size distribution as mutagenicity (same size filter, same source dataset)
            self.n_nodes = torch.Tensor([0.0000, 0.0000, 0.0000, 0.0000, 0.0003, 0.0020, 0.0030, 0.0045, 0.0078,
        0.0086, 0.0134, 0.0164, 0.0192, 0.0177, 0.0275, 0.0300, 0.0316, 0.0371,
        0.0396, 0.0288, 0.0374, 0.0346, 0.0298, 0.0275, 0.0353, 0.0331, 0.0404,
        0.0318, 0.0419, 0.0358, 0.0417, 0.0295, 0.0326, 0.0265, 0.0300, 0.0305,
        0.0260, 0.0149, 0.0230, 0.0154, 0.0134, 0.0111, 0.0144, 0.0106, 0.0111,
        0.0061, 0.0058, 0.0063, 0.0068, 0.0056, 0.0035])
            # Same atom type distribution as mutagenicity (same source dataset)
            self.node_types = torch.Tensor([4.2015e-01, 7.9549e-02, 1.1706e-02, 4.2743e-01, 4.9092e-02, 2.1513e-03,
        2.8587e-03, 4.9518e-03, 9.5936e-04, 1.5505e-04, 8.4307e-04, 1.3567e-04,
        1.9381e-05])
            # Stored classes: 0=no bond, 1=single, 2=double, 3=triple, 4=aromatic(never in Kekulé)
            self.edge_types = torch.Tensor([0.8558, 0.1173, 0.0268, 0.0001, 0.0])

        elif dataset_name == 'aids':
            # AIDS graphs filtered to max 50 nodes (no frequency filtering).
            # Most graphs have 5–40 nodes; peaks around 15-20.
            # NOTE: Update these values after first download() by computing exact marginals.
            self.n_nodes = torch.Tensor([
            0.0000e+00, 0.0000e+00, 1.0565e-03, 1.0565e-03, 6.3391e-03, 2.1130e-03, 1.4791e-02, 2.4828e-02, 6.4976e-02,
            1.1516e-01, 1.8119e-01, 2.4564e-01, 1.8965e-01, 1.5848e-03, 0.0000e+00, 2.1130e-03, 2.6413e-03, 3.1696e-03,
            2.6413e-03, 6.8674e-03, 1.4263e-02, 1.0037e-02, 1.0037e-02, 1.3735e-02, 6.3391e-03, 3.1696e-03, 4.2261e-03,
            6.8674e-03, 2.6413e-03, 1.5848e-03, 2.1130e-03, 2.1130e-03, 2.6413e-03, 6.3391e-03, 6.3391e-03, 5.2826e-03,
            4.2261e-03, 2.1130e-03, 3.6978e-03, 2.6413e-03, 5.2826e-04, 1.5848e-03, 1.0565e-03, 5.2826e-04, 3.1696e-03,
            1.5848e-03, 3.1696e-03, 4.7544e-03, 3.1696e-03, 2.6413e-03, 1.5848e-03])
            # Placeholder: uniform over 38 atom-type dims. Update after download().
            self.node_types = torch.Tensor([
            6.1923e-01, 1.6540e-01, 1.4176e-01, 1.4617e-02, 3.1178e-03, 4.0410e-02, 6.4785e-04, 3.3202e-03, 7.2883e-04,
            1.7006e-03, 7.6932e-04, 4.0896e-03, 1.2147e-04, 1.0933e-03, 1.6196e-04, 2.8344e-04, 5.6687e-04, 5.2638e-04,
            1.2147e-04, 1.2147e-04, 2.8344e-04, 4.0491e-05, 1.6196e-04, 4.0491e-05, 8.0981e-05, 4.0491e-05, 4.0491e-05,
            4.0491e-05, 8.0981e-05, 4.0491e-05, 1.2147e-04, 4.0491e-05, 8.0981e-05, 8.0981e-05, 4.0491e-05])
            self.edge_types = torch.tensor([0.8180, 0.1820]) # similar sparsity to NCI1

        elif dataset_name == 'spmotif':
            _loaded = False
            _datadir = getattr(dataset_cfg, 'datadir', None) if dataset_cfg is not None else None
            if _datadir is not None:
                _base = pathlib.Path(os.path.realpath(__file__)).parents[2]
                _raw = os.path.join(_base, _datadir, 'raw')
                _f_n = os.path.join(_raw, 'n_nodes_dist.npy')
                _f_e = os.path.join(_raw, 'edge_type_marginals.npy')
                if all(os.path.exists(p) for p in (_f_n, _f_e)):
                    self.n_nodes = torch.Tensor(np.load(_f_n))
                    self.node_types = torch.tensor([1])      # single node type
                    self.edge_types = torch.Tensor(np.load(_f_e))
                    _loaded = True
                    print(f'[SPMotif] Loaded marginals from {_raw}: '
                          f'max {len(self.n_nodes) - 1} nodes, '
                          f'edge_types={np.round(self.edge_types.numpy(), 4).tolist()}')
            if not _loaded:
                print('[SPMotif] WARNING: marginals not found on disk, using fallback.')
                self.n_nodes = torch.ones(28) / 28.0
                self.node_types = torch.tensor([1])
                self.edge_types = torch.Tensor([0.80, 0.20])

        elif dataset_name == 'benzene':
            # Read the marginals written by download() instead of hard-coding them.
            # The aids/nci1 branches above carry hand-copied placeholders that must be
            # refreshed by hand after every re-download; loading from disk removes that
            # failure mode. The literals below are only a fallback for the case where
            # download() has not run yet.
            _loaded = False
            _datadir = getattr(dataset_cfg, 'datadir', None) if dataset_cfg is not None else None
            if _datadir is not None:
                _base = pathlib.Path(os.path.realpath(__file__)).parents[2]
                _raw = os.path.join(_base, _datadir, 'raw')
                _f_n = os.path.join(_raw, 'n_nodes_dist.npy')
                _f_x = os.path.join(_raw, 'node_type_marginals.npy')
                _f_e = os.path.join(_raw, 'edge_type_marginals.npy')
                if all(os.path.exists(p) for p in (_f_n, _f_x, _f_e)):
                    self.n_nodes = torch.Tensor(np.load(_f_n))
                    self.node_types = torch.Tensor(np.load(_f_x))
                    self.edge_types = torch.Tensor(np.load(_f_e))
                    _loaded = True
                    print(f'[Benzene] Loaded marginals from {_raw}: '
                          f'{len(self.node_types)} atom types, '
                          f'max {len(self.n_nodes) - 1} nodes')
            if not _loaded:
                print('[Benzene] WARNING: marginals not found on disk, using fallback. '
                      'Run download() first.')
                self.n_nodes = torch.ones(26) / 26.0
                self.node_types = torch.ones(8) / 8.0
                self.edge_types = torch.Tensor([0.90, 0.055, 0.007, 0.0005, 0.0375])

        elif dataset_name in ('proteins', 'imdb_binary', 'twitter'):
            # Marginals are measured and written by download(); loading them from disk
            # avoids the hand-copied literals that the aids/nci1 branches above carry.
            _tag = dataset_name.upper()
            _loaded = False
            _datadir = getattr(dataset_cfg, 'datadir', None) if dataset_cfg is not None else None
            if _datadir is not None:
                _base = pathlib.Path(os.path.realpath(__file__)).parents[2]
                _raw = os.path.join(_base, _datadir, 'raw')
                _f_n = os.path.join(_raw, 'n_nodes_dist.npy')
                _f_x = os.path.join(_raw, 'node_type_marginals.npy')
                _f_e = os.path.join(_raw, 'edge_type_marginals.npy')
                if all(os.path.exists(p) for p in (_f_n, _f_x, _f_e)):
                    self.n_nodes = torch.Tensor(np.load(_f_n))
                    self.node_types = torch.Tensor(np.load(_f_x))
                    self.edge_types = torch.Tensor(np.load(_f_e))
                    _loaded = True
                    print(f'[{_tag}] Loaded marginals from {_raw}: '
                          f'{len(self.node_types)} node type(s), '
                          f'max {len(self.n_nodes) - 1} nodes')
            if not _loaded:
                print(f'[{_tag}] WARNING: marginals not found on disk, using fallback. '
                      'Run download() first.')
                self.n_nodes = torch.ones(51) / 51.0
                _nt = 3 if dataset_name == 'proteins' else 1
                self.node_types = torch.ones(_nt) / _nt
                self.edge_types = torch.Tensor([0.90, 0.10])

        super().complete_infos(self.n_nodes, self.node_types)

