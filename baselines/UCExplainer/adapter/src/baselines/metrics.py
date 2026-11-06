"""Counterfactual evaluation metrics matching GDCE/BASELINE_PLAN.md §5.

Helpers are direct ports of `GDCE/src/inversion_mutag_mol.py`
(`compute_ged`, `compute_sparsity`, `_degree_mmd`, `_clustering_mmd`,
`_spectral_mmd`, `eval_mmd`, `graph_to_smiles`) — duplicated here so
COMBINEX runs without depending on GDCE's importable Python modules.
"""

from __future__ import annotations

from typing import List, Optional

import networkx as nx
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Graph Edit Distance
# ---------------------------------------------------------------------------

def edge_index_to_nx(edge_index: torch.Tensor, n_nodes: int) -> nx.Graph:
    """Undirected nx.Graph from a PyG edge_index. Skips self-loops."""
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    if edge_index.numel() > 0:
        ei = edge_index.cpu().numpy()
        for i, j in zip(ei[0], ei[1]):
            if int(i) < int(j):
                G.add_edge(int(i), int(j))
    return G


def compute_ged(
    orig_edge_index: torch.Tensor,
    cf_edge_index: torch.Tensor,
    n_nodes_orig: int,
    n_nodes_cf: Optional[int] = None,
    timeout: float = 5.0,
) -> int:
    """nx.graph_edit_distance with a hard timeout. Returns -1 on timeout/exception."""
    if n_nodes_cf is None:
        n_nodes_cf = n_nodes_orig
    G_orig = edge_index_to_nx(orig_edge_index, n_nodes_orig)
    G_cf = edge_index_to_nx(cf_edge_index, n_nodes_cf)
    try:
        ged = nx.graph_edit_distance(G_orig, G_cf, timeout=timeout)
        return -1 if ged is None else int(ged)
    except Exception:
        return -1


# ---------------------------------------------------------------------------
# Sparsity (node, edge, edge-type)
# ---------------------------------------------------------------------------

def _node_types_from_onehot(x: torch.Tensor) -> torch.Tensor:
    """argmax along feature dim → atom-type indices."""
    return x.argmax(dim=-1).cpu()


def _edge_dict(edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor]) -> dict:
    """Map (i, j) with i<j → bond-type index (or 1 if no edge_attr)."""
    d: dict = {}
    if edge_index.numel() == 0:
        return d
    ei = edge_index.cpu()
    if edge_attr is not None and edge_attr.numel() > 0:
        ea = edge_attr.cpu()
        ea = ea.argmax(dim=-1) if ea.dim() > 1 else ea
    else:
        ea = None
    for k, (src, dst) in enumerate(zip(ei[0].tolist(), ei[1].tolist())):
        if src < dst:
            d[(src, dst)] = int(ea[k]) if ea is not None else 1
    return d


def compute_sparsity(
    orig_x: torch.Tensor,
    cf_x: torch.Tensor,
    orig_edge_index: torch.Tensor,
    cf_edge_index: torch.Tensor,
    orig_edge_attr: Optional[torch.Tensor] = None,
    cf_edge_attr: Optional[torch.Tensor] = None,
) -> tuple[float, float, float, int]:
    """Returns (node_sparsity, edge_sparsity, edgetype_sparsity, n_orig_edges_undirected).

    Formulas mirror BASELINE_PLAN.md §5.4-5.6.
    """
    orig_types = _node_types_from_onehot(orig_x)
    cf_types = _node_types_from_onehot(cf_x)
    n_orig = len(orig_types)
    n_cf = len(cf_types)
    n_common = min(n_orig, n_cf)

    changed_nodes = sum(
        orig_types[i].item() != cf_types[i].item() for i in range(n_common)
    )
    changed_nodes += abs(n_orig - n_cf)
    n_total_nodes = max(n_orig, n_cf)
    node_sparsity = changed_nodes / n_total_nodes if n_total_nodes > 0 else 0.0

    orig_edges = _edge_dict(orig_edge_index, orig_edge_attr)
    cf_edges = _edge_dict(cf_edge_index, cf_edge_attr)
    n_orig_edges = len(orig_edges)

    orig_keys = set(orig_edges)
    cf_keys = set(cf_edges)
    structural_changed = len(orig_keys.symmetric_difference(cf_keys))
    edge_sparsity = structural_changed / n_orig_edges if n_orig_edges > 0 else 0.0

    if orig_edge_attr is not None and cf_edge_attr is not None:
        common_keys = orig_keys & cf_keys
        type_changed = sum(orig_edges[k] != cf_edges[k] for k in common_keys)
        edgetype_sparsity = (
            type_changed / n_orig_edges if n_orig_edges > 0 else 0.0
        )
    else:
        edgetype_sparsity = float("nan")

    return node_sparsity, edge_sparsity, edgetype_sparsity, n_orig_edges


# ---------------------------------------------------------------------------
# MMD (degree / clustering / spectral)
# ---------------------------------------------------------------------------

def _mmd_pad(x: np.ndarray, y: np.ndarray):
    n = max(len(x), len(y))
    if len(x) < n:
        x = np.hstack((x, np.zeros(n - len(x))))
    if len(y) < n:
        y = np.hstack((y, np.zeros(n - len(y))))
    return x, y


def _gaussian_emd(x, y, sigma: float = 1.0, distance_scaling: float = 1.0):
    import pyemd
    from scipy.linalg import toeplitz

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x, y = _mmd_pad(x, y)
    n = len(x)
    d_mat = toeplitz(range(n)).astype(float) / distance_scaling
    emd_val = pyemd.emd(x, y, d_mat)
    return float(np.exp(-emd_val * emd_val / (2 * sigma * sigma)))


def _disc(s1, s2, kernel, **kwargs):
    d = 0.0
    for a in s1:
        for b in s2:
            d += kernel(a, b, **kwargs)
    total = len(s1) * len(s2)
    return d / total if total > 0 else 1e6


def _compute_mmd(s1, s2, kernel, is_hist: bool = True, **kwargs):
    if is_hist:
        s1 = [s / np.sum(s) for s in s1 if np.sum(s) > 0]
        s2 = [s / np.sum(s) for s in s2 if np.sum(s) > 0]
    if not s1 or not s2:
        return float("nan")
    return (
        _disc(s1, s1, kernel, **kwargs)
        + _disc(s2, s2, kernel, **kwargs)
        - 2 * _disc(s1, s2, kernel, **kwargs)
    )


def _degree_mmd(ref: List[nx.Graph], pred: List[nx.Graph]) -> float:
    pred = [G for G in pred if G.number_of_nodes() > 0]
    if not ref or not pred:
        return float("nan")
    s_ref = [np.array(nx.degree_histogram(G)) for G in ref]
    s_pred = [np.array(nx.degree_histogram(G)) for G in pred]
    return _compute_mmd(s_ref, s_pred, kernel=_gaussian_emd)


def _spectral_mmd(ref: List[nx.Graph], pred: List[nx.Graph]) -> float:
    from scipy.linalg import eigvalsh

    pred = [G for G in pred if G.number_of_nodes() > 0]
    if not ref or not pred:
        return float("nan")

    def _spec(G):
        eigs = eigvalsh(nx.normalized_laplacian_matrix(G).todense())
        pmf, _ = np.histogram(eigs, bins=200, range=(-1e-5, 2), density=False)
        return pmf

    s_ref = [_spec(G) for G in ref]
    s_pred = [_spec(G) for G in pred]
    return _compute_mmd(s_ref, s_pred, kernel=_gaussian_emd)


def _clustering_mmd(
    ref: List[nx.Graph], pred: List[nx.Graph], bins: int = 100
) -> float:
    pred = [G for G in pred if G.number_of_nodes() > 0]
    if not ref or not pred:
        return float("nan")

    def _clust(G):
        hist, _ = np.histogram(
            list(nx.clustering(G).values()),
            bins=bins, range=(0.0, 1.0), density=False,
        )
        return hist

    s_ref = [_clust(G) for G in ref]
    s_pred = [_clust(G) for G in pred]
    return _compute_mmd(
        s_ref, s_pred, kernel=_gaussian_emd,
        sigma=1.0 / 10, distance_scaling=bins,
    )


def eval_mmd(ref_graphs: List[nx.Graph], cf_graphs: List[nx.Graph]) -> dict:
    return {
        "degree": _degree_mmd(ref_graphs, cf_graphs),
        "cluster": _clustering_mmd(ref_graphs, cf_graphs),
        "spectral": _spectral_mmd(ref_graphs, cf_graphs),
    }


# ---------------------------------------------------------------------------
# Stratified sampling (BASELINE_PLAN §4.2)
# ---------------------------------------------------------------------------

def stratified_sample_indices(
    pred_labels: np.ndarray, seed: int, n_samples: int
) -> np.ndarray:
    """Class-stratified sampling proportional to predicted-class distribution.

    Mirrors the inversion scripts: numpy.random.default_rng(seed), no replacement.
    """
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(pred_labels, return_counts=True)
    n_total = len(pred_labels)
    indices = []
    for cls, cnt in zip(classes, counts):
        n_cls = int(round(n_samples * cnt / n_total))
        cls_idx = np.where(pred_labels == cls)[0]
        n_take = min(n_cls, len(cls_idx))
        if n_take > 0:
            indices.extend(rng.choice(cls_idx, size=n_take, replace=False))
    # If rounding lost samples, top up with uniformly drawn extras.
    if len(indices) < n_samples:
        remaining = np.setdiff1d(np.arange(n_total), np.asarray(indices))
        n_extra = min(n_samples - len(indices), len(remaining))
        if n_extra > 0:
            indices.extend(rng.choice(remaining, size=n_extra, replace=False))
    return np.asarray(indices[:n_samples], dtype=np.int64)
