"""Metric helpers shared between the Mutag and NCI1 evaluation scripts.

Copies / adapts:
  - eval_mmd (degree / cluster / spectral MMD)  -> from GDCE inversion scripts
  - compute_ged                                  -> from GDCE inversion scripts
  - compute_sparsity_binary, compute_sparsity_typed -> simplified versions
  - graph_to_smiles (Mutag only)                 -> from GDCE inversion_mutag_mol.py
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import networkx as nx
import torch


# --------------------------------------------------------------------- MMD

def _mmd_pad(x, y):
    n = max(len(x), len(y))
    if len(x) < n:
        x = np.hstack((x, np.zeros(n - len(x))))
    if len(y) < n:
        y = np.hstack((y, np.zeros(n - len(y))))
    return x, y


def _gaussian_emd(x, y, sigma: float = 1.0, distance_scaling: float = 1.0) -> float:
    import pyemd
    from scipy.linalg import toeplitz
    x = x.astype(float); y = y.astype(float)
    x, y = _mmd_pad(x, y)
    n = len(x)
    d_mat = toeplitz(range(n)).astype(float) / distance_scaling
    e = pyemd.emd(x, y, d_mat)
    return float(np.exp(-e * e / (2 * sigma * sigma)))


def _disc(s1, s2, kernel, **kw) -> float:
    if not s1 or not s2:
        return 1e6
    total = 0.0
    for a in s1:
        for b in s2:
            total += kernel(a, b, **kw)
    return total / (len(s1) * len(s2))


def _mmd(s1, s2, kernel, is_hist=True, **kw) -> float:
    if is_hist:
        s1 = [s / np.sum(s) for s in s1 if np.sum(s) > 0]
        s2 = [s / np.sum(s) for s in s2 if np.sum(s) > 0]
    if not s1 or not s2:
        return float('nan')
    return _disc(s1, s1, kernel, **kw) + _disc(s2, s2, kernel, **kw) - 2 * _disc(s1, s2, kernel, **kw)


def _degree_mmd(ref, pred) -> float:
    pred = [G for G in pred if G.number_of_nodes() > 0]
    if not ref or not pred:
        return float('nan')
    return _mmd([np.array(nx.degree_histogram(G)) for G in ref],
                [np.array(nx.degree_histogram(G)) for G in pred],
                kernel=_gaussian_emd)


def _spectral_mmd(ref, pred) -> float:
    from scipy.linalg import eigvalsh
    pred = [G for G in pred if G.number_of_nodes() > 0]
    if not ref or not pred:
        return float('nan')

    def _spec(G):
        eigs = eigvalsh(nx.normalized_laplacian_matrix(G).todense())
        pmf, _ = np.histogram(eigs, bins=200, range=(-1e-5, 2), density=False)
        return pmf

    return _mmd([_spec(G) for G in ref], [_spec(G) for G in pred], kernel=_gaussian_emd)


def _cluster_mmd(ref, pred, bins: int = 100) -> float:
    pred = [G for G in pred if G.number_of_nodes() > 0]
    if not ref or not pred:
        return float('nan')

    def _clust(G):
        hist, _ = np.histogram(list(nx.clustering(G).values()),
                               bins=bins, range=(0.0, 1.0), density=False)
        return hist

    return _mmd([_clust(G) for G in ref], [_clust(G) for G in pred],
                kernel=_gaussian_emd, sigma=1.0 / 10, distance_scaling=bins)


def eval_mmd(ref: List[nx.Graph], cf: List[nx.Graph]) -> Dict[str, float]:
    return {
        'degree':   _degree_mmd(ref, cf),
        'cluster':  _cluster_mmd(ref, cf),
        'spectral': _spectral_mmd(ref, cf),
    }


# ----------------------------------------------------- edge_index → nx.Graph

def edge_index_to_nx(edge_index: torch.Tensor, n_nodes: int) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    if edge_index.numel() == 0:
        return G
    ei = edge_index.cpu().numpy()
    for i, j in zip(ei[0], ei[1]):
        if int(i) < int(j):
            G.add_edge(int(i), int(j))
    return G


# ----------------------------------------------------------- GED + Sparsity

def compute_ged(orig_edge_index: torch.Tensor, cf_edge_index: torch.Tensor,
                n_nodes: int, timeout: int = 5) -> int:
    G_orig = edge_index_to_nx(orig_edge_index, n_nodes)
    G_cf   = edge_index_to_nx(cf_edge_index,   n_nodes)
    try:
        ged = nx.graph_edit_distance(G_orig, G_cf, timeout=timeout)
        return int(ged) if ged is not None else -1
    except Exception:
        return -1


def _edge_dict(edge_index: torch.Tensor,
               edge_attr: Optional[torch.Tensor] = None) -> Dict[Tuple[int, int], int]:
    """Build {(i,j): bond_class}; i<j. bond_class is 0 if edge_attr not given."""
    d: Dict[Tuple[int, int], int] = {}
    if edge_index.numel() == 0:
        return d
    ei = edge_index.cpu()
    if edge_attr is None:
        for k in range(ei.size(1)):
            src, dst = int(ei[0, k]), int(ei[1, k])
            if src < dst:
                d[(src, dst)] = 0
    else:
        ea = edge_attr.cpu()
        ea = ea.argmax(dim=-1) if ea.dim() > 1 else ea.long()
        for k in range(ei.size(1)):
            src, dst = int(ei[0, k]), int(ei[1, k])
            if src < dst:
                d[(src, dst)] = int(ea[k])
    return d


def compute_sparsity(orig_edge_index: torch.Tensor, cf_edge_index: torch.Tensor,
                     orig_node_types: Optional[torch.Tensor] = None,
                     cf_node_types: Optional[torch.Tensor] = None,
                     orig_edge_attr: Optional[torch.Tensor] = None,
                     cf_edge_attr: Optional[torch.Tensor] = None
                     ) -> Tuple[float, float, float]:
    """Returns (node_sparsity, edge_sparsity, edgetype_sparsity).

    - node_sparsity: fraction of nodes whose atom type changed
    - edge_sparsity: |orig_edges △ cf_edges| / n_orig_edges (COMBINEX-style)
    - edgetype_sparsity: edges present in both, but bond type changed / n_orig_edges
      (returns 0.0 if no edge_attr provided)
    """
    # nodes
    if orig_node_types is not None and cf_node_types is not None:
        n_o, n_c = len(orig_node_types), len(cf_node_types)
        n_common = min(n_o, n_c)
        changed = sum(orig_node_types[i].item() != cf_node_types[i].item()
                      for i in range(n_common))
        changed += abs(n_o - n_c)
        node_sp = changed / max(n_o, n_c) if max(n_o, n_c) > 0 else 0.0
    else:
        node_sp = 0.0

    orig_edges = _edge_dict(orig_edge_index, orig_edge_attr)
    cf_edges   = _edge_dict(cf_edge_index,   cf_edge_attr)
    n_orig = len(orig_edges)

    o_keys, c_keys = set(orig_edges), set(cf_edges)
    edge_sp = len(o_keys.symmetric_difference(c_keys)) / n_orig if n_orig > 0 else 0.0

    if orig_edge_attr is not None and cf_edge_attr is not None:
        common = o_keys & c_keys
        type_changed = sum(orig_edges[k] != cf_edges[k] for k in common)
        edgetype_sp = type_changed / n_orig if n_orig > 0 else 0.0
    else:
        edgetype_sp = 0.0

    return node_sp, edge_sp, edgetype_sp


# ---------------------------------------------------------- SMILES (Mutag)

_BOND_MAP_CACHE = None


def _bond_map():
    global _BOND_MAP_CACHE
    if _BOND_MAP_CACHE is None:
        from rdkit import Chem
        _BOND_MAP_CACHE = {
            1: Chem.rdchem.BondType.SINGLE,
            2: Chem.rdchem.BondType.DOUBLE,
            3: Chem.rdchem.BondType.TRIPLE,
            4: Chem.rdchem.BondType.AROMATIC,
        }
    return _BOND_MAP_CACHE


def graph_to_smiles(node_types: torch.Tensor, edge_index: torch.Tensor,
                    edge_attr: torch.Tensor, atom_decoder: dict,
                    strict: bool = True) -> Optional[str]:
    """Build SMILES from a remapped Mutagenicity graph. Returns None on failure.

    atom_decoder: {str(remapped_idx) -> element_symbol}.
    edge_attr: (E, 5) one-hot, 0=no bond (skipped) / 1=single / 2=double / 3=triple / 4=aromatic.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import RWMol
    except ImportError:
        return None

    from collections import defaultdict

    bond_types = _bond_map()
    mol = RWMol()
    node_syms = []
    for idx in node_types.cpu().tolist():
        sym = atom_decoder.get(str(idx), atom_decoder.get(idx, 'C'))
        mol.AddAtom(Chem.Atom(sym))
        node_syms.append(sym)
    n_atoms = int(node_types.numel())

    bond_types_map = {}   # (i, j) with i<j -> int bond type
    if edge_index.numel() > 0:
        ei = edge_index.cpu().numpy()
        ea = edge_attr.argmax(dim=-1).cpu().numpy() if edge_attr.dim() > 1 else edge_attr.cpu().numpy().astype(int)
        added = set()
        for (src, dst), bt in zip(zip(ei[0], ei[1]), ea):
            i, j = int(src), int(dst)
            # RSGG-CE may add CF edges to padding nodes that have no atom;
            # such edges can't be a bond in the molecule, so skip them.
            if i >= n_atoms or j >= n_atoms:
                continue
            if i >= j or (i, j) in added:
                continue
            if int(bt) in bond_types:
                mol.AddBond(i, j, bond_types[int(bt)])
                bond_types_map[(i, j)] = int(bt)
                added.add((i, j))

    # Assign formal charges for nitro groups: N with >=2 O neighbors -> N+, O-.
    # IDENTICAL to GDCE's inversion_mutag_mol.graph_to_smiles so the strict-valid
    # criterion matches GDCE exactly. The Mutagenicity dataset stores no formal
    # charges, so nitroaromatics (common among mutagens) would otherwise fail
    # strict valence sanitization and be wrongly counted as invalid.
    nbrs = defaultdict(list)
    for (i, j), bt in bond_types_map.items():
        nbrs[i].append((j, bt))
        nbrs[j].append((i, bt))
    for atom_idx, sym in enumerate(node_syms):
        if sym == 'N':
            o_nbrs = [(nb, bt) for nb, bt in nbrs[atom_idx] if node_syms[nb] == 'O']
            if len(o_nbrs) >= 2:
                mol.GetAtomWithIdx(atom_idx).SetFormalCharge(1)
                for nb, bt in o_nbrs:
                    if bt == 1:  # single bond -> O-
                        mol.GetAtomWithIdx(nb).SetFormalCharge(-1)
                        break

    try:
        if strict:
            Chem.SanitizeMol(mol)
        else:
            Chem.SanitizeMol(
                mol,
                Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES
            )
        return Chem.MolToSmiles(mol)
    except Exception:
        return None
