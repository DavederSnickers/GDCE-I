"""Graph-level metrics for the paired noise-treatment comparison (review response).

Pure numpy / RDKit with no torch or networkx dependency, so the same functions run inside
the GDCE container (python 3.9) and in a local analysis environment.

Graphs are passed as
    nt : (n,)   int array of node-type indices
    et : (n, n) int array of bond classes (0 = no bond, 1 single, 2 double,
                3 triple, 4 aromatic), symmetric, zero diagonal
Source and counterfactual share the node indexing (the diffusion model never adds or
removes nodes), so edits are aligned position-wise and need no graph matching.
"""

from collections import defaultdict

import numpy as np

BOND_AROMATIC = 4


# ---------------------------------------------------------------------------
# aligned edits
# ---------------------------------------------------------------------------

def edit_masks(nt_src, et_src, nt_cf, et_cf):
    """Boolean masks of changed node positions (n,) and changed upper-triangle pairs (n, n)."""
    node_mask = nt_src != nt_cf
    pair_mask = np.triu(et_src != et_cf, 1)
    return node_mask, pair_mask


def edit_counts(nt_src, et_src, nt_cf, et_cf, heavy=None):
    """Counts of aligned edits, split by kind, plus the paper's three sparsity rates.

    heavy: optional (n,) bool mask of heavy (non-hydrogen) atoms in the SOURCE. Edits
    counted as "heavy" touch only heavy atoms: a node edit on a heavy atom, or a pair
    edit whose two endpoints are both heavy.
    """
    node_mask, pair_mask = edit_masks(nt_src, et_src, nt_cf, et_cf)
    s = et_src[pair_mask]
    c = et_cf[pair_mask]
    n_add = int(((s == 0) & (c > 0)).sum())
    n_del = int(((s > 0) & (c == 0)).sum())
    n_bt = int(((s > 0) & (c > 0)).sum())
    n_node = int(node_mask.sum())
    n = len(nt_src)
    n_src_edges = int(np.triu(et_src > 0, 1).sum())
    out = {
        'n_node': n_node, 'n_add': n_add, 'n_del': n_del, 'n_bt': n_bt,
        'n_total': n_node + n_add + n_del + n_bt,
        # paper conventions: modification RATES (1 - rate is the reported sparsity)
        'node_rate': n_node / n if n else 0.0,
        'edge_rate': (n_add + n_del) / n_src_edges if n_src_edges else 0.0,
        'edgetype_rate': n_bt / n_src_edges if n_src_edges else 0.0,
    }
    if heavy is not None:
        hp = np.outer(heavy, heavy)
        out['n_node_heavy'] = int((node_mask & heavy).sum())
        out['n_pair_heavy'] = int((pair_mask & hp).sum())
        out['n_total_heavy'] = out['n_node_heavy'] + out['n_pair_heavy']
    return out


def corruption_split(nt_src, et_src, nt_ref, et_ref, nt_cf, et_cf):
    """Split the counterfactual's edits by whether the reference state G^tau had corrupted them.

    corrupted  positions where G^tau differs from G (what the reverse pass must restore)
    restored   corrupted positions that the output brings back to G
    spontaneous edits at positions G^tau had NOT corrupted
    """
    cn, cp = edit_masks(nt_src, et_src, nt_ref, et_ref)
    en, ep = edit_masks(nt_src, et_src, nt_cf, et_cf)
    n_corrupt = int(cn.sum() + cp.sum())
    n_restored = int((cn & ~en).sum() + (cp & ~ep).sum())
    n_spont = int((en & ~cn).sum() + (ep & ~cp).sum())
    return {'n_corrupt': n_corrupt, 'n_restored': n_restored, 'n_spont': n_spont}


def shared_edits(nt_src, et_src, nt_a, et_a, nt_b, et_b):
    """Edits of graph a that graph b makes identically (same position, same new value)."""
    an, ap = edit_masks(nt_src, et_src, nt_a, et_a)
    same_n = an & (nt_a == nt_b)
    same_p = ap & np.triu(et_a == et_b, 1)
    return int(same_n.sum() + same_p.sum())


def edits_touching(nt_src, et_src, nt_cf, et_cf, atoms):
    """Number of aligned edits that touch a given atom set (node edit on it, or pair edit
    with at least one endpoint in it)."""
    if len(atoms) == 0:
        return 0
    sel = np.zeros(len(nt_src), dtype=bool)
    sel[list(atoms)] = True
    node_mask, pair_mask = edit_masks(nt_src, et_src, nt_cf, et_cf)
    touch_pair = pair_mask & (sel[:, None] | sel[None, :])
    return int((node_mask & sel).sum() + touch_pair.sum())


# ---------------------------------------------------------------------------
# connectivity
# ---------------------------------------------------------------------------

def components(et, heavy=None):
    """(#connected components, #components containing a heavy atom)."""
    n = et.shape[0]
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    ii, jj = np.nonzero(np.triu(et > 0, 1))
    for i, j in zip(ii.tolist(), jj.tolist()):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
    roots = [find(i) for i in range(n)]
    n_comp = len(set(roots))
    if heavy is None:
        return n_comp, n_comp
    n_heavy_comp = len({r for r, h in zip(roots, heavy) if h})
    return n_comp, n_heavy_comp


# ---------------------------------------------------------------------------
# motifs
# ---------------------------------------------------------------------------

def _adjacency(et):
    adj = defaultdict(set)
    ii, jj = np.nonzero(np.triu(et > 0, 1))
    for i, j in zip(ii.tolist(), jj.tolist()):
        adj[i].add(j)
        adj[j].add(i)
    return adj


def benzene_rings(nt, et, carbon=0):
    """Chordless 6-cycles of carbon atoms whose bonds are all aromatic, or alternate
    single/double (Kekule benzene).

    On the 2393 Benzene test molecules this rule reproduces the dataset label exactly
    (ring present <=> y = 1), so it serves as a ground-truth oracle for counterfactuals.
    """
    n = len(nt)
    is_c = (nt == carbon)
    ok_bond = (et == 1) | (et == 2) | (et == BOND_AROMATIC)
    adj = defaultdict(set)
    ii, jj = np.nonzero(np.triu(ok_bond, 1))
    for i, j in zip(ii.tolist(), jj.tolist()):
        if is_c[i] and is_c[j]:
            adj[i].add(j)
            adj[j].add(i)

    rings = set()
    for s in sorted(adj):
        # paths s -> ... of 6 distinct carbons with all later nodes > s (dedup by min node)
        stack = [(s, (s,))]
        while stack:
            v, path = stack.pop()
            if len(path) == 6:
                if s in adj[v]:
                    rings.add(path)
                continue
            for w in adj[v]:
                if w > s and w not in path:
                    stack.append((w, path + (w,)))

    out = []
    seen = set()
    for cyc in rings:
        key = frozenset(cyc)
        if key in seen:
            continue
        # chordless: no bond between non-consecutive ring atoms
        chord = False
        for a in range(6):
            for b in range(a + 2, 6):
                if a == 0 and b == 5:
                    continue
                if et[cyc[a], cyc[b]] > 0:
                    chord = True
        if chord:
            continue
        ts = [int(et[cyc[k], cyc[(k + 1) % 6]]) for k in range(6)]
        aromatic = all(t == BOND_AROMATIC for t in ts)
        kekule = sorted(ts) == [1, 1, 1, 2, 2, 2] and all(ts[k] != ts[(k + 1) % 6] for k in range(6))
        if aromatic or kekule:
            seen.add(key)
            out.append(tuple(sorted(cyc)))
    return out


def nitro_groups(nt, et, n_idx, o_idx, c_idx):
    """Aromatic-nitro groups as in inversion_mutag_nitro.has_aromatic_nitro_group:
    an N with >= 2 O neighbours and >= 1 C neighbour. Returns a list of atom-index sets
    (N, its O neighbours, its C neighbours)."""
    adj = _adjacency(et)
    groups = []
    for a in range(len(nt)):
        if int(nt[a]) != n_idx:
            continue
        o_nb = [b for b in adj[a] if int(nt[b]) == o_idx]
        c_nb = [b for b in adj[a] if int(nt[b]) == c_idx]
        if len(o_nb) >= 2 and len(c_nb) >= 1:
            groups.append({a, *o_nb, *c_nb})
    return groups


# ---------------------------------------------------------------------------
# RDKit
# ---------------------------------------------------------------------------

def build_mol(nt, et, decoder):
    """RWMol from the graph, with the nitro charge convention of graph_to_smiles
    (N with >= 2 O neighbours -> N+, one singly bonded O -> O-)."""
    from rdkit import Chem
    bt = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE,
          3: Chem.BondType.TRIPLE, 4: Chem.BondType.AROMATIC}
    mol = Chem.RWMol()
    syms = []
    for t in nt.tolist():
        sym = decoder.get(str(t), decoder.get(t, 'C'))
        mol.AddAtom(Chem.Atom(sym))
        syms.append(sym)
    nbrs = defaultdict(list)
    ii, jj = np.nonzero(np.triu(et > 0, 1))
    for i, j in zip(ii.tolist(), jj.tolist()):
        b = int(et[i, j])
        if b in bt:
            mol.AddBond(i, j, bt[b])
            nbrs[i].append((j, b))
            nbrs[j].append((i, b))
    for a, sym in enumerate(syms):
        if sym == 'N':
            o_nb = [(nb, b) for nb, b in nbrs[a] if syms[nb] == 'O']
            if len(o_nb) >= 2:
                mol.GetAtomWithIdx(a).SetFormalCharge(1)
                for nb, b in o_nb:
                    if b == 1:
                        mol.GetAtomWithIdx(nb).SetFormalCharge(-1)
                        break
    return mol


def strict_smiles(nt, et, decoder):
    """SMILES after full RDKit sanitisation, or None (= the paper's SMILES validity)."""
    from rdkit import Chem
    try:
        mol = build_mol(nt, et, decoder)
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def graph_fingerprint(nt, et, decoder, radius=2, n_bits=2048):
    """Morgan fingerprint computed WITHOUT sanitisation, so that source and counterfactual
    are fingerprinted by the same procedure whether or not RDKit accepts the graph."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    try:
        mol = build_mol(nt, et, decoder)
        for bond in mol.GetBonds():
            if bond.GetBondType() == Chem.BondType.AROMATIC:
                bond.SetIsAromatic(True)
                bond.GetBeginAtom().SetIsAromatic(True)
                bond.GetEndAtom().SetIsAromatic(True)
        mol.UpdatePropertyCache(strict=False)
        Chem.FastFindRings(mol)
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    except Exception:
        return None
