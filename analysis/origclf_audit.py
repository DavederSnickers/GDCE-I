"""Audit of the original-classifier runs (R1.4, David 2026-11-02): do the counterfactuals behind the
strong C2Explainer, CF2 and UCExplainer cells follow the protocol (discrete graphs, no one-hot
violations, flip confirmed by re-scoring), and what kind of edits produce them?

Per run, over the counted flips (attacked classifier re-applied to the dumped graph): structural
sanity of the dumped graphs, edit counts against the source, bonds added to hydrogen, RDKit
validity with the failure reason, and fragmentation of the valid counterfactuals. Bond orders of
edge-only methods follow the table convention (kept edge: input order, added edge: single).

Usage: python -I origclf_audit.py <origclf advflip dir> <Table-2 advflip dir>
"""
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import origclf_table as ot  # noqa: E402
import review_metrics as rm  # noqa: E402  (path set by origclf_table)

RUNS = [  # (label, folder index: 0 = original classifiers, 1 = Table 2)
    ('C2_tuned', 0), ('C2_mutag_gcn', 1), ('GDCE_c2gcn', 0),
    ('CF2_tuned', 0), ('CF2_mutag_gcn', 1), ('GDCE_cf2gcn', 0),
    ('UCX_tuned', 0), ('UCX_mutag_gine', 1), ('GDCE_ucxgine', 0),
]
H = [k for k, v in ot.DEC.items() if v == 'H']
H = int(H[0]) if H else None


def failure(nt, et):
    """None if RDKit sanitises the graph, else a short reason."""
    from rdkit import Chem
    try:
        Chem.SanitizeMol(rm.build_mol(nt, et, ot.DEC))
        return None
    except Exception as e:  # noqa: BLE001
        msg = str(e).lower()
        if 'valence' in msg:
            return 'valence'
        if 'kekulize' in msg or 'aromatic' in msg:
            return 'aromaticity'
        return 'other'


def audit(path, sources):
    d = pd.read_csv(path)
    off, _ = ot.pick_offset(d, sources)
    edge_only = d.bond_source.iloc[0] in ('inherited', 'binary')
    raw = Counter()
    rows = []
    for r in d.itertuples():
        src_nt, src_et = sources[int(r.data_idx) - off]
        nt = np.array(ot.json.loads(r.node_types) if isinstance(r.node_types, str) else r.node_types)
        edges = ot.json.loads(r.edges) if isinstance(r.edges, str) else r.edges
        # structural sanity of the dumped (discrete) graph
        pairs = [(min(a, b), max(a, b)) for a, b, _ in edges]
        raw['graphs'] += 1
        raw['self_loops'] += sum(a == b for a, b, _ in edges)
        raw['duplicate_pairs'] += len(pairs) - len(set(pairs))
        raw['bad_index'] += sum(not (0 <= a < len(nt) and 0 <= b < len(nt)) for a, b, _ in edges)
        raw['bad_bond_type'] += sum(int(t) not in (1, 2, 3, 4) for *_, t in edges)
        raw['bad_atom_type'] += int(((nt < 0) | (nt >= len(ot.DEC))).sum())
        if int(r.adv_flip_base) != 1:
            continue
        _, et = ot.cf_graph(r)
        if edge_only and len(nt) == len(src_nt):
            et = np.where(et > 0, np.where(src_et > 0, src_et, 1), 0)
        same_size = len(nt) == len(src_nt)
        e0 = {(i, j) for i, j in zip(*np.nonzero(np.triu(src_et, 1)))}
        e1 = {(i, j) for i, j in zip(*np.nonzero(np.triu(et, 1)))}
        added, deleted = (e1 - e0, e0 - e1) if same_size else (set(), set())
        retyped = sum(src_et[i, j] != et[i, j] for i, j in e0 & e1) if same_size else 0
        atoms = int((nt != src_nt).sum()) if same_size else np.nan
        on_h = sum(H in (nt[i], nt[j]) for i, j in added) if H is not None else 0
        reason = failure(nt, et)
        smi = rm.strict_smiles(nt, et, ot.DEC) if reason is None else None
        rows.append({'run': r.run, 'ctrl': int(r.adv_flip_ctrl), 'added': len(added),
                     'deleted': len(deleted), 'retyped': retyped, 'atoms': atoms, 'add_on_H': on_h,
                     'n_edges': len(e0), 'valid': reason is None, 'reason': reason or 'valid',
                     'fragments': smi.count('.') + 1 if smi else np.nan})
    return pd.DataFrame(rows), raw


def main(folders):
    sources = ot.load_sources()
    pd.set_option('display.width', 250)
    out = []
    for label, k in RUNS:
        g, raw = audit(os.path.join(folders[k], f'{label}.csv'), sources)
        edits = g.added + g.deleted + g.retyped
        added_total = g.added.sum()
        out.append({
            'run': label, 'dumped': raw['graphs'], 'flips': len(g),
            'selfloop/dup/badidx/badtype': f"{raw['self_loops']}/{raw['duplicate_pairs']}/"
                                           f"{raw['bad_index']}/{raw['bad_bond_type'] + raw['bad_atom_type']}",
            'edits_mean': round(edits.mean(), 2), 'edits_median': edits.median(),
            '1_edit': round((edits == 1).mean(), 3),
            'added': round(g.added.mean(), 2), 'deleted': round(g.deleted.mean(), 2),
            'retyped': round(g.retyped.mean(), 2), 'atoms_changed': round(g.atoms.mean(), 2),
            'added_on_H': round(g.add_on_H.sum() / added_total, 3) if added_total else np.nan,
            'valid': round(g.valid.mean(), 3),
            'valence_fail': round((g.reason == 'valence').mean(), 3),
            'arom_fail': round((g.reason == 'aromaticity').mean(), 3),
            'valid_fragmented': round((g[g.valid].fragments > 1).mean(), 3),
            'fragments_mean': round(g[g.valid].fragments.mean(), 2),
        })
    print(pd.DataFrame(out).to_string(index=False))


if __name__ == '__main__':
    main(sys.argv[1:3])
