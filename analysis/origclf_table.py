"""Original-classifier ablation (R1.4): the table, computed with ONE metric implementation for all
methods from the re-scored counterfactual dumps (adversarial_flip.py output, one row per dumped
counterfactual with adv_flip_base / adv_flip_ctrl).

Conventions of Table 2: each run has 100 inputs; FR, NAFR and SMILES (= NAFR & valid SMILES) are
fractions of all inputs, the sparsities are conditioned on the flips; per-run means, then
mean +- sample std over the five runs. NA = mean NAFR / mean FR, as in Table 2. A flip counts only if the attacked classifier, re-applied
to the dumped graph, gives the target class (UCExplainer: after the one-hot projection).
Sparsity = 1 - mean modification rate over the flips (rates as inversion_mutag_mol.compute_sparsity:
changed atoms / n, |E xor E'| / |E|, changed bond types of kept edges / |E|; EdgeSp clamped at 0).
Validity = RDKit sanitisation (review_metrics.strict_smiles); for edge-only methods the dumped bond
orders are replaced as in the audit: a kept edge keeps its input order, an added edge is single.

Usage: python origclf_table.py <advflip dir> [<reference advflip dir of Table 2>]
"""
import ast
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, './src')
import review_metrics as rm  # noqa: E402
from rdkit import RDLogger  # noqa: E402

RDLogger.DisableLog('rdApp.*')
pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 30)

DATA = './data/mutagenicity_mol/raw'
DEC = json.load(open(os.path.join(DATA, 'atom_decoder.json')))
N_RUNS, N_PER_RUN = 5, 100
OFFSETS = (0, 2377, 2377 + 792)    # test index, index behind train (GRETEL/XPlore's GDCE loader),
                                   # or index into train+val+test (COMBINEX)

# rows of the ablation table: label -> (classifier block, method shown)
ROWS = [('C2_default', 'c2gcn', 'C2Explainer (default)'), ('C2_tuned', 'c2gcn', 'C2Explainer (tuned)'),
        ('GDCE_c2gcn', 'c2gcn', 'GDCE-I'),
        ('CF2_default', 'cf2gcn', 'CF2 (default)'), ('CF2_tuned', 'cf2gcn', 'CF2 (tuned)'),
        ('GDCE_cf2gcn', 'cf2gcn', 'GDCE-I'),
        ('D4_june', 'd4native', 'D4Explainer (June model, alpha_cf 0.5)'), ('GDCE_d4native', 'd4native', 'GDCE-I'),
        ('GDCE_d4gcn', 'd4gcn', 'GDCE-I (re-implemented D4 classifier)'),
        ('UCX_default', 'ucxgine', 'UCExplainer (default)'), ('UCX_tuned', 'ucxgine', 'UCExplainer (tuned)'),
        ('GDCE_ucxgine', 'ucxgine', 'GDCE-I'),
        ('XPL_default', 'xplgcn', 'XPlore (default)'), ('XPL_tuned', 'xplgcn', 'XPlore (tuned)'),
        ('GDCE_xplgcn', 'xplgcn', 'GDCE-I')]
REFERENCE = [('C2_mutag_gcn', 'C2Explainer'), ('CF2_mutag_gcn', 'CF2'), ('XPL_mutag_gcn', 'XPlore'),
             ('D4_mutag_flip', 'D4Explainer'),
             ('UCX_mutag_gine', 'UCExplainer (GINE)'), ('GDCE_mutag_gcn', 'GDCE-I (GCN)'),
             ('GDCE_mutag_gine', 'GDCE-I (GINE)')]


def load_sources(split='test'):
    df = pd.read_csv(os.path.join(DATA, f'{split}.csv'))
    out = []
    for _, row in df.iterrows():
        adj = np.fromstring(str(row['adjacency_matrix']).strip('[]'), sep=',')
        n = int(round(np.sqrt(len(adj))))
        et = np.fromstring(str(row['edge_type_matrix']).strip('[]'), sep=',').astype(np.int64)
        et = et.reshape(n, n) * (adj.reshape(n, n) > 0)
        np.fill_diagonal(et, 0)
        out.append((np.array(ast.literal_eval(str(row['node_types'])), dtype=np.int64), et))
    return out


def cf_graph(row):
    nt = np.array(json.loads(row.node_types) if isinstance(row.node_types, str) else row.node_types)
    edges = json.loads(row.edges) if isinstance(row.edges, str) else row.edges
    et = np.zeros((len(nt), len(nt)), dtype=np.int64)
    for a, b, t in edges:
        if a != b:
            et[a, b] = et[b, a] = t
    return nt, et


def rates(src, cf):
    """Modification rates as inversion_mutag_mol.compute_sparsity (node / edge / edge type)."""
    (nt0, et0), (nt1, et1) = src, cf
    n0, n1 = len(nt0), len(nt1)
    m = min(n0, n1)
    node = (int((nt0[:m] != nt1[:m]).sum()) + abs(n0 - n1)) / max(n0, n1)
    e0 = {(i, j): int(et0[i, j]) for i, j in zip(*np.nonzero(np.triu(et0, 1)))}
    e1 = {(i, j): int(et1[i, j]) for i, j in zip(*np.nonzero(np.triu(et1, 1)))}
    n_e = max(len(e0), 1)
    edge = len(set(e0) ^ set(e1)) / n_e
    etype = sum(e0[k] != e1[k] for k in set(e0) & set(e1)) / n_e
    return node, edge, etype


def pick_offset(d, sources):
    """The index space of data_idx: the one in which every dumped graph has its source's size."""
    best = None
    for off in OFFSETS:
        idx = d.data_idx.astype(int) - off
        if (idx < 0).any() or (idx >= len(sources)).any():
            continue
        size_ok = np.mean([len(sources[i][0]) == len(cf_graph(r)[0]) for i, r in zip(idx, d.itertuples())])
        if best is None or size_ok > best[1]:
            best = (off, size_ok)
    return best


def evaluate(path, sources, n_runs=N_RUNS):
    d = pd.read_csv(path)
    off, size_ok = pick_offset(d, sources)
    # edge-only methods: 'inherited' (C2, CF2, D4 dumps) or 'binary' (XPlore dumps)
    edge_only = d.bond_source.iloc[0] in ('inherited', 'binary')
    recs = []
    for r in d.itertuples():
        src = sources[int(r.data_idx) - off]
        nt, et = cf_graph(r)
        if edge_only and len(nt) == len(src[0]):
            et = np.where(et > 0, np.where(src[1] > 0, src[1], 1), 0)
        node, edge, etype = rates(src, (nt, et)) if len(nt) == len(src[0]) else (np.nan,) * 3
        recs.append({'run': r.run, 'base': int(r.adv_flip_base), 'ctrl': int(r.adv_flip_ctrl),
                     'valid': rm.strict_smiles(nt, et, DEC) is not None, 'node': node, 'edge': edge,
                     'etype': etype, 'same_atoms': len(nt) == len(src[0]) and bool((nt == src[0]).all())})
    g = pd.DataFrame(recs)
    per = []
    for run in range(n_runs):
        x = g[(g.run == run) & (g.base == 1)]
        per.append({'FR': len(x) / N_PER_RUN, 'NAFR': x.ctrl.sum() / N_PER_RUN,
                    'NA': x.ctrl.mean() if len(x) else np.nan,
                    'SMILES': (x.ctrl.astype(bool) & x.valid).sum() / N_PER_RUN,
                    'FR&SMILES': x.valid.sum() / N_PER_RUN,
                    'EdgeSp': max(1 - x.edge.mean(), 0.0) if len(x) else np.nan,
                    'NodeSp': 1 - x.node.mean() if len(x) else np.nan,
                    'EdgeTypeSp': 1 - x.etype.mean() if len(x) else np.nan,
                    'EdgeRate': x.edge.mean() if len(x) else np.nan})
    per = pd.DataFrame(per)
    info = {'n_dumped': len(g), 'gate': g.base.mean(), 'offset': off, 'size_match': size_ok,
            'atoms_identical': g.same_atoms.mean(), 'edge_only': edge_only}
    return per, info


def fmt(per, col, std=True):
    m, s = per[col].mean(), per[col].std(ddof=1)
    return f'{m:.3f}' if not std or np.isnan(s) else f'{m:.3f}+-{s:.3f}'


def block(rows, folder, sources):
    out = []
    for label, *names in rows:
        path = os.path.join(folder, f'{label}.csv')
        if not os.path.exists(path):
            out.append({'label': label, 'method': names[-1], 'status': 'missing'})
            continue
        per, info = evaluate(path, sources)
        row = {'label': label, 'method': names[-1]}
        for c in ['FR', 'NAFR', 'EdgeSp', 'NodeSp', 'EdgeTypeSp', 'SMILES', 'FR&SMILES']:
            row[c] = fmt(per, c)
        # Table 2's NA is the ratio of the run averages, NAFR / FR (reproduces every NA cell)
        row['NA'] = f"{per.NAFR.mean() / per.FR.mean():.3f}" if per.FR.mean() > 0 else 'nan'
        row['EdgeRate'] = f"{per.EdgeRate.mean():.3f}"
        row.update({k: (round(v, 3) if isinstance(v, float) else v) for k, v in info.items()})
        out.append(row)
    return pd.DataFrame(out)


if __name__ == '__main__':
    sources = load_sources()
    print(block(ROWS, sys.argv[1], sources).to_string(index=False))
    if len(sys.argv) > 2:
        print('\n--- Table 2 runs (shared judge), same implementation ---')
        print(block(REFERENCE, sys.argv[2], sources).to_string(index=False))
