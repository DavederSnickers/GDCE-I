"""Re-measure existing counterfactual dumps with the review metrics.

review_paired_tau.py evaluates the noise treatments on a fixed tau grid; this script
applies the same per-graph metrics to the counterfactuals the paper actually reports
(the first flip of the dynamic budget search, as written by CFDumpWriter), so the
Table 3 operating points can be compared on the same footing.

Usage (repo root, inside the container):
    PYTHONPATH=$PWD python src/review_dump_metrics.py --dataset mutagenicity_mol \
        --judge $GDCE_JUDGES/judge_mutagenicity_mol_gcn \
        --dump noinv=<path> --dump naive=<path> --dump posterior=<path> \
        --out results_review/dump_metrics/mutag_gcn_s3.csv
"""

import argparse
import ast
import csv
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'shared_judge'))
from shared_judge import load_judge  # noqa: E402
from review_paired_tau import judge_probs, greedy_necessary  # noqa: E402
import review_metrics as rm  # noqa: E402


def load_test_graphs(data_root, dataset):
    """Test split as (node_types, bond matrix), in CSV order (= GDCE's test data_idx)."""
    df = pd.read_csv(os.path.join(data_root, dataset, 'raw', 'test.csv'))
    graphs = []
    for _, row in df.iterrows():
        adj = np.fromstring(str(row['adjacency_matrix']).strip('[]'), sep=',')
        n = int(round(np.sqrt(len(adj))))
        et = np.fromstring(str(row['edge_type_matrix']).strip('[]'), sep=',').astype(np.int64)
        et = et.reshape(n, n) * (adj.reshape(n, n) > 0)
        np.fill_diagonal(et, 0)
        nt = np.array(ast.literal_eval(str(row['node_types'])), dtype=np.int64)
        graphs.append((nt, et))
    return graphs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--data_root', default='data')
    ap.add_argument('--judge', required=True)
    ap.add_argument('--ctrl', default=None)
    ap.add_argument('--dump', action='append', required=True, help='variant=path')
    ap.add_argument('--out', required=True)
    ap.add_argument('--greedy_max', type=int, default=150)
    args = ap.parse_args()

    from rdkit import RDLogger, DataStructs
    RDLogger.DisableLog('rdApp.*')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(os.path.join(args.data_root, args.dataset, 'raw', 'atom_decoder.json')) as f:
        decoder = json.load(f)
    sym2idx = {v: int(k) for k, v in decoder.items()}
    h_idx = sym2idx.get('H')
    judge = load_judge(args.judge, device=str(device))
    ctrl = load_judge(args.ctrl or args.judge + '_seed1', device=str(device))
    dx = judge.num_features
    graphs = load_test_graphs(args.data_root, args.dataset)
    fps = [rm.graph_fingerprint(nt, et, decoder) for nt, et in graphs]
    print(f'{len(graphs)} test graphs, dx={dx}, device={device}')

    rows = []
    for spec in args.dump:
        variant, path = spec.split('=', 1)
        n_bad = 0
        with open(path) as f:
            lines = [json.loads(line) for line in f if line.strip()]
        for d in lines:
            nt_src, et_src = graphs[d['data_idx']]
            n = len(nt_src)
            if d['n_nodes'] != n:
                n_bad += 1
                continue
            nt_cf = np.array(d['node_types'], dtype=np.int64)
            et_cf = np.zeros((n, n), dtype=np.int64)
            for i, j, b in d['edges']:
                et_cf[i, j] = et_cf[j, i] = b
            tgt = int(d['target_class'])
            NT = torch.tensor(np.stack([nt_src, nt_cf]), device=device)
            ET = torch.tensor(np.stack([et_src, et_cf]), device=device)
            pb = judge_probs(judge, NT, ET, dx).cpu().numpy()
            pc = judge_probs(ctrl, NT, ET, dx).cpu().numpy()
            heavy = (nt_src != h_idx) if h_idx is not None else np.ones(n, dtype=bool)
            ec = rm.edit_counts(nt_src, et_src, nt_cf, et_cf, heavy)
            n_comp, n_heavy_comp = rm.components(et_cf, heavy)
            n_comp_src, n_heavy_comp_src = rm.components(et_src, heavy)
            fp = rm.graph_fingerprint(nt_cf, et_cf, decoder)
            tani = DataStructs.TanimotoSimilarity(fp, fps[d['data_idx']]) if fp is not None else float('nan')
            row = {'variant': variant, 'run': d['run'], 'seed': d['seed'],
                   'sample_idx': d['sample_idx'], 'data_idx': d['data_idx'],
                   't': d.get('t', -1), 'n_nodes': n, 'src_class': d['source_class'],
                   'tgt_class': tgt, 'src_pred_check': int(pb[0].argmax()),
                   'flip': int(pb[1].argmax() == tgt), 'ctrl_flip': int(pc[1].argmax() == tgt),
                   'base_p_tgt': float(pb[1][tgt]), **ec,
                   'smiles_valid': int(rm.strict_smiles(nt_cf, et_cf, decoder) is not None),
                   'n_comp': n_comp, 'n_heavy_comp': n_heavy_comp,
                   'n_comp_src': n_comp_src, 'n_heavy_comp_src': n_heavy_comp_src,
                   'tani_src': tani}
            motif = set()
            if args.dataset == 'benzene':
                motif = {a for r in rm.benzene_rings(nt_src, et_src, sym2idx['C']) for a in r}
                row['gt_src'] = int(len(motif) > 0)
                row['gt_cf'] = int(len(rm.benzene_rings(nt_cf, et_cf, sym2idx['C'])) > 0)
                row['ring_touch'] = rm.edits_touching(nt_src, et_src, nt_cf, et_cf, motif)
            else:
                for g in rm.nitro_groups(nt_src, et_src, sym2idx['N'], sym2idx['O'], sym2idx['C']):
                    motif |= g
                row['nitro_src'] = int(len(motif) > 0)
                row['nitro_cf'] = int(len(rm.nitro_groups(
                    nt_cf, et_cf, sym2idx['N'], sym2idx['O'], sym2idx['C'])) > 0)
                row['nitro_touch'] = rm.edits_touching(nt_src, et_src, nt_cf, et_cf, motif)
            row['greedy_k'] = row['greedy_nec'] = row['greedy_nec_heavy'] = row['greedy_nec_motif'] = -1
            if row['flip']:
                gr = greedy_necessary(judge, NT[0], ET[0], NT[1], ET[1], tgt, dx, args.greedy_max)
                if gr is not None:
                    gk, gnec, gnodes, gpairs = gr
                    row['greedy_k'], row['greedy_nec'] = gk, gnec
                    row['greedy_nec_heavy'] = (sum(1 for a in gnodes if heavy[a]) +
                                               sum(1 for a, b in gpairs if heavy[a] and heavy[b]))
                    row['greedy_nec_motif'] = (sum(1 for a in gnodes if a in motif) +
                                               sum(1 for a, b in gpairs if a in motif or b in motif))
            rows.append(row)
        print(f'  {variant}: {len(lines)} dump lines, {n_bad} size mismatches')

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(args.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys, restval='')
        w.writeheader()
        w.writerows(rows)
    df = pd.DataFrame(rows)
    print(df.groupby('variant')[['flip', 'ctrl_flip', 'n_total', 'n_total_heavy', 'greedy_nec',
                                 'smiles_valid', 'tani_src']].mean().round(3).to_string())
    print(f'-> {args.out}')


if __name__ == '__main__':
    main()
