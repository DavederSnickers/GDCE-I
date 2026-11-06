"""R2.3 audit: recompute FR / NA / NAFR / SMILES cells of Tables 2-4 from the per-CF adv-flip CSVs,
with one validity function for all methods, plus a fragment-aware variant.

valid      RDKit sanitisation of the whole graph (review_metrics.strict_smiles, nitro charges)
conn       the counterfactual has no more heavy-atom components than its source graph
"""
import gzip, json, sys
import numpy as np, pandas as pd
sys.path.insert(0, './src')
import review_metrics as rm
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)

DEC = {'mutag': json.load(open('results/mutag_s3/meta.json'))['atom_decoder'],
       'benzene': json.load(open('results/benzene_s3/meta.json'))['atom_decoder']}
H = {'mutag': 3, 'benzene': None}


def sources(ds):
    out = {}
    with gzip.open(f'results/{ds}_s3/graphs.jsonl.gz', 'rt') as f:
        for line in f:
            d = json.loads(line)
            if d['variant'] == 'posterior' and d['cond'] == 'src' and d['tau'] == 1:
                n = len(d['node_types']); et = np.zeros((n, n), dtype=np.int64)
                for a, b, t in d['edges']:
                    et[a, b] = et[b, a] = t
                out[d['data_idx']] = (np.array(d['node_types']), et)
    return out


def graph(row):
    nt = np.array(json.loads(row.node_types)); n = len(nt)
    et = np.zeros((n, n), dtype=np.int64)
    for a, b, t in json.loads(row.edges):
        if a != b:
            et[a, b] = et[b, a] = t
    return nt, et


def heavy_comp(nt, et, h):
    heavy = (nt != h) if h is not None else np.ones(len(nt), dtype=bool)
    return rm.components(et, heavy)[1]


FILES = {'mutag': ['GDCE_mutag_gcn', 'GDCE_mutag_gine', 'C2_mutag_gcn', 'CF2_mutag_gcn', 'D4_mutag_flip',
                   'UCX_mutag_gcn', 'UCX_mutag_gine'],
         'benzene': ['GDCE_benzene_gcn', 'GDCE_benzene_gine', 'C2_benzene_gcn', 'CF2_benzene_gcn', 'D4_benzene_flip',
                     'UCX_benzene_gcn', 'UCX_benzene_gine']}
rows = []
for ds in ['mutag', 'benzene']:
    src = sources(ds)
    src_nodes = {k: len(v[0]) for k, v in src.items()}
    for name in FILES[ds]:
        d = pd.read_csv(f'advflip/{name}.csv')
        d = d[d.adv_flip_base == 1].copy()          # flips confirmed on the decoded graph (UCX: projected)
        valid, conn, matched = [], [], []
        for r in d.itertuples():
            nt, et = graph(r)
            s = src.get(r.data_idx)
            if r.bond_source == 'inherited':
                # paper convention: a surviving edge keeps the input bond order, an added edge
                # is a single bond (the dumps store every edge of these methods as single)
                assert s is not None and len(s[0]) == len(nt), (name, r.data_idx)
                et = np.where(et > 0, np.where(s[1] > 0, s[1], 1), 0)
            valid.append(rm.strict_smiles(nt, et, DEC[ds]) is not None)
            if s is not None and len(s[0]) == len(nt):
                conn.append(heavy_comp(nt, et, H[ds]) <= heavy_comp(s[0], s[1], H[ds])); matched.append(True)
            else:   # UCX uses another index; fall back to "single heavy-atom molecule"
                conn.append(heavy_comp(nt, et, H[ds]) <= 1); matched.append(False)
        d['valid'] = np.array(valid); d['conn'] = np.array(conn); d['vc'] = d.valid & d.conn
        d['ctrl'] = d.adv_flip_ctrl == 1
        per = []
        for run in range(5):
            x = d[d.run == run]
            per.append({'FR': len(x) / 100, 'NAFR': x.ctrl.sum() / 100,
                        'NA': x.ctrl.mean(),
                        'valid|flip': x.valid.mean(), 'vc|flip': x.vc.mean(),
                        'FR&valid': x.valid.sum() / 100, 'NAFR&valid': (x.ctrl & x.valid).sum() / 100,
                        'FRxvalid|flip': len(x) / 100 * x.valid.mean(),
                        'FR&vc': x.vc.sum() / 100, 'NAFR&vc': (x.ctrl & x.vc).sum() / 100})
        per = pd.DataFrame(per)
        row = {'ds': ds, 'method': name, 'bonds': d.bond_source.iloc[0], 'n_flips': len(d), 'src_matched': np.mean(matched).round(3)}
        for c in per.columns:
            row[c] = per[c].mean()
        row['NAFR&valid_sd'] = per['NAFR&valid'].std(ddof=0)
        if 'smiles_valid' in d:
            row['csv_smiles_valid|flip'] = d.smiles_valid.astype(float).mean()
        rows.append(row)
out = pd.DataFrame(rows)
print(out.round(3).to_string(index=False))
out.to_csv('results/audit_tables.csv', index=False)
