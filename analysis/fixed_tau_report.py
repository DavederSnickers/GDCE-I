"""Fixed-budget comparison (no inversion vs. posterior inversion) at isolated tau values.

Usage: python fixed_tau_report.py <chains.csv>
Per tau and method: the paper's metrics (FR, NAFR, NA, three sparsities, SMILES) plus
aligned edits, Tanimoto to the source and the placebo replay under the source class.
A paired block compares the two methods on the graphs where both flip.
"""
import os
import sys
import numpy as np
import pandas as pd
import analyze_paired as ap

pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 40)

df = pd.concat([ap.load(p) for p in sys.argv[1:]], ignore_index=True)
if 'TAUS' in os.environ:    # e.g. TAUS=10,50,100 to pick isolated budgets from a fine grid
    df = df[df.tau.isin([int(t) for t in os.environ['TAUS'].split(',')])]
METHODS = [v for v in ['noinv', 'naive', 'posterior'] if v in set(df.variant)]
tgt = df[df.cond == 'tgt']
src = df[df.cond == 'src']
n = len(df[['run', 'sample_idx']].drop_duplicates())
print(f'{sys.argv[1:]}: {n} graphs, tau {sorted(df.tau.unique())}, variants {METHODS}')

rows = []
for tau in sorted(df.tau.unique()):
    for v in METHODS:
        t = tgt[(tgt.variant == v) & (tgt.tau == tau)]
        s = src[(src.variant == v) & (src.tau == tau)]
        f = t[t.flip == 1]
        rows.append({
            'tau': tau, 'method': v, 'flips': f'{len(f)}/{len(t)}',
            'FR': t.flip.mean(), 'NAFR': (t.flip & t.ctrl_flip).mean(),
            'NA': f.ctrl_flip.mean(),
            'EdgeSp': 1 - f.edge_rate.mean(), 'NodeSp': 1 - f.node_rate.mean(),
            'ETypeSp': 1 - f.edgetype_rate.mean(),
            'SMILES|flip': f.smiles_valid.mean(), 'FR&SMILES': (t.flip & t.smiles_valid).mean(),
            'edits|flip': f.n_total.mean(), 'tani|flip': f.tani_src.mean(),
            'plc_edits': s.n_total.mean(), 'plc_FR': s.flip.mean(),
            'attr_FR': t.flip.mean() - s.flip.mean(),
        })
T = pd.DataFrame(rows)
print('\n--- per tau (target replay; plc_* = replay under the SOURCE class; attr_FR = FR - plc_FR) ---')
print(T.round(3).to_string(index=False))

print('\n--- paired, same tau, graphs where BOTH methods flip ---')
rng = np.random.default_rng(0)
pr = []
for tau in sorted(df.tau.unique()):
    A = tgt[(tgt.variant == 'posterior') & (tgt.tau == tau) & (tgt.flip == 1)].set_index(['run', 'sample_idx'])
    B = tgt[(tgt.variant == 'noinv') & (tgt.tau == tau) & (tgt.flip == 1)].set_index(['run', 'sample_idx'])
    idx = A.index.intersection(B.index)
    r = {'tau': tau, 'n_both': len(idx), 'only_posterior': len(A.index.difference(B.index)),
         'only_noinv': len(B.index.difference(A.index))}
    if len(idx):
        x, y = A.loc[idx, 'n_total'].values.astype(float), B.loc[idx, 'n_total'].values.astype(float)
        d = x - y
        boots = [rng.choice(d, len(d)).mean() for _ in range(2000)]
        r.update({'edits_post': x.mean(), 'edits_noinv': y.mean(),
                  'diff_CI95': f'[{np.percentile(boots, 2.5):.1f}, {np.percentile(boots, 97.5):.1f}]',
                  'tani_post': A.loc[idx, 'tani_src'].mean(), 'tani_noinv': B.loc[idx, 'tani_src'].mean(),
                  'NA_post': A.loc[idx, 'ctrl_flip'].mean(), 'NA_noinv': B.loc[idx, 'ctrl_flip'].mean()})
    pr.append(r)
print(pd.DataFrame(pr).round(3).to_string(index=False))
print('\nposterior replay under the source class exact for all graphs and tau:',
      bool(src[src.variant == 'posterior'].exact.all()))
