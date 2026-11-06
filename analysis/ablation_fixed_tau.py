"""Inversion ablation at FIXED budgets, with the metric conventions of Table 3.

Usage: python ablation_fixed_tau.py <chains.csv> [<chains.csv> ...]   (env TAUS=10,50,100,200 optional)

Per (tau, variant) and run (100 graphs): FR, NAFR, NA (control-judge transfer among flips),
EdgeSp / NodeSp / EdgeTypeSp = 1 - mean modification rate over flips (rates as in
inversion_mutag_mol.compute_sparsity: |E xor E'| / |E|, changed atoms / n, changed bond types / |E|;
EdgeSp clamped at 0), FR&SMILES (= the SMILES column of Tables 3/4), SMILES|flip, and the
placebo flip rate of the replay under the source class. Reported as mean +- sample std over runs.
"""
import os
import sys
import numpy as np
import pandas as pd
import analyze_paired as ap

pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 40)

df = pd.concat([ap.load(p) for p in sys.argv[1:]], ignore_index=True)
if 'TAUS' in os.environ:
    df = df[df.tau.isin([int(t) for t in os.environ['TAUS'].split(',')])]
METHODS = [v for v in ['noinv', 'naive', 'posterior'] if v in set(df.variant)]
NAMES = {'noinv': 'No inversion', 'naive': 'Naive inversion', 'posterior': 'Posterior (GDCE-I)'}
n_graphs = len(df[['run', 'sample_idx']].drop_duplicates())
print(f'{len(sys.argv) - 1} file(s), {n_graphs} graphs, runs {sorted(df.run.unique())}, '
      f'tau {sorted(int(t) for t in df.tau.unique())}, variants {METHODS}')
print('posterior replay under the source class exact for all graphs and tau:',
      bool(df[(df.variant == 'posterior') & (df.cond == 'src')].exact.all()))

per_run = []
for (tau, v, run), t in df[df.cond == 'tgt'].groupby(['tau', 'variant', 'run']):
    s = df[(df.cond == 'src') & (df.tau == tau) & (df.variant == v) & (df.run == run)]
    f = t[t.flip == 1]
    per_run.append({
        'tau': int(tau), 'variant': v, 'run': run, 'n': len(t),
        'FR': t.flip.mean(),
        'NAFR': (t.flip & t.ctrl_flip).mean(),
        'NA': f.ctrl_flip.mean() if len(f) else np.nan,
        'EdgeSp': max(1 - f.edge_rate.mean(), 0.0) if len(f) else np.nan,
        'NodeSp': 1 - f.node_rate.mean() if len(f) else np.nan,
        'EdgeTypeSp': 1 - f.edgetype_rate.mean() if len(f) else np.nan,
        'FR&SMILES': (t.flip & t.smiles_valid).mean(),
        'SMILES|flip': f.smiles_valid.mean() if len(f) else np.nan,
        'PlaceboFR': s.flip.mean(),
        'Edits|flip': f.n_total.mean() if len(f) else np.nan,
    })
P = pd.DataFrame(per_run)
COLS = ['FR', 'NAFR', 'NA', 'EdgeSp', 'NodeSp', 'EdgeTypeSp', 'FR&SMILES', 'SMILES|flip', 'PlaceboFR',
        'Edits|flip']
agg = P.groupby(['tau', 'variant'])[COLS].agg(['mean', 'std'])

print('\n=== mean +- sample std over runs (per-run means of 100 graphs) ===')
for tau in sorted(P.tau.unique()):
    print(f'\n--- tau = {tau} ---')
    rows = []
    for v in METHODS:
        r = {'variant': NAMES[v]}
        for c in COLS:
            m, sd = agg.loc[(tau, v), (c, 'mean')], agg.loc[(tau, v), (c, 'std')]
            r[c] = f'{m:.1f}+-{sd:.1f}' if c == 'Edits|flip' else f'{m:.3f}+-{sd:.3f}'
        rows.append(r)
    print(pd.DataFrame(rows).to_string(index=False))

print('\n=== paired, same tau, graphs where both flip (pooled over runs) ===')
tgt = df[df.cond == 'tgt']
rng = np.random.default_rng(0)
out = []
for tau in sorted(P.tau.unique()):
    A = tgt[(tgt.variant == 'posterior') & (tgt.tau == tau) & (tgt.flip == 1)].set_index(['run', 'sample_idx'])
    for other in [m for m in METHODS if m != 'posterior']:
        B = tgt[(tgt.variant == other) & (tgt.tau == tau) & (tgt.flip == 1)].set_index(['run', 'sample_idx'])
        idx = A.index.intersection(B.index)
        x, y = A.loc[idx, 'n_total'].values.astype(float), B.loc[idx, 'n_total'].values.astype(float)
        d = x - y
        boots = [rng.choice(d, len(d)).mean() for _ in range(2000)] if len(d) else [np.nan]
        out.append({'tau': int(tau), 'vs': other, 'n_both': len(idx), 'edits_posterior': x.mean(),
                    f'edits_other': y.mean(),
                    'diff_CI95': f'[{np.percentile(boots, 2.5):.1f}, {np.percentile(boots, 97.5):.1f}]'})
print(pd.DataFrame(out).round(2).to_string(index=False))
