"""Paper-grid (110 tau) replication of the Table 3 operating point + lottery test."""
import sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import analyze_paired as ap

pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 60)

df = pd.concat([ap.load(p) for p in sys.argv[1:]], ignore_index=True)
keys = df[['run', 'sample_idx']].drop_duplicates()
print('graphs', len(keys), 'rows', len(df))
print('posterior placebo exact (all tau):', df[(df.variant == 'posterior') & (df.cond == 'src')].exact.mean())

first, n = ap.first_flip(df)
print('\n=== first flip on the paper grid (= Table 3 operating point) ===')
print(ap.summarize_first(first, n).round(3).to_string(index=False))

# compare with Table 3 dumps on the same graphs
dump = pd.read_csv('results/dump/mutag_gcn_table3.csv', keep_default_na=False, na_values=[''])
dump['heavy_conn'] = (dump.n_heavy_comp <= dump.n_heavy_comp_src).astype(int)
print('\n=== Table 3 dumps (original pipeline), same metrics ===')
print(dump.groupby('variant')[['flip', 'ctrl_flip', 'n_total', 'n_total_heavy', 'smiles_valid', 'heavy_conn', 'tani_src']].mean().round(3))
print('dump t median:', dump.groupby('variant').t.median().to_dict())
m = first.merge(dump[['run', 'sample_idx', 'variant', 'data_idx']], on=['run', 'sample_idx', 'variant'], how='left', suffixes=('', '_d'))
print('same graphs as dumps (data_idx match):', (m.data_idx == m.data_idx_d).mean())

# paired NA / edits on graphs where both flip
for b in ['noinv', 'naive']:
    A = first[first.variant == 'posterior'].set_index(['run', 'sample_idx'])
    B = first[first.variant == b].set_index(['run', 'sample_idx'])
    idx = A.index.intersection(B.index)
    x, y = A.loc[idx], B.loc[idx]
    bb, cc, p = ap.mcnemar(x.ctrl_flip.values, y.ctrl_flip.values)
    w = wilcoxon(x.n_total.values, y.n_total.values)
    print(f'\nposterior vs {b} (both flip, n={len(idx)}): NA {x.ctrl_flip.mean():.3f} vs {y.ctrl_flip.mean():.3f} '
          f'McNemar b={bb} c={cc} p={p:.3g}; edits {x.n_total.mean():.2f} vs {y.n_total.mean():.2f} Wilcoxon p={w.pvalue:.2g}; '
          f'tau median {x.tau.median()} vs {y.tau.median()}')

print('\n=== grid thinning (lottery test) ===')
print(ap.grid_thinning(df, steps=(1, 2, 3, 5, 10, 20, 50)).round(3).to_string(index=False))

pt, ps = ap.placebo(df)
print('\n=== placebo grid search (110 tau) ===')
print(ps.round(3).to_string(index=False))
sel = pt[pt.tau.isin([1, 2, 3, 5, 8, 11, 15, 20, 30, 51, 75, 100, 150, 200])]
print(sel.pivot(index='tau', columns='variant', values=['FR_tgt', 'FR_placebo', 'attributable']).round(3).to_string())

t = ap.tau_table(df)
print('\n=== matched FR ===')
print(ap.matched_fr(t).round(3).to_string(index=False))
print('\n=== monotonicity ===')
print(ap.monotonicity(df).round(3).to_string(index=False))
cm = ap.cumulative_vs_marginal(t, first, n)
print('\n=== cumulative vs marginal ===')
print(cm.loc[[1, 2, 3, 5, 10, 20, 30, 50, 75, 100, 150, 200]].round(3).to_string())
print('\n=== same tau, all chains: edits / FR ===')
print(t[t.tau.isin([1, 3, 5, 11, 20, 30, 51, 75, 100])].pivot(index='tau', columns='variant', values=['FR', 'edits_all', 'edits|flip', 'NA|flip']).round(3).to_string())
first.to_csv('results/paper_first.csv', index=False)
