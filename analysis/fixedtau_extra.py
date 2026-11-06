"""Extra columns of the fixed-budget ablation (tau 10/50/100/200, 5x100 Mutag) for the answer to R1.1:
Tanimoto to the input over flips, placebo edits, target-attributable FR (FR - placebo FR), paired Tanimoto.

Usage: python fixedtau_extra.py <chains.csv> [...]
Input: hydra results_review/paired_tau/fixedtau_ablation_mutag_s3_r{0..4}/chains.csv (same as ablation_fixed_tau.py)."""
import sys
import numpy as np
import pandas as pd

df = pd.concat([pd.read_csv(p, keep_default_na=False, na_values=['']) for p in sys.argv[1:]],
               ignore_index=True)
df['tani_src'] = pd.to_numeric(df['tani_src'], errors='coerce')
tgt, src = df[df.cond == 'tgt'], df[df.cond == 'src']

rows = []
for (tau, v, run), t in tgt.groupby(['tau', 'variant', 'run']):
    s = src[(src.tau == tau) & (src.variant == v) & (src.run == run)]
    f = t[t.flip == 1]
    rows.append({'tau': tau, 'variant': v, 'run': run,
                 'FR': t.flip.mean(), 'PlaceboFR': s.flip.mean(),
                 'AttribFR': t.flip.mean() - s.flip.mean(),
                 'Tani|flip': f.tani_src.mean(),
                 'Edits|flip': f.n_total.mean(),
                 'PlaceboEdits': s.n_total.mean(),
                 'TgtEditsAll': t.n_total.mean()})
P = pd.DataFrame(rows)
agg = P.groupby(['tau', 'variant']).agg(['mean', 'std']).drop(columns='run')
pd.set_option('display.width', 250)
for tau in sorted(P.tau.unique()):
    print(f'\n--- tau = {tau} ---')
    out = {}
    for v in ['noinv', 'naive', 'posterior']:
        out[v] = {c: f"{agg.loc[(tau, v), (c, 'mean')]:.3f}+-{agg.loc[(tau, v), (c, 'std')]:.3f}"
                  for c in ['FR', 'PlaceboFR', 'AttribFR', 'Tani|flip', 'Edits|flip', 'PlaceboEdits',
                            'TgtEditsAll']}
    print(pd.DataFrame(out).T.to_string())

print('\n=== paired: graphs where both flip at the same tau (pooled), Tanimoto to input ===')
key = ['run', 'sample_idx', 'tau']
for other in ['noinv', 'naive']:
    a = tgt[tgt.variant == 'posterior'].set_index(key)
    b = tgt[tgt.variant == other].set_index(key)
    j = a[['flip', 'tani_src', 'n_total', 'smiles_valid', 'ctrl_flip']].join(
        b[['flip', 'tani_src', 'n_total', 'smiles_valid', 'ctrl_flip']], lsuffix='_p', rsuffix='_o')
    j = j[(j.flip_p == 1) & (j.flip_o == 1)].dropna(subset=['tani_src_p', 'tani_src_o'])
    rng = np.random.default_rng(0)
    for tau, d in j.groupby(level='tau'):
        diff = (d.tani_src_p - d.tani_src_o).values
        bs = [rng.choice(diff, len(diff)).mean() for _ in range(2000)]
        print(f'tau {tau:>3} vs {other:5s} n={len(d):3d}  tani post {d.tani_src_p.mean():.3f}  '
              f'other {d.tani_src_o.mean():.3f}  diff CI95 [{np.percentile(bs, 2.5):.3f}, '
              f'{np.percentile(bs, 97.5):.3f}]  smiles {d.smiles_valid_p.mean():.3f} vs '
              f'{d.smiles_valid_o.mean():.3f}  ctrl {d.ctrl_flip_p.mean():.3f} vs {d.ctrl_flip_o.mean():.3f}')
