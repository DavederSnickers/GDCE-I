"""DICE comparison (jobs 5130017-26) in the metric conventions of the revised Table 3.

Same per-run conventions as reviews/analysis/ablation_fixed_tau.py: per-run means over 100 graphs,
mean +- sample std over the 5 runs; EdgeSp/NodeSp = 1 - mean modification rate over flips
(EdgeSp clamped at 0); FR&SMILES over all inputs; PlaceboFR = flip rate of the replay under the
source class; AttrFR = FR - PlaceboFR; Exact = share of source-class replays that return the input.
Usage: python dice_table3_format.py '<glob of chains.csv>' <label>
  e.g. on hydra: python dice_table3_format.py 'results_review/paired_tau/dice_mutag_s3_r*/chains.csv' mutag
Output of 2026-10-30: results/dice_table3_{mutag,benzene}.txt (letter table of R2.1 = tau 100 block).
"""
import glob
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, 'analysis')
import analyze_paired as ap  # noqa: E402

pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 40)

pattern, ds = sys.argv[1], sys.argv[2]
paths = sorted(glob.glob(pattern))
df = pd.concat([ap.load(p) for p in paths], ignore_index=True)
VARIANTS = ['noinv', 'posterior', 'posteriorI', 'diceI_20_80', 'diceM_20_80', 'diceM_100_0',
            'diceM_2_98', 'diceI_5_95', 'diceM_100_0_f5', 'diceM_100_0_f10']
VARIANTS = [v for v in VARIANTS if v in set(df.variant)]
TAUS = [11, 51, 100, 200]
print(f'{ds}: {len(paths)} runs, {len(df[["run", "sample_idx"]].drop_duplicates())} graphs')

rows = []
for (tau, v, run), t in df[(df.cond == 'tgt') & df.tau.isin(TAUS) & df.variant.isin(VARIANTS)].groupby(
        ['tau', 'variant', 'run']):
    s = df[(df.cond == 'src') & (df.tau == tau) & (df.variant == v) & (df.run == run)]
    f = t[t.flip == 1]
    rows.append({
        'tau': int(tau), 'variant': v, 'run': run,
        'FR': t.flip.mean(),
        'NAFR': (t.flip & t.ctrl_flip).mean(),
        'EdgeSp': max(1 - f.edge_rate.mean(), 0.0) if len(f) else np.nan,
        'NodeSp': 1 - f.node_rate.mean() if len(f) else np.nan,
        'FR&SMILES': (t.flip & t.smiles_valid).mean(),
        'PlaceboFR': s.flip.mean(),
        'AttrFR': t.flip.mean() - s.flip.mean(),
        'Exact': s.exact.mean(),
        'EdgeSp_plc_all': max(1 - s.edge_rate.mean(), 0.0),
        'Edits|flip': f.n_total.mean() if len(f) else np.nan,
    })
P = pd.DataFrame(rows)
COLS = ['FR', 'NAFR', 'EdgeSp', 'NodeSp', 'FR&SMILES', 'PlaceboFR', 'AttrFR', 'Exact', 'EdgeSp_plc_all',
        'Edits|flip']
agg = P.groupby(['tau', 'variant'])[COLS].agg(['mean', 'std'])
for tau in TAUS:
    print(f'\n--- {ds}, tau = {tau} ---')
    out = []
    for v in VARIANTS:
        if (tau, v) not in agg.index:
            continue
        r = {'variant': v}
        for c in COLS:
            m, sd = agg.loc[(tau, v), (c, 'mean')], agg.loc[(tau, v), (c, 'std')]
            r[c] = f'{m:.1f}+-{sd:.1f}' if c == 'Edits|flip' else f'{m:.3f}+-{sd:.3f}'
        out.append(r)
    print(pd.DataFrame(out).to_string(index=False))
