"""Recompute the last metric column of Table 3 (budget search) as NAFR & SMILES.

Sources (local):
  ablation/retrained/advflip/{noinv,fresh,posterior}_s3.csv   adv-flip dumps of the three
      variants under the budget search (confirmed flips only; adv_flip_base/_ctrl, smiles_valid)
  ablation/retrained/GDCE_{noinv,fresh,posterior}_s3.csv       the run dumps (t_used, success)
Conventions of Tables 2/3: fractions of ALL inputs (100 per run), per-run means, then mean and
sample std over the 5 runs. NA = mean NAFR / mean FR.
"""
import pandas as pd

ROOT = 'runs/table3'
N = 100
VAR = [('noinv', 'No inversion'), ('fresh', 'Naive inversion'), ('posterior', 'Posterior (GDCE-I)')]

for key, name in VAR:
    a = pd.read_csv(f'{ROOT}/advflip/{key}_s3.csv')
    d = pd.read_csv(f'{ROOT}/GDCE_{key}_s3.csv')
    runs = sorted(a.run.unique())
    fr, nafr, nas, frs = [], [], [], []
    for r in runs:
        g = a[a.run == r]
        base = g.adv_flip_base.astype(bool)
        ctrl = g.adv_flip_ctrl.astype(bool)
        val = g.smiles_valid.astype(bool)
        fr.append(base.sum() / N)
        nafr.append((base & ctrl).sum() / N)
        nas.append((base & ctrl & val).sum() / N)
        frs.append((base & val).sum() / N)
    fr, nafr, nas, frs = (pd.Series(x) for x in (fr, nafr, nas, frs))
    med = d[d.success == 1].t_used.median()
    print(f'{name:22s} runs={list(runs)}')
    print(f'   FR            {fr.mean():.3f} +- {fr.std():.3f}')
    print(f'   NAFR          {nafr.mean():.3f} +- {nafr.std():.3f}')
    print(f'   NA            {nafr.mean()/fr.mean():.3f}')
    print(f'   FR & SMILES   {frs.mean():.3f} +- {frs.std():.3f}   <- old "SMILES" column')
    print(f'   NAFR & SMILES {nas.mean():.3f} +- {nas.std():.3f}   <- new column')
    print(f'   median tau    {med:.0f}')
