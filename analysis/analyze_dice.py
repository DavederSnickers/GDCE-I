"""DICE vs posterior inversion: analysis of review_paired_tau.py runs with DICE variants.

Usage: python analyze_dice.py <chains.csv> [<chains.csv> ...]   (one dataset per call)
"""
import sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import analyze_paired as ap

pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 80)
pd.set_option('display.max_rows', 500)

ORDER = ['noinv', 'posterior', 'posteriorI',
         'diceM_2_98', 'diceM_5_95', 'diceM_10_90', 'diceM_20_80',
         'diceM_100_0', 'diceM_100_0_f2', 'diceM_100_0_f5', 'diceM_100_0_f10', 'diceM_100_0_f20',
         'diceI_5_95', 'diceI_10_90', 'diceI_20_80']


def overview(df, V, n):
    first, _ = ap.first_flip(df)
    rows = []
    for v in V:
        t = df[(df.variant == v) & (df.cond == 'tgt')]
        s = df[(df.variant == v) & (df.cond == 'src')]
        f = first[first.variant == v]
        r = {'variant': v,
             'plc_exact': s.exact.mean(),
             'plc_edits@51': s[s.tau == 51].n_total.mean(),
             'plc_search': s.groupby(['run', 'sample_idx']).flip.max().sum() / n}
        for tau in (11, 51, 100):
            tt, ss = t[t.tau == tau], s[s.tau == tau]
            r[f'FR@{tau}'] = tt.flip.mean()
            r[f'attr@{tau}'] = tt.flip.mean() - ss.flip.mean()
            r[f'edits@{tau}'] = tt.n_total.mean()
        r['srch_FR'] = len(f) / n
        r['srch_NA'] = f.ctrl_flip.mean()
        r['srch_tau'] = f.tau.median()
        r['srch_edits'] = f.n_total.mean()
        r['srch_valid'] = f.smiles_valid.mean()
        r['srch_conn'] = f.heavy_conn.mean()
        r['srch_tani'] = f.tani_src.mean()
        if 'gt_cf' in f:
            gt = (f.gt_cf == f.tgt_class)
            r['GT|flip'] = gt.mean()
            r['GT_CF_rate'] = gt.sum() / n
            r['ring_frac'] = (f.ring_touch / f.n_total.clip(lower=1)).mean()
        rows.append(r)
    return pd.DataFrame(rows)


def per_tau(df, V, taus):
    t = ap.tau_table(df)
    t = t[t.tau.isin(taus)]
    t['attr'] = t.FR - t.src_flip
    cols = ['FR', 'attr', 'src_flip', 'src_edits', 'edits_all', 'edits|flip', 'NA|flip', 'valid|flip',
            'conn|flip', 'tani|flip']
    if 'GT|flip' in t:
        cols += ['GT|flip', 'ring_frac|flip']
    return t, {c: t.pivot(index='tau', columns='variant', values=c)[[v for v in V]] for c in cols}


def paired_vs_posterior(df, V, taus=(11, 51, 100)):
    """same tau, graphs where BOTH flip: edits and Tanimoto, posterior vs each other variant"""
    tgt = df[df.cond == 'tgt']
    rows = []
    for v in V:
        if v == 'posterior':
            continue
        for tau in taus:
            A = tgt[(tgt.variant == 'posterior') & (tgt.tau == tau) & (tgt.flip == 1)].set_index(['run', 'sample_idx'])
            B = tgt[(tgt.variant == v) & (tgt.tau == tau) & (tgt.flip == 1)].set_index(['run', 'sample_idx'])
            idx = A.index.intersection(B.index)
            if len(idx) < 10:
                continue
            x, y = A.loc[idx, 'n_total'].values, B.loc[idx, 'n_total'].values
            p = wilcoxon(x, y).pvalue if np.any(x != y) else 1.0
            rows.append({'other': v, 'tau': tau, 'n_both': len(idx),
                         'edits_post': x.mean(), 'edits_other': y.mean(), 'wilcoxon_p': p,
                         'tani_post': A.loc[idx, 'tani_src'].mean(), 'tani_other': B.loc[idx, 'tani_src'].mean()})
    return pd.DataFrame(rows)


def edits_vs_tau_profile(df, V):
    """does the budget tau control the edit size? edits at tau=1 vs tau=200 and monotonicity"""
    m = ap.monotonicity(df)
    t = df[df.cond == 'tgt']
    e1 = t[t.tau == 1].groupby('variant').n_total.mean()
    e200 = t[t.tau == t.tau.max()].groupby('variant').n_total.mean()
    m['edits@tau1'] = m.variant.map(e1)
    m[f'edits@tau{t.tau.max()}'] = m.variant.map(e200)
    return m


def main():
    df = pd.concat([ap.load(p) for p in sys.argv[1:]], ignore_index=True)
    V = [v for v in ORDER if v in set(df.variant)] + sorted(set(df.variant) - set(ORDER))
    ap.V = V
    n = len(df[['run', 'sample_idx']].drop_duplicates())
    print('=' * 120)
    print(sys.argv[1:], 'graphs:', n, 'variants:', V)
    print('\n--- overview (plc_* = replay under the SOURCE class; srch_* = first flip of the 17-tau budget search) ---')
    print(overview(df, V, n).round(3).to_string(index=False))
    t, tabs = per_tau(df, V, sorted(df.tau.unique()))
    for c, tab in tabs.items():
        print(f'\n--- per tau: {c} ---')
        print(tab.round(3).to_string())
    print('\n--- matched flip rate: edits per flip where each variant reaches FR level ---')
    mf = ap.matched_fr(ap.tau_table(df), levels=(0.2, 0.3, 0.5))
    print(mf.pivot(index='variant', columns='FR_level', values='edits|flip').reindex(V).round(2).to_string())
    print('\n--- paired same tau (both flip): posterior vs other ---')
    print(paired_vs_posterior(df, V).round(4).to_string(index=False))
    print('\n--- is tau a dial for edit size? ---')
    print(edits_vs_tau_profile(df, V).round(3).to_string(index=False))


if __name__ == '__main__':
    main()
