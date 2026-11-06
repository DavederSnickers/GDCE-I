"""Analysis of review_paired_tau.py output (chains.csv).

Usage: python analyze_paired.py <chains.csv> [<chains.csv> ...]
"""
import sys
import numpy as np
import pandas as pd

pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 60)
V = ['noinv', 'naive', 'posterior']


def load(path):
    df = pd.read_csv(path, keep_default_na=False, na_values=[''])
    for c in ['tani_src', 'tani_other_max', 'shared_with_src', 'greedy_k', 'greedy_nec',
              'greedy_nec_heavy', 'greedy_nec_motif']:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df['heavy_conn'] = (df.n_heavy_comp <= df.n_heavy_comp_src).astype(int)
    df['valid_conn'] = ((df.smiles_valid == 1) & (df.heavy_conn == 1)).astype(int)
    return df


def per_run_mean(d, col, by):
    """mean over the 5 runs of the per-run mean (paper convention), plus std over runs"""
    g = d.groupby(by + ['run'])[col].mean().groupby(by)
    return g.mean(), g.std(ddof=0)


def tau_table(df):
    tgt = df[df.cond == 'tgt'].copy()
    src = df[df.cond == 'src'].copy()
    out = []
    for (v, tau), d in tgt.groupby(['variant', 'tau']):
        s = src[(src.variant == v) & (src.tau == tau)]
        fl = d[d.flip == 1]
        gr = fl[fl.greedy_k > 0]
        row = {
            'variant': v, 'tau': tau, 'n': len(d),
            'FR': d.flip.mean(),
            'NAFR': (d.flip & d.ctrl_flip).mean(),
            'NA|flip': fl.ctrl_flip.mean() if len(fl) else np.nan,
            'edits_all': d.n_total.mean(),
            'edits|flip': fl.n_total.mean() if len(fl) else np.nan,
            'heavy|flip': fl.n_total_heavy.mean() if len(fl) else np.nan,
            'spont|flip': fl.n_spont.mean() if len(fl) else np.nan,
            'shared_src|flip': fl.shared_with_src.mean() if len(fl) else np.nan,
            'nec|flip': gr.greedy_nec.mean() if len(gr) else np.nan,
            'redund|flip': (1 - gr.greedy_nec / gr.greedy_k).mean() if len(gr) else np.nan,
            'valid|flip': fl.smiles_valid.mean() if len(fl) else np.nan,
            'conn|flip': fl.heavy_conn.mean() if len(fl) else np.nan,
            'FR&valid': (d.flip & d.smiles_valid).mean(),
            'FR&valid&conn': (d.flip & d.valid_conn).mean(),
            'NAFR&valid&conn': (d.flip & d.ctrl_flip & d.valid_conn).mean(),
            'tani|flip': fl.tani_src.mean() if len(fl) else np.nan,
            'src_edits': s.n_total.mean(),
            'src_exact': s.exact.mean(),
            'src_flip': s.flip.mean(),
            'corrupt': d.n_corrupt.mean(),
        }
        if 'gt_cf' in d:
            row['GTflip'] = (d.flip & (d.gt_cf == d.tgt_class)).mean()
            row['GT|flip'] = (fl.gt_cf == fl.tgt_class).mean() if len(fl) else np.nan
            row['ring_frac|flip'] = (fl.ring_touch / fl.n_total.clip(lower=1)).mean() if len(fl) else np.nan
            row['src_GTflip'] = (s.gt_cf != s.gt_src).mean()
        if 'nitro_src' in d:
            nd = d[(d.nitro_src == 1) & (d.src_class == 0)]
            nfl = nd[nd.flip == 1]
            row['nitro_n'] = len(nd)
            row['nitro_FR'] = nd.flip.mean() if len(nd) else np.nan
            row['nitro_removed|flip'] = (nfl.nitro_cf == 0).mean() if len(nfl) else np.nan
            row['nitro_touch|flip'] = (nfl.nitro_touch > 0).mean() if len(nfl) else np.nan
        out.append(row)
    t = pd.DataFrame(out)
    t['variant'] = pd.Categorical(t.variant, V)
    return t.sort_values(['tau', 'variant'])


def first_flip(df):
    """grid budget search: per (run, sample, variant) the smallest grid tau that flips"""
    tgt = df[df.cond == 'tgt']
    fl = tgt[tgt.flip == 1].sort_values('tau')
    first = fl.groupby(['run', 'sample_idx', 'variant']).head(1)
    keys = tgt[['run', 'sample_idx']].drop_duplicates()
    return first, len(keys)


def summarize_first(first, n_total):
    rows = []
    for v in V:
        f = first[first.variant == v]
        gr = f[f.greedy_k > 0]
        r = {'variant': v, 'FR': len(f) / n_total,
             'NAFR': f.ctrl_flip.sum() / n_total,
             'NA|flip': f.ctrl_flip.mean(),
             'median_tau': f.tau.median(),
             'edits': f.n_total.mean(), 'heavy': f.n_total_heavy.mean(),
             'edge_sp': 1 - f.edge_rate.mean(), 'node_sp': 1 - f.node_rate.mean(),
             'et_sp': 1 - f.edgetype_rate.mean(),
             'nec': gr.greedy_nec.mean(), 'redund': (1 - gr.greedy_nec / gr.greedy_k).mean(),
             'shared_src': f.shared_with_src.mean(),
             'valid': f.smiles_valid.mean(), 'conn': f.heavy_conn.mean(),
             'FR&valid&conn': f.valid_conn.sum() / n_total,
             'tani': f.tani_src.mean()}
        if 'gt_cf' in f:
            r['GT|flip'] = (f.gt_cf == f.tgt_class).mean()
            r['ring_frac'] = (f.ring_touch / f.n_total.clip(lower=1)).mean()
        rows.append(r)
    return pd.DataFrame(rows)


def matched_fr(t, levels=(0.3, 0.5, 0.7, 0.8, 0.9)):
    """edits of all chains (successful or not) at the tau where each variant reaches FR level"""
    rows = []
    for v in V:
        d = t[t.variant == v].sort_values('tau')
        fr = d.FR.values
        for L in levels:
            if fr.max() < L:
                rows.append({'variant': v, 'FR_level': L, 'tau*': np.nan})
                continue
            # first crossing, linear interpolation in tau-index space
            k = int(np.argmax(fr >= L))
            if k == 0:
                w = 0.0
                k0 = 0
            else:
                k0 = k - 1
                w = (L - fr[k0]) / (fr[k] - fr[k0]) if fr[k] != fr[k0] else 1.0
            def interp(col):
                a, b = d[col].values[k0], d[col].values[k]
                return a + w * (b - a)
            rows.append({'variant': v, 'FR_level': L, 'tau*': interp('tau'),
                         'edits|flip': interp('edits|flip'), 'heavy|flip': interp('heavy|flip'),
                         'nec|flip': interp('nec|flip'), 'NA|flip': interp('NA|flip'),
                         'tani|flip': interp('tani|flip'), 'valid|flip': interp('valid|flip')})
    return pd.DataFrame(rows)


def paired_same_tau(df, a='posterior', b='noinv', taus=(11, 20, 30, 51, 75, 100)):
    """on graphs where BOTH variants flip at the same tau: paired differences"""
    tgt = df[df.cond == 'tgt']
    rows = []
    rng = np.random.default_rng(0)
    for tau in taus:
        A = tgt[(tgt.variant == a) & (tgt.tau == tau) & (tgt.flip == 1)].set_index(['run', 'sample_idx'])
        B = tgt[(tgt.variant == b) & (tgt.tau == tau) & (tgt.flip == 1)].set_index(['run', 'sample_idx'])
        idx = A.index.intersection(B.index)
        if len(idx) < 5:
            continue
        r = {'tau': tau, 'n_both': len(idx)}
        for col in ['n_total', 'n_total_heavy', 'greedy_nec', 'tani_src', 'smiles_valid', 'heavy_conn', 'ctrl_flip']:
            x = A.loc[idx, col].astype(float).values
            y = B.loc[idx, col].astype(float).values
            m = ~(np.isnan(x) | np.isnan(y))
            if col == 'greedy_nec':
                m &= (x >= 0) & (y >= 0)
            dlt = x[m] - y[m]
            boots = [rng.choice(dlt, len(dlt)).mean() for _ in range(2000)] if len(dlt) else [np.nan]
            r[f'{col}:{a}'] = x[m].mean()
            r[f'{col}:{b}'] = y[m].mean()
            r[f'{col}:CI'] = f'[{np.percentile(boots, 2.5):.2f},{np.percentile(boots, 97.5):.2f}]'
        rows.append(r)
    return pd.DataFrame(rows)


def budget_curves(df, ks=(1, 2, 3, 5, 8, 10, 15, 20)):
    """P(exists grid tau with flip and <= k edits) per variant (best over the grid)"""
    tgt = df[df.cond == 'tgt']
    n = len(tgt[['run', 'sample_idx']].drop_duplicates())
    fl = tgt[tgt.flip == 1]
    best = fl.groupby(['run', 'sample_idx', 'variant']).n_total.min().reset_index()
    best_h = fl.groupby(['run', 'sample_idx', 'variant']).n_total_heavy.min().reset_index()
    nafl = fl[fl.ctrl_flip == 1].groupby(['run', 'sample_idx', 'variant']).n_total.min().reset_index()
    rows = []
    for v in V:
        r = {'variant': v}
        for k in ks:
            r[f'<= {k}'] = (best[best.variant == v].n_total <= k).sum() / n
        rows.append(r)
    rows2 = []
    for v in V:
        r = {'variant': v}
        for k in ks:
            r[f'<= {k}'] = (nafl[nafl.variant == v].n_total <= k).sum() / n
        rows2.append(r)
    return pd.DataFrame(rows), pd.DataFrame(rows2)


def monotonicity(df):
    """Is tau a faithful dial for edit size?  Per (graph, variant) over the tau grid:
    Spearman(tau, edits), share of grid steps where edits DROP when tau grows, and whether
    the first flip on the grid is also the minimum-edit flip on the grid."""
    from scipy.stats import spearmanr
    tgt = df[df.cond == 'tgt']
    rows = []
    for v in V:
        d = tgt[tgt.variant == v]
        rhos, drops, first_is_min, excess, zero_at_1 = [], [], [], [], []
        for _, g in d.groupby(['run', 'sample_idx']):
            g = g.sort_values('tau')
            e = g.n_total.values
            if np.ptp(e) > 0:
                rhos.append(spearmanr(g.tau.values, e).correlation)
            drops.append((np.diff(e) < 0).mean())
            zero_at_1.append(int(e[0] == 0))
            f = g[g.flip == 1]
            if len(f):
                first_is_min.append(int(f.n_total.iloc[0] == f.n_total.min()))
                excess.append(f.n_total.iloc[0] - f.n_total.min())
        rows.append({'variant': v, 'spearman(tau,edits)': np.nanmean(rhos),
                     'share_steps_edits_drop': np.mean(drops),
                     'edits=0 at tau=1': np.mean(zero_at_1),
                     'first_flip_is_min_edit': np.mean(first_is_min),
                     'excess_edits_first_vs_min': np.mean(excess)})
    return pd.DataFrame(rows)


def cumulative_vs_marginal(t, first, n_total):
    """budget search with independent draws: cumulative first-flip rate vs per-tau FR"""
    rows = []
    for v in V:
        f = first[first.variant == v]
        for tau in sorted(t.tau.unique()):
            rows.append({'variant': v, 'tau': tau,
                         'FR(tau)': float(t[(t.variant == v) & (t.tau == tau)].FR.iloc[0]),
                         'P(first flip <= tau)': (f.tau <= tau).sum() / n_total})
    r = pd.DataFrame(rows)
    return r.pivot(index='tau', columns='variant')


def placebo(df):
    """Target vs placebo (replay under the SOURCE class) at every tau, and for the grid search."""
    rows = []
    for v in V:
        for tau in sorted(df.tau.unique()):
            a = df[(df.variant == v) & (df.tau == tau) & (df.cond == 'tgt')]
            b = df[(df.variant == v) & (df.tau == tau) & (df.cond == 'src')]
            rows.append({'variant': v, 'tau': tau, 'FR_tgt': a.flip.mean(), 'FR_placebo': b.flip.mean(),
                         'attributable': a.flip.mean() - b.flip.mean(),
                         'NAFR_tgt': (a.flip & a.ctrl_flip).mean(),
                         'NAFR_placebo': (b.flip & b.ctrl_flip).mean(),
                         'edits_tgt': a.n_total.mean(), 'edits_placebo': b.n_total.mean()})
    per_tau = pd.DataFrame(rows)
    keys = df[['run', 'sample_idx']].drop_duplicates()
    n = len(keys)
    srch = []
    for v in V:
        t = df[(df.variant == v) & (df.cond == 'tgt')]
        s = df[(df.variant == v) & (df.cond == 'src')]
        srch.append({'variant': v,
                     'search_success_tgt': t.groupby(['run', 'sample_idx']).flip.max().sum() / n,
                     'search_success_placebo': s.groupby(['run', 'sample_idx']).flip.max().sum() / n,
                     'search_success_placebo_NA': (s.flip & s.ctrl_flip).groupby([s.run, s.sample_idx]).max().sum() / n})
    return per_tau, pd.DataFrame(srch)


def mcnemar(x, y):
    """exact two-sided McNemar on paired binary outcomes"""
    from scipy.stats import binomtest
    b = int(((x == 1) & (y == 0)).sum())
    c = int(((x == 0) & (y == 1)).sum())
    p = binomtest(b, b + c, 0.5).pvalue if b + c > 0 else 1.0
    return b, c, p


def grid_thinning(df, steps=(1, 2, 5, 10, 25)):
    """Budget search on thinned grids: keep every k-th tau of 1..100 (plus 110..200).
    For independent draws (noinv) fewer attempts should cost edits; for a nested,
    monotone dial (posterior) thinning should mostly shift tau, not edit size."""
    tgt = df[df.cond == 'tgt']
    all_taus = sorted(tgt.tau.unique())
    n = len(tgt[['run', 'sample_idx']].drop_duplicates())
    rows = []
    for k in steps:
        keep = set([t for t in all_taus if t <= 100 and (t - 1) % k == 0] + [t for t in all_taus if t > 100])
        d = tgt[tgt.tau.isin(keep) & (tgt.flip == 1)].sort_values('tau')
        first = d.groupby(['run', 'sample_idx', 'variant']).head(1)
        for v in V:
            f = first[first.variant == v]
            rows.append({'every_k': k, 'n_taus': len(keep), 'variant': v, 'FR': len(f) / n,
                         'NA|flip': f.ctrl_flip.mean(), 'median_tau': f.tau.median(),
                         'edits': f.n_total.mean(), 'heavy': f.n_total_heavy.mean(),
                         'edge_sp': 1 - f.edge_rate.mean(), 'tani': f.tani_src.mean()})
    return pd.DataFrame(rows)


def main():
    for path in sys.argv[1:]:
        df = load(path)
        print('=' * 100)
        print(path, df.shape, 'graphs:', len(df[['run', 'sample_idx']].drop_duplicates()))
        t = tau_table(df)
        cols1 = ['variant', 'tau', 'FR', 'NAFR', 'NA|flip', 'edits_all', 'edits|flip', 'heavy|flip',
                 'spont|flip', 'shared_src|flip', 'nec|flip', 'redund|flip', 'src_edits', 'src_exact', 'src_flip', 'corrupt']
        print('\n--- per tau (target replay; src_* = replay under the source class) ---')
        print(t[cols1].round(3).to_string(index=False))
        cols2 = ['variant', 'tau', 'valid|flip', 'conn|flip', 'FR&valid', 'FR&valid&conn', 'NAFR&valid&conn', 'tani|flip']
        extra = [c for c in ['GTflip', 'GT|flip', 'ring_frac|flip', 'src_GTflip', 'nitro_n', 'nitro_FR',
                             'nitro_removed|flip', 'nitro_touch|flip'] if c in t]
        print(t[cols2 + extra].round(3).to_string(index=False))
        first, n = first_flip(df)
        print('\n--- grid budget search (first flip on the grid), n =', n)
        print(summarize_first(first, n).round(3).to_string(index=False))
        print('\n--- matched flip rate (interpolated along tau) ---')
        print(matched_fr(t).round(3).to_string(index=False))
        print('\n--- paired, same tau, graphs where both flip: posterior vs noinv ---')
        print(paired_same_tau(df).round(3).to_string(index=False))
        print('\n--- paired, same tau: posterior vs naive ---')
        print(paired_same_tau(df, 'posterior', 'naive').round(3).to_string(index=False))
        pt, ps = placebo(df)
        print('\n--- target vs placebo (replay under the source class) ---')
        print(pt.round(3).to_string(index=False))
        print(ps.round(3).to_string(index=False))
        if df.tau.nunique() > 50:
            print('\n--- budget search on thinned grids (lottery effect of independent draws) ---')
            print(grid_thinning(df).round(3).to_string(index=False))
        print('\n--- is tau a faithful dial for edit size? ---')
        print(monotonicity(df).round(3).to_string(index=False))
        print('\n--- per-tau flip rate vs cumulative first-flip rate of the grid search ---')
        print(cumulative_vs_marginal(t, first, n).round(3).to_string())
        b1, b2 = budget_curves(df)
        print('\n--- P(flip with <= k aligned edits at some grid tau) ---')
        print(b1.round(3).to_string(index=False))
        print('--- same, flip confirmed by the control judge ---')
        print(b2.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
