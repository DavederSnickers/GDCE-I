"""Revised Table 3 (inversion ablation at FIXED budgets) as LaTeX in the style of the manuscript.

Usage: python fixedtau_table3_tex.py <out_dir> <chains.csv> [<chains.csv> ...]
Input: hydra results_review/paired_tau/fixedtau_ablation_mutag_s3_r{0..4}/chains.csv.
Writes <out_dir>/table3_fixedtau.tex (manuscript, replaces tab:ablation_inversion) and
<out_dir>/table3_fixedtau_letter.tex (response letter: no \\label, literal table references).

Conventions of Tables 2/3: per-run means over the 100 inputs of a run, then mean and sample std
over the 5 runs. FR, NAFR and NAFR & SMILES (as in Table 2) are fractions of all inputs; EdgeSp /
NodeSp / EdgeTypeSp = 1 - mean modification rate over the flips (EdgeSp clamped at 0, as in
ablation_fixed_tau.py).
NA = mean NAFR / mean FR (ratio of the run averages, as in Table 2), its std is the sample std of
the per-run ratio. Bold = column-wise best within one budget (ties at the printed precision).
"""
import os
import sys

import numpy as np
import pandas as pd

TAUS = [10, 50, 100, 200]
VARIANTS = [('noinv', 'No inversion'), ('naive', 'Naive inversion'),
            ('posterior', r'\textbf{Posterior (GDCE-I)}')]
# column -> (decimals, higher is better)
COLS = {'FR': (3, True), 'NAFR': (3, True), 'EdgeSp': (3, True), 'NodeSp': (3, True),
        'EdgeTypeSp': (3, True), 'NA': (3, True), 'NAFR&SMILES': (3, True)}

CAPTION = (r"Inversion ablation at fixed budgets $\tau$ on Mutagenicity against the shared GCN "
           r"classifier ($5\times100$, $s=3$), reported under the metrics of {TAB2}. All three "
           r"variants come from the same training run and checkpoint. For every input they are "
           r"replayed from the same reference state $\hat{{G}}^\tau$ under the same target condition "
           r"and differ only in the noise treatment. Mean and sample standard deviation over "
           r"the five runs; the column-wise best per budget is in \textbf{{bold}}.")


def per_run(df):
    rows = []
    tgt = df[df.cond == 'tgt']
    for (tau, v, run), t in tgt.groupby(['tau', 'variant', 'run']):
        f = t[t.flip == 1]
        rows.append({
            'tau': int(tau), 'variant': v, 'run': run,
            'FR': t.flip.mean(),
            'NAFR': (t.flip & t.ctrl_flip).mean(),
            'NA_run': f.ctrl_flip.mean(),
            'EdgeSp': max(1 - f.edge_rate.mean(), 0.0),
            'NodeSp': 1 - f.node_rate.mean(),
            'EdgeTypeSp': 1 - f.edgetype_rate.mean(),
            'NAFR&SMILES': (t.flip & t.ctrl_flip & t.smiles_valid).mean(),
        })
    return pd.DataFrame(rows)


def aggregate(P):
    out = {}
    for (tau, v), g in P.groupby(['tau', 'variant']):
        r = {c: (g[c].mean(), g[c].std()) for c in COLS if c != 'NA'}
        r['NA'] = (g.NAFR.mean() / g.FR.mean(), g.NA_run.std())
        out[(tau, v)] = r
    return out


def body(agg):
    lines = []
    for i, tau in enumerate(TAUS):
        best = {}
        for c, (d, up) in COLS.items():
            vals = [round(agg[(tau, v)][c][0], d) for v, _ in VARIANTS]
            best[c] = max(vals) if up else min(vals)
        for j, (v, name) in enumerate(VARIANTS):
            cells = []
            for c, (d, _) in COLS.items():
                m, s = agg[(tau, v)][c]
                num = f'{m:.{d}f}'
                num = rf'\mathbf{{{num}}}' if round(m, d) == best[c] else num
                cells.append(rf'${num}_{{\pm {s:.{d}f}}}$')
            lead = f'${tau}$' if j == 0 else ''
            lines.append(f'{lead} & {name} & ' + ' & '.join(cells) + r' \\')
        if i < len(TAUS) - 1:
            lines.append(r'\midrule')
    return '\n'.join(lines)


HEAD = r"""\setlength{\tabcolsep}{4pt}
\resizebox{\linewidth}{!}{
\begin{tabular}{cl | cc | ccc | cc}
\toprule
\multicolumn{2}{c|}{} & \multicolumn{7}{c}{\cellcolor{CadetBlue!20}\textbf{desiderata}} \\
\cmidrule(lr){3-9}
\multicolumn{2}{c|}{} & \multicolumn{2}{c|}{\cellcolor{CadetBlue!20}\textbf{sufficiency}} & \multicolumn{3}{c|}{\cellcolor{CadetBlue!20}\textbf{understandability}} & \multicolumn{2}{c}{\cellcolor{CadetBlue!20}\textbf{fidelity}} \\
$\tau$ & \textbf{Variant} & \cellcolor{CadetBlue!20}FR ($\uparrow$) & \multicolumn{1}{c|}{\cellcolor{CadetBlue!20}NAFR ($\uparrow$)} & \cellcolor{CadetBlue!20}EdgeSp. ($\uparrow$) & \cellcolor{CadetBlue!20}NodeSp. ($\uparrow$) & \multicolumn{1}{c|}{\cellcolor{CadetBlue!20}EdgeTypeSp. ($\uparrow$)} & \cellcolor{CadetBlue!20}NA ($\uparrow$) & \cellcolor{CadetBlue!20}NAFR\,\&\,SMILES ($\uparrow$) \\
\midrule
"""
FOOT = r"""
\bottomrule
\end{tabular}}"""


def main():
    out_dir, paths = sys.argv[1], sys.argv[2:]
    df = pd.concat([pd.read_csv(p, keep_default_na=False, na_values=['']) for p in paths],
                   ignore_index=True)
    df = df[df.tau.isin(TAUS)]
    assert sorted(df.run.unique()) == [0, 1, 2, 3, 4], df.run.unique()
    agg = aggregate(per_run(df))
    table = HEAD + body(agg) + FOOT

    paper = ("\\begin{table}[H]\n\\centering\n\\small\n" + table + "\n\\caption{"
             + CAPTION.format(TAB2=r"Table~\ref{tab:quantitative_results_main}")
             + "}\n\\label{tab:ablation_inversion}\n\\end{table}\n")
    letter = ("\\begin{table}[H]\n\\centering\n\\small\n\\parbox{\\linewidth}{\\textbf{Revised Table 3.} "
              + CAPTION.format(TAB2="Table~2") + "}\n\\par\\medskip\n" + table + "\n\\end{table}\n")
    for name, text in [('table3_fixedtau.tex', paper), ('table3_fixedtau_letter.tex', letter)]:
        with open(os.path.join(out_dir, name), 'w') as fh:
            fh.write('% generated by reviews/analysis/fixedtau_table3_tex.py -- do not edit by hand\n' + text)
    print(paper)
    for tau in TAUS:
        print(tau, {v: f"NA {agg[(tau, v)]['NA'][0]:.3f}" for v, _ in VARIANTS})


if __name__ == '__main__':
    main()
