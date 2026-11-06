"""LaTeX table of the original-classifier ablation (R1.4) for the Supplementary Material.

Same metric implementation as origclf_table.py (one evaluation of the re-scored dumps for every
row); the shared-classifier rows are the Table-2 runs re-scored with the same code. Columns and
conventions as Table 2: FR, NAFR and NAFR & SMILES over all inputs, sparsities over the flips,
NA = mean NAFR / mean FR; '--' where a method cannot edit node or bond types.

Usage: python -I origclf_table_tex.py <origclf advflip dir> <Table-2 advflip dir> > table.tex
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import origclf_table as ot  # noqa: E402

# (dump label, folder index, classifier cell, method cell); folder 0 = original classifiers,
# folder 1 = Table-2 runs on the shared classifiers
BLOCKS = [
    [('C2_mutag_gcn', 1, 'Shared GCN', 'C2Explainer'),
     ('C2_default', 0, "C2Explainer's GCN", 'C2Explainer (default)'),
     ('C2_tuned', 0, '', 'C2Explainer (tuned)'),
     ('GDCE_c2gcn', 0, '', 'GDCE-I')],
    [('CF2_mutag_gcn', 1, 'Shared GCN', '$CF^2$'),
     ('CF2_default', 0, "$CF^2$'s GCN", '$CF^2$ (default)'),
     ('CF2_tuned', 0, '', '$CF^2$ (tuned)'),
     ('GDCE_cf2gcn', 0, '', 'GDCE-I')],
    [('XPL_mutag_gcn', 1, 'Shared GCN', 'XPlore'),
     ('XPL_default', 0, "XPlore's GCN", 'XPlore (default)'),
     ('XPL_tuned', 0, '', 'XPlore (tuned)'),
     ('GDCE_xplgcn', 0, '', 'GDCE-I')],
    [('D4_mutag_flip', 1, 'Shared GCN', 'D4Explainer'),
     ('D4_june', 0, "D4Explainer's GCN", r'D4Explainer ($\alpha_{cf} = 0.5$)'),
     ('GDCE_d4native', 0, '', 'GDCE-I')],
    [('UCX_mutag_gine', 1, 'Shared GINE', 'UCExplainer'),
     ('UCX_default', 0, "UCExplainer's GINE", 'UCExplainer (default)'),
     ('UCX_tuned', 0, '', 'UCExplainer (tuned)'),
     ('GDCE_ucxgine', 0, '', 'GDCE-I')],
    [('GDCE_mutag_gcn', 1, 'Shared GCN', 'GDCE-I'),
     ('GDCE_mutag_gine', 1, 'Shared GINE', 'GDCE-I')],
]
EDGE_ONLY = ('C2', 'CF2', 'D4', 'XPL')


def cell(per, col):
    m, s = per[col].mean(), per[col].std(ddof=1)
    return f'${m:.3f}_{{\\pm {s:.3f}}}$'


def row(label, folder, sources):
    per, info = ot.evaluate(os.path.join(folder, f'{label}.csv'), sources)
    edge_only = label.startswith(EDGE_ONLY)
    edge = cell(per, 'EdgeSp')
    if per.EdgeRate.mean() > 1:            # clamped at 0, as in Table 2
        edge = r'$0.000^{\dagger}$'
    na = per.NAFR.mean() / per.FR.mean()
    return [cell(per, 'FR'), cell(per, 'NAFR'), edge,
            '--' if edge_only else cell(per, 'NodeSp'),
            '--' if edge_only else cell(per, 'EdgeTypeSp'),
            f'${na:.3f}$', cell(per, 'SMILES')]


def main(folders):
    sources = ot.load_sources()
    lines = [r'\begin{table}[H]', r'\centering', r'\small', r'\setlength{\tabcolsep}{4pt}',
             r'\resizebox{\linewidth}{!}{', r'\begin{tabular}{l l | c c | c c c | c c}', r'\toprule',
             r'\textbf{Classifier} & \textbf{Method} & FR ($\uparrow$) & NAFR ($\uparrow$) & '
             r'EdgeSp. ($\uparrow$) & NodeSp. ($\uparrow$) & EdgeTypeSp. ($\uparrow$) & NA ($\uparrow$) & '
             r'NAFR\,\&\,SMILES ($\uparrow$) \\']
    for block in BLOCKS:
        lines.append(r'\midrule')
        for label, k, clf, method in block:
            if not os.path.exists(os.path.join(folders[k], f'{label}.csv')):   # run still going
                lines.append(' & '.join([clf, method, r'\multicolumn{7}{c}{\textit{pending}}']) + r' \\')
                continue
            vals = row(label, folders[k], sources)
            lines.append(' & '.join([clf, method] + vals) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}}', r'\end{table}']
    print('\n'.join(lines))


if __name__ == '__main__':
    main(sys.argv[1:3])
