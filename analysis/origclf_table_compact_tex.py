"""Compact LaTeX version of the original-classifier table (R1.4, David 2026-11-02): per baseline one
row with its tuned run against its own classifier and one row with GDCE-I distilled from the same
classifier; every value is followed, in parentheses, by the same method under the shared protocol.

Same metric implementation as origclf_table.py / origclf_table_tex.py. Means over the five runs;
the largest sample standard deviation of all shown values is printed for the caption.

Usage: python -I origclf_table_compact_tex.py <origclf advflip dir> <Table-2 advflip dir> > table.tex
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import origclf_table as ot  # noqa: E402

# blocks of (method cell, label of the run on the original classifier, label of the shared-protocol
# run); original-classifier dumps live in the origclf folder, shared ones in the Table-2 folder
BLOCKS = [
    [('C2Explainer', 'C2_tuned', 'C2_mutag_gcn'), ('GDCE-I', 'GDCE_c2gcn', 'GDCE_mutag_gcn')],
    [('$CF^2$', 'CF2_tuned', 'CF2_mutag_gcn'), ('GDCE-I', 'GDCE_cf2gcn', 'GDCE_mutag_gcn')],
    [('XPlore', 'XPL_tuned', 'XPL_mutag_gcn'), ('GDCE-I', 'GDCE_xplgcn', 'GDCE_mutag_gcn')],
    [(r'D4Explainer$^{\ddagger}$', 'D4_june', 'D4_mutag_flip'),
     ('GDCE-I', 'GDCE_d4native', 'GDCE_mutag_gcn')],
    [(r'UCExplainer$^{\S}$', 'UCX_tuned', 'UCX_mutag_gine'),
     (r'GDCE-I$^{\S}$', 'GDCE_ucxgine', 'GDCE_mutag_gine')],
]
EDGE_ONLY = ('C2', 'CF2', 'D4', 'XPL')
COLS = ['FR', 'NAFR', 'EdgeSp', 'NodeSp', 'EdgeTypeSp', 'NA', 'SMILES']


def values(path, label, sources, stds):
    """Mean per column ('--' where the method cannot edit that part), or None if the run is missing."""
    if not os.path.exists(path):
        return None
    per, _ = ot.evaluate(path, sources)
    out = {}
    for c in COLS:
        if c in ('NodeSp', 'EdgeTypeSp') and label.startswith(EDGE_ONLY):
            out[c] = '--'
        elif c == 'NA':
            out[c] = f'{per.NAFR.mean() / per.FR.mean():.3f}'
        elif c == 'EdgeSp' and per.EdgeRate.mean() > 1:      # clamped at 0, as in Table 2
            out[c] = r'0.000^{\dagger}'
        else:
            out[c] = f'{per[c].mean():.3f}'
            stds.append(per[c].std(ddof=1))
    return out


def cell(own, shared):
    """Value on the original classifier, followed by the shared-protocol value (macro \\cs)."""
    if own == '--' and shared == '--':
        return '--'
    o = r'\textit{tbd}' if own is None else f'${own}$'
    return rf'\cs{{{o}}}{{${shared}$}}'


def main(origclf, table2):
    sources = ot.load_sources()
    stds = []
    lines = [r'\begin{table}[t]', r'\centering', r'\small', r'\setlength{\tabcolsep}{4pt}',
             r'\newcommand{\cs}[2]{#1\,{\scriptsize(#2)}}',
             r'\resizebox{\linewidth}{!}{', r'\begin{tabular}{l | c c | c c c | c c}', r'\toprule',
             r'\textbf{Method} & FR ($\uparrow$) & NAFR ($\uparrow$) & EdgeSp. ($\uparrow$) & '
             r'NodeSp. ($\uparrow$) & EdgeTypeSp. ($\uparrow$) & NA ($\uparrow$) & '
             r'NAFR\,\&\,SMILES ($\uparrow$) \\']
    for block in BLOCKS:
        lines.append(r'\midrule')
        for method, own, shared in block:
            o = values(os.path.join(origclf, f'{own}.csv'), own, sources, stds)
            s = values(os.path.join(table2, f'{shared}.csv'), shared, sources, stds)
            cells = [cell(None if o is None else o[c], s[c]) for c in COLS]
            lines.append(' & '.join([method] + cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}}', r'\end{table}']
    print('\n'.join(lines))
    print(f'% largest sample standard deviation of the shown means: {np.nanmax(stds):.3f}')


if __name__ == '__main__':
    main(*sys.argv[1:3])
