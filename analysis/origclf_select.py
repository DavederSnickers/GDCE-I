"""Original-classifier ablation (R1.4): pick each baseline's tuned configuration on the VALIDATION
split (1 run x 100 graphs) and write origclf_tuned.env for submit_origclf_baselines.sh PHASE=test.

Rule agreed with David (2026-10-27): highest validation FR; every configuration within 0.02 of it
counts as a tie, and among the ties the sparsest one wins: the lowest mean modification rate over
the counted flips. For UCExplainer the harness flag is not used once the re-scored dumps exist
(<ROOT>/ucx_rescored/): its 'projected' check classifies a DIRECTED graph (one edge-mask parameter per
directed edge), while the dump and the paper protocol use the undirected molecule. The harness
columns edge_/node_/edgetype_sparsity are modification RATES (GDCE's
compute_sparsity; the table reports 1 - rate), averaged over the columns a harness has. FR is
the flip rate under the attacked classifier; for UCExplainer the flip has to survive the projection
to one-hot features (projected_success), as in the paper's protocol.

Usage: python origclf_select.py <dir with c2/ cf2/ ucx/ d4/ xpl/ copies of results/origclf/val> <out.env>
"""
import glob
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # origclf_table, also under python -I
ROOT, OUT = sys.argv[1], sys.argv[2]
TOL = 0.02
RATES = ['edge_sparsity', 'node_sparsity', 'edgetype_sparsity']   # modification rates despite the name


def score(csv, success_col):
    d = pd.read_csv(csv)
    flips = d[d[success_col].astype(float) == 1]
    cols = [c for c in RATES if c in d.columns and flips[c].notna().any()]
    rate = flips[cols].astype(float).mean(axis=1).mean() if len(flips) else float('nan')
    return {'n': len(d), 'FR': d[success_col].astype(float).mean(), 'mod_rate': rate,
            'rate_cols': '+'.join(c.split('_')[0] for c in cols)}


def rescored(method):
    """FR and modification rate from the dumps re-scored by adversarial_flip.py (<ROOT>/<method>_rescored/),
    i.e. the paper protocol: the attacked classifier must flip on the dumped, undirected graph."""
    folder = os.path.join(ROOT, f'{method}_rescored')
    if not os.path.isdir(folder):
        return None
    import origclf_table as ot
    sources = ot.load_sources('val')
    out = {}
    for csv in sorted(glob.glob(os.path.join(folder, '*.csv'))):
        tag = os.path.basename(csv)[:-4]
        if tag == 'summary':
            continue
        per, _ = ot.evaluate(csv, sources, n_runs=1)
        rate = 1 - per[['EdgeSp', 'NodeSp', 'EdgeTypeSp']].iloc[0].mean()
        out[tag] = {'FR': float(per.FR.iloc[0]), 'mod_rate': float(rate), 'rate_cols': 'rescored dump'}
    return out


def table(method, pattern, success_col, parse):
    rows = []
    rs = rescored(method)
    for csv in sorted(glob.glob(os.path.join(ROOT, method, pattern))):
        tag = csv.split(os.sep)[len(os.path.join(ROOT, method).split(os.sep))]
        if tag.endswith('prep'):
            continue
        row = {'tag': tag, **parse(tag), **score(csv, success_col)}
        if rs is not None:
            row['harness_FR'] = row['FR']
            row.update(rs[tag])
        rows.append(row)
    t = pd.DataFrame(rows)
    if t.empty:
        return t, None
    best = t.FR.max()
    t['tie'] = t.FR >= best - TOL
    pick = t[t.tie].sort_values(['mod_rate', 'FR'], ascending=[True, False]).iloc[0]
    t['chosen'] = t.tag == pick.tag
    return t.sort_values(['FR', 'mod_rate'], ascending=[False, True]), pick


def c2_parse(tag):
    m = re.match(r'c2_sm(\w+?)_b([\d.]+)_lr([\d.]+)$', tag)
    return {'subgraph_mode': m.group(1), 'beta': m.group(2), 'lr': m.group(3)}


def cf2_parse(tag):
    m = re.match(r'cf2_a([\d.]+)_l([\d.]+)$', tag)
    return {'alp': m.group(1), 'lam': m.group(2)}


def ucx_parse(tag):
    m = re.match(r'ucx_lr([\d.]+)_(\w+)$', tag)
    return {'lr': m.group(1), 'policy': m.group(2)}


def d4_parse(tag):
    m = re.match(r'd4_a(\d+)_(flip|modrate)$', tag)
    return {'alpha_label': m.group(1), 'checkpoint': m.group(2)}


def xpl_parse(tag):
    m = re.match(r'xpl_a([\d.]+)_b([\d.]+)$', tag)
    return {'alpha': m.group(1), 'beta': m.group(2)}


# configurations the baselines ship with; if one of them wins, no separate tuned run is needed
DEFAULTS = {'c2': 'c2_smfalse_b1.0_lr0.1', 'cf2': 'cf2_a0.6_l1000', 'ucx': 'ucx_lr0.1_linear',
            'd4': 'd4_a0001_flip', 'xpl': 'xpl_a1.0_b0.5'}
env = []
pd.set_option('display.width', 200)
for method, pattern, col, parse in [
        ('c2', '*/shared_judge/c2_results.csv', 'success', c2_parse),
        ('cf2', '*/inversion_results.csv', 'success', cf2_parse),
        ('ucx', '*/inversion_results.csv', 'projected_success', ucx_parse),
        ('d4', '*/inversion_results.csv', 'success', d4_parse),
        ('xpl', '*/xplore_results.csv', 'success', xpl_parse)]:
    t, pick = table(method, pattern, col, parse)
    print(f'\n=== {method}: {len(t)} configurations on the validation split')
    if pick is None:
        continue
    print(t.round(3).to_string(index=False))
    print(f'--> chosen: {pick.tag}')
    if pick.tag == DEFAULTS[method]:
        print('    (= the shipped default, no separate tuned test run)')
        env.append({'c2': 'C2_TUNED=""', 'cf2': 'CF2_TUNED=""', 'ucx': 'UCX_TUNED=""',
                    'd4': 'D4_TUNED_ALPHA=""\nD4_TUNED_KIND=""', 'xpl': 'XPL_TUNED=""'}[method])
    elif method == 'c2':
        env.append(f'C2_TUNED="subgraph_mode={pick.subgraph_mode} beta={pick.beta} lr={pick.lr}"')
    elif method == 'cf2':
        env.append(f'CF2_TUNED="--alp {pick.alp} --lam {pick.lam}"')
    elif method == 'ucx':
        env.append(f'UCX_TUNED="--override optimizer.lr={pick.lr} --override scheduler.policy={pick.policy}"')
    elif method == 'xpl':
        env.append(f'XPL_TUNED="alpha={pick.alpha} beta={pick.beta}"')
    else:
        alpha = {'0001': '0.001', '001': '0.01', '01': '0.1'}[pick.alpha_label]
        env.append(f'D4_TUNED_ALPHA={alpha}')
        env.append(f'D4_TUNED_KIND={pick.checkpoint}')
with open(OUT, 'w') as f:
    f.write('# written by reviews/analysis/origclf_select.py (validation split, FR then sparsity)\n')
    f.write('\n'.join(env) + '\n')
print(f'\nwrote {OUT}:\n' + '\n'.join(env))
