"""R1.3 / R2.4: exhaustive minimum-edit counterfactuals vs GDCE-I (same graphs, same GCN judge).
Usage: python analyze_min_edit.py <dataset tag: mutag|benzene>"""
import sys
import numpy as np, pandas as pd
import analyze_paired as ap
ds = sys.argv[1]
m = pd.read_csv(f'results/min_edit/{ds}_gcn_s3.csv')
n = len(m)
print('=' * 100); print(ds, 'graphs', n)
print('min_k distribution (-1 = no flip with <= 2 edits):', m.min_k.value_counts().sort_index().to_dict())
print('beam upper bound for graphs without <= 2-edit flip:', m[m.min_k < 0].ub_k.value_counts().sort_index().to_dict())
k1 = m[m.min_k == 1]
print(f'graphs with a 1-edit CF: {len(k1)/n:.3f}; number of distinct 1-edit CFs per such graph: median {k1.k1_flips.median():.0f}, mean {k1.k1_flips.mean():.1f}')
print(f'  1-edit CFs by kind (share of all flipping single edits): node {k1.k1_node.sum()/k1.k1_flips.sum():.3f}  del {k1.k1_del.sum()/k1.k1_flips.sum():.3f}  add {k1.k1_add.sum()/k1.k1_flips.sum():.3f}')
for k in (1, 2):
    has = m[m[f'k{k}_flips'] > 0]
    # existence of a valid / valid-connected / transferred CF at exactly k edits (sampled check, cap 1000)
    print(f'k={k}: graphs with >=1 flip {len(has)/n:.3f} | among them: some CF valid {np.mean(has[f"k{k}_valid"] > 0):.3f}, '
          f'valid&no-new-fragment {np.mean(has[f"k{k}_validconn"] > 0):.3f}, transferred to control judge {np.mean(has[f"k{k}_ctrl"] > 0):.3f}, '
          f'valid&conn&ctrl {np.mean(has[f"k{k}_validconn_ctrl"] > 0):.3f}')
    tot = has[f'k{k}_flips'].sum(); chk = has[f'k{k}_checked'].sum()
    print(f'      share of all k={k} flips: ctrl {has[f"k{k}_ctrl"].sum()/tot:.3f}; of checked: valid {has[f"k{k}_valid"].sum()/chk:.3f}, valid&conn {has[f"k{k}_validconn"].sum()/chk:.3f}')
# smallest k <= 2 with a valid & connected (& transferred) CF
def first_k(row, key):
    for k in (1, 2):
        if row[f'k{k}_{key}'] > 0:
            return k
    return -1
m['min_k_vc'] = m.apply(lambda r: first_k(r, 'validconn'), axis=1)
m['min_k_vcc'] = m.apply(lambda r: first_k(r, 'validconn_ctrl'), axis=1)
print('smallest k<=2 with a valid & fragment-free flip:', m.min_k_vc.value_counts().sort_index().to_dict())
print('smallest k<=2 with a valid & fragment-free flip that the control judge confirms:', m.min_k_vcc.value_counts().sort_index().to_dict())

# ---- GDCE-I in the same edit space (node substitutions + edge insertions/deletions) ----
key = ['run', 'sample_idx']
def gcn_edits(d):
    return d.n_node + d.n_add + d.n_del
srcs = {'17-tau grid': f'results/{ds}_s3/chains.csv'}
if ds == 'mutag':
    srcs['paper grid (110 tau)'] = None
for label, path in srcs.items():
    if path is None:
        first = pd.read_csv('results/paper_first.csv')
    else:
        df = ap.load(path)
        first, _ = ap.first_flip(df)
    for v in ['posterior', 'noinv']:
        f = first[first.variant == v].copy()
        f['gcn_edits'] = gcn_edits(f)
        j = m.merge(f[key + ['gcn_edits', 'n_bt', 'n_total', 'greedy_nec', 'tau']], on=key, how='inner')
        jm = j[j.min_k > 0]
        print(f'[{label}] {v}: n={len(j)} GDCE edits (GCN-visible) mean {j.gcn_edits.mean():.2f} median {j.gcn_edits.median():.0f} '
              f'(+ bond-type changes {j.n_bt.mean():.2f}); exhaustive minimum {jm.min_k.mean():.2f}; '
              f'GDCE == minimum in {np.mean(jm.gcn_edits == jm.min_k):.3f}, within +1 in {np.mean(jm.gcn_edits <= jm.min_k + 1):.3f}; '
              f'greedy-necessary core {j[j.greedy_nec >= 0].greedy_nec.mean():.2f}')
m.to_csv(f'results/min_edit/{ds}_annotated.csv', index=False)
print('\nexamples of 1-edit counterfactuals:'); print(k1.k1_examples.head(8).to_string(index=False))
print('\nexamples of valid & fragment-free 1-edit CFs:'); print(k1[k1.k1_validconn > 0].k1_ex_validconn.head(8).to_string(index=False))
