"""R1.5: what does a GDCE-I counterfactual do to the aromatic nitro group?
Paired-tau Mutagenicity run (17-tau grid), first flip of each variant, graphs whose source
carries the motif and is judged mutagenic (class 0 in mutagenicity_mol)."""
import gzip, json, sys
import numpy as np, pandas as pd
sys.path.insert(0, './src')
import review_metrics as rm
import analyze_paired as ap
from rdkit import RDLogger; RDLogger.DisableLog('rdApp.*')
C, O, N, H = 0, 1, 4, 3
dec = json.load(open('results/mutag_s3/meta.json'))['atom_decoder']

df = ap.load('results/mutag_s3/chains.csv')
first, n = ap.first_flip(df)
first = first[(first.nitro_src == 1) & (first.src_class == 0)]
want = {(r.run, r.sample_idx, r.variant, r.tau) for r in first.itertuples()}
graphs, src = {}, {}
with gzip.open('results/mutag_s3/graphs.jsonl.gz', 'rt') as f:
    for line in f:
        d = json.loads(line)
        key = (d['run'], d['sample_idx'], d['variant'], d['tau'])
        if d['cond'] == 'tgt' and key in want:
            graphs[key] = d
        if d['variant'] == 'posterior' and d['cond'] == 'src' and d['tau'] == 1:
            src[(d['run'], d['sample_idx'])] = d

def dense(d):
    nt = np.array(d['node_types']); k = len(nt); et = np.zeros((k, k), dtype=int)
    for a, b, t in d['edges']:
        et[a, b] = et[b, a] = t
    return nt, et

def fate(nt0, et0, nt1, et1, grp):
    """classify what happened to one nitro group {N, O, O, C}"""
    n_at = [a for a in grp if nt0[a] == N][0]
    os_ = [a for a in grp if nt0[a] == O]
    cs = [a for a in grp if nt0[a] == C]
    if nt1[n_at] != N:
        return f'N -> {dec[str(nt1[n_at])]}'
    o_ok = [o for o in os_ if nt1[o] == O and et1[n_at, o] > 0]
    if len(o_ok) < 2:
        return 'O removed/changed (no longer NO2)'
    if all(et1[n_at, c] > 0 and nt1[c] == C for c in cs):
        # group intact; did the aryl carbon / ring change?
        return 'NO2 intact'
    heavy_nb = [b for b in np.nonzero(et1[n_at])[0] if nt1[b] != H and b not in os_]
    if not heavy_nb:
        return 'NO2 detached (fragment)'
    return 'NO2 re-attached elsewhere'

rows = []
for r in first.itertuples():
    key = (r.run, r.sample_idx, r.variant, r.tau)
    nt0, et0 = dense(src[(r.run, r.sample_idx)])
    nt1, et1 = dense(graphs[key])
    groups = rm.nitro_groups(nt0, et0, N, O, C)
    fates = [fate(nt0, et0, nt1, et1, g) for g in groups]
    # summarise per graph: the most drastic fate among its nitro groups
    order = ['NO2 detached (fragment)', 'NO2 re-attached elsewhere', 'O removed/changed (no longer NO2)']
    main = next((f for f in fates if f != 'NO2 intact'), 'NO2 intact')
    rows.append({'variant': r.variant, 'tau': r.tau, 'fate': main, 'n_groups': len(groups),
                 'valid': r.smiles_valid, 'conn': r.heavy_conn, 'edits': r.n_total, 'ctrl': r.ctrl_flip,
                 'nitro_cf': r.nitro_cf})
res = pd.DataFrame(rows)
for v in ['posterior', 'noinv', 'naive']:
    x = res[res.variant == v]
    print(f'==== {v}: {len(x)} flipped nitro-mutagenic graphs (of {len(set((a,b) for a,b,c,d in want if c==v))} with a flip)')
    t = x.fate.value_counts()
    print(pd.DataFrame({'n': t, 'share': (t / len(x)).round(3)}).to_string())
    print('  CF still contains an aromatic-nitro group (any):', round(x.nitro_cf.mean(), 3), '| valid', round(x.valid.mean(), 3),
          '| no new heavy fragment', round(x.conn.mean(), 3), '| valid & no new fragment', round((x.valid & x.conn).mean(), 3),
          '| ctrl flip', round(x.ctrl.mean(), 3))
    print(x.groupby('fate')[['valid', 'conn', 'edits']].mean().round(3).to_string())
n_nitro = df[(df.nitro_src == 1) & (df.src_class == 0)][['run', 'sample_idx']].drop_duplicates().shape[0]
print('nitro-mutagenic source graphs in the 5x100 sample:', n_nitro)
