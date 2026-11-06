"""What are the new heavy-atom fragments in counterfactuals? (paired runs, first flip on the 17-tau grid)"""
import gzip, json, sys
from collections import Counter
import numpy as np, pandas as pd
sys.path.insert(0, './src')
import review_metrics as rm
import analyze_paired as ap

def comps(nt, et, heavy):
    n = len(nt); parent = list(range(n))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    ii, jj = np.nonzero(np.triu(et > 0, 1))
    for i, j in zip(ii, jj):
        ri, rj = find(i), find(j)
        if ri != rj: parent[ri] = rj
    groups = {}
    for a in range(n):
        groups.setdefault(find(a), []).append(a)
    return [g for g in groups.values() if any(heavy[a] for a in g)]

for ds, h in [('mutag', 3), ('benzene', None)]:
    dec = json.load(open(f'results/{ds}_s3/meta.json'))['atom_decoder']
    df = ap.load(f'results/{ds}_s3/chains.csv')
    first, n = ap.first_flip(df)
    want = {(r.run, r.sample_idx, r.variant, r.tau) for r in first.itertuples()}
    src, cf = {}, {}
    with gzip.open(f'results/{ds}_s3/graphs.jsonl.gz', 'rt') as f:
        for line in f:
            d = json.loads(line)
            k = (d['run'], d['sample_idx'], d['variant'], d['tau'])
            if d['cond'] == 'tgt' and k in want: cf[k] = d
            if d['variant'] == 'posterior' and d['cond'] == 'src' and d['tau'] == 1: src[(d['run'], d['sample_idx'])] = d
    def dense(d):
        nt = np.array(d['node_types']); m = len(nt); et = np.zeros((m, m), dtype=int)
        for a, b, t in d['edges']: et[a, b] = et[b, a] = t
        return nt, et
    for v in ['posterior', 'noinv']:
        sizes, kinds, n_new = Counter(), Counter(), []
        for k in want:
            if k[2] != v: continue
            nt0, et0 = dense(src[k[:2]]); nt1, et1 = dense(cf[k])
            heavy1 = (nt1 != h) if h is not None else np.ones(len(nt1), bool)
            heavy0 = (nt0 != h) if h is not None else np.ones(len(nt0), bool)
            c0 = comps(nt0, et0, heavy0); c1 = comps(nt1, et1, heavy1)
            main = max(len([a for a in g if heavy1[a]]) for g in c1)
            extra = sorted(c1, key=lambda g: -sum(heavy1[a] for a in g))[1:]
            n_new.append(len(c1) - len(c0))
            for g in extra[: max(0, len(c1) - len(c0))] if len(c1) > len(c0) else []:
                hv = [a for a in g if heavy1[a]]
                sizes[min(len(hv), 4)] += 1
                if len(hv) == 1:
                    kinds[dec[str(nt1[hv[0]])]] += 1
        tot = sum(sizes.values())
        print(f'{ds} {v}: graphs with new heavy fragments {np.mean(np.array(n_new) > 0):.3f}; new fragments {tot};'
              f' heavy-atom size 1/2/3/>=4: {[round(sizes[s] / tot, 3) for s in (1, 2, 3, 4)] if tot else "-"};'
              f' single-atom fragments by element: {dict(kinds.most_common(6))}')
