"""Candidate examples for a regenerated Fig. 4: GDCE-I (posterior, shared GCN judge, s=3) counterfactuals of
nitro-carrying molecules judged mutagenic, first flip on the 17-tau grid, valid and without new fragments."""
import gzip, json, sys
import numpy as np, pandas as pd, torch, torch.nn.functional as F
sys.path.insert(0, './src'); sys.path.insert(0, './src/shared_judge')
import review_metrics as rm
import analyze_paired as ap
from shared_judge import load_judge
from rdkit import RDLogger; RDLogger.DisableLog('rdApp.*')
dec = json.load(open('results/mutag_s3/meta.json'))['atom_decoder']
J = load_judge('judges/judge_mutagenicity_mol_gcn')
Jc = load_judge('judges/judge_mutagenicity_mol_gcn_seed1')
def p(judge, nt, et):
    x = F.one_hot(torch.tensor(nt), 13).float(); a, b = np.nonzero(et > 0)
    with torch.no_grad():
        return F.softmax(judge(x, torch.tensor(np.stack([a, b])), torch.zeros(len(nt), dtype=torch.long)), -1)[0].numpy()
df = ap.load('results/mutag_s3/chains.csv')
first, n = ap.first_flip(df)
f = first[(first.variant == 'posterior') & (first.nitro_src == 1) & (first.src_class == 0) & (first.valid_conn == 1)]
want = {(r.run, r.sample_idx, r.tau) for r in f.itertuples()}
cf, src = {}, {}
with gzip.open('results/mutag_s3/graphs.jsonl.gz', 'rt') as fh:
    for line in fh:
        d = json.loads(line)
        if d['variant'] == 'posterior' and d['cond'] == 'tgt' and (d['run'], d['sample_idx'], d['tau']) in want:
            cf[(d['run'], d['sample_idx'])] = d
        if d['variant'] == 'posterior' and d['cond'] == 'src' and d['tau'] == 1:
            src[(d['run'], d['sample_idx'])] = d
def dense(d):
    nt = np.array(d['node_types']); m = len(nt); et = np.zeros((m, m), dtype=int)
    for a, b, t in d['edges']: et[a, b] = et[b, a] = t
    return nt, et
rows = []
for r in f.sort_values('n_total').itertuples():
    k = (r.run, r.sample_idx)
    nt0, et0 = dense(src[k]); nt1, et1 = dense(cf[k])
    p0, p0c, p1, p1c = p(J, nt0, et0), p(Jc, nt0, et0), p(J, nt1, et1), p(Jc, nt1, et1)
    nm, pm = rm.edit_masks(nt0, et0, nt1, et1)
    node_ed = [f'{dec[str(nt0[a])]}{a}->{dec[str(nt1[a])]}' for a in np.nonzero(nm)[0]]
    ii, jj = np.nonzero(pm)
    pair_ed = [f'{dec[str(nt0[a])]}{a}-{dec[str(nt0[b])]}{b}:{et0[a,b]}->{et1[a,b]}' for a, b in zip(ii, jj)]
    rows.append({'run': r.run, 'idx': r.data_idx, 'tau': r.tau, 'edits': r.n_total,
                 'p_mut_src(f)': round(float(p0[0]), 3), 'p_mut_src(f_sur)': round(float(p0c[0]), 3),
                 'p_mut_cf(f)': round(float(p1[0]), 3), 'p_mut_cf(f_sur)': round(float(p1c[0]), 3),
                 'src': rm.strict_smiles(nt0, et0, dec), 'cf': rm.strict_smiles(nt1, et1, dec),
                 'edits_list': '; '.join(node_ed + pair_ed)})
res = pd.DataFrame(rows).drop_duplicates('idx')
pd.set_option('display.width', 300); pd.set_option('display.max_colwidth', 200)
print(len(res), 'candidates')
print(res.head(12).to_string(index=False))
res.to_csv('results/fig4_candidates.csv', index=False)
