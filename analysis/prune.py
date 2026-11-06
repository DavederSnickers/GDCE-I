"""GDCE-I + greedy pruning: revert every edit of a counterfactual that the judge does not need.
Plain greedy (as greedy_necessary in review_paired_tau.py) and a validity-preserving variant that only
accepts reversions after which the molecule still sanitises and gains no heavy-atom fragment.
Usage: python prune.py mutag|benzene"""
import ast, gzip, json, sys
import numpy as np, pandas as pd, torch, torch.nn.functional as F
sys.path.insert(0, './src'); sys.path.insert(0, './src/shared_judge')
import review_metrics as rm
import analyze_paired as ap
from shared_judge import load_judge
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog('rdApp.*')
ds = sys.argv[1]
DSDIR = {'mutag': 'mutagenicity_mol', 'benzene': 'benzene'}[ds]
H = {'mutag': 3, 'benzene': None}[ds]
dec = json.load(open(f'results/{ds}_s3/meta.json'))['atom_decoder']
J = load_judge(f'judges/judge_{DSDIR}_gcn')
Jc = load_judge(f'judges/judge_{DSDIR}_gcn_seed1')
dx = J.num_features

def probs(judge, NT, ET):
    B, n = NT.shape
    x = F.one_hot(torch.tensor(NT).reshape(-1), dx).float()
    b, i, j = np.nonzero(ET > 0)
    ei = torch.tensor(np.stack([b * n + i, b * n + j]))
    batch = torch.arange(B).repeat_interleave(n)
    with torch.no_grad():
        return F.softmax(judge(x, ei, batch), -1).numpy()

def envs(nt, et):
    try:
        mol = rm.build_mol(nt, et, dec); Chem.SanitizeMol(mol); mol = Chem.RemoveHs(mol, sanitize=True)
    except Exception:
        return None
    info = {}; AllChem.GetMorganFingerprint(mol, 1, bitInfo=info)
    heavy = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1}
    return ({k for k, v in info.items() if any(r == 1 and a in heavy for a, r in v)} |
            {('r0', k) for k, v in info.items() if any(r == 0 and a in heavy for a, r in v)})
trainf = './data/mutagenicity_mol/raw/train.csv' if ds == 'mutag' else 'results/benzene_train.csv'
TRAIN = set()
for r in pd.read_csv(trainf).itertuples():
    nt = np.array(ast.literal_eval(r.node_types)); n = len(nt)
    et = np.array(ast.literal_eval(r.edge_type_matrix)).reshape(n, n).astype(int); np.fill_diagonal(et, 0)
    e = envs(nt, et)
    if e is not None:
        TRAIN |= e

def quality(nt, et, heavy, c0, tgt):
    e = envs(nt, et)
    valid = e is not None
    conn = rm.components(et, heavy)[1] <= c0
    ctrl = int(probs(Jc, nt[None], et[None])[0].argmax() == tgt)
    return valid, conn, ctrl, (None if e is None else int(len(e - TRAIN) > 0))

def prune(nt0, et0, nt1, et1, tgt, heavy, c0, keep_valid):
    nt, et = nt1.copy(), et1.copy()
    while True:
        nodes = np.nonzero(nt != nt0)[0]
        ii, jj = np.nonzero(np.triu(et != et0, 1))
        cands = [('n', a, -1) for a in nodes] + [('e', a, b) for a, b in zip(ii, jj)]
        if not cands:
            break
        NT = np.repeat(nt[None], len(cands), 0); ET = np.repeat(et[None], len(cands), 0)
        for r, (k, a, b) in enumerate(cands):
            if k == 'n':
                NT[r, a] = nt0[a]
            else:
                ET[r, a, b] = ET[r, b, a] = et0[a, b]
        p = probs(J, NT, ET)[:, tgt]
        order = np.argsort(-p)
        took = False
        for r in order:
            if p[r] <= 0.5:
                break
            if keep_valid:
                ok = rm.strict_smiles(NT[r], ET[r], dec) is not None and rm.components(ET[r], heavy)[1] <= c0
                if not ok:
                    continue
            nt, et = NT[r], ET[r]; took = True
            break
        if not took:
            break
    return nt, et

df = ap.load(f'results/{ds}_s3/chains.csv')
first, n = ap.first_flip(df)
want = {(r.run, r.sample_idx, r.variant, r.tau) for r in first.itertuples() if r.variant in ('posterior', 'noinv')}
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
rows = []
for k in sorted(want):
    nt0, et0 = dense(src[k[:2]]); nt1, et1 = dense(cf[k])
    tgt = int(cf[k]['tgt_class'])
    heavy = (nt0 != H) if H is not None else np.ones(len(nt0), bool)
    c0 = rm.components(et0, heavy)[1]
    for mode, kv in (('raw', None), ('pruned', False), ('pruned_valid', True)):
        if kv is None:
            nt, et = nt1, et1
        else:
            nt, et = prune(nt0, et0, nt1, et1, tgt, heavy, c0, kv)
        v, c, ct, nov = quality(nt, et, heavy, c0, tgt)
        nm, pm = rm.edit_masks(nt0, et0, nt, et)
        ec = rm.edit_counts(nt0, et0, nt, et)
        rows.append({'variant': k[2], 'mode': mode, 'edits_gcn': ec['n_node'] + ec['n_add'] + ec['n_del'], 'edits_all': ec['n_total'],
                     'valid': int(v), 'conn': int(c), 'ctrl': ct, 'novel': nov,
                     'ok': int(v and c and ct == 1 and nov == 0)})
R = pd.DataFrame(rows)
R.to_csv(f'results/prune_{ds}.csv', index=False)
print(ds, 'graphs', n)
g = R.groupby(['variant', 'mode'])
out = g.agg(n=('valid', 'size'), edits_gcn=('edits_gcn', 'mean'), edits_all=('edits_all', 'mean'), valid=('valid', 'mean'),
            conn=('conn', 'mean'), ctrl=('ctrl', 'mean'))
out['novel|valid'] = g.apply(lambda x: x[x.valid == 1].novel.mean())
out['ok_rate(all graphs)'] = g.ok.sum() / n
print(out.round(3).to_string())
