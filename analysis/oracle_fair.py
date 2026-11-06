"""Fair single-edit search: enumerate all 1-edit flips of the shared GCN judge, keep only those that pass the checks
a method may use (RDKit validity, no new heavy fragment, no atom environment unseen in training), pick the one with
the highest target probability, and only THEN ask the control judge. Compare with GDCE-I (raw / pruned).
Usage: python oracle_fair.py mutag|benzene"""
import ast, gzip, json, sys
import numpy as np, pandas as pd, torch
torch.set_num_threads(8)
sys.path.insert(0, './src'); sys.path.insert(0, './src/shared_judge')
import review_metrics as rm
import review_min_edit as me
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
src = {}
with gzip.open(f'results/{ds}_s3/graphs.jsonl.gz', 'rt') as f:
    for line in f:
        d = json.loads(line)
        if d['variant'] == 'posterior' and d['cond'] == 'src' and d['tau'] == 1:
            src[(d['run'], d['sample_idx'])] = d
rows = []
for (run, si), d in src.items():
    nt0 = np.array(d['node_types']); m = len(nt0); et0 = np.zeros((m, m), dtype=int)
    for a, b, t in d['edges']: et0[a, b] = et0[b, a] = t
    tgt = int(d['tgt_class'])
    heavy = (nt0 != H) if H is not None else np.ones(m, bool)
    c0 = rm.components(et0, heavy)[1]
    ed = me.Edits(nt0, et0, dx, 'cpu')
    single = torch.arange(ed.S).unsqueeze(1)
    p = me.evaluate(J, ed, single, dx, tgt, 4096).numpy()
    fl = np.nonzero(p > 0.5)[0]
    best_any, best_ok = None, None
    if len(fl):
        best_any = int(fl[np.argmax(p[fl])])
        for e in fl[np.argsort(-p[fl])]:
            nt1, et1 = ed.graph([int(e)], et0)
            E = envs(nt1, et1)
            if E is None or len(E - TRAIN) > 0 or rm.components(et1, heavy)[1] > c0:
                continue
            best_ok = int(e); break
    def ctrl_of(e):
        if e is None:
            return 0
        return int(me.evaluate(Jc, ed, torch.tensor([[e]]), dx, tgt, 8)[0] > 0.5)
    rows.append({'run': run, 'sample_idx': si, 'has_flip': int(len(fl) > 0), 'has_filtered': int(best_ok is not None),
                 'filtered_ctrl': ctrl_of(best_ok), 'top_ctrl': ctrl_of(best_any),
                 'top_valid': int(best_any is not None and envs(*ed.graph([best_any], et0)) is not None),
                 'filtered_edit': ed.describe(best_ok, nt0, et0, dec) if best_ok is not None else ''})
R = pd.DataFrame(rows)
R.to_csv(f'results/oracle_fair_{ds}.csv', index=False)
n = len(R)
print(ds, 'graphs', n)
print(f'single-edit flip exists: {R.has_flip.mean():.3f}')
print(f'highest-probability single-edit flip (no filter): valid {R.top_valid.sum()/n:.3f}, control-confirmed {R.top_ctrl.sum()/n:.3f} of all graphs')
print(f'filtered (valid, no new fragment, no unseen env) single-edit flip exists: {R.has_filtered.mean():.3f}; '
      f'... and confirmed by the control judge: {R.filtered_ctrl.sum()/n:.3f} of all graphs')
kinds = R[R.has_filtered == 1].filtered_edit.str.replace(r'\d+', '', regex=True).value_counts().head(12)
print('most common filtered single edits:'); print(kinds.to_string())
