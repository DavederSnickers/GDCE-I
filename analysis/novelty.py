"""Distributional plausibility proxy (R1.2 / R1.3): share of counterfactuals whose sanitised molecule contains an
atom environment (Morgan radius 1, implicit H) that occurs in NO training molecule.
Compared: test sources, GDCE-I / no-inversion CFs (paired runs, first flip on the 17-tau grid), exhaustive
1-edit CFs of the shared GCN judge, and the baselines' CFs (shared-judge dumps).
Usage: python novelty.py mutag|benzene"""
import ast, gzip, json, sys
import numpy as np, pandas as pd, torch
sys.path.insert(0, './src'); sys.path.insert(0, './src/shared_judge')
import review_metrics as rm
import review_min_edit as me
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

def envs(nt, et):
    """radius-1 Morgan environment ids of the sanitised molecule (explicit H nodes made implicit), or None"""
    try:
        mol = rm.build_mol(nt, et, dec); Chem.SanitizeMol(mol); mol = Chem.RemoveHs(mol, sanitize=True)
    except Exception:
        return None
    info = {}
    AllChem.GetMorganFingerprint(mol, 1, bitInfo=info)
    # heavy atoms only: a detached explicit-H node ([HH]) is a representation artefact, not chemistry
    heavy = {a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1}
    return ({k for k, v in info.items() if any(r == 1 and a in heavy for a, r in v)} |
            {('r0', k) for k, v in info.items() if any(r == 0 and a in heavy for a, r in v)})

def dense_from_csv(row):
    nt = np.array(ast.literal_eval(row.node_types)); n = len(nt)
    et = np.array(ast.literal_eval(row.edge_type_matrix)).reshape(n, n).astype(int); np.fill_diagonal(et, 0)
    return nt, et

train = pd.read_csv(f'./data/{DSDIR}/raw/train.csv') if ds == 'mutag' else None
if train is None:   # benzene ships only test.csv locally -> fetch train.csv from the cluster beforehand
    train = pd.read_csv(f'results/{ds}_train.csv')
TRAIN = set(); n_ok = 0
for r in train.itertuples():
    e = envs(*dense_from_csv(r))
    if e is not None:
        TRAIN |= e; n_ok += 1
print(f'{ds}: training molecules sanitised {n_ok}/{len(train)}; distinct radius-0/1 environments {len(TRAIN)}')

def novel(nt, et):
    e = envs(nt, et)
    return None if e is None else int(len(e - TRAIN) > 0)

# ---- sources and GDCE-I / noinv counterfactuals (17-tau grid, first flip) ----
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
res = {}
nov_src = [novel(*dense(d)) for d in src.values()]
nov_src = [x for x in nov_src if x is not None]
res['test sources'] = (len(nov_src), np.mean(nov_src))
rows = []
for r in first.itertuples():
    nt, et = dense(cf[(r.run, r.sample_idx, r.variant, r.tau)])
    rows.append({'variant': r.variant, 'run': r.run, 'sample_idx': r.sample_idx, 'novel': novel(nt, et),
                 'valid': r.smiles_valid, 'conn': r.heavy_conn, 'ctrl': r.ctrl_flip, 'edits': r.n_node + r.n_add + r.n_del})
G = pd.DataFrame(rows)

# ---- exhaustive 1-edit counterfactuals of the same graphs ----
ex = []
for (run, si), d in src.items():
    nt0, et0 = dense(d)
    tgt = int(d['tgt_class'])
    heavy = (nt0 != H) if H is not None else np.ones(len(nt0), bool)
    c0 = rm.components(et0, heavy)[1]
    ed = me.Edits(nt0, et0, dx, 'cpu')
    single = torch.arange(ed.S).unsqueeze(1)
    p = me.evaluate(J, ed, single, dx, tgt, 4096)
    fl = torch.nonzero(p > 0.5, as_tuple=True)[0]
    if len(fl) == 0:
        ex.append({'run': run, 'sample_idx': si, 'n1': 0}); continue
    pc = (me.evaluate(Jc, ed, single[fl], dx, tgt, 4096) > 0.5).numpy()
    n_v = n_vc = n_vcc = n_plaus = n_nov = 0
    for e, c in zip(fl.tolist(), pc):
        nt1, et1 = ed.graph([e], et0)
        nv = novel(nt1, et1)
        if nv is None:
            continue
        n_v += 1; n_nov += nv
        conn = rm.components(et1, heavy)[1] <= c0
        if conn:
            n_vc += 1
            if c:
                n_vcc += 1
                n_plaus += int(nv == 0)
    ex.append({'run': run, 'sample_idx': si, 'n1': len(fl), 'n1_valid': n_v, 'n1_valid_novel': n_nov,
               'n1_vc': n_vc, 'n1_vcc': n_vcc, 'n1_plausible': n_plaus})
X = pd.DataFrame(ex)
X.to_csv(f'results/min_edit/{ds}_k1_novelty.csv', index=False)

print('\n--- share of VALID counterfactuals with >=1 atom environment unseen in training ---')
print(f'test sources (reference): n={res["test sources"][0]}  novel {res["test sources"][1]:.3f}')
for v in ['posterior', 'noinv']:
    g = G[(G.variant == v) & (G.valid == 1)]
    print(f'{v:10s} first flip: valid n={len(g)} novel {g.novel.mean():.3f} | valid&conn n={int(g.conn.sum())} novel {g[g.conn == 1].novel.mean():.3f}')
x1 = X[X.n1 > 0]
print(f'exhaustive 1-edit CFs: graphs with a 1-edit flip {len(x1)}/{len(X)}; valid 1-edit flips {int(x1.n1_valid.sum())}, '
      f'novel {x1.n1_valid_novel.sum() / x1.n1_valid.sum():.3f}')
print(f'graphs where a valid, fragment-free, control-confirmed 1-edit CF exists: {np.mean(x1.n1_vcc > 0) * len(x1) / len(X):.3f} of all graphs;'
      f' ... that is also free of unseen environments: {np.mean(x1.n1_plausible > 0) * len(x1) / len(X):.3f}')
# plausible-CF rate of GDCE-I's first flip: flip & valid & conn & ctrl & no novel env
for v in ['posterior', 'noinv']:
    g = G[G.variant == v]
    ok = (g.valid == 1) & (g.conn == 1) & (g.ctrl == 1) & (g.novel == 0)
    print(f'{v:10s} first flip: flip & valid & no new fragment & control-confirmed & no unseen environment: {ok.sum() / n:.3f} of all graphs')

# ---- baselines (shared-judge dumps, inherited bond orders) ----
print('\n--- baselines: share of valid CFs with unseen environments ---')
for name in [f'GDCE_{ds}_gcn', f'C2_{ds}_gcn', f'CF2_{ds}_gcn', f'D4_{ds}_flip', f'UCX_{ds}_gcn']:
    d = pd.read_csv(f'advflip/{name}.csv'); d = d[d.adv_flip_base == 1]
    vals = []
    for r in d.itertuples():
        nt = np.array(json.loads(r.node_types)); m = len(nt); et = np.zeros((m, m), dtype=int)
        for a, b, t in json.loads(r.edges):
            if a != b: et[a, b] = et[b, a] = t
        s = src.get((r.run, r.sample_idx))
        if r.bond_source == 'inherited' and s is not None and s['data_idx'] == r.data_idx:
            et0 = dense(s)[1]
            if len(et0) == m:
                et = np.where(et > 0, np.where(et0 > 0, et0, 1), 0)
        nv = novel(nt, et)
        if nv is not None:
            vals.append(nv)
    print(f'{name:18s} valid CFs {len(vals):4d}  novel {np.mean(vals) if vals else float("nan"):.3f}')
