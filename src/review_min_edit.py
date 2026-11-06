"""Exhaustive minimum-edit counterfactuals for the shared judge (review response R1.3 / R2.4).

The budget search of GDCE-I returns the counterfactual at the smallest diffusion budget tau
that flips the judge. This script asks how far that is from the smallest number of graph
edits that flips the judge at all. For every source graph of a paired-tau run (the same
5 x 100 test graphs as GDCE-I, read from its graphs.jsonl.gz) it enumerates ALL
counterfactuals with k <= kmax edits and records

    min_k        the smallest k <= kmax that flips the judge (-1: none)
    k<k>_*       how many k-edit flips exist, how many of them flip the control judge too,
                 and, on a random sample of at most check_cap of them, how many are valid
                 molecules (RDKit sanitisation as in review_metrics.strict_smiles) and add
                 no heavy-atom fragment to the source
    ub_k         for graphs without a flip at k <= kmax: an upper bound on the minimum from
                 a beam search over single edits, ranked by the judge's target probability

Edit space of the GCN judge (binary adjacency): node-type substitution (any of the dx
types) and edge insertion / deletion; an inserted edge is a single bond. Bond-type changes
are invisible to this judge and are never needed for a minimal flip.

Usage (repo root, inside the container, GPU):
    python src/review_min_edit.py --run_dir results_review/paired_tau/mutag_gcn_s3 \
        --out results_review/min_edit/mutag_gcn_s3.csv
Smoke test: add --limit 3
"""

import argparse
import csv
import gzip
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'shared_judge'))
from shared_judge import load_judge  # noqa: E402

import review_metrics as rm  # noqa: E402


def read_sources(run_dir):
    """Source graphs of a paired-tau run: the posterior replay under the source class at the
    smallest tau reproduces the input exactly (exact == 1 for every graph in chains.csv)."""
    meta = json.load(open(os.path.join(run_dir, 'meta.json')))
    tau0 = min(meta['tau_grid'])
    out = []
    with gzip.open(os.path.join(run_dir, 'graphs.jsonl.gz'), 'rt') as f:
        for line in f:
            d = json.loads(line)
            if d['variant'] != 'posterior' or d['cond'] != 'src' or d['tau'] != tau0:
                continue
            nt = np.array(d['node_types'], dtype=np.int64)
            n = len(nt)
            et = np.zeros((n, n), dtype=np.int64)
            for a, b, t in d['edges']:
                et[a, b] = et[b, a] = t
            out.append({'run': d['run'], 'sample_idx': d['sample_idx'], 'data_idx': d['data_idx'],
                        'src_class': d['src_class'], 'tgt_class': d['tgt_class'], 'nt': nt, 'et': et})
    return meta, out


class Edits:
    """The single edits of one source graph, as parallel code tensors (kind, a, b):
    kind 0 = node a becomes type b; kind 1 = the pair (a, b) is toggled (insert / delete)."""

    def __init__(self, nt, et, dx, device):
        n = len(nt)
        kind, a, b = [], [], []
        for i in range(n):
            for t in range(dx):
                if t != nt[i]:
                    kind.append(0), a.append(i), b.append(t)
        for i in range(n):
            for j in range(i + 1, n):
                kind.append(1), a.append(i), b.append(j)
        self.kind_np, self.a_np, self.b_np = np.array(kind), np.array(a), np.array(b)
        self.kind = torch.tensor(kind, device=device)
        self.a = torch.tensor(a, device=device)
        self.b = torch.tensor(b, device=device)
        self.S = len(kind)
        self.nt = torch.tensor(nt, device=device)
        self.adj = torch.tensor((et > 0).astype(np.uint8), device=device)
        self.n = n

    def conflict(self, p, q):
        """True where single edits p and q touch the same node type or the same pair."""
        return (self.kind[p] == self.kind[q]) & (self.a[p] == self.a[q]) & \
            ((self.kind[p] == 0) | (self.b[p] == self.b[q]))

    def apply(self, idx):
        """idx: (B, k) single-edit indices, -1 = no edit. Returns NT (B, n), ADJ (B, n, n)."""
        B, k = idx.shape
        NT = self.nt.unsqueeze(0).repeat(B, 1)
        ADJ = self.adj.unsqueeze(0).repeat(B, 1, 1)
        rows = torch.arange(B, device=idx.device)
        for slot in range(k):
            e = idx[:, slot]
            ok = e >= 0
            e = e.clamp(min=0)
            kd, a, b = self.kind[e], self.a[e], self.b[e]
            m = ok & (kd == 0)
            NT[rows[m], a[m]] = b[m]
            m = ok & (kd == 1)
            r, i, j = rows[m], a[m], b[m]
            new = 1 - ADJ[r, i, j]
            ADJ[r, i, j] = new
            ADJ[r, j, i] = new
        return NT, ADJ

    def graph(self, edit_ids, et_src):
        """numpy (nt, et) of one counterfactual; an inserted edge is a single bond."""
        nt = self.nt.cpu().numpy().copy()
        et = et_src.copy()
        for e in edit_ids:
            if e < 0:
                continue
            kd, a, b = self.kind_np[e], self.a_np[e], self.b_np[e]
            if kd == 0:
                nt[a] = b
            else:
                et[a, b] = et[b, a] = 0 if et_src[a, b] > 0 else 1
        return nt, et

    def describe(self, e, nt_src, et_src, decoder):
        kd, a, b = self.kind_np[e], self.a_np[e], self.b_np[e]
        sym = lambda t: decoder.get(str(int(t)), str(int(t)))
        if kd == 0:
            return f'{sym(nt_src[a])}{a}->{sym(b)}'
        op = 'del' if et_src[a, b] > 0 else 'add'
        return f'{op}:{sym(nt_src[a])}{a}-{sym(nt_src[b])}{b}'


@torch.no_grad()
def judge_p(judge, NT, ADJ, dx, tgt):
    """Target-class probability of a binary-adjacency judge for B graphs of equal size."""
    B, n = NT.shape
    x = F.one_hot(NT.reshape(-1), dx).float()
    bb, i, j = torch.nonzero(ADJ, as_tuple=True)
    ei = torch.stack([bb * n + i, bb * n + j])
    batch = torch.arange(B, device=NT.device).repeat_interleave(n)
    return F.softmax(judge(x, ei, batch), dim=-1)[:, tgt]


@torch.no_grad()
def evaluate(judge, ed, idx, dx, tgt, chunk):
    """Target probabilities for the edit sets idx (B, k), in chunks."""
    out = []
    for s in range(0, idx.shape[0], chunk):
        NT, ADJ = ed.apply(idx[s:s + chunk])
        out.append(judge_p(judge, NT, ADJ, dx, tgt))
    return torch.cat(out) if out else torch.zeros(0, device=idx.device)


def check_sample(ed, sets, et_src, nt_src, heavy, n_heavy_comp_src, decoder, cap, rng):
    """RDKit validity and heavy-atom connectivity on at most cap of the edit sets."""
    m = len(sets)
    pick = np.arange(m) if m <= cap else rng.choice(m, size=cap, replace=False)
    valid, conn = np.zeros(len(pick), dtype=bool), np.zeros(len(pick), dtype=bool)
    for r, p in enumerate(pick):
        nt, et = ed.graph(sets[p], et_src)
        valid[r] = rm.strict_smiles(nt, et, decoder) is not None
        conn[r] = rm.components(et, heavy)[1] <= n_heavy_comp_src
    return pick, valid, conn


def beam_search(judge, ed, seeds, dx, tgt, chunk, width, depth):
    """Upper bound on the minimal flip size: grow the best edit sets one single edit at a time.
    seeds: (W, k0) edit sets to start from. Returns (depth of the first flip, its edit set)."""
    beam = seeds
    S = ed.S
    for k in range(beam.shape[1] + 1, depth + 1):
        W = beam.shape[0]
        new = torch.arange(S, device=beam.device).repeat(W)
        base = beam.repeat_interleave(S, dim=0)
        ok = torch.ones(len(new), dtype=torch.bool, device=beam.device)
        for slot in range(base.shape[1]):
            ok &= ~ed.conflict(base[:, slot], new) & (base[:, slot] != new)
        cand = torch.cat([base[ok], new[ok].unsqueeze(1)], dim=1)
        cand = torch.unique(torch.sort(cand, dim=1).values, dim=0)
        p = evaluate(judge, ed, cand, dx, tgt, chunk)
        hit = torch.nonzero(p > 0.5, as_tuple=True)[0]
        if len(hit):
            return k, cand[hit[0]].tolist()
        top = torch.topk(p, min(width, len(p))).indices
        beam = cand[top]
    return -1, []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run_dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--kmax', type=int, default=2)
    ap.add_argument('--beam', type=int, default=64)
    ap.add_argument('--depth', type=int, default=12)
    ap.add_argument('--check_cap', type=int, default=1000)
    ap.add_argument('--chunk_elems', type=int, default=2 ** 24)
    ap.add_argument('--limit', type=int, default=None)
    args = ap.parse_args()
    assert args.kmax in (1, 2)

    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    meta, sources = read_sources(args.run_dir)
    if args.limit:
        sources = sources[:args.limit]
    decoder = meta['atom_decoder']
    sym2idx = {v: int(k) for k, v in decoder.items()}
    h_idx = sym2idx.get('H', None)
    judge = load_judge(meta['judge'], device=str(device))
    ctrl = load_judge(meta['ctrl'], device=str(device))
    assert judge.edge_dim == 0 and ctrl.edge_dim == 0, 'edit space assumes a binary-adjacency judge'
    dx = judge.num_features
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    print(f'MIN-EDIT  {args.run_dir}  graphs={len(sources)}  kmax={args.kmax}  judge={meta["judge"]}')

    rng = np.random.default_rng(0)
    writer, fout = None, open(args.out, 'w', newline='')
    t_start = time.time()
    for g_i, src in enumerate(sources):
        t0 = time.time()
        nt_src, et_src, tgt = src['nt'], src['et'], int(src['tgt_class'])
        n = len(nt_src)
        heavy = (nt_src != h_idx) if h_idx is not None else np.ones(n, dtype=bool)
        n_heavy_comp_src = rm.components(et_src, heavy)[1]
        ed = Edits(nt_src, et_src, dx, device)
        chunk = min(65536, max(256, args.chunk_elems // (n * n)))
        p0 = float(evaluate(judge, ed, torch.full((1, 1), -1, device=device), dx, tgt, chunk)[0])
        row = {'run': src['run'], 'sample_idx': src['sample_idx'], 'data_idx': src['data_idx'],
               'n': n, 'n_heavy': int(heavy.sum()), 'src_class': src['src_class'], 'tgt_class': tgt,
               'p_tgt_src': p0, 'n_singles': ed.S,
               'src_valid': int(rm.strict_smiles(nt_src, et_src, decoder) is not None),
               'src_heavy_comp': n_heavy_comp_src}

        # ---- k = 1 ----
        single = torch.arange(ed.S, device=device).unsqueeze(1)
        p1 = evaluate(judge, ed, single, dx, tgt, chunk)
        levels = {1: (single, p1)}
        # ---- k = 2: all pairs of non-conflicting single edits ----
        if args.kmax >= 2:
            iu = torch.triu_indices(ed.S, ed.S, 1, device=device)
            keep = ~ed.conflict(iu[0], iu[1])
            pairs = iu[:, keep].T.contiguous()
            p2 = evaluate(judge, ed, pairs, dx, tgt, chunk)
            levels[2] = (pairs, p2)
            row['n_pairs'] = int(pairs.shape[0])
        row['min_k'] = -1
        for k, (sets, p) in levels.items():
            fl = torch.nonzero(p > 0.5, as_tuple=True)[0]
            row[f'k{k}_flips'] = int(len(fl))
            if len(fl) and row['min_k'] < 0:
                row['min_k'] = k
            pc = evaluate(ctrl, ed, sets[fl], dx, tgt, chunk) if len(fl) else torch.zeros(0)
            ctrl_ok = (pc > 0.5).cpu().numpy()
            row[f'k{k}_ctrl'] = int(ctrl_ok.sum())
            fsets = sets[fl].cpu().numpy()
            pick, valid, conn = check_sample(ed, fsets, et_src, nt_src, heavy, n_heavy_comp_src,
                                             decoder, args.check_cap, rng)
            row[f'k{k}_checked'] = int(len(pick))
            row[f'k{k}_valid'] = int(valid.sum())
            row[f'k{k}_validconn'] = int((valid & conn).sum())
            row[f'k{k}_validconn_ctrl'] = int((valid & conn & ctrl_ok[pick]).sum())
            if k == 1:
                kinds = ed.kind_np[fsets[:, 0]]
                tog = fsets[:, 0][kinds == 1]
                dele = et_src[ed.a_np[tog], ed.b_np[tog]] > 0
                row['k1_node'] = int((kinds == 0).sum())
                row['k1_del'] = int(dele.sum())
                row['k1_add'] = int((~dele).sum())
            ex = [' + '.join(ed.describe(e, nt_src, et_src, decoder) for e in s if e >= 0)
                  for s in fsets[:5]]
            row[f'k{k}_examples'] = ' | '.join(ex)
            vc = pick[valid & conn]
            row[f'k{k}_ex_validconn'] = ' | '.join(
                ' + '.join(ed.describe(e, nt_src, et_src, decoder) for e in fsets[p] if e >= 0)
                for p in vc[:5])
        # ---- upper bound by beam search where no flip with <= kmax edits exists ----
        row['ub_k'], row['ub_edits'] = row['min_k'], ''
        if row['min_k'] < 0:
            sets, p = levels[max(levels)]
            top = torch.topk(p, min(args.beam, len(p))).indices
            k_ub, best = beam_search(judge, ed, sets[top], dx, tgt, chunk, args.beam, args.depth)
            row['ub_k'] = k_ub
            row['ub_edits'] = ' + '.join(ed.describe(e, nt_src, et_src, decoder) for e in best)
        row['t_sec'] = round(time.time() - t0, 2)

        if writer is None:
            writer = csv.DictWriter(fout, fieldnames=list(row.keys()), restval='')
            writer.writeheader()
        writer.writerow(row)
        fout.flush()
        print(f'  [{g_i:4d}] run={src["run"]} idx={src["data_idx"]:4d} n={n:2d} {src["src_class"]}->{tgt} '
              f'p0={p0:.3f} min_k={row["min_k"]} ub={row["ub_k"]} k1={row["k1_flips"]} '
              f'k2={row.get("k2_flips", "-")} {row["t_sec"]}s (total {time.time() - t_start:.0f}s)')
    fout.close()
    print(f'DONE {len(sources)} graphs in {time.time() - t_start:.0f}s -> {args.out}')


if __name__ == '__main__':
    main()
