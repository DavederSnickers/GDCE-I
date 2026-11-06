"""How much can the node-feature channel ALONE do against the shared judge?

XPlore++ adds a node-feature perturbation on top of the edge search and gets a
WORSE flip rate than the edge-only variant (0.06 vs 0.92 on mutagenicity_mol).
Two explanations are compatible with that, and they call for opposite
conclusions:

  (A) the judge barely responds to node features, so the extra channel is dead
      weight that only destabilises the search;
  (B) the channel is potent, and XPlore++'s optimiser drives it the wrong way.

This probe settles it without XPlore in the loop. It applies exactly the
transformation XPlore++ is capable of -- N_v_bar = N * x, an elementwise product
with the one-hot node features, i.e. a per-node rescaling -- and measures how
often the judge's prediction moves. The graph's topology is never touched.

Three regimes, all reachable by XPlore++'s `P_node_hat`:
  uniform   every node scaled by the same s (s < 1 shrinks, s > 1 amplifies)
  per-node  each node scaled independently, s ~ U[1-m, 1+m]
  gating    a random fraction of nodes zeroed (this is XPlore+'s mechanism, and
            also where XPlore++ lands when an entry crosses zero)

If flips are easy here, (A) is out.

Usage (in the container, from the GRETEL root):
    python -m src.wrappers.feature_channel_probe --config configs/mutagenicity_mol.xplore.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--n', type=int, default=300, help='test graphs to probe')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(args.config)

    from src.wrappers.gdce_bootstrap import ensure_gdce_on_src_path
    ensure_gdce_on_src_path(cfg.gdce_path)
    _sj = os.path.join(cfg.gdce_path, 'src', 'shared_judge')
    if _sj not in sys.path:
        sys.path.insert(0, _sj)

    from shared_judge import load_judge
    from src.wrappers.dataset_loader import load_gdce_dataset

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    judge = load_judge(str(cfg.shared_judge_path), device=device).eval()
    for p in judge.parameters():
        p.requires_grad_(False)

    ds = load_gdce_dataset(cfg.gdce_path, cfg.dataset_name, cfg.datadir,
                           name='probe', max_nodes=int(cfg.n_nodes))
    test_ids = ds.splits[0]['test']
    rng = np.random.default_rng(args.seed)
    pick = rng.choice(len(test_ids), size=min(args.n, len(test_ids)), replace=False)
    insts = [ds.instances[test_ids[i]] for i in pick]

    def graph(inst):
        n = int(inst.n_real_nodes)
        adj = inst.to_numpy_array()[:n, :n]
        adj = adj - np.diag(np.diag(adj))
        s, d = np.nonzero(adj)
        ei = (torch.empty((2, 0), dtype=torch.long, device=device) if len(s) == 0
              else torch.tensor(np.stack([s, d]), dtype=torch.long, device=device))
        x = inst.gdce_x[:n].to(device).float()
        return x, ei

    def pred(x, ei):
        with torch.no_grad():
            b = torch.zeros(x.shape[0], dtype=torch.long, device=device)
            return int(judge(x, ei, b, edge_attr=None,
                             edge_weights=None).argmax(-1).item())

    graphs = [graph(i) for i in insts]
    base = [pred(x, ei) for x, ei in graphs]
    print(f"{len(graphs)} test graphs; base class counts "
          f"{ {c: base.count(c) for c in set(base)} }")

    def report(name, make_scale):
        flips = 0
        for (x, ei), b in zip(graphs, base):
            s = make_scale(x.shape[0]).to(device).float().unsqueeze(-1)
            if pred(x * s, ei) != b:
                flips += 1
        print(f"    {name:<34} {flips:4d}/{len(graphs)}  ({flips/len(graphs):.1%})")

    print("\n  UNIFORM rescaling  x -> s*x   (topology untouched)")
    for s in [0.0, 0.1, 0.25, 0.5, 0.9, 1.1, 2.0, 5.0, 10.0, 100.0]:
        report(f"s = {s}", lambda n, s=s: torch.full((n,), float(s)))

    print("\n  PER-NODE rescaling  s_i ~ U[1-m, 1+m]")
    for m in [0.25, 0.5, 0.9, 1.0, 2.0]:
        report(f"m = {m}",
               lambda n, m=m: torch.tensor(rng.uniform(1 - m, 1 + m, size=n)))

    print("\n  GATING  a random fraction of nodes zeroed (XPlore+'s mechanism)")
    for f in [0.05, 0.1, 0.25, 0.5, 1.0]:
        def gate(n, f=f):
            s = np.ones(n)
            k = max(1, int(round(f * n)))
            s[rng.choice(n, size=min(k, n), replace=False)] = 0.0
            return torch.tensor(s)
        report(f"fraction = {f}", gate)

    print("\n  BEST-OF-100 random per-node rescalings, m = 1.0")
    hits = 0
    for (x, ei), b in zip(graphs, base):
        for _ in range(100):
            s = torch.tensor(rng.uniform(0.0, 2.0, size=x.shape[0]),
                             device=device).float().unsqueeze(-1)
            if pred(x * s, ei) != b:
                hits += 1
                break
    print(f"    {'any of 100 draws flips':<34} {hits:4d}/{len(graphs)}  "
          f"({hits/len(graphs):.1%})")
    print("\n  A high number in the last block means the feature channel is potent "
          "and\n  a search that fails to exploit it is failing for its own reasons.")


if __name__ == '__main__':
    main()
