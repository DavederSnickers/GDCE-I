"""What actually happens inside XPlore++'s optimisation loop, per iteration.

Two hypotheses about XPlore++'s collapse have already been tested and killed:

  * "the shared judge cannot see node features" -- refuted by
    feature_channel_probe: 93.3% of mutagenicity graphs flip under a per-node
    rescaling alone, topology untouched. The channel is potent.
  * "the feature update descends the objective while the edge update ascends it,
    so the two fight" -- refuted by XPlore_signfix: ascending the feature update
    gives FR 0.002, WORSE than the authors' 0.062. Direction is not the issue.

So the remaining question is what the feature channel does to the search itself.
This script logs, for a handful of instances and every iteration: the range of
P_node_hat, the norm of the perturbed feature matrix, the judge's logits and
prediction, how many edges are currently on, and whether anything has gone
non-finite. It answers "saturation, divergence or NaN?" by looking, not by
reasoning about the code.

Usage (in the container, from the GRETEL root):
    python -m src.wrappers.xplore_trace --config configs/mutagenicity_mol.xplore.yaml \\
        --variant xplore++ --n 6 --K 60
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
    ap.add_argument('--variant', default='xplore++')
    ap.add_argument('--n', type=int, default=6)
    ap.add_argument('--K', type=int, default=60)
    ap.add_argument('--alpha', type=float, default=1.0)
    ap.add_argument('--every', type=int, default=5)
    ap.add_argument('--signfix', action='store_true')
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

    from src.wrappers.dataset_loader import load_gdce_dataset
    from src.wrappers.xplore_oracle import XPloreJudgeOracle
    from src.wrappers.xplore.graph_instance import GraphInstance
    from src.wrappers.xplore_eval import to_v2_instance, VARIANTS
    if args.signfix:
        from src.wrappers.xplore.XPlore_signfix import XPlore
    else:
        from src.wrappers.xplore.XPlore import XPlore

    oracle = XPloreJudgeOracle(judge_prefix=str(cfg.shared_judge_path), device='cpu')
    ds = load_gdce_dataset(cfg.gdce_path, cfg.dataset_name, cfg.datadir,
                           name='trace', max_nodes=int(cfg.n_nodes))
    test_ids = ds.splits[0]['test']

    params = dict(VARIANTS[args.variant])
    params.update(dict(alpha=args.alpha, K=args.K, beta=float(cfg.beta),
                       gamma_edge=float(cfg.gamma_edge), change_all_feat=False,
                       directed=False, dataset_classes=2, node_classification=False,
                       multi_label_classification=False, decay_alpha=False,
                       debugging=False, visualize=False, chem_flag=False,
                       device='cpu'))

    class _DS:
        name = str(cfg.dataset_name); dataset_name = name
        num_classes = 2; manipulators: list = []

    xp = XPlore(context=None, local_config={'dataset': _DS(), 'oracle': oracle,
                                            'parameters': params})
    xp.CHEMICAL_DATASETS = {str(cfg.dataset_name)}

    # Wrap the loss so we can look at the state of every iteration from outside,
    # without touching the authors' file. __calculate_loss is name-mangled.
    orig_loss = getattr(xp, '_XPlore__calculate_loss')
    log = []

    def traced(instance):
        loss = orig_loss(instance)
        pn = getattr(xp, 'P_node_hat', None)
        log.append(dict(
            k=xp.k,
            loss=float(loss.detach()) if torch.isfinite(loss.detach()) else float('nan'),
            logits=[round(float(v), 3) for v in xp.g_v_logits.detach().flatten()],
            pred=int(xp.g_v_bar_pred),
            edges_on=int((xp.A_v_bar.detach() != 0).sum()) // 2,
            phat=(float(xp.P_hat.detach().min()), float(xp.P_hat.detach().max())),
            pnode=(None if pn is None else
                   (float(pn.detach().min()), float(pn.detach().max()))),
            xnorm=float(xp.N_v_bar.detach().abs().sum()),
            finite=bool(torch.isfinite(xp.g_v_logits.detach()).all()),
        ))
        return loss

    setattr(xp, '_XPlore__calculate_loss', traced)

    print(f"variant={args.variant} alpha={args.alpha} K={args.K} "
          f"signfix={args.signfix}")
    for iid in test_ids[:args.n]:
        inst = ds.instances[iid]
        gi = to_v2_instance(inst, int(inst.n_real_nodes), GraphInstance)
        log.clear()
        torch.manual_seed(0); np.random.seed(0)
        cf = xp.explain(gi)
        base = oracle.predict_int(gi)
        got = oracle.predict_int(cf)
        n0 = int(np.count_nonzero(np.triu(np.asarray(gi.data), 1)))
        print(f"\n--- iid={iid}  n={gi.data.shape[0]}  edges={n0}  "
              f"f(G)={base} -> f(G')={got}  {'FLIP' if got != base else 'no flip'} "
              f" (ran {len(log)} its)")
        print(f"    {'k':>4} {'loss':>11} {'logits':>18} {'pred':>4} {'edges':>6} "
              f"{'P_hat':>17} {'P_node':>19} {'|x|_1':>9} fin")
        for r in log[::max(1, args.every)] + ([log[-1]] if log else []):
            pn = ('--' if r['pnode'] is None
                  else f"[{r['pnode'][0]:.2f},{r['pnode'][1]:.2f}]")
            print(f"    {r['k']:>4} {r['loss']:>11.4f} "
                  f"{str(r['logits']):>18} {r['pred']:>4} {r['edges_on']:>6} "
                  f"[{r['phat'][0]:.2f},{r['phat'][1]:.2f}]".rjust(18)
                  + f" {pn:>19} {r['xnorm']:>9.2f} {'y' if r['finite'] else 'N'}")


if __name__ == '__main__':
    main()
