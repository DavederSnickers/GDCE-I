"""Independent gegencheck of saved (original, counterfactual) pairs.

Loads a counterfactuals.pt written by the run_c2_*_evaluation.py scripts and
verifies, WITHOUT trusting the run's own metrics, that the counterfactuals are
genuine DISCRETE graphs:

  * every claimed success differs from its original by > 0 undirected edges
    (catches phantom 0-edit "flips");
  * the independent undirected edit distance matches the stored GED;
  * stored cf_pred actually differs from orig_pred (binary flip);
  * (optional --reclassify) re-runs the classifier on original vs CF and
    confirms the prediction flips on the discrete graph, independently of the run.

Usage:
    apprun python verify_counterfactuals.py c2_mutag_paper_results/gcn/counterfactuals.pt
    apprun python verify_counterfactuals.py <pt> --reclassify --classifier gcn \
        --ckpt models/GCN_MUTAG_sd.pt --in_dim 7
    apprun python verify_counterfactuals.py <pt> --reclassify --classifier gine \
        --ckpt /path/to/gine.ckpt --gdce_path <GDCE_ROOT>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / 'src'))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('pt', help='path to counterfactuals.pt')
    p.add_argument('--reclassify', action='store_true',
                   help='also rebuild the classifier and re-confirm the flip')
    p.add_argument('--classifier', choices=['gcn', 'gine'], default='gcn')
    p.add_argument('--ckpt', default=None, help='classifier checkpoint (for --reclassify)')
    p.add_argument('--in_dim', type=int, default=None, help='GCN input dim (gcn)')
    p.add_argument('--nhid', type=int, default=20)
    p.add_argument('--nout', type=int, default=20)
    p.add_argument('--self_loops', choices=['none', 'full'], default='none',
                   help="gcn: 'none' for MUTAG, 'full' for GDCE *_mol/nci1")
    p.add_argument('--gdce_path', default='<GDCE_ROOT>')
    return p.parse_args()


def undirected_edges(ei) -> set:
    """Set of undirected edges {(i,j) | i<j}, self-loops excluded (GED convention)."""
    s = set()
    if ei is None or ei.numel() == 0:
        return s
    arr = ei.cpu().numpy()
    for a, b in zip(arr[0], arr[1]):
        a, b = int(a), int(b)
        if a == b:
            continue
        s.add((a, b) if a < b else (b, a))
    return s


def build_classifier(args, in_dim, device):
    if args.classifier == 'gcn':
        from wrappers.paper_gcn import PaperGCNWrapperC2
        return PaperGCNWrapperC2(args.ckpt, in_dim=in_dim, nhid=args.nhid,
                                 nout=args.nout, self_loops=args.self_loops).to(device).eval()
    else:
        if args.gdce_path not in sys.path:
            sys.path.insert(0, args.gdce_path)
        from wrappers.mutag_gine_wrapper import MutagGINEWrapperC2
        return MutagGINEWrapperC2(args.ckpt).to(device).eval()


def main() -> None:
    args = parse_args()
    recs = torch.load(args.pt, weights_only=False)
    print(f"Loaded {len(recs)} records from {args.pt}\n")

    n_succ = 0
    phantoms = []          # success but identical graphs (edit distance 0)
    ged_mismatch = []      # independent edits != stored ged
    pred_inconsistent = [] # cf_pred == orig_pred but marked success
    edits_all = []

    for r in recs:
        if not r.get('success'):
            continue
        n_succ += 1
        eo = undirected_edges(r['original']['edge_index'])
        ec = undirected_edges(r['counterfactual']['edge_index'])
        edits = len(eo ^ ec)
        edits_all.append(edits)
        if edits == 0:
            phantoms.append(r['data_idx'])
        ged = r.get('ged')
        if ged is not None and ged >= 0 and edits != ged:
            ged_mismatch.append((r['data_idx'], ged, edits))
        if r.get('cf_pred') == r.get('orig_pred'):
            pred_inconsistent.append(r['data_idx'])

    print("=" * 64)
    print("STRUCTURAL CHECK (model-free)")
    print("=" * 64)
    print(f"  successes              : {n_succ}")
    if edits_all:
        import statistics
        print(f"  independent edit dist  : min={min(edits_all)} "
              f"mean={statistics.mean(edits_all):.2f} max={max(edits_all)}")
    print(f"  PHANTOM (0-edit flips) : {len(phantoms)}"
          + (f"  e.g. data_idx={phantoms[:10]}" if phantoms else "  -> none (good)"))
    print(f"  GED mismatches         : {len(ged_mismatch)}"
          + (f"  e.g. (idx,ged,edits)={ged_mismatch[:5]}" if ged_mismatch else "  -> none (good)"))
    print(f"  pred inconsistencies   : {len(pred_inconsistent)}"
          + (f"  e.g. data_idx={pred_inconsistent[:10]}" if pred_inconsistent else "  -> none (good)"))

    verdict_ok = not phantoms and not pred_inconsistent
    print(f"\n  STRUCTURAL VERDICT: {'OK - genuine distinct discrete CFs' if verdict_ok else 'SUSPICIOUS (see flags above)'}")

    if not args.reclassify:
        print("\n(run with --reclassify --ckpt ... to also re-confirm the flips with the model)")
        return

    if not args.ckpt:
        raise SystemExit("--reclassify requires --ckpt")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    in_dim = args.in_dim or int(recs[0]['original']['x'].size(1))
    model = build_classifier(args, in_dim, device)
    needs_orig_graph = hasattr(model, 'set_orig_graph')

    print("\n" + "=" * 64)
    print("RECLASSIFY CHECK (independent model re-run on discrete graphs)")
    print("=" * 64)
    confirmed = flip_fail = pred_mismatch = 0
    for r in recs:
        if not r.get('success'):
            continue
        x = r['original']['x'].to(device)
        oei = r['original']['edge_index'].to(device)
        cei = r['counterfactual']['edge_index'].to(device)
        n = x.size(0)
        with torch.no_grad():
            if needs_orig_graph:
                oea = r['original']['edge_attr']
                ea = oea.to(device) if oea is not None else torch.zeros((oei.size(1), model.edge_dim), device=device)
                model.set_orig_graph(oei, ea, n)
            op = int(model(x, oei).argmax(-1))
            if needs_orig_graph:
                model.set_orig_graph(oei, ea, n)   # cf bonds derive from the original lookup
            cp = int(model(x, cei).argmax(-1))
        if cp != op:
            confirmed += 1
        else:
            flip_fail += 1
        if op != r.get('orig_pred') or cp != r.get('cf_pred'):
            pred_mismatch += 1

    print(f"  re-confirmed flips     : {confirmed}/{n_succ}")
    print(f"  flip did NOT hold      : {flip_fail}"
          + ("  <-- these are not real counterfactuals!" if flip_fail else "  -> none (good)"))
    print(f"  pred != stored         : {pred_mismatch}"
          + ("  <-- nondeterminism / representation mismatch" if pred_mismatch else "  -> none (good)"))
    print(f"\n  RECLASSIFY VERDICT: {'OK - flips are real on the discrete graph' if flip_fail == 0 else 'SUSPICIOUS'}")


if __name__ == '__main__':
    main()
