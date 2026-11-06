import argparse
import os
import sys

import torch
from torch_geometric.loader import DataLoader
from constants import feature_dict, task_type, dataset_choices
from explainers import *
from gnns import *
from utils.dataset import get_datasets


# Paper-reported optimal hyperparameters (D4Explainer appendix, Table 7).
# Applied automatically when the CLI value matches the parser default,
# so explicit --flag overrides still win.
PAPER_HPARAMS = {
    "BA_shapes":      {"n_hidden": 64,  "train_batchsize": 4,  "alpha_cf": 0.005, "epoch": 1500},
    "Tree_Cycle":     {"n_hidden": 64,  "train_batchsize": 32, "alpha_cf": 0.1,   "epoch": 1500},
    "Tree_Grids":     {"n_hidden": 128, "train_batchsize": 32, "alpha_cf": 0.05,  "epoch": 1500, "num_layers": 8},
    "cornell":        {"n_hidden": 128, "train_batchsize": 4,  "alpha_cf": 0.05,  "epoch": 1500},
    "ba3":            {"n_hidden": 128, "train_batchsize": 32, "alpha_cf": 0.05,  "epoch": 1500},
    "mutag":          {"n_hidden": 64,  "train_batchsize": 2,  "alpha_cf": 0.001, "epoch": 1500},
    "mutag_filtered": {"n_hidden": 64,  "train_batchsize": 2,  "alpha_cf": 0.001, "epoch": 1500},
    "gdce_mutag":     {"n_hidden": 64,  "train_batchsize": 2,  "alpha_cf": 0.001, "epoch": 1500},
    "gdce_nci1":      {"n_hidden": 128, "train_batchsize": 32, "alpha_cf": 0.01,  "epoch": 1500},
    "gdce_benzene":   {"n_hidden": 128, "train_batchsize": 32, "alpha_cf": 0.01,  "epoch": 1500},
    # No paper entry for PROTEINS/TWITTER -- D4 never reported them. Both are
    # binary-adjacency graph classification like NCI1, so the NCI1 row is the
    # closest defensible default; TWITTER's 39k training graphs make 1500 epochs
    # unnecessary and expensive, hence the shorter schedule.
    "gdce_proteins":  {"n_hidden": 128, "train_batchsize": 32, "alpha_cf": 0.01,  "epoch": 1500},
    "gdce_twitter":   {"n_hidden": 128, "train_batchsize": 32, "alpha_cf": 0.01,  "epoch": 300},
    "bbbp":           {"n_hidden": 128, "train_batchsize": 16, "alpha_cf": 0.005, "epoch": 1500},
    "NCI1":           {"n_hidden": 128, "train_batchsize": 32, "alpha_cf": 0.01,  "epoch": 1500},
    "nci1_filtered":  {"n_hidden": 128, "train_batchsize": 32, "alpha_cf": 0.01,  "epoch": 1500},
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train explainers")
    parser.add_argument("--cuda", type=int, default=0, help="GPU device.")
    parser.add_argument("--root", type=str, default="results/", help="Result directory.")
    parser.add_argument("--dataset", type=str, default="Tree_Cycle", choices=dataset_choices)
    parser.add_argument("--verbose", type=int, default=10)
    parser.add_argument("--gnn_type", type=str, default="gcn")
    parser.add_argument("--gnn_path", type=str, default=None,
                        help="Override the oracle path. Point it at a shared-judge "
                             "prefix (<prefix>.npz + <prefix>.json) to train against "
                             "the classifier shared by all compared methods.")
    parser.add_argument("--task", type=str, default="nc")

    parser.add_argument("--train_batchsize", type=int, default=32)
    parser.add_argument("--test_batchsize", type=int, default=32)
    parser.add_argument("--sigma_length", type=int, default=10)
    parser.add_argument("--epoch", type=int, default=800)
    parser.add_argument("--feature_in", type=int)
    parser.add_argument("--data_size", type=int, default=-1)

    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--alpha_cf", type=float, default=0.5)
    parser.add_argument("--alpha_label", type=str, default=None,
                        help="Suffix used in checkpoint filenames "
                             "(best_model_<metric>_alpha_<label>.pt). "
                             "Defaults to alpha_cf with the decimal point removed, "
                             "e.g. 0.5 -> '05', 0.001 -> '0001'.")
    parser.add_argument("--dropout", type=float, default=0.001)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--lr_decay", type=float, default=0.999)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--prob_low", type=float, default=0.0)
    parser.add_argument("--prob_high", type=float, default=0.4)
    parser.add_argument("--sparsity_level", type=float, default=2.5)

    parser.add_argument("--normalization", type=str, default="instance")
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--layers_per_conv", type=int, default=1)
    parser.add_argument("--n_hidden", type=int, default=64)
    parser.add_argument("--cat_output", type=bool, default=True)
    parser.add_argument("--residual", type=bool, default=False)
    parser.add_argument("--noise_mlp", type=bool, default=True)
    parser.add_argument("--simplified", type=bool, default=False)

    return parser, parser.parse_args()


parser, args = parse_args()
args.noise_list = None

# Apply paper-optimal hyperparameters for known datasets, but only for fields
# the user did not explicitly override on the command line.
explicit_args = {a.lstrip('-') for a in sys.argv[1:] if a.startswith('--')}
hparams = PAPER_HPARAMS.get(args.dataset, {})
for k, v in hparams.items():
    if k not in explicit_args:
        setattr(args, k, v)
        print(f"[paper-hparam] {args.dataset}: {k}={v}")
    else:
        print(f"[user-override] {args.dataset}: {k}={getattr(args, k)} (paper value was {v})")

if args.alpha_label is None:
    args.alpha_label = str(args.alpha_cf).replace(".", "")
print(f"[checkpoint label] alpha_cf={args.alpha_cf} -> alpha_label='{args.alpha_label}'")

args.device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
args.feature_in = feature_dict[args.dataset]
args.task = task_type[args.dataset]
train_dataset, val_dataset, test_dataset = get_datasets(name=args.dataset)

train_dataset = train_dataset[: args.data_size]
gnn_path = args.gnn_path or f"param/gnns/{args.dataset}_{args.gnn_type}.pt"
explainer = DiffExplainer(args.device, gnn_path)

# Train D4Explainer over train_dataset and evaluate
explainer.explain_graph_task(args, train_dataset, val_dataset)

# Test D4Explainer on test_dataset.
#
# `explain_evaluation` loads "best_model.pth", but training writes the
# alpha-suffixed names (best_model_{modrate,flip}_alpha_<label>.pt). Where that
# unsuffixed file is absent this loop would raise FileNotFoundError *after* a
# completed training run, marking the whole job FAILED and hiding real failures.
# Training checkpoints are the deliverable here; the actual evaluation is done by
# run_d4_*_evaluation.py, so skip with a clear message instead of crashing.
_exp_dir = os.path.join(args.root, args.dataset)
_legacy_ckpt = os.path.join(_exp_dir, "best_model.pth")
if os.path.exists(_legacy_ckpt):
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
    for graph in test_loader:
        explanation, y_ori, y_exp, modif_r = explainer.explain_evaluation(args, graph)
else:
    print(f"[main] training done. Skipping the built-in test loop: {_legacy_ckpt} "
          f"does not exist (training saved alpha-suffixed checkpoints instead). "
          f"Run run_d4_*_evaluation.py for the evaluation.")