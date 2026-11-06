"""Train a COMBINEX oracle (GCN_G) on MUTAG-G, NCI1-G or BENZENE-G — no explainer step.

This is the Phase-A counterpart of `run_combinex_{mutag,nci1}_evaluation.py`:
trains the oracle once per dataset, dumps the state_dict to
`data/models/<dataset>_<model>_<hidden_layers>_epochs_<N>.pt`
(same path convention as `GraphTrainer.start_training`), then exits.

Usage (from repo root, inside the apptainer image):

    apptainer exec --nv vm/combinex.sif python train_oracle.py --dataset MUTAG-G
    apptainer exec --nv vm/combinex.sif python train_oracle.py --dataset NCI1-G

Override the GDCE CSV root via env var COMBINEX_GDCE_DATA_ROOT.
"""

from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from src.datasets.dataset import DataInfo
from src.oracles.train.train import GraphTrainer
from src.utils.dataset import get_dataset
from src.utils.models import get_model


def _load_cfg(dataset_yaml: str, model_yaml: str, dropout: float = None, lr: float = None):
    config_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "config"))
    overrides = [
        f"dataset={dataset_yaml}",
        f"model={model_yaml}",
        "task=graph",
        "explainer=combined",
        "logger.mode=offline",
    ]
    # Per-run dropout override (keeps config/model/gcn.yaml — shared with MUTAG-G —
    # untouched). Architecture is unchanged; only the dropout hyperparameter is tuned.
    if dropout is not None:
        overrides.append(f"model.dropout={dropout}")
    # Per-run learning-rate override. The optimizer uses a CONSTANT lr (no scheduler),
    # so a lower lr stabilizes the final-epoch test accuracy (BatchNorm noise).
    if lr is not None:
        overrides.append(f"trainer.lr={lr}")
    # NCI1-G uses BatchNorm (harder dataset; matches the eval-time build below).
    # MUTAG-G stays batchnorm=false so its existing oracle checkpoint still loads.
    if dataset_yaml in ("nci1_g", "benzene_g"):
        overrides.append("model.batchnorm=true")
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(config_name="config", overrides=overrides)
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", required=True, choices=("MUTAG-G", "NCI1-G", "BENZENE-G"),
        help="COMBINEX dataset name",
    )
    parser.add_argument("--model", default="gcn", choices=("gcn", "gine"))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--dropout", type=float, default=None,
        help="Override model dropout (default: config value 0.5). Architecture-preserving "
             "hyperparameter tune; e.g. 0.1 for NCI1 to reduce over-regularization.",
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Override trainer.lr (default: config value 0.001). Lower (e.g. 3e-4) "
             "stabilizes final-epoch accuracy since the optimizer has no LR scheduler.",
    )
    args = parser.parse_args()

    dataset_yaml = {"MUTAG-G": "mutag_g", "NCI1-G": "nci1_g",
                    "BENZENE-G": "benzene_g"}[args.dataset]
    cfg = _load_cfg(dataset_yaml, args.model, dropout=args.dropout, lr=args.lr)
    cfg.general.seed = args.seed

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() and cfg.device == "cuda" else "cpu"

    with wandb.init(
        project=cfg.logger.project,
        group="oracle_train",
        mode=cfg.logger.mode,
        config=OmegaConf.to_container(cfg),
        name=f"oracle_{args.dataset}_{cfg.model.name}",
    ):
        data = get_dataset(args.dataset)
        datainfo = DataInfo(cfg, data)

        oracle_cls = get_model(name=cfg.model.name, task=cfg.task.name)
        oracle_kwargs = dict(
            num_features=datainfo.num_features,
            num_classes=datainfo.num_classes,
            cfg=cfg,
        )
        # GINE needs the bond-type width; 0 for NCI1 (no edge attributes).
        if cfg.model.name == "GINE":
            oracle_kwargs["edge_attr_dim"] = int(getattr(datainfo, "edge_attr_dim", 0))
            print(f"[train_oracle] GINE edge_attr_dim={oracle_kwargs['edge_attr_dim']}")
        oracle = oracle_cls(**oracle_kwargs).to(device)

        # GraphTrainer writes to data/models/<name>_<model>_<hidden>_epochs_<N>.pt
        # at the end of start_training().
        os.makedirs("data/models", exist_ok=True)
        trainer = GraphTrainer(cfg=cfg, dataset=data, model=oracle, loss=F.cross_entropy)
        trainer.start_training()

    print(
        f"[train_oracle] done — checkpoint at "
        f"data/models/{cfg.dataset.name}_{cfg.model.name}_"
        f"{cfg.model.hidden_layers}_epochs_{cfg.trainer.epochs}.pt"
    )


if __name__ == "__main__":
    main()
