"""Smoke tests for the UCExplainer port (edge-attribute perturbation + GINE oracle).

Run inside the apptainer image, from the repo root:

    apptainer exec --nv --bind /home/space:/home/space vm/combinex.sif \
        python smoke_test_ucx.py

Checks, in order:
  S1/S2  datasets load with the expected shapes and edge-attribute metadata
  S3     GINENet_G forward/embedding pass, with and without edge attributes
  S4     GraphPerturber actually builds an E_attr parameter and perturbs bonds
  S5     one full CombinedExplainer.explain() call on a single small graph

Exits non-zero on the first failure.
"""

from __future__ import annotations

import os
import sys
import traceback

import torch

os.environ.setdefault("WANDB_MODE", "disabled")

FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}" + (f" — {detail}" if detail else ""), flush=True)
    if not condition:
        FAILURES.append(label)


def _cfg(dataset_yaml: str, model_yaml: str = "gine"):
    from hydra import compose, initialize_config_dir

    config_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "config"))
    overrides = [
        f"dataset={dataset_yaml}",
        f"model={model_yaml}",
        "task=graph",
        "explainer=combined",
        "logger.mode=offline",
    ]
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        return compose(config_name="config", overrides=overrides)


def main() -> int:
    from src.datasets.dataset import DataInfo
    from src.utils.dataset import get_dataset
    from src.utils.models import get_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}, torch={torch.__version__}", flush=True)

    expectations = {
        # dataset          n_node_types  n_edge_types
        # mutagenicity_mol has 13 atom types (node_type_map.npy), benzene 8.
        "MUTAG-G": (13, 5),
        "BENZENE-G": (8, 5),
        "NCI1-G": (10, 0),
    }
    yaml_of = {"MUTAG-G": "mutag_g", "BENZENE-G": "benzene_g", "NCI1-G": "nci1_g"}

    loaded = {}

    for name, (exp_nodes, exp_edges) in expectations.items():
        print(f"\n=== S1/S2: {name} ===", flush=True)
        try:
            data = get_dataset(name)
            cfg = _cfg(yaml_of[name])
            datainfo = DataInfo(cfg, data)
            sample = data.dataset[0]
            loaded[name] = (data, cfg, datainfo, sample)

            check(f"{name}: node feature width",
                  exp_nodes is None or datainfo.num_features == exp_nodes,
                  f"num_features={datainfo.num_features}")
            check(f"{name}: edge_attr_dim",
                  int(getattr(datainfo, "edge_attr_dim", 0)) == exp_edges,
                  f"edge_attr_dim={getattr(datainfo, 'edge_attr_dim', 0)}")

            if exp_edges > 0:
                ea = getattr(sample, "edge_attr", None)
                check(f"{name}: sample carries edge_attr",
                      ea is not None and ea.shape[1] == exp_edges,
                      f"shape={None if ea is None else tuple(ea.shape)}")
                check(f"{name}: discrete_edge_attr_mask present",
                      hasattr(datainfo, "discrete_edge_attr_mask"))
                if ea is not None and ea.numel():
                    sums = ea.sum(dim=-1)
                    check(f"{name}: input bonds are valid one-hot",
                          bool(((sums - 1.0).abs() < 1e-6).all()),
                          f"row sums in [{sums.min():.2f}, {sums.max():.2f}]")
            else:
                check(f"{name}: no edge-attribute metadata",
                      not hasattr(datainfo, "discrete_edge_attr_mask"))
        except Exception:
            traceback.print_exc()
            FAILURES.append(f"{name}: dataset load")

    for name in ("MUTAG-G", "NCI1-G"):
        if name not in loaded:
            continue
        print(f"\n=== S3: GINENet_G forward on {name} ===", flush=True)
        try:
            data, cfg, datainfo, sample = loaded[name]
            oracle_cls = get_model(name="GINE", task="Graph")
            oracle = oracle_cls(
                num_features=datainfo.num_features,
                num_classes=datainfo.num_classes,
                cfg=cfg,
                edge_attr_dim=int(getattr(datainfo, "edge_attr_dim", 0)),
            ).to(device).eval()

            x = sample.x.to(device)
            ei = sample.edge_index.to(device)
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=device)
            ea = getattr(sample, "edge_attr", None)
            kw = {"edge_attr": ea.to(device)} if ea is not None else {}

            with torch.no_grad():
                out = oracle(x, ei, batch, **kw)
                emb = oracle.get_embedding_repr(x, ei, batch, **kw)
            check(f"{name}: forward output shape", tuple(out.shape) == (1, datainfo.num_classes),
                  f"{tuple(out.shape)}")
            check(f"{name}: embedding shape", emb.shape[0] == 1, f"{tuple(emb.shape)}")

            # edge_weights path (used by the perturber) must work too.
            with torch.no_grad():
                out_w = oracle(x, ei, batch,
                               edge_weights=torch.ones(ei.shape[1], device=device), **kw)
            check(f"{name}: forward with edge_weights", tuple(out_w.shape) == (1, datainfo.num_classes))
        except Exception:
            traceback.print_exc()
            FAILURES.append(f"{name}: GINE forward")

    # ---- S4/S5: perturber + one real explain() call on MUTAG-G ----
    if "MUTAG-G" in loaded:
        print("\n=== S4/S5: perturber + explain() on MUTAG-G ===", flush=True)
        try:
            from src.baselines.combinex_eval import _make_factual
            from src.graph_level_explainer.explainer.combined.combined_explainer import (
                CombinedExplainer,
            )

            data, cfg, datainfo, _ = loaded["MUTAG-G"]
            # Pick a small graph so the test is quick.
            sample = min((data.dataset[i] for i in range(50)), key=lambda g: g.x.shape[0])

            oracle_cls = get_model(name="GINE", task="Graph")
            oracle = oracle_cls(
                num_features=datainfo.num_features,
                num_classes=datainfo.num_classes,
                cfg=cfg,
                edge_attr_dim=int(datainfo.edge_attr_dim),
            ).to(device).eval()

            emb_mean = torch.zeros(cfg.model.hidden_layers[-1])
            with torch.no_grad():
                pred = int(oracle(
                    sample.x.to(device), sample.edge_index.to(device),
                    torch.zeros(sample.x.shape[0], dtype=torch.long, device=device),
                    edge_attr=sample.edge_attr.to(device),
                ).argmax(dim=-1).item())
            factual = _make_factual(sample, pred, 1 - pred, emb_mean, device)
            check("factual carries edge_attr", getattr(factual, "edge_attr", None) is not None)

            cfg.optimizer.num_epochs = 30
            cfg.timeout = 120
            explainer = CombinedExplainer(cfg, datainfo)

            # Inspect the perturber directly before running the loop.
            from src.graph_level_explainer.explainer.combined.graph_perturber import (
                GraphPerturber,
            )
            gp = GraphPerturber(cfg=cfg, model=oracle, datainfo=datainfo,
                                graph=factual, device=device).to(device)
            check("perturber sees edge attributes", gp.has_edge_attrs)
            check("perturber built E_attr", hasattr(gp, "E_attr"))
            check("oracle exposes edge_attr", gp.model_has_edge_attr)
            check("oracle exposes edge_weights", gp.model_has_edge_weights)
            check("E_attr trainable (p_lambda != 0)",
                  gp.E_attr.requires_grad, f"p_lambda={cfg.explainer.p_lambda!r}")
            check("P_x trainable (beta != 0)",
                  gp.P_x.requires_grad, f"beta={cfg.explainer.beta!r}")

            cf = explainer.explain(graph=factual, oracle=oracle)
            if cf is None:
                print("  [INFO] no counterfactual within 30 epochs — "
                      "acceptable for a smoke test, the loop itself ran.", flush=True)
            else:
                check("CF has edge_attr", getattr(cf, "edge_attr", None) is not None)
                if getattr(cf, "edge_attr", None) is not None:
                    check("CF edge_attr aligned with edge_index",
                          cf.edge_attr.shape[0] == cf.edge_index.shape[1],
                          f"{cf.edge_attr.shape[0]} vs {cf.edge_index.shape[1]}")
                    sums = cf.edge_attr.detach().cpu().sum(dim=-1)
                    n_viol = int(((sums - 1.0).abs() > 1e-6).sum().item())
                    print(f"  [INFO] CF bond one-hot violations: {n_viol}/{len(sums)}",
                          flush=True)
                sums_x = cf.x.detach().cpu().sum(dim=-1)
                n_viol_x = int(((sums_x - 1.0).abs() > 1e-6).sum().item())
                print(f"  [INFO] CF node one-hot violations: {n_viol_x}/{len(sums_x)}",
                      flush=True)
        except Exception:
            traceback.print_exc()
            FAILURES.append("MUTAG-G: explain()")

    print("\n" + "=" * 60, flush=True)
    if FAILURES:
        print(f"SMOKE TESTS FAILED ({len(FAILURES)}):", flush=True)
        for f in FAILURES:
            print(f"  - {f}", flush=True)
        return 1
    print("ALL SMOKE TESTS PASSED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
