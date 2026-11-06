"""Shared evaluation core for `run_combinex_{mutag,nci1}_evaluation.py`.

Loads a pretrained COMBINEX oracle (saved by `train_oracle.py`), pre-classifies
the test set, draws 5 stratified seeds × N samples (BASELINE_PLAN §4.2), runs
the CombinedExplainer single-process per sample, and writes the two CSV
artifacts (inversion_results.csv, run_summary.csv).

SMILES validity is computed (strict mode only) for MUTAG; EdgeType Sparsity is
reported as NaN because COMBINEX' counterfactuals are binary-adjacency only
(no bond-type prediction) — see BASELINE_PLAN §3.4.
"""

from __future__ import annotations

import os
import time
import traceback
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch_geometric.data import Data

from src.baselines.cf_dump import CFDumpWriter, edges_from_edge_index, node_types_from_x
from src.baselines.metrics import (
    compute_ged,
    compute_sparsity,
    stratified_sample_indices,
)
from src.datasets.dataset import DataInfo
from src.graph_level_explainer.explainer.combined.combined_explainer import (
    CombinedExplainer,
)
from src.utils.dataset import get_dataset
from src.utils.models import get_model


_DEFAULT_SEEDS: tuple[int, ...] = (42, 43, 44, 45, 46)

# COMBINEX dataset name -> config/dataset/<file>.yaml, and -> the GDCE dataset
# name that the CF dumps are keyed by (so dumps from different repos join
# without a lookup table). PROTEINS-GDCE is not PROTEINS-G: the latter is the
# raw TUDataset loader with a different split.
_DATASET_YAML = {"MUTAG-G": "mutag_g", "NCI1-G": "nci1_g",
                 "BENZENE-G": "benzene_g", "PROTEINS-GDCE": "proteins_gdce",
                 "TWITTER-G": "twitter_g"}
_GDCE_NAME = {"MUTAG-G": "mutagenicity_mol", "BENZENE-G": "benzene",
              "NCI1-G": "nci1", "PROTEINS-GDCE": "proteins",
              "TWITTER-G": "twitter"}
_SUPPORTED_DATASETS = tuple(_DATASET_YAML)


@dataclass
class EvalConfig:
    dataset_name: str             # "MUTAG-G", "NCI1-G", "BENZENE-G",
                                  # "PROTEINS-GDCE" or "TWITTER-G"
    model_name: str               # "gcn"
    oracle_checkpoint: str        # path to .pt state_dict
    output_dir: str               # where CSVs are written
    num_runs: int = 5
    num_samples: int = 100
    seeds: Optional[tuple[int, ...]] = None
    # Per-sample explainer caps. Defaults match COMBINEX' config but you can
    # override to trade quality for runtime — full BASELINE_PLAN runs at 500/500
    # take ~33 h on one GPU per dataset.
    explainer_num_epochs: Optional[int] = None
    explainer_timeout_s: Optional[int] = None
    # Wenn > 0: druckt fuer die ersten N erfolgreichen Samples pro Run die
    # konkreten geaenderten Knotenzeilen (orig, V_pert, projiziert).
    print_cf_features: int = 0
    # Extra Hydra overrides for the explainer (e.g. "optimizer.lr=0.01",
    # "scheduler.initial_alpha=0.5"); used by the hyperparameter tuning runs.
    overrides: tuple[str, ...] = ()

    def __post_init__(self):
        if self.dataset_name not in _SUPPORTED_DATASETS:
            raise ValueError(f"unsupported dataset {self.dataset_name!r}")
        if self.seeds is None:
            # Use the BASELINE_PLAN defaults, sliced/extended to num_runs.
            if self.num_runs <= len(_DEFAULT_SEEDS):
                self.seeds = _DEFAULT_SEEDS[: self.num_runs]
            else:
                extra = tuple(
                    _DEFAULT_SEEDS[-1] + 1 + i
                    for i in range(self.num_runs - len(_DEFAULT_SEEDS))
                )
                self.seeds = _DEFAULT_SEEDS + extra
        if len(self.seeds) != self.num_runs:
            raise ValueError(
                f"len(seeds)={len(self.seeds)} must equal num_runs={self.num_runs}"
            )


def _load_combinex_cfg(dataset_yaml: str, model_yaml: str, extra=()):
    """Compose a COMBINEX-style config tree via Hydra."""
    from hydra import compose, initialize_config_dir

    config_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "config")
    )
    overrides = [
        f"dataset={dataset_yaml}",
        f"model={model_yaml}",
        "task=graph",
        "explainer=combined",
        "logger.mode=offline",
    ]
    # Must match train_oracle.py: NCI1-G oracle is built with BatchNorm, so the
    # eval-time model must too (strict load_state_dict). MUTAG-G stays off.
    # (No effect when the oracle is a shared judge -- that one is loaded from its
    # own export and never built from cfg.model.)
    if dataset_yaml in ("nci1_g", "benzene_g", "proteins_gdce", "twitter_g"):
        overrides.append("model.batchnorm=true")
    overrides.extend(extra)
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(config_name="config", overrides=overrides)
    return cfg


def _takes_edge_attr(model) -> bool:
    """Does this oracle's forward accept edge attributes? (GINE yes, GCN no)."""
    import inspect

    return "edge_attr" in inspect.signature(model.forward).parameters


def _is_shared_judge(path: str) -> bool:
    """A shared judge is addressed by prefix: <prefix>.npz + <prefix>.json."""
    return os.path.exists(str(path) + ".json") and os.path.exists(str(path) + ".npz")


def _build_oracle(cfg, datainfo: DataInfo, ckpt_path: str, device: str):
    # The shared judge already matches COMBINEX's calling convention exactly —
    # forward(x, edge_index, batch, edge_attr=..., edge_weights=...) plus
    # get_embedding_repr — so it needs no wrapper. The GraphPerturber's signature
    # inspection picks up both edge_weights and edge_attr from it.
    if _is_shared_judge(ckpt_path):
        import sys as _sys
        _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from shared_judge import load_judge

        print(f"[combinex_eval] using SHARED JUDGE at {ckpt_path}", flush=True)
        return load_judge(ckpt_path, device=device)

    oracle_cls = get_model(name=cfg.model.name, task=cfg.task.name)
    kwargs = dict(
        num_features=datainfo.num_features,
        num_classes=datainfo.num_classes,
        cfg=cfg,
    )
    # Must mirror train_oracle.py, otherwise the strict load_state_dict fails on
    # the GINE message-MLP's input width.
    if cfg.model.name == "GINE":
        kwargs["edge_attr_dim"] = int(getattr(datainfo, "edge_attr_dim", 0))
    oracle = oracle_cls(**kwargs)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    oracle.load_state_dict(state)
    oracle.to(device).eval()
    return oracle


def _oracle_forward(oracle, x, edge_index, batch, edge_attr=None):
    """Call the oracle, passing edge_attr only if it accepts and has one."""
    if edge_attr is not None and _takes_edge_attr(oracle):
        return oracle(x, edge_index, batch, edge_attr=edge_attr)
    return oracle(x, edge_index, batch)


def _preclassify_test(dataset, test_mask: torch.Tensor, oracle, device: str):
    """Run the oracle on every test graph; return (preds, embeddings)."""
    from torch_geometric.loader import DataLoader

    test_dataset = dataset[test_mask]
    loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    takes_ea = _takes_edge_attr(oracle)
    preds = []
    embs = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            ea = getattr(batch, "edge_attr", None) if takes_ea else None
            kw = {"edge_attr": ea} if ea is not None else {}
            out = oracle(batch.x, batch.edge_index, batch.batch, **kw)
            preds.append(out.argmax(dim=-1).cpu())
            emb = oracle.get_embedding_repr(batch.x, batch.edge_index, batch.batch, **kw)
            embs.append(emb.detach().cpu())
    preds_tensor = torch.cat(preds, dim=0)
    return preds_tensor.numpy(), torch.cat(embs, dim=0)


def _make_factual(
    sample: Data, pred: int, target: int, embedding_mean: torch.Tensor, device: str
) -> Data:
    """Build the per-sample `factual` Data structure the CombinedExplainer expects."""
    x = sample.x.clone()
    n = x.shape[0]
    kwargs = dict(
        x=x,
        edge_index=sample.edge_index.clone(),
        y=torch.tensor(int(pred), dtype=torch.long),
        y_ground=sample.y.clone() if sample.y is not None else torch.tensor(int(pred)),
        targets=torch.tensor(int(target), dtype=torch.long),
        x_projection=embedding_mean.clone(),
        batch=torch.zeros(n, dtype=torch.long),
    )
    # The perturber needs the bond types to build its E_attr parameter.
    orig_ea = getattr(sample, "edge_attr", None)
    if orig_ea is not None:
        kwargs["edge_attr"] = orig_ea.clone()
    return Data(**kwargs).to(device)


def _classify_cf(oracle, cf: Data, device: str) -> int:
    """Re-classify a generated counterfactual with the oracle."""
    n = cf.x.shape[0]
    batch = torch.zeros(n, dtype=torch.long, device=device)
    ea = getattr(cf, "edge_attr", None)
    with torch.no_grad():
        out = _oracle_forward(
            oracle, cf.x.to(device), cf.edge_index.to(device), batch,
            edge_attr=ea.to(device) if ea is not None else None,
        )
    return int(out.argmax(dim=-1).item())


def _load_atom_decoder(pyg_dataset) -> Optional[dict]:
    """Load atom_decoder.json next to the GDCE-style raw CSVs.

    Returns None if RDKit or the decoder file aren't available -- caller falls
    back to NaN for SMILES metrics.
    """
    try:
        import json
        raw_dir = getattr(pyg_dataset, "raw_dir", None)
        if raw_dir is None:
            return None
        path = os.path.join(raw_dir, "atom_decoder.json")
        if not os.path.exists(path):
            return None
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _inherit_edge_attr(
    orig_edge_index: torch.Tensor,
    orig_edge_attr: Optional[torch.Tensor],
    cf_edge_index: torch.Tensor,
) -> Optional[torch.Tensor]:
    """For each (src, dst) in cf_edge_index, copy the original edge_attr row.

    COMBINEX only filters edges (never adds) -- the bond types of surviving
    edges are unchanged. Returns None if no edge_attr is available.
    """
    if orig_edge_attr is None or orig_edge_index.numel() == 0:
        return None
    pair_to_col = {
        (int(s), int(d)): k
        for k, (s, d) in enumerate(
            zip(orig_edge_index[0].tolist(), orig_edge_index[1].tolist())
        )
    }
    n_cf = cf_edge_index.shape[1]
    if n_cf == 0:
        return torch.zeros((0, orig_edge_attr.shape[1]), dtype=orig_edge_attr.dtype)
    cf_attr = torch.zeros((n_cf, orig_edge_attr.shape[1]), dtype=orig_edge_attr.dtype)
    for k, (s, d) in enumerate(
        zip(cf_edge_index[0].tolist(), cf_edge_index[1].tolist())
    ):
        col = pair_to_col.get((int(s), int(d)))
        if col is not None:
            cf_attr[k] = orig_edge_attr[col]
    return cf_attr


def _graph_to_smiles_inline(
    node_types_tensor: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    atom_decoder: dict,
) -> Optional[str]:
    """Inlined port of GDCE.src.inversion_mutag_mol.graph_to_smiles (strict).

    Returns SMILES string on success, None if RDKit sanitization fails (i.e.
    the projected molecule is chemically invalid).
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import RWMol
    except ImportError:
        return None

    _BOND_TYPES = {
        1: Chem.rdchem.BondType.SINGLE,
        2: Chem.rdchem.BondType.DOUBLE,
        3: Chem.rdchem.BondType.TRIPLE,
        4: Chem.rdchem.BondType.AROMATIC,
    }

    from collections import defaultdict

    mol = RWMol()
    atom_indices = node_types_tensor.cpu().tolist()
    node_syms = []
    for idx in atom_indices:
        symbol = atom_decoder.get(str(idx), atom_decoder.get(idx, "C"))
        mol.AddAtom(Chem.Atom(symbol))
        node_syms.append(symbol)

    bond_types_map = {}  # (i, j) with i<j -> int bond type
    if edge_index.numel() > 0 and edge_attr is not None and edge_attr.numel() > 0:
        ei = edge_index.cpu().numpy()
        ea = edge_attr.argmax(dim=-1).cpu().numpy()
        added = set()
        for (src, dst), bt in zip(zip(ei[0], ei[1]), ea):
            i, j = int(src), int(dst)
            if i >= j or (i, j) in added:
                continue
            if int(bt) > 0 and int(bt) in _BOND_TYPES:
                mol.AddBond(i, j, _BOND_TYPES[int(bt)])
                bond_types_map[(i, j)] = int(bt)
                added.add((i, j))

    # Assign formal charges for nitro groups: N with >=2 O neighbors -> N+, O-.
    # The Mutagenicity dataset stores no formal charges, so nitroaromatics (very
    # common among mutagens) would otherwise fail strict valence sanitization and
    # be wrongly counted as invalid. Mirrors GDCE.graph_to_smiles.
    nbrs = defaultdict(list)
    for (i, j), bt in bond_types_map.items():
        nbrs[i].append((j, bt))
        nbrs[j].append((i, bt))
    for atom_idx, sym in enumerate(node_syms):
        if sym == "N":
            o_nbrs = [(nb, bt) for nb, bt in nbrs[atom_idx] if node_syms[nb] == "O"]
            if len(o_nbrs) >= 2:
                mol.GetAtomWithIdx(atom_idx).SetFormalCharge(1)
                for nb, bt in o_nbrs:
                    if bt == 1:  # single bond -> O-
                        mol.GetAtomWithIdx(nb).SetFormalCharge(-1)
                        break

    try:
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def _compute_smiles_strict(
    x_proj: torch.Tensor,
    cf_edge_index: torch.Tensor,
    inherited_edge_attr: Optional[torch.Tensor],
    atom_decoder: Optional[dict],
) -> Optional[str]:
    """Strict RDKit SMILES of the projected CF, or None if it cannot be built
    (no RDKit, no atom_decoder, no edge_attr, or strict sanitization fails).

    Callers must gate on `atom_decoder is not None and inherited_edge_attr is
    not None` to distinguish "not attempted" from "attempted but invalid".
    """
    if atom_decoder is None or inherited_edge_attr is None:
        return None
    node_types = x_proj.detach().cpu().argmax(dim=-1)
    return _graph_to_smiles_inline(
        node_types, cf_edge_index, inherited_edge_attr, atom_decoder
    )


def _validate_atom_decoder(atom_decoder: dict, num_node_types: int) -> None:
    """Print the loaded atom_decoder mapping and warn loudly on any mismatch.

    The Mutagenicity remapping is index-sensitive: a stale/wrong decoder
    (e.g. an old `atom_decoder.json.wrong.bak` left on the cluster) would make
    `_graph_to_smiles_inline` silently relabel atoms or fall back to carbon,
    producing wrong-but-valid SMILES. Surfacing the full mapping makes that
    failure mode visible at the top of every run.
    """
    try:
        keys = sorted(int(k) for k in atom_decoder.keys())
    except (TypeError, ValueError):
        print(f"[combinex_eval] !!! atom_decoder has non-integer keys: "
              f"{list(atom_decoder.keys())}", flush=True)
        return
    mapping = ", ".join(f"{k}->{atom_decoder[str(k)]}" for k in keys)
    print(f"[combinex_eval] atom_decoder mapping: {mapping}", flush=True)
    missing = set(range(num_node_types)) - set(keys)
    if missing:
        print(
            f"[combinex_eval] !!! WARNING: atom_decoder is missing indices "
            f"{sorted(missing)} (num_node_types={num_node_types}). Atoms with "
            f"these indices SILENTLY fall back to carbon in graph_to_smiles — "
            f"fix the decoder before trusting SMILES metrics.",
            flush=True,
        )
    if len(keys) != num_node_types:
        print(
            f"[combinex_eval] !!! WARNING: atom_decoder has {len(keys)} entries "
            f"but the dataset has num_node_types={num_node_types}.",
            flush=True,
        )


def _print_stratified_summary(
    per_sample_rows: list,
    n_total: int,
    is_mutag: bool,
) -> None:
    """Stratified final result: count, share and mean sparsities per subset.

    The mean sparsities are formed over the stratum INTERSECT {samples with
    defined sparsities (i.e. success=True)}, because edge/node_sparsity is NaN
    for unsuccessful samples."""
    if not per_sample_rows:
        print("[summary] no samples", flush=True)
        return

    df = pd.DataFrame(per_sample_rows)
    # numeric columns, for mean()
    for col in ("edge_sparsity", "node_sparsity", "row_change_rate",
                "onehot_violation_rate", "ged", "mod_rate"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    def _mean(sub: pd.DataFrame, col: str) -> str:
        if col not in sub.columns:
            return "n/a"
        vals = sub[col].dropna()
        if len(vals) == 0:
            return "n/a"
        return f"{vals.mean():.3f}"

    def _line(label: str, mask: pd.Series) -> str:
        sub = df[mask]
        cnt = len(sub)
        rate = cnt / n_total if n_total > 0 else float("nan")
        return (
            f"  {label:<40s} | "
            f"{cnt:>5d} ({rate*100:>5.1f}%) | "
            f"edge_sp={_mean(sub, 'edge_sparsity'):>6s} | "
            f"node_sp={_mean(sub, 'node_sparsity'):>6s} | "
            f"row_chg={_mean(sub, 'row_change_rate'):>6s} | "
            f"oh_viol={_mean(sub, 'onehot_violation_rate'):>6s}"
        )

    print("\n" + "=" * 130, flush=True)
    print(f"[STRATIFIED SUMMARY]  n_total = {n_total}", flush=True)
    print("=" * 130, flush=True)
    print(_line("Structurally valid (V_pert is one-hot)",
                df["structurally_valid"].fillna(False).astype(bool)),
          flush=True)
    print(_line("COMBINEX-valid (raw flip)",
                df["success"].fillna(False).astype(bool)),
          flush=True)
    print(_line("Projected valid (flip survives argmax)",
                df["projected_success"].fillna(False).astype(bool)),
          flush=True)
    if is_mutag and "smiles_valid_strict" in df.columns:
        sv = pd.to_numeric(df["smiles_valid_strict"], errors="coerce")
        print(_line("SMILES valid (after projection)",
                    sv.fillna(0).astype(bool)),
              flush=True)
        # Intersected: projected_flip AND SMILES valid -- the "real" CF rate.
        ps = df["projected_success"].fillna(False).astype(bool)
        print(_line("Projected flip AND SMILES valid",
                    ps & sv.fillna(0).astype(bool)),
              flush=True)
    print("=" * 130, flush=True)
    print(
        "  Note: edge_sp/node_sp/row_chg/oh_viol means are over the stratum ∩ "
        "{success=True} (NaN-safe).",
        flush=True,
    )
    print("=" * 130 + "\n", flush=True)


def _onehot_project(x: torch.Tensor) -> torch.Tensor:
    """Argmax projection onto a valid one-hot: exactly one 1 per row, at the
    position of the maximum. Ties break to the lowest index (argmax default)."""
    idx = x.argmax(dim=-1)
    out = torch.zeros_like(x)
    out.scatter_(1, idx.unsqueeze(1), 1.0)
    return out


def _onehot_violation_rate(t: Optional[torch.Tensor]) -> float:
    """Share of rows that do NOT contain exactly one 1 (multi-hot / zero-hot).

    That is the failure being looked for: the perturbation writes several 1s --
    or none -- into a vector that ought to be a valid one-hot.
    """
    if t is None or t.shape[0] == 0:
        return float("nan")
    row_sums = t.detach().cpu().sum(dim=-1)
    violations = int(((row_sums - 1.0).abs() > 1e-6).sum().item())
    return violations / t.shape[0]


def _classify_projected(
    oracle, cf: Data, device: str
) -> tuple[int, torch.Tensor, Optional[torch.Tensor]]:
    """Project V_pert AND edge_attr onto a valid one-hot and reclassify.

    Both have to be projected: since the explainer perturbs edge attributes as
    well, a flip can rest on off-manifold bond vectors just as much as on
    off-manifold node features.
    """
    x_proj = _onehot_project(cf.x.detach().cpu()).to(device)
    ea = getattr(cf, "edge_attr", None)
    ea_proj = (
        _onehot_project(ea.detach().cpu()).to(device)
        if ea is not None and ea.numel() > 0
        else None
    )
    n = x_proj.shape[0]
    batch = torch.zeros(n, dtype=torch.long, device=device)
    with torch.no_grad():
        out = _oracle_forward(
            oracle, x_proj, cf.edge_index.to(device), batch, edge_attr=ea_proj
        )
    return int(out.argmax(dim=-1).item()), x_proj, ea_proj


def _print_cf_feature_diff(
    sample_label: str,
    orig_x: torch.Tensor,
    v_pert: torch.Tensor,
    x_proj: torch.Tensor,
    max_rows: int = 16,
) -> None:
    """Print the changed node rows side by side: orig | V_pert | proj.

    Shows only rows that actually changed, and marks multi-hot/zero-hot rows
    explicitly.
    """
    orig = orig_x.detach().cpu()
    pert = v_pert.detach().cpu()
    proj = x_proj.detach().cpu()
    changed = (orig != pert).any(dim=-1).nonzero(as_tuple=True)[0].tolist()
    if not changed:
        print(f"    [{sample_label}] no row changed (V_pert == orig).", flush=True)
        return
    print(
        f"    [{sample_label}] {len(changed)} of {orig.shape[0]} rows changed:",
        flush=True,
    )
    for i in changed[:max_rows]:
        o = orig[i].int().tolist()
        p = pert[i].int().tolist()
        q = proj[i].int().tolist()
        row_sum = int(pert[i].sum().item())
        if row_sum == 0:
            tag = "ZERO-HOT "
        elif row_sum > 1:
            tag = "MULTI-HOT"
        else:
            tag = "ONE-HOT  "
        # argmax indices for human reading
        o_arg = int(orig[i].argmax().item())
        p_arg = int(pert[i].argmax().item())
        q_arg = int(proj[i].argmax().item())
        print(
            f"      node {i:>3} [{tag}] sum(V)={row_sum}  "
            f"orig(arg={o_arg})={o}  V_pert(arg={p_arg})={p}  proj(arg={q_arg})={q}",
            flush=True,
        )
    if len(changed) > max_rows:
        print(f"      ... and {len(changed) - max_rows} more changed rows", flush=True)


def _print_cf_edge_attr_diff(
    sample_label: str,
    orig_edge_attr: Optional[torch.Tensor],
    cf_edge_attr: Optional[torch.Tensor],
    max_rows: int = 16,
) -> None:
    """The edge counterpart of _print_cf_feature_diff.

    Prints the changed edge_attr rows and marks MULTI-HOT/ZERO-HOT -- exactly
    the cases where several 1s were written into one bond-type vector.
    """
    if cf_edge_attr is None or cf_edge_attr.numel() == 0:
        print(f"    [{sample_label}] no edge_attr on the counterfactual.", flush=True)
        return
    cf = cf_edge_attr.detach().cpu()
    n_viol = 0
    printed = 0
    print(
        f"    [{sample_label}] edge_attr: {cf.shape[0]} edges, "
        f"{cf.shape[1]} bond types:",
        flush=True,
    )
    for k in range(cf.shape[0]):
        row_sum = int(round(float(cf[k].sum().item())))
        if row_sum == 1:
            continue
        n_viol += 1
        if printed >= max_rows:
            continue
        tag = "ZERO-HOT " if row_sum == 0 else "MULTI-HOT"
        print(
            f"      edge {k:>3} [{tag}] sum={row_sum}  cf={cf[k].int().tolist()}",
            flush=True,
        )
        printed += 1
    if n_viol == 0:
        print("      all edge_attr rows are valid one-hot.", flush=True)
    elif n_viol > max_rows:
        print(f"      ... and {n_viol - max_rows} more violating edges", flush=True)


def _row_metrics(
    factual: Data,
    counterfactual: Optional[Data],
    is_mutagenicity: bool,
    n_orig_edges_undirected_hint: Optional[int] = None,
) -> dict:
    """Compute the per-sample metric row defined in BASELINE_PLAN §7."""
    base = {
        "ged": -1,
        "mod_rate": float("nan"),
        "node_sparsity": float("nan"),
        "edge_sparsity": float("nan"),
        "n_orig_edges": -1,
        # Share of nodes with any feature change at all (catches non-one-hot
        # perturbations that the argmax-based node_sparsity misses).
        "row_change_rate": float("nan"),
        # Share of rows in V_pert that are NOT exactly one 1 (multi-hot /
        # zero-hot), i.e. the off-manifold rate.
        "onehot_violation_rate": float("nan"),
        # The same for the bond-type vectors.
        "edge_onehot_violation_rate": float("nan"),
    }
    if is_mutagenicity:
        base["edgetype_sparsity"] = float("nan")
        base["smiles_valid"] = float("nan")

    if counterfactual is None:
        return base

    n_orig = int(factual.x.shape[0])
    n_cf = int(counterfactual.x.shape[0])

    node_sparsity, edge_sparsity, edgetype_sparsity, n_orig_edges = compute_sparsity(
        orig_x=factual.x,
        cf_x=counterfactual.x,
        orig_edge_index=factual.edge_index,
        cf_edge_index=counterfactual.edge_index,
        orig_edge_attr=getattr(factual, "edge_attr", None),
        cf_edge_attr=getattr(counterfactual, "edge_attr", None),
    )

    ged = compute_ged(
        factual.edge_index, counterfactual.edge_index,
        n_nodes_orig=n_orig, n_nodes_cf=n_cf, timeout=5.0,
    )
    mod_rate = (ged / n_orig_edges) if (ged >= 0 and n_orig_edges > 0) else float("nan")

    # --- row_change_rate: ehrliche Knoten-Aenderungsrate (jede Feature-Diff zaehlt) ---
    orig_x_cpu = factual.x.detach().cpu()
    cf_x_cpu = counterfactual.x.detach().cpu()
    n_common = min(orig_x_cpu.shape[0], cf_x_cpu.shape[0])
    n_total_nodes = max(orig_x_cpu.shape[0], cf_x_cpu.shape[0])
    rows_changed = 0
    if n_common > 0:
        rows_changed = int(
            (orig_x_cpu[:n_common] != cf_x_cpu[:n_common]).any(dim=-1).sum().item()
        )
    rows_changed += abs(orig_x_cpu.shape[0] - cf_x_cpu.shape[0])
    row_change_rate = rows_changed / n_total_nodes if n_total_nodes > 0 else float("nan")

    # --- onehot_violation_rate: V_pert rows that are not exactly one 1 ---
    onehot_violation_rate = _onehot_violation_rate(cf_x_cpu)
    # --- the same for the counterfactual's bond-type vectors ---
    edge_onehot_violation_rate = _onehot_violation_rate(
        getattr(counterfactual, "edge_attr", None)
    )

    base.update({
        "ged": int(ged),
        "mod_rate": float(mod_rate),
        "node_sparsity": float(node_sparsity),
        "edge_sparsity": float(edge_sparsity),
        "n_orig_edges": int(n_orig_edges),
        "row_change_rate": float(row_change_rate),
        "onehot_violation_rate": float(onehot_violation_rate),
        "edge_onehot_violation_rate": float(edge_onehot_violation_rate),
    })
    if is_mutagenicity:
        # edgetype_sparsity is real now: with a GINE oracle the explainer
        # perturbs bond types itself, so compute_sparsity gets a genuine
        # cf_edge_attr instead of the old inherited one (NaN when absent).
        base["edgetype_sparsity"] = float(edgetype_sparsity)
        base["smiles_valid"] = float("nan")
    return base


def run_evaluation(ev: EvalConfig) -> None:
    os.makedirs(ev.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- 1) config + dataset ----
    dataset_yaml = _DATASET_YAML[ev.dataset_name]
    cfg = _load_combinex_cfg(dataset_yaml, ev.model_name, ev.overrides)
    cfg.timeout = int(getattr(cfg, "timeout", 500))
    if ev.explainer_num_epochs is not None:
        cfg.optimizer.num_epochs = int(ev.explainer_num_epochs)
    if ev.explainer_timeout_s is not None:
        cfg.timeout = int(ev.explainer_timeout_s)
    print(
        f"[combinex_eval] per-sample budget: "
        f"num_epochs={cfg.optimizer.num_epochs}, timeout={cfg.timeout}s"
    )

    data = get_dataset(ev.dataset_name)
    datainfo = DataInfo(cfg, data)
    pyg_dataset = data.dataset
    test_mask = data.test_mask

    # ---- 2) oracle ----
    oracle = _build_oracle(cfg, datainfo, ev.oracle_checkpoint, device)

    # ---- 3) pre-classify test set ----
    preds, embs = _preclassify_test(pyg_dataset, test_mask, oracle, device)
    embedding_mean = embs.mean(dim=0)
    datainfo.distribution_mean_projection = embedding_mean
    test_indices_global = np.where(test_mask.cpu().numpy())[0]
    test_subset = pyg_dataset[test_mask]

    # Molecular datasets: both carry 5 one-hot bond types and an
    # atom_decoder.json, so SMILES + edge-type metrics apply to both.
    is_mol = ev.dataset_name in ("MUTAG-G", "BENZENE-G")
    is_mutag = is_mol
    method = "combinex_combined"

    atom_decoder = _load_atom_decoder(pyg_dataset) if is_mutag else None
    if is_mutag:
        print(
            f"[combinex_eval] atom_decoder: "
            f"{'loaded (' + str(len(atom_decoder)) + ' atoms)' if atom_decoder else 'NOT FOUND -> SMILES metrics will be N/A'}",
            flush=True,
        )
        if atom_decoder:
            _validate_atom_decoder(atom_decoder, int(datainfo.num_features))

    per_sample_rows = []
    per_run_rows = []

    # Graph-level counterfactual dump so a second judge can score this run later
    # without repeating it (~13 h/dataset). Always the PROJECTED graph -- the raw
    # V_pert is multi-hot and no classifier has ever seen anything like it.
    cf_writer = CFDumpWriter(
        os.path.join(ev.output_dir, "cf_dump.jsonl"),
        method="ucexplainer",
        # Dump the GDCE dataset name, not COMBINEX' internal one, so dumps from
        # different repos are joinable without a lookup table.
        dataset=_GDCE_NAME.get(ev.dataset_name, ev.dataset_name),
        judge=ev.oracle_checkpoint,
        bond_source=("predicted" if is_mol else "binary"),
        variant=ev.model_name,
    )

    print(
        f"[combinex_eval] starting {ev.num_runs} run(s) × {ev.num_samples} sample(s) "
        f"on {ev.dataset_name}",
        flush=True,
    )
    for run_i, seed in enumerate(ev.seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)

        sample_idx_local = stratified_sample_indices(preds, seed=seed, n_samples=ev.num_samples)

        per_run_metrics = []
        n_success = 0

        print(
            f"\n[run {run_i + 1}/{ev.num_runs}] seed={seed} — "
            f"{len(sample_idx_local)} samples queued",
            flush=True,
        )

        for s_i, local_idx in enumerate(sample_idx_local):
            sample = test_subset[int(local_idx)]
            pred = int(preds[int(local_idx)])
            target = 1 - pred
            n_nodes = int(sample.x.shape[0])

            factual = _make_factual(sample, pred, target, embedding_mean, device)

            # SMILES of the unmodified input graph (strict). Computed for EVERY
            # molecular sample, regardless of whether the flip succeeded, so
            # that the CSV shows the source molecule next to the counterfactual.
            smiles_orig: Optional[str] = None
            if is_mutag and atom_decoder is not None:
                orig_ea_full = getattr(sample, "edge_attr", None)
                if orig_ea_full is not None:
                    smiles_orig = _graph_to_smiles_inline(
                        sample.x.detach().cpu().argmax(dim=-1),
                        sample.edge_index,
                        orig_ea_full,
                        atom_decoder,
                    )

            t0 = time.time()
            try:
                explainer = CombinedExplainer(cfg, datainfo)
                cf = explainer.explain(graph=factual, oracle=oracle)
            except Exception:
                traceback.print_exc()
                cf = None
            elapsed = time.time() - t0

            success = False
            cf_pred = -1
            projected_cf_pred = -1
            projected_success = False
            x_proj = None
            ea_proj = None
            # structurally_valid: V_pert is a valid one-hot graph (no
            # multi-/zero-hot node AND no multi-/zero-hot bond). Computed on the
            # RAW explainer output, so also without a successful flip.
            structurally_valid = False
            smiles_valid_strict: Optional[bool] = None
            smiles_n_fragments: Optional[int] = None
            smiles_connected: Optional[bool] = None
            smiles_cf: Optional[str] = None
            if cf is not None and cf.edge_index.shape[1] > 0:
                # --- structural validity auf raw V_pert UND raw edge_attr ---
                if cf.x.shape[0] > 0:
                    node_viol = _onehot_violation_rate(cf.x)
                    edge_viol = _onehot_violation_rate(getattr(cf, "edge_attr", None))
                    structurally_valid = (
                        node_viol == 0.0
                        and (edge_viol == 0.0 or edge_viol != edge_viol)  # NaN = kein edge_attr
                    )

                cf_pred = _classify_cf(oracle, cf, device)
                success = (cf_pred == target)
                if success:
                    # Argmax projection back onto a valid one-hot (nodes AND
                    # edges) and reclassify: does the flip survive the
                    # on-manifold projection?
                    projected_cf_pred, x_proj, ea_proj = _classify_projected(
                        oracle, cf, device
                    )
                    projected_success = (projected_cf_pred == target)

                    # SMILES validity only makes sense on the projected
                    # (valid one-hot) graph. Bond types now come from the
                    # counterfactual itself (projected); only if the explainer
                    # produced none do we fall back on those inherited from the
                    # original.
                    if is_mol and atom_decoder is not None:
                        orig_ea = getattr(sample, "edge_attr", None)
                        inherited_ea = (
                            ea_proj.detach().cpu()
                            if ea_proj is not None
                            else _inherit_edge_attr(
                                sample.edge_index, orig_ea, cf.edge_index
                            )
                        )
                        if inherited_ea is not None:
                            s_strict = _compute_smiles_strict(
                                x_proj, cf.edge_index, inherited_ea, atom_decoder
                            )
                            smiles_valid_strict = s_strict is not None
                            # Keep the CF SMILES only if strictly valid, else None.
                            smiles_cf = s_strict if smiles_valid_strict else None
                            # n_fragments: disconnected SMILES has '.'-separated
                            # components; a single connected molecule has 1.
                            # Mirrors GDCE.inversion_mutag_mol (n_fragments,
                            # smiles_connected).
                            smiles_n_fragments = (
                                s_strict.count(".") + 1 if smiles_valid_strict else -1
                            )
                            smiles_connected = bool(
                                smiles_valid_strict and smiles_n_fragments == 1
                            )

                    # Dump the projected graph. `own_pred` records what the oracle
                    # said about THAT graph, so a later re-scoring can condition on
                    # the flips this run actually claims (projected_success).
                    cf_writer.write(
                        run=run_i, seed=int(seed), sample_idx=int(s_i),
                        data_idx=int(test_indices_global[int(local_idx)]),
                        source_class=pred, target_class=target,
                        node_types=node_types_from_x(x_proj),
                        edges=edges_from_edge_index(cf.edge_index, ea_proj),
                        own_pred=int(projected_cf_pred), projected=True,
                        raw_pred=int(cf_pred),
                        structurally_valid=bool(structurally_valid),
                    )

                    if ev.print_cf_features > 0 and n_success < ev.print_cf_features:
                        label = (
                            f"run{run_i + 1}/sample{s_i + 1} "
                            f"data_idx={int(test_indices_global[int(local_idx)])}"
                        )
                        _print_cf_feature_diff(
                            sample_label=label,
                            orig_x=factual.x,
                            v_pert=cf.x,
                            x_proj=x_proj,
                        )
                        _print_cf_edge_attr_diff(
                            sample_label=label,
                            orig_edge_attr=getattr(sample, "edge_attr", None),
                            cf_edge_attr=getattr(cf, "edge_attr", None),
                        )
            if not success:
                cf = None

            row = _row_metrics(factual, cf, is_mutagenicity=is_mutag)

            flip_marker = "FLIP" if success else "----"
            data_idx_global = int(test_indices_global[int(local_idx)])
            if success:
                proj_marker = "FLIP" if projected_success else "----"
                print(
                    f"  [{s_i + 1:>3}/{ev.num_samples}] data_idx={data_idx_global:>5}  "
                    f"n={n_nodes:>2}  {pred}->{cf_pred} ({flip_marker})  "
                    f"proj={projected_cf_pred} ({proj_marker})  "
                    f"ged={row['ged']:>3}  mod_rate={row['mod_rate']:.3f}  "
                    f"node_sp={row['node_sparsity']:.3f}  edge_sp={row['edge_sparsity']:.3f}  "
                    f"row_chg={row['row_change_rate']:.3f}  "
                    f"oh_viol={row['onehot_violation_rate']:.3f}  "
                    f"({elapsed:.1f}s)",
                    flush=True,
                )
            else:
                print(
                    f"  [{s_i + 1:>3}/{ev.num_samples}] data_idx={data_idx_global:>5}  "
                    f"n={n_nodes:>2}  {pred}->? ({flip_marker})  "
                    f"no valid CF  ({elapsed:.1f}s)",
                    flush=True,
                )
            # Store the SMILES flags as floats (NaN when not measurable), so
            # that the pandas aggregation works cleanly.
            row.update({
                "run": run_i,
                "seed": int(seed),
                "method": method,
                "sample_idx": int(s_i),
                "data_idx": int(test_indices_global[int(local_idx)]),
                "source_class": pred,
                "target_class": target,
                "success": bool(success),
                "projected_success": bool(projected_success),
                "projected_cf_pred": int(projected_cf_pred),
                "structurally_valid": bool(structurally_valid),
                "smiles_valid_strict": (
                    float("nan") if smiles_valid_strict is None
                    else float(bool(smiles_valid_strict))
                ),
                "smiles_n_fragments": (
                    float("nan") if smiles_n_fragments is None
                    else float(smiles_n_fragments)
                ),
                "smiles_connected": (
                    float("nan") if smiles_connected is None
                    else float(bool(smiles_connected))
                ),
                # Plain-text SMILES: the original always, where it can be
                # produced; the counterfactual only on a projected flip AND
                # strict validity. An empty string means absent.
                "smiles_orig": smiles_orig if smiles_orig is not None else "",
                "smiles_cf": smiles_cf if smiles_cf is not None else "",
                "wall_time_s": float(elapsed),
            })
            per_sample_rows.append(row)
            per_run_metrics.append(row)
            n_success += int(success)

        # ---- run-level aggregation ----
        df_run = pd.DataFrame(per_run_metrics)
        df_succ = df_run[df_run["success"]]
        df_geds = df_succ[df_succ["ged"] >= 0]

        n_projected_success = int(df_succ["projected_success"].sum()) if len(df_succ) else 0
        run_summary = {
            "run": run_i,
            "seed": int(seed),
            "method": method,
            "accuracy": n_success / ev.num_samples,
            # Share of the successful counterfactuals whose flip survives the
            # argmax projection (V_pert -> valid one-hot). Measures how much of
            # COMBINEX's success is on-manifold.
            "projected_accuracy": (
                n_projected_success / ev.num_samples
                if ev.num_samples > 0 else float("nan")
            ),
            "avg_ged": df_geds["ged"].mean() if len(df_geds) else float("nan"),
            "avg_mod_rate": df_geds["mod_rate"].mean() if len(df_geds) else float("nan"),
            "avg_node_sparsity": df_succ["node_sparsity"].mean() if len(df_succ) else float("nan"),
            "avg_edge_sparsity": df_succ["edge_sparsity"].mean() if len(df_succ) else float("nan"),
            "avg_row_change_rate": (
                df_succ["row_change_rate"].mean() if len(df_succ) else float("nan")
            ),
            "avg_onehot_violation_rate": (
                df_succ["onehot_violation_rate"].mean() if len(df_succ) else float("nan")
            ),
            "avg_edge_onehot_violation_rate": (
                df_succ["edge_onehot_violation_rate"].mean()
                if len(df_succ) and "edge_onehot_violation_rate" in df_succ.columns
                else float("nan")
            ),
            "n_total": ev.num_samples,
            "n_success": n_success,
            "n_projected_success": n_projected_success,
        }
        if is_mutag:
            # SMILES rates over the successful counterfactuals (NaN-safe --
            # None entries are converted to NaN and correctly ignored by
            # .mean()).
            sv = pd.to_numeric(df_succ.get("smiles_valid_strict"), errors="coerce") \
                if "smiles_valid_strict" in df_succ.columns else pd.Series(dtype=float)
            conn = pd.to_numeric(df_succ.get("smiles_connected"), errors="coerce") \
                if "smiles_connected" in df_succ.columns else pd.Series(dtype=float)
            frags = pd.to_numeric(df_succ.get("smiles_n_fragments"), errors="coerce") \
                if "smiles_n_fragments" in df_succ.columns else pd.Series(dtype=float)
            n_flip = len(df_succ)
            # valid_avg_fragments: mean fragment count over strict-valid CFs only
            # (n_fragments is -1 for invalid). Mirrors GDCE valid_avg_fragments.
            valid_frags = frags[(sv.fillna(0).astype(bool)) & (frags > 0)]
            run_summary.update({
                # Real now (GINE oracle => the explainer perturbs bond types).
                "avg_edgetype_sparsity": (
                    df_succ["edgetype_sparsity"].mean()
                    if len(df_succ) and "edgetype_sparsity" in df_succ.columns
                    else float("nan")
                ),
                "smiles_valid_rate": float(sv.mean()) if len(sv.dropna()) else float("nan"),
                # connected, strict-valid SMILES as a fraction of ALL flips
                # (BASELINE_PLAN/graph_counterfactual.md §5: valid_connected_over_flip).
                "valid_connected_over_flip": (
                    float(conn.fillna(0).sum()) / n_flip if n_flip > 0 else float("nan")
                ),
                "valid_avg_fragments": (
                    float(valid_frags.mean()) if len(valid_frags) else float("nan")
                ),
                "n_structurally_valid": int(
                    df_run["structurally_valid"].fillna(False).astype(bool).sum()
                ) if "structurally_valid" in df_run.columns else 0,
            })
        per_run_rows.append(run_summary)

        print(
            f"[run {run_i + 1}/{ev.num_runs}] seed={seed} — "
            f"acc={run_summary['accuracy']:.3f} "
            f"({n_success}/{ev.num_samples})  "
            f"proj_acc={run_summary['projected_accuracy']:.3f} "
            f"({n_projected_success}/{ev.num_samples})  "
            f"avg_ged={run_summary['avg_ged']:.2f}  "
            f"avg_mod_rate={run_summary['avg_mod_rate']:.3f}  "
            f"avg_node_sp={run_summary['avg_node_sparsity']:.3f}  "
            f"avg_edge_sp={run_summary['avg_edge_sparsity']:.3f}  "
            f"avg_row_chg={run_summary['avg_row_change_rate']:.3f}  "
            f"avg_oh_viol={run_summary['avg_onehot_violation_rate']:.3f}  "
            f"avg_edge_oh_viol={run_summary['avg_edge_onehot_violation_rate']:.3f}",
            flush=True,
        )
        if is_mutag:
            print(
                f"[run {run_i + 1}/{ev.num_runs}] seed={seed} — SMILES: "
                f"valid_rate={run_summary['smiles_valid_rate']:.3f}  "
                f"connected_over_flip={run_summary['valid_connected_over_flip']:.3f}  "
                f"avg_fragments={run_summary['valid_avg_fragments']:.2f}",
                flush=True,
            )

    # ---- stratifizierte Zusammenfassung ueber alle Runs ----
    _print_stratified_summary(
        per_sample_rows,
        n_total=ev.num_runs * ev.num_samples,
        is_mutag=is_mutag,
    )

    cf_writer.close()

    # ---- write CSVs ----
    inv_path = os.path.join(ev.output_dir, "inversion_results.csv")
    pd.DataFrame(per_sample_rows).to_csv(inv_path, index=False)

    run_path = os.path.join(ev.output_dir, "run_summary.csv")
    pd.DataFrame(per_run_rows).to_csv(run_path, index=False)

    print(f"[done] wrote {inv_path}, {run_path}")
