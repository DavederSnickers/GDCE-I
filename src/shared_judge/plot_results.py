"""Figures for the shared-judge benchmark.

Figure 1 — cross-method comparison on the SHARED GCN judge (the one column every
method has): validity, structural change (GED), chemical validity.
Figure 2 — UCExplainer's one-hot diagnostics across all five of its runs.

The validity panel deliberately plots UCExplainer's *projected* validity as the
solid bar, with the raw flip rate as a hatched extension. CF2 and C2Explainer need
no projection: they only edit the edge set and leave node vectors untouched, so
their counterfactuals are on-manifold by construction and raw == projected. Plotting
the raw numbers side by side would compare a rate that counts malformed graphs
against two that do not.

    python plot_results.py --res_dir <dir with ucx/ cff/ c2/> --out_dir <dir>
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

# --- design tokens (validated categorical slots 1-3, light surface) -----------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#dedddaff"
SERIES = {"UCExplainer": "#2a78d6", "CF²": "#eb6834", "C2Explainer": "#1baf7a"}

DATASETS = ["mutagenicity", "benzene", "nci1"]
DS_LABEL = {"mutagenicity": "Mutagenicity", "benzene": "Benzene", "nci1": "NCI1"}


def _load(res_dir: str) -> pd.DataFrame:
    src = [
        ("UCExplainer", "mutagenicity", "GCN",  f"{res_dir}/ucx/mutag_gcn/inversion_results.csv"),
        ("UCExplainer", "mutagenicity", "GINE", f"{res_dir}/ucx/mutag_gine/inversion_results.csv"),
        ("UCExplainer", "benzene",      "GCN",  f"{res_dir}/ucx/benzene_gcn/inversion_results.csv"),
        ("UCExplainer", "benzene",      "GINE", f"{res_dir}/ucx/benzene_gine/inversion_results.csv"),
        ("UCExplainer", "nci1",         "GCN",  f"{res_dir}/ucx/nci1_gcn/inversion_results.csv"),
        ("CF²",         "mutagenicity", "GCN",  f"{res_dir}/cff/mutag_gcn/inversion_results.csv"),
        ("CF²",         "benzene",      "GCN",  f"{res_dir}/cff/benzene_gcn/inversion_results.csv"),
        ("CF²",         "nci1",         "GCN",  f"{res_dir}/cff/nci1_gcn/inversion_results.csv"),
        ("C2Explainer", "mutagenicity", "GCN",  f"{res_dir}/c2/mutag_gcn/shared_judge/c2_results.csv"),
        ("C2Explainer", "benzene",      "GCN",  f"{res_dir}/c2/benzene_gcn/shared_judge/c2_results.csv"),
        ("C2Explainer", "nci1",         "GCN",  f"{res_dir}/c2/nci1_gcn/shared_judge/c2_results.csv"),
    ]
    rows = []
    for method, ds, judge, path in src:
        if not os.path.exists(path):
            print(f"  skip (missing): {path}")
            continue
        df = pd.read_csv(path)
        succ = df["success"].astype(bool)

        def per_run(col, mask=None):
            if col not in df.columns:
                return np.nan
            d = df if mask is None else df[mask]
            if not len(d):
                return np.nan
            return pd.to_numeric(d.groupby("run")[col].mean(), errors="coerce").mean()

        smiles_col = ("smiles_valid_strict" if "smiles_valid_strict" in df.columns
                      else "smiles_valid")
        rows.append({
            "method": method, "dataset": ds, "judge": judge,
            "validity_raw": succ.groupby(df["run"]).mean().mean(),
            "validity_proj": per_run("projected_success"),
            "ged": per_run("ged", succ & (pd.to_numeric(df["ged"], errors="coerce") >= 0)),
            "smiles": per_run(smiles_col, succ),
            "oh_node": per_run("onehot_violation_rate", succ),
            "oh_edge": per_run("edge_onehot_violation_rate", succ),
            "struct_ok": per_run("structurally_valid"),
        })
    return pd.DataFrame(rows)


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def _bars(ax, sub, value_col, methods, fmt, hatch_col=None):
    """Grouped bars: one group per dataset, one bar per method."""
    n = len(methods)
    width = 0.78 / n
    for i, m in enumerate(methods):
        xs, vals, tops = [], [], []
        for j, ds in enumerate(DATASETS):
            r = sub[(sub.method == m) & (sub.dataset == ds)]
            v = float(r[value_col].iloc[0]) if len(r) else np.nan
            t = float(r[hatch_col].iloc[0]) if (hatch_col and len(r)) else np.nan
            xs.append(j - 0.39 + width * (i + 0.5))
            vals.append(v)
            tops.append(t)
        # hatched extension = the part of the raw rate that does not survive
        # projection back onto valid one-hot graphs
        if hatch_col:
            for x, v, t in zip(xs, vals, tops):
                if not np.isnan(t) and not np.isnan(v) and t > v:
                    ax.bar(x, t - v, bottom=v, width=width * 0.92,
                           facecolor="none", edgecolor=SERIES[m], lw=1.0,
                           hatch="////", zorder=2)
        ax.bar(xs, vals, width=width * 0.92, color=SERIES[m], zorder=3,
               linewidth=0)
        for x, v, t in zip(xs, vals, tops):
            if np.isnan(v):
                continue
            # A hatched extension occupies the space above the bar, so the value
            # goes inside it instead of colliding with the hatch.
            inside = (not np.isnan(t)) and t > v and v > 0.08
            ax.text(x, v - 0.012 if inside else v, fmt(v), ha="center",
                    va="top" if inside else "bottom", fontsize=8,
                    color=SURFACE if inside else INK_2, zorder=4)
    ax.set_xticks(range(len(DATASETS)))
    ax.set_xticklabels([DS_LABEL[d] for d in DATASETS], color=INK, fontsize=10)


def figure_one(d: pd.DataFrame, out_dir: str):
    sub = d[d.judge == "GCN"].copy()
    # UCExplainer's honest number is the projected one; the others need no projection
    sub["validity_plot"] = np.where(sub.method == "UCExplainer",
                                    sub.validity_proj, sub.validity_raw)
    sub["validity_top"] = np.where(sub.method == "UCExplainer",
                                   sub.validity_raw, np.nan)
    methods = ["UCExplainer", "CF²", "C2Explainer"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), facecolor=SURFACE)
    for ax in axes:
        _style(ax)

    _bars(axes[0], sub, "validity_plot", methods, lambda v: f"{v:.2f}",
          hatch_col="validity_top")
    axes[0].set_title("Counterfactual validity", color=INK, fontsize=11.5,
                      loc="left", pad=12)
    axes[0].set_ylim(0, 1.12)
    axes[0].set_ylabel("share of samples", color=INK_2, fontsize=9)

    _bars(axes[1], sub, "ged", methods, lambda v: f"{v:.0f}")
    axes[1].set_title("Structural change (graph edit distance)", color=INK,
                      fontsize=11.5, loc="left", pad=12)
    axes[1].set_ylabel("edits", color=INK_2, fontsize=9)

    mol = sub[sub.dataset != "nci1"]
    _bars(axes[2], mol, "smiles", methods, lambda v: f"{v:.2f}")
    axes[2].set_title("Chemical validity (SMILES)", color=INK, fontsize=11.5,
                      loc="left", pad=12)
    axes[2].set_ylim(0, 1.12)
    axes[2].set_ylabel("share of successful CFs", color=INK_2, fontsize=9)
    axes[2].text(2, 0.5, "NCI1:\nno bond types", ha="center", va="center",
                 fontsize=8.5, color=INK_2, style="italic")

    handles = [Patch(facecolor=SERIES[m], label=m) for m in methods]
    handles.append(Patch(facecolor="none", edgecolor=SERIES["UCExplainer"],
                         hatch="////", label="raw flip rate (does not survive\n"
                                             "projection onto valid one-hot)"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=9, labelcolor=INK_2, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("All methods measured by the SAME classifier per dataset "
                 "(shared GCN judge)", color=INK, fontsize=12.5, x=0.011,
                 ha="left", y=1.0)
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_dir}/shared_judge_comparison.{ext}", dpi=200,
                    facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def figure_two(d: pd.DataFrame, out_dir: str):
    ucx = d[d.method == "UCExplainer"].copy()
    ucx["run"] = ucx.dataset.map(DS_LABEL) + "\n" + ucx.judge + " judge"
    ucx["_ds"] = ucx.dataset.map({d: i for i, d in enumerate(DATASETS)})
    ucx = ucx.sort_values(["_ds", "judge"])

    fig, ax = plt.subplots(figsize=(9, 4.2), facecolor=SURFACE)
    _style(ax)
    x = np.arange(len(ucx))
    w = 0.26
    bars = [
        ("malformed node vectors", "oh_node", "#2a78d6"),
        ("malformed bond vectors", "oh_edge", "#eb6834"),
        ("fully valid one-hot CF", "struct_ok", "#1baf7a"),
    ]
    for i, (label, col, c) in enumerate(bars):
        vals = ucx[col].to_numpy(dtype=float)
        ax.bar(x - w + i * w, vals, width=w * 0.9, color=c, label=label, zorder=3)
        for xx, v in zip(x - w + i * w, vals):
            if not np.isnan(v):
                ax.text(xx, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8,
                        color=INK_2, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(ucx["run"], color=INK, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("share", color=INK_2, fontsize=9)
    ax.set_title("UCExplainer: how many counterfactuals are well-formed graphs?",
                 color=INK, fontsize=12, loc="left", pad=12)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, ncol=3,
              loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_dir}/ucexplainer_onehot.{ext}", dpi=200,
                    facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--res_dir", required=True)
    p.add_argument("--out_dir", required=True)
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    d = _load(a.res_dir)
    print(d.to_string(index=False))
    figure_one(d, a.out_dir)
    figure_two(d, a.out_dir)
    d.to_csv(f"{a.out_dir}/plot_data.csv", index=False)
    print(f"\nwrote {a.out_dir}/shared_judge_comparison.{{png,pdf}}")
    print(f"wrote {a.out_dir}/ucexplainer_onehot.{{png,pdf}}")
    print(f"wrote {a.out_dir}/plot_data.csv")


if __name__ == "__main__":
    main()
