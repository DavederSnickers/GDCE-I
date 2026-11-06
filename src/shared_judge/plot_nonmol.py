"""Figures for the non-molecular benchmark (PROTEINS, TWITTER).

Three figures, each answering one question:

  1. flip_and_robustness — does the method flip the judge, and does the flip
     survive an independently trained control judge?
  2. change_profile — WHERE does a method change the graph: node labels or
     edges? This is what separates the edge-only methods (C2, CF2, D4) from the
     ones that can rewrite node features (GDCE-I, UCX), and it explains the
     TWITTER results better than any aggregate number.
  3. twitter_word_shortcut — TWITTER nodes are words, so a counterfactual is a
     word substitution. Plotting the target-word distribution shows that one
     flip direction collapses onto a single token.

Colour semantics differ per figure and are stated in each: figures 1 uses hue for
the dataset, figure 2 for the changed component, figure 3 is single-hue. The
palette slots are the validated categorical 1/2 for the light surface, the same
ones src/shared_judge/plot_results.py already uses, so a method keeps its meaning
across the paper.

Two conventions carried over from plot_results.py, both load-bearing:
  * UCX's flip rate is plotted as projected (solid) + raw (hatched extension).
    Its raw rate counts counterfactuals whose node vectors are not one-hot; the
    other methods cannot produce those, so the raw numbers are not comparable.
  * The AdversarialFlipRate panel draws the judge-agreement floor per dataset.
    Without it a transfer rate is uninterpretable.

    python src/shared_judge/plot_nonmol.py \
        --summary_dir <dir with {method}_{dataset}.csv> \
        --adv_csv <adv_flip summary> \
        --words_csv <word-flip summary> \
        --out_dir comparison_results/figures
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# --- design tokens (validated categorical slots 1-2, light surface) -----------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_3 = "#8a8983"
GRID = "#dedddaff"
BLUE = "#2a78d6"
ORANGE = "#eb6834"

DATASETS = ["proteins", "twitter"]
DS_LABEL = {"proteins": "PROTEINS", "twitter": "TWITTER"}
DS_COLOR = {"proteins": BLUE, "twitter": ORANGE}
# Sample budget differs by dataset on purpose: PROTEINS has only 174 test graphs,
# so one run covers the whole split; TWITTER draws 5x100 out of 13006.
DS_BUDGET = {"proteins": "1 x 174 (full test split)", "twitter": "5 x 100"}

# Judge agreement on untouched test graphs -- the floor a transfer rate is read
# against. Below it, counterfactuals are more model-specific than the two judges
# are to each other. From src/judge_agreement.py.
FLOOR = {"proteins": 0.9598, "twitter": 0.8855}

METHODS = ["GDCE-I", "UCX", "CF2", "C2", "D4"]
METHOD_LABEL = {"GDCE-I": "GDCE-I", "UCX": "UCExplainer", "CF2": "CF²",
                "C2": "C2Explainer", "D4": "D4Explainer"}
# file stem in --summary_dir -> method key
FILE_METHOD = {"gdce": "GDCE-I", "ucx": "UCX", "cf2": "CF2", "c2": "C2",
               "d4modrate": "D4"}
# label prefix in the adv-flip summary -> method key
ADV_METHOD = {"GDCE": "GDCE-I", "UCX": "UCX", "CF2": "CF2", "C2": "C2",
              "D4modrate": "D4", "D4flip": None}


def _f(row, key):
    v = row.get(key)
    if v in (None, "", "nan", "NaN"):
        return float("nan")
    return float(v)


def _mean(rows, key):
    vs = [_f(r, key) for r in rows]
    vs = [v for v in vs if v == v]
    return st.mean(vs) if vs else float("nan")


def _std(rows, key):
    vs = [_f(r, key) for r in rows]
    vs = [v for v in vs if v == v]
    return st.stdev(vs) if len(vs) > 1 else 0.0


def load_summaries(summary_dir):
    """{(method, dataset): {...}} from the per-run summary CSVs."""
    out = {}
    for stem, method in FILE_METHOD.items():
        for ds in DATASETS:
            path = os.path.join(summary_dir, f"{stem}_{ds}.csv")
            if not os.path.exists(path):
                print(f"  skip (missing): {path}")
                continue
            with open(path) as fh:
                rows = list(csv.DictReader(fh))
            if not rows:
                print(f"  skip (empty): {path}")
                continue
            rec = {
                "accuracy": _mean(rows, "accuracy"),
                "accuracy_std": _std(rows, "accuracy"),
                "node_sparsity": _mean(rows, "avg_node_sparsity"),
                "edge_sparsity": _mean(rows, "avg_edge_sparsity"),
                "n_success": sum(int(r["n_success"]) for r in rows),
                "n_total": sum(int(r["n_total"]) for r in rows),
            }
            # Only COMBINEX reports a projected rate; for the others the raw rate
            # already IS the projected one (they never leave the one-hot manifold).
            if "projected_accuracy" in rows[0]:
                rec["projected_accuracy"] = _mean(rows, "projected_accuracy")
            out[(method, ds)] = rec
    return out


def load_adv(adv_csv):
    """{(method, dataset): {...}} from the AdversarialFlipRate summary."""
    out = {}
    if not os.path.exists(adv_csv):
        print(f"  skip (missing): {adv_csv}")
        return out
    with open(adv_csv) as fh:
        for r in csv.DictReader(fh):
            ds = r["dataset"]
            if ds not in DATASETS:
                continue
            prefix = r["label"].split("/")[0].replace("_proteins", "").replace("_twitter", "")
            method = ADV_METHOD.get(prefix)
            if method is None:
                continue
            out[(method, ds)] = {
                "adv": _f(r, "adversarial_flip_rate"),
                "transfer": _f(r, "transfer_rate"),
                "transfer_std": _f(r, "transfer_std"),
                "gate": _f(r, "fidelity_gate"),
                "n_confirmed": int(r["n_confirmed"]),
            }
    return out


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8, length=0)


def _bars(ax, values, present, width=0.38):
    """Grouped bars, one group per method, one bar per dataset."""
    xs = range(len(METHODS))
    for k, ds in enumerate(DATASETS):
        off = (k - 0.5) * (width + 0.02)
        for i, m in enumerate(METHODS):
            v = values.get((m, ds))
            if v is None or v != v:
                # Absent is not zero. Say so, otherwise the reader reads a value.
                ax.text(i + off, 0.004, "n/a", ha="center", va="bottom",
                        fontsize=6, color=INK_3, rotation=90)
                continue
            ax.bar(i + off, v, width, color=DS_COLOR[ds], zorder=3,
                   linewidth=0.8, edgecolor=SURFACE)
            if v == 0:
                # A zero-height bar is invisible; the reader cannot tell it from
                # a missing one, and the two mean very different things here.
                ax.text(i + off, 0.004, "0", ha="center", va="bottom",
                        fontsize=7, color=DS_COLOR[ds], weight="bold")
    ax.set_xticks(list(xs))
    ax.set_xticklabels([METHOD_LABEL[m] for m in METHODS], fontsize=8,
                       color=INK_2, rotation=18, ha="right")


def fig_flip_and_robustness(summ, adv, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))
    fig.patch.set_facecolor(SURFACE)

    # -- panel A: flip rate ---------------------------------------------------
    ax = axes[0]
    _style(ax)
    width = 0.38
    for k, ds in enumerate(DATASETS):
        off = (k - 0.5) * (width + 0.02)
        for i, m in enumerate(METHODS):
            rec = summ.get((m, ds))
            if rec is None:
                ax.text(i + off, 0.012, "n/a", ha="center", va="bottom",
                        fontsize=6.5, color=INK_3, rotation=90)
                continue
            raw = rec["accuracy"]
            proj = rec.get("projected_accuracy", raw)
            ax.bar(i + off, proj, width, color=DS_COLOR[ds], zorder=3,
                   linewidth=0.8, edgecolor=SURFACE)
            if proj + 1e-9 < raw:
                # Hatched extension = the part that does not survive projection.
                ax.bar(i + off, raw - proj, width, bottom=proj, zorder=3,
                       facecolor="none", edgecolor=DS_COLOR[ds], hatch="////",
                       linewidth=0.8)
            elif raw == 0:
                ax.text(i + off, 0.008, "0", ha="center", va="bottom",
                        fontsize=7, color=DS_COLOR[ds], weight="bold")
    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels([METHOD_LABEL[m] for m in METHODS], fontsize=8,
                       color=INK_2, rotation=18, ha="right")
    ax.set_ylim(0, 1.16)
    ax.set_ylabel("flip rate", fontsize=9, color=INK)
    ax.set_title("A  Counterfactuals that flip the judge", fontsize=9.5,
                 color=INK, loc="left", pad=10)
    ax.legend(handles=[
        Patch(facecolor="none", edgecolor=INK_2, hatch="////",
              label="raw flip, lost on projection to one-hot"),
    ], fontsize=7, frameon=False, loc="upper right", labelcolor=INK_2,
        handlelength=1.6, borderpad=0.1)

    # -- panel B: adversarial flip rate ---------------------------------------
    ax = axes[1]
    _style(ax)
    _bars(ax, {k: v["adv"] for k, v in adv.items()}, adv)
    for ds in DATASETS:
        # The floor is 1 - agreement: the AdvFlipRate two judges already show on
        # untouched graphs. Below it a method is indistinguishable from noise.
        ax.axhline(1 - FLOOR[ds], color=DS_COLOR[ds], linewidth=1.2,
                   linestyle=(0, (4, 3)), zorder=2)
        # Right edge: the D4 column is the emptiest part of the panel.
        ax.text(len(METHODS) - 0.38, 1 - FLOOR[ds] + 0.006,
                f"floor {DS_LABEL[ds]}", fontsize=6.5, color=DS_COLOR[ds],
                va="bottom", ha="right")
    ax.set_xlim(-0.62, len(METHODS) - 0.35)
    ax.set_ylim(0, 0.40)
    ax.set_ylabel("AdversarialFlipRate  (lower is better)", fontsize=9, color=INK)
    ax.set_title("B  Flips that a control judge does not confirm", fontsize=9.5,
                 color=INK, loc="left", pad=10)
    # UCX's bar is conditioned on a 26% fidelity gate -- flag it on the mark.
    for k, ds in enumerate(DATASETS):
        rec = adv.get(("UCX", ds))
        if rec and rec["gate"] == rec["gate"] and rec["gate"] < 0.9:
            ax.text(METHODS.index("UCX") + (k - 0.5) * 0.40,
                    rec["adv"] + 0.008, f"gate\n{rec['gate']:.2f}", fontsize=6,
                    color=INK_3, ha="center", va="bottom", linespacing=0.95)

    # One shared legend for both panels: hue means the same thing in each.
    fig.legend(handles=[Patch(facecolor=BLUE, label=DS_LABEL["proteins"]),
                        Patch(facecolor=ORANGE, label=DS_LABEL["twitter"])],
               fontsize=8, frameon=False, labelcolor=INK_2, ncol=2,
               loc="lower left", bbox_to_anchor=(0.005, 0.885))
    fig.text(0.005, 0.015,
             "PROTEINS: 1 run x 174 = full test split (no run-to-run spread).  "
             "TWITTER: 5 runs x 100 of 13006.  Shared GCN judge (seed 0); "
             "control judge = same recipe, seed 1.",
             fontsize=6.8, color=INK_3)
    fig.tight_layout(rect=(0, 0.045, 1, 0.885))
    _save(fig, out_dir, "nonmol_flip_and_robustness")


def fig_change_profile(summ, out_dir):
    """Where the edit lands: node labels vs edges."""
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    width = 0.38
    for j, ds in enumerate(DATASETS):
        ax = axes[j]
        _style(ax)
        for i, m in enumerate(METHODS):
            rec = summ.get((m, ds))
            if rec is None:
                ax.text(i, 0.008, "n/a", ha="center", va="bottom", fontsize=6.5,
                        color=INK_3, rotation=90)
                continue
            if rec["n_success"] == 0:
                # No counterfactual at all -- there is nothing to decompose, and
                # a bare gap would read as "changes nothing", which is the
                # opposite of what happened.
                ax.text(i, 0.008, "no flips", ha="center", va="bottom",
                        fontsize=6.5, color=INK_3, rotation=90)
                continue
            for off, key, col in ((-0.5, "node_sparsity", BLUE),
                                  (0.5, "edge_sparsity", ORANGE)):
                v = rec[key]
                x = i + off * (width + 0.02)
                if v != v:
                    # Not reported by that method's eval script -- distinct from
                    # a reported zero, and the reader must not conflate them.
                    ax.text(x, 0.008, "n.r.", ha="center", va="bottom",
                            fontsize=6, color=INK_3, rotation=90)
                    continue
                ax.bar(x, v, width, color=col, zorder=3, linewidth=0.8,
                       edgecolor=SURFACE)
                if v == 0:
                    # An exact zero is the finding here (edge-only methods never
                    # touch node labels); an invisible bar would hide it.
                    ax.text(x, 0.008, "0", ha="center", va="bottom",
                            fontsize=7, color=col, weight="bold")
        ax.set_xticks(range(len(METHODS)))
        ax.set_xticklabels([METHOD_LABEL[m] for m in METHODS], fontsize=8,
                           color=INK_2, rotation=18, ha="right")
        ax.set_title(f"{DS_LABEL[ds]}", fontsize=9.5, color=INK, loc="left", pad=10)
        if j == 0:
            ax.set_ylabel("fraction changed", fontsize=9, color=INK)
            ax.legend(handles=[Patch(facecolor=BLUE, label="node labels"),
                               Patch(facecolor=ORANGE, label="edges")],
                      fontsize=7.5, frameon=False, loc="upper right",
                      labelcolor=INK_2)
    fig.suptitle("Where each method edits the graph", fontsize=10, color=INK,
                 x=0.008, ha="left", y=0.99)
    fig.text(0.005, 0.015,
             "C2, CF² and D4 only mask edges -- they cannot change node labels at all. "
             "On TWITTER, where the label lives in the words, that is why they fail.",
             fontsize=6.8, color=INK_3)
    fig.tight_layout(rect=(0, 0.05, 1, 0.945))
    _save(fig, out_dir, "nonmol_change_profile")


def fig_word_shortcut(words_csv, out_dir, top=12):
    """TWITTER target-word distribution per flip direction."""
    if not os.path.exists(words_csv):
        print(f"  skip (missing): {words_csv}")
        return
    with open(words_csv) as fh:
        rows = [r for r in csv.DictReader(fh) if r["direction"] != "ALL"]
    if not rows:
        print(f"  skip (no directional rows): {words_csv}")
        return
    by_dir = {}
    for r in rows:
        by_dir.setdefault(r["direction"], {})
        w = r["to_word"]
        by_dir[r["direction"]][w] = by_dir[r["direction"]].get(w, 0) + int(r["count"])

    # 0 = Positive, 1 = Negative (verified against the test-split word
    # distribution; the class names in the configs were inverted until 2026-08-05).
    DIR_LABEL = {"0->1": "Positive → Negative", "1->0": "Negative → Positive"}

    dirs = sorted(by_dir)
    fig, axes = plt.subplots(1, len(dirs), figsize=(9.6, 3.9))
    if len(dirs) == 1:
        axes = [axes]
    fig.patch.set_facecolor(SURFACE)
    for ax, d in zip(axes, dirs):
        counts = sorted(by_dir[d].items(), key=lambda kv: kv[1], reverse=True)[:top]
        total = sum(by_dir[d].values())
        labels = [w for w, _ in counts][::-1]
        vals = [100 * c / total for _, c in counts][::-1]
        ax.set_facecolor(SURFACE)
        ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=INK_2, labelsize=8, length=0)
        ax.barh(range(len(vals)), vals, height=0.62, color=BLUE, zorder=3)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8, color=INK_2)
        # Direct-label the leader only; a number on every bar is noise.
        ax.text(vals[-1] - 1.2, len(vals) - 1, f"{vals[-1]:.0f}%", va="center",
                ha="right", fontsize=8, color=SURFACE, zorder=4, weight="bold")
        ax.set_xlabel("share of substitutions into this word (%)", fontsize=8.5,
                      color=INK_2)
        ax.set_title(f"{DIR_LABEL.get(d, d)}   ({total} substitutions, "
                     f"{len(by_dir[d])} distinct targets)",
                     fontsize=9, color=INK, loc="left", pad=10)
        ax.set_xlim(0, max(60, max(vals) * 1.15))
    fig.suptitle("GDCE-I on TWITTER: which word the counterfactual writes",
                 fontsize=10, color=INK, x=0.008, ha="left", y=0.99)
    fig.text(0.005, 0.055,
             "A TWITTER node is a word, so a counterfactual is a word substitution. "
             "Both panels share the x scale.",
             fontsize=6.8, color=INK_3)
    fig.text(0.005, 0.015,
             "One direction collapses onto a single token -- and the "
             "AdversarialFlipRate does not see it, because the control judge "
             "learned the same shortcut.",
             fontsize=6.8, color=INK_3)
    fig.tight_layout(rect=(0, 0.095, 1, 0.945))
    _save(fig, out_dir, "twitter_word_shortcut")


def _save(fig, out_dir, stem):
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(out_dir, f"{stem}.{ext}")
        fig.savefig(path, dpi=200, facecolor=SURFACE)
        print(f"  wrote {path}")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--summary_dir", required=True)
    p.add_argument("--adv_csv", required=True)
    p.add_argument("--words_csv", required=True)
    p.add_argument("--out_dir", default="comparison_results/figures")
    args = p.parse_args()

    summ = load_summaries(args.summary_dir)
    adv = load_adv(args.adv_csv)
    print(f"loaded {len(summ)} summary cells, {len(adv)} adv-flip cells")

    fig_flip_and_robustness(summ, adv, args.out_dir)
    fig_change_profile(summ, args.out_dir)
    fig_word_shortcut(args.words_csv, args.out_dir)


if __name__ == "__main__":
    main()
