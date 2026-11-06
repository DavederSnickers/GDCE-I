"""Decode TWITTER counterfactuals back into words: which word became which?

A TWITTER graph is a tweet -- nodes are words, edges their co-occurrence -- so a
counterfactual that relabels a node is literally a word substitution. The metrics
(GED, mod_rate) cannot show that: GDCE-I's TWITTER run flips with GED 0.15, i.e.
almost no edge is touched and essentially the whole effect sits in the node
labels. This script reads a cf_dump.jsonl and reports the substitutions.

Index chain, all three steps are needed and none of them is the identity:

    cf node_type  --node_type_map.npy-->  TU node label  --readme.txt-->  word

`node_type_map.npy` holds the sorted TU labels that survived the >=500 frequency
filter (238 of 1323), so its position IS the remapped index. The word list lives
in the "Node Label Conversion" block of the TUDataset readme.

Usage:

    python src/shared_judge/twitter_word_flips.py \
        --dump <path/to/cf_dump.jsonl> \
        --data_root data \
        --out results_word_flips/<label>

Writes <out>_pairs.csv (one row per substituted word) and <out>_summary.csv
(aggregated by from->to and direction), and prints the top substitutions.
"""

from __future__ import annotations

import argparse
import ast
import csv
import glob
import json
import os
import sys
from collections import Counter, defaultdict

csv.field_size_limit(min(sys.maxsize, 2 ** 31 - 1))


def load_word_map(readme_path):
    """Parse the readme's 'Node Label Conversion' block -> {label:int -> word:str}.

    Lines look like "\t0\tapple". Component headers are ignored: TWITTER has a
    single component, and its labels are contiguous 0..1322, which is what makes
    the label equal to PyG's one-hot column index in the first place.
    """
    words = {}
    in_block = False
    with open(readme_path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if "Node Label Conversion" in line:
                in_block = True
                continue
            if not in_block:
                continue
            if line.startswith("==="):
                break
            parts = line.rstrip("\n").split("\t")
            # ['', '0', 'apple'] for a mapping line; anything else is prose.
            if len(parts) >= 3 and parts[0] == "" and parts[1].strip().isdigit():
                words[int(parts[1])] = parts[2].strip()
    if not words:
        raise ValueError(f"no word mappings found in {readme_path}")
    return words


def find_readme(data_root):
    pat = os.path.join(data_root, "twitter", "raw", "TU", "*", "raw", "readme.txt")
    hits = sorted(glob.glob(pat))
    if not hits:
        raise FileNotFoundError(f"no TU readme under {pat}")
    return hits[0]


def _read_node_types(path):
    out = []
    with open(path) as fh:
        for row in csv.DictReader(fh):
            out.append([int(t) for t in ast.literal_eval(row["node_types"])])
    return out


def load_test_node_types(data_root):
    """Row order of data/twitter/raw/test.csv IS the test-split index."""
    return _read_node_types(os.path.join(data_root, "twitter", "raw", "test.csv"))


def split_offset(data_root):
    """len(train) + len(val) -- the offset a global dataset index carries."""
    n = 0
    for split in ("train", "val"):
        with open(os.path.join(data_root, "twitter", "raw", f"{split}.csv")) as fh:
            n += sum(1 for _ in csv.DictReader(fh))
    return n


def detect_offset(records, test_types, candidates):
    """Pick the index offset under which the dumps' n_nodes actually match.

    The cf_dump contract says data_idx indexes the test split, but COMBINEX
    writes a global dataset index (train+val+test) instead. Guessing wrong does
    not fail loudly -- it compares against a different tweet and invents word
    substitutions -- so the offset is verified against n_nodes rather than
    assumed, and an ambiguous result is an error.
    """
    scores = {}
    for off in candidates:
        ok = 0
        for rec in records:
            i = int(rec["data_idx"]) - off
            if 0 <= i < len(test_types) and len(test_types[i]) == rec["n_nodes"]:
                ok += 1
        scores[off] = ok / len(records) if records else 0.0
    best = max(scores, key=scores.get)
    return best, scores


def load_type_map(data_root):
    import numpy as np
    return [int(v) for v in np.load(os.path.join(data_root, "twitter", "raw",
                                                 "node_type_map.npy"))]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dump", required=True, help="cf_dump.jsonl (dataset must be twitter)")
    p.add_argument("--data_root", default="data")
    p.add_argument("--out", required=True, help="output prefix, e.g. results_word_flips/gdce")
    p.add_argument("--top", type=int, default=25)
    p.add_argument("--index_offset", type=int, default=None,
                   help="subtract this from data_idx; default: auto-detect "
                        "between 0 (test-split index) and len(train)+len(val) "
                        "(global index, as COMBINEX writes)")
    p.add_argument("--min_match", type=float, default=0.95,
                   help="required fraction of records whose n_nodes matches "
                        "under the chosen offset (default 0.95)")
    args = p.parse_args()

    type_map = load_type_map(args.data_root)
    words = load_word_map(find_readme(args.data_root))
    test_types = load_test_node_types(args.data_root)
    print(f"vocabulary: {len(words)} TU labels, {len(type_map)} kept types, "
          f"{len(test_types)} test graphs")

    with open(args.dump) as fh:
        records = [json.loads(l) for l in fh if l.strip()]
    records = [r for r in records if r.get("dataset") == "twitter"]
    if not records:
        raise SystemExit(f"no twitter records in {args.dump}")

    if args.index_offset is not None:
        offset = args.index_offset
        _, scores = detect_offset(records, test_types, [offset])
    else:
        offset, scores = detect_offset(
            records, test_types, [0, split_offset(args.data_root)])
    print(f"index offset: {offset}  (n_nodes match rate per candidate: "
          + ", ".join(f"{o}:{s:.1%}" for o, s in sorted(scores.items())) + ")")
    if scores[offset] < args.min_match:
        raise SystemExit(
            f"ERROR: only {scores[offset]:.1%} of records line up with the test "
            f"split under offset {offset} (need {args.min_match:.0%}). Refusing "
            f"to report substitutions against graphs that are probably not the "
            f"originals.")

    def word_of(remapped_idx):
        tu_label = type_map[remapped_idx]
        return words.get(tu_label, f"<label {tu_label}>")

    pair_rows = []
    subs = Counter()                       # (from_word, to_word) -> count
    subs_by_dir = defaultdict(Counter)     # "0->1" -> Counter
    n_records = n_usable = n_identical = 0
    n_subs_per_cf = []
    skipped = Counter()

    for rec in records:
        n_records += 1
        idx = int(rec["data_idx"]) - offset
        if not 0 <= idx < len(test_types):
            skipped["data_idx out of range"] += 1
            continue
        orig = test_types[idx]
        cf = [int(v) for v in rec["node_types"]]
        if len(orig) != len(cf):
            # Node count must match position-wise or the comparison is meaningless.
            skipped[f"n_nodes mismatch ({len(orig)} vs {len(cf)})"] += 1
            continue
        n_usable += 1

        direction = f"{int(rec['source_class'])}->{int(rec['target_class'])}"
        changed = [(pos, o, c) for pos, (o, c) in enumerate(zip(orig, cf)) if o != c]
        n_subs_per_cf.append(len(changed))
        if not changed:
            n_identical += 1
        for pos, o, c in changed:
            w_from, w_to = word_of(o), word_of(c)
            subs[(w_from, w_to)] += 1
            subs_by_dir[direction][(w_from, w_to)] += 1
            pair_rows.append({
                "method": rec.get("method", ""), "variant": rec.get("variant", ""),
                "run": rec.get("run", -1), "seed": rec.get("seed", -1),
                "sample_idx": rec.get("sample_idx", -1), "data_idx": idx,
                "direction": direction, "node_pos": pos,
                "from_type": o, "to_type": c,
                "from_word": w_from, "to_word": w_to,
                "n_nodes": len(orig), "n_edges_cf": len(rec.get("edges", [])),
            })

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    pairs_csv = f"{args.out}_pairs.csv"
    fields = ["method", "variant", "run", "seed", "sample_idx", "data_idx",
              "direction", "node_pos", "from_type", "to_type", "from_word",
              "to_word", "n_nodes", "n_edges_cf"]
    with open(pairs_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(pair_rows)

    summary_csv = f"{args.out}_summary.csv"
    with open(summary_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["direction", "from_word", "to_word", "count"])
        for (a, b), n in subs.most_common():
            w.writerow(["ALL", a, b, n])
        for direction in sorted(subs_by_dir):
            for (a, b), n in subs_by_dir[direction].most_common():
                w.writerow([direction, a, b, n])

    mean_subs = (sum(n_subs_per_cf) / len(n_subs_per_cf)) if n_subs_per_cf else float("nan")
    print(f"\nrecords={n_records}  usable={n_usable}  "
          f"unchanged node labels={n_identical}")
    print(f"word substitutions: {len(pair_rows)} total, "
          f"{mean_subs:.2f} per counterfactual, {len(subs)} distinct pairs")
    for reason, n in skipped.most_common():
        print(f"  [skip] {reason}: {n}")

    print(f"\nTop {args.top} substitutions (all directions):")
    for (a, b), n in subs.most_common(args.top):
        print(f"  {n:5d}  {a} -> {b}")
    for direction in sorted(subs_by_dir):
        print(f"\nTop {args.top} for {direction}:")
        for (a, b), n in subs_by_dir[direction].most_common(args.top):
            print(f"  {n:5d}  {a} -> {b}")

    print(f"\nwrote {pairs_csv}")
    print(f"wrote {summary_csv}")


if __name__ == "__main__":
    main()
