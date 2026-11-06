#!/bin/bash
# Copy GDCE's preprocessed PROTEINS/TWITTER CSVs into D4Explainer's data tree.
#
# Every method in the comparison reads the same preprocessed CSVs and the same
# train/val/test split -- that, not a shared oracle, is where fairness comes from.
# Copies rather than symlinks, so a later re-preprocessing in GDCE cannot silently
# change what D4 was evaluated on.
#
# Run from the D4Explainer repo root ON THE CLUSTER:  bash prepare_d4_nonmol_data.sh

set -eu

GDCE=${GDCE_DATA:?set GDCE_DATA}

for ds in proteins twitter; do
    dst="data/GDCE_$(echo "$ds" | tr '[:lower:]' '[:upper:]')/raw"
    mkdir -p "$dst"
    for f in train.csv val.csv test.csv; do
        cp -v "${GDCE}/${ds}/raw/${f}" "${dst}/${f}"
    done
done

echo "=== done"
wc -l data/GDCE_PROTEINS/raw/*.csv data/GDCE_TWITTER/raw/*.csv
