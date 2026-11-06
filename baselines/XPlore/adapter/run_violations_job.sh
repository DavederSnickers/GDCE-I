#!/bin/bash
# Valence-violation breakdown for the molecular cf_dumps, XPlore vs RSGG-CE.
cd ${XPLORE_ROOT:?set XPLORE_ROOT} || exit 1
export APPTAINERENV_PYTHONPATH=${XPLORE_ROOT:?set XPLORE_ROOT}
run() {
  echo ""
  echo "############ $1 ############"
  apptainer run --nv --bind /home/space:/home/space \
    ${XPLORE_ROOT:?set XPLORE_ROOT}/vm/gretel.sif \
    python -m src.wrappers.smiles_violations --dump "$2" --config "$3" 2>&1 \
    | grep -viE 'warning|parameter expansion|gocryptfs|pkg_resources|Updates to this image'
}
for ds in mutagenicity_mol benzene; do
  run "xplore / $ds"  "xplore_${ds}_results/cf_dump.jsonl"        "configs/${ds}.xplore.yaml"
  run "rsgg-ce / $ds" "rsgg_${ds}_shared_results/cf_dump.jsonl"   "configs/${ds}.xplore.yaml"
done
echo "VIOLATIONS_DONE"
