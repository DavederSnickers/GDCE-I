#!/bin/bash
cd ${XPLORE_ROOT:?set XPLORE_ROOT} || exit 1
export APPTAINERENV_PYTHONPATH=${XPLORE_ROOT:?set XPLORE_ROOT}
for c in configs/mutagenicity_mol.xplore.yaml configs/benzene.xplore.yaml; do
  echo ""; echo "############ $c ############"
  apptainer run --nv --bind /home/space:/home/space ${XPLORE_ROOT:?set XPLORE_ROOT}/vm/gretel.sif \
    python -m src.wrappers.feature_channel_probe --config "$c" 2>&1 \
    | grep -viE 'warning|parameter expansion|gocryptfs|pkg_resources|Updates to this image|it/s\]'
done
echo PROBE_DONE
