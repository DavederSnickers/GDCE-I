#!/bin/bash
cd ${XPLORE_ROOT:?set XPLORE_ROOT} || exit 1
export APPTAINERENV_PYTHONPATH=${XPLORE_ROOT:?set XPLORE_ROOT}
R="apptainer run --bind /home/space:/home/space ${XPLORE_ROOT:?set XPLORE_ROOT}/vm/gretel.sif"
F='grep -viE warning|parameter expansion|gocryptfs|pkg_resources|Updates to this image|it/s]'
echo "######## BASELINE: variant=xplore (edges only) ########"
$R python -m src.wrappers.xplore_trace --config configs/mutagenicity_mol.xplore.yaml --variant xplore --n 4 --K 60 --alpha 1.0 --every 10 2>&1 | grep -viE 'warning|parameter expansion|gocryptfs|pkg_resources|Updates to this image|it/s\]'
echo ""; echo "######## variant=xplore++ , authors' sign (descent) ########"
$R python -m src.wrappers.xplore_trace --config configs/mutagenicity_mol.xplore.yaml --variant xplore++ --n 4 --K 60 --alpha 1.0 --every 10 2>&1 | grep -viE 'warning|parameter expansion|gocryptfs|pkg_resources|Updates to this image|it/s\]'
echo ""; echo "######## variant=xplore++ , sign-flipped (ascent) ########"
$R python -m src.wrappers.xplore_trace --config configs/mutagenicity_mol.xplore.yaml --variant xplore++ --n 4 --K 60 --alpha 1.0 --every 10 --signfix 2>&1 | grep -viE 'warning|parameter expansion|gocryptfs|pkg_resources|Updates to this image|it/s\]'
echo TRACE_DONE
