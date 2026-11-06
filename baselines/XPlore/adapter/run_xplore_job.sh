#!/bin/bash
# XPlore under the shared-judge protocol. PYTHONPATH is pinned to GRETEL here on
# purpose: GDCE's src/ is a regular package and would shadow GRETEL's namespace
# package of the same name, so inheriting a GDCE-flavoured PYTHONPATH from the
# submitting shell breaks 'import src.wrappers'. GDCE's modules are merged in
# later by gdce_bootstrap, which appends to src.__path__ instead.
cd ${XPLORE_ROOT:?set XPLORE_ROOT} || exit 1
export APPTAINERENV_PYTHONPATH=${XPLORE_ROOT:?set XPLORE_ROOT}
apptainer run --nv --bind /home/space:/home/space \
    ${XPLORE_ROOT:?set XPLORE_ROOT}/vm/gretel.sif python run_xplore_evaluation.py "$@"
