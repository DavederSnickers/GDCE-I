"""Make GDCE's `src.*` importable alongside GRETEL's `src.*`.

Both projects use a top-level package named `src`. GRETEL's `src` is a
*namespace* package (no __init__.py); GDCE's `src` is a *regular* package
(empty __init__.py). Their submodule names do NOT collide
(GRETEL: dataset/oracle/explainer/utils/...; GDCE: models/datasets/metrics/...),
so we can serve both under one `src` namespace by appending GDCE's src
directory to the already-imported GRETEL namespace package's __path__.

This must run AFTER GRETEL's `src` is first imported (so `src` is resolved as a
namespace package) and BEFORE any `from src.models...` / `from src.datasets...`
(GDCE) import. Do NOT add gdce_path to sys.path for this purpose — that could
let GDCE's regular `src` package shadow GRETEL's namespace package.
"""

import os


def ensure_gdce_on_src_path(gdce_path: str) -> None:
    import src  # GRETEL namespace package (already in sys.modules)
    gdce_src = os.path.join(gdce_path, 'src')
    if not os.path.isdir(gdce_src):
        raise FileNotFoundError(f"GDCE src not found: {gdce_src}")
    # NOTE: src.__path__ is a _NamespacePath, not a plain list. Appending to it is
    # NOT durable: when sys.path later changes (GDCE's dataset / hydra imports do
    # this), the _NamespacePath recalculates itself from the parent path and DROPS
    # the appended gdce_src, so a subsequent `from src.models...` (the GDCE oracle
    # build, which runs after dataset loading) fails with ModuleNotFoundError.
    # Freezing __path__ into a plain list makes the GDCE entry survive sys.path
    # mutations. Idempotent: a later call sees gdce_src already present and skips.
    current = list(src.__path__)
    if gdce_src not in current:
        src.__path__ = current + [gdce_src]
