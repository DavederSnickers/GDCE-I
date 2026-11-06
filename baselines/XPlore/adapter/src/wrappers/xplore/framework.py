"""The four pieces of GRETEL v2 that XPlore.py imports, reduced to what it uses.

XPlore ships inside GRETEL v2, whose `Explainer` base pulls in `Configurable` ->
`Context` -> `flufl.lock` + `jsonc_parser` + the whole JSONC config/factory
machinery. None of that survives contact with this repo (GRETEL v1) and none of
it is part of the algorithm, so the port supplies the four symbols directly and
leaves `XPlore.py` itself byte-identical below its import header.

What the real base class does, and what is kept here:

  * `Explainer.__init__` binds `self.dataset` and `self.oracle` out of the config
    dict, then `Configurable.__init__` calls `check_configuration()` followed by
    `init()`. XPlore relies on that order -- `check_configuration` fills in every
    default parameter that `init` then reads -- so the shim reproduces it exactly.
  * `retake_oracle(cfg)` is a one-line dict lookup in GRETEL v2 as well.
  * `BaseManipulator` and `NodeCentrality` are imported by XPlore but only ever
    touched through `type(m).__name__` over `self.dataset.manipulators`. We run
    with no manipulators (the shared judge reads one-hot node types, not
    centrality features), so the classes need to exist and nothing more.
"""

from types import SimpleNamespace


class Explainer:
    """GRETEL v2's `src.core.explainer_base.Explainer`, minus Context."""

    def __init__(self, context, local_config):
        self.context = context
        self.dataset = local_config['dataset']
        self.oracle = local_config['oracle']
        self.local_config = local_config
        self.check_configuration()
        self.init()

    def check_configuration(self):
        self.local_config['parameters'] = self.local_config.get('parameters', {})
        self.local_config['parameters']['fold_id'] = \
            self.local_config['parameters'].get('fold_id', -1)
        self.fold_id = self.local_config['parameters']['fold_id']

    def init(self):
        pass

    def explain(self, instance):
        raise NotImplementedError


def retake_oracle(cfg):
    return cfg['oracle']


class BaseManipulator:
    """Marker only -- see the module docstring."""


class NodeCentrality(BaseManipulator):
    """Marker only -- see the module docstring."""


def torch_diffusion_namespace():
    """A stand-in for the `src` package as XPlore dereferences it.

    The single use is `isinstance(self.oracle, src.oracle.nn.torch_diffusion.
    OracleTorch)`. `OracleTorch` here is a class nothing inherits from, so the
    test is always False and XPlore takes its ordinary (non-diffusion) path.
    """
    class OracleTorch:
        pass

    return SimpleNamespace(
        oracle=SimpleNamespace(
            nn=SimpleNamespace(
                torch_diffusion=SimpleNamespace(OracleTorch=OracleTorch))))
