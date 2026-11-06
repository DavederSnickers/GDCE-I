"""GRETEL `Oracle` wrappers around GDCE's classifier checkpoints.

These let RSGG-CE (GraphCounteRGANExplainer) query the *same* classifier GDCE /
C2Explainer use, so the resulting metrics are comparable. The heavy lifting
(checkpoint loading, bond-type reconstruction for Mutag, self-loop handling for
NCI1) is delegated to the battle-tested C2 wrappers
(`mutag_gine_wrapper.MutagGINEWrapperC2`, `nci1_gin_wrapper.NCI1GINWrapperC2`).

RSGG-CE feeds the oracle topology-only DataInstances (built from an adjacency
matrix). To classify them with a feature-dependent GNN we need node (and, for
Mutag, bond) features:

  * INFERENCE (per evaluation sample): call `set_current_graph(orig_instance)`
    before `explain()`. The original molecule's atom features + bond lookup are
    cached. Since RSGG-CE only edits the *edge set* on a fixed node set, both the
    original and every candidate counterfactual share these node features, so the
    classification is faithful. New edges get the default single bond (Mutag).

  * TRAINING (RSGG `__fit`): no per-sample context is set. The oracle then uses
    uniform node features + default bonds (feature-agnostic). This only scales the
    GAN's discriminator loss; counterfactual *selection* and all *reported* metrics
    re-classify with the real classifier under the correct per-sample context.
"""

import numpy as np
import torch
import torch.nn.functional as F

from src.oracle.oracle_base import Oracle


class _GDCEOracleBase(Oracle):

    def __init__(self, id, ckpt_path, node_dim, oracle_store_path='', config_dict=None):
        super().__init__(id, oracle_store_path, config_dict)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.node_dim = int(node_dim)
        self._wrapper = self._build_wrapper(ckpt_path)
        self._wrapper.to(self.device).eval()
        self._cache = None   # {'x': Tensor(n_real, node_dim), 'n_real': int}

    # ---- subclass hooks ----
    def _build_wrapper(self, ckpt_path):
        raise NotImplementedError

    def _prepare_context(self, instance, has_cache: bool):
        """Set up per-forward classifier state (e.g. Mutag bond lookup)."""
        pass

    def _finalize_edges(self, edge_index: torch.Tensor, n_real: int) -> torch.Tensor:
        return edge_index

    # ---- per-sample context (inference) ----
    def set_current_graph(self, instance) -> None:
        self._cache = {
            'x': instance.gdce_x.to(self.device).float(),
            'n_real': int(instance.n_real_nodes),
            'instance': instance,
        }

    def clear_current_graph(self) -> None:
        self._cache = None

    # ---- helpers ----
    def _resolve(self, instance):
        if self._cache is not None:
            return self._cache['x'], self._cache['n_real'], True
        n_real = int(getattr(instance, 'n_real_nodes', instance.graph.number_of_nodes()))
        x = torch.full((n_real, self.node_dim), 1.0 / self.node_dim,
                       dtype=torch.float, device=self.device)
        return x, n_real, False

    def _edges_from_instance(self, instance, n_real: int) -> torch.Tensor:
        adj = instance.to_numpy_array()[:n_real, :n_real]
        src, dst = np.nonzero(np.triu(adj, k=1))           # molecular edges, no self-loops
        if len(src) == 0:
            return torch.empty((2, 0), dtype=torch.long, device=self.device)
        ei = np.stack([np.concatenate([src, dst]), np.concatenate([dst, src])])
        return torch.tensor(ei, dtype=torch.long, device=self.device)

    # ---- GRETEL Oracle interface ----
    def embedd(self, instance):
        x, n_real, has_cache = self._resolve(instance)
        self._prepare_context(instance, has_cache)
        ei = self._edges_from_instance(instance, n_real)
        ei = self._finalize_edges(ei, n_real)
        with torch.no_grad():
            logits = self._wrapper(x, ei)
        return logits

    def _real_predict(self, embedding):
        return int(embedding.argmax(dim=-1).item())

    def _real_predict_proba(self, embedding):
        return F.softmax(embedding, dim=-1).reshape(-1).detach().cpu().numpy()

    def fit(self, dataset, split_i=0):
        pass            # pretrained GDCE checkpoint; nothing to train

    def read_oracle(self, oracle_name=None):
        pass

    def write_oracle(self):
        pass


class GDCEMutagOracle(_GDCEOracleBase):
    """Wraps MutagGINELightning (GINEConv, 5 bond types)."""

    def __init__(self, id, ckpt_path, node_dim, oracle_store_path='', config_dict=None):
        super().__init__(id, ckpt_path, node_dim, oracle_store_path, config_dict)
        self._name = 'gdce_mutag_gine'

    def _build_wrapper(self, ckpt_path):
        from src.wrappers.mutag_gine_wrapper import MutagGINEWrapperC2
        return MutagGINEWrapperC2(ckpt_path)

    def _prepare_context(self, instance, has_cache: bool):
        if has_cache:
            orig = self._cache['instance']
            self._wrapper.set_orig_graph(
                orig.gdce_edge_index.to(self.device),
                orig.gdce_edge_attr.to(self.device),
                self._cache['n_real'])
        else:
            # training: empty lookup -> every edge gets the default single bond
            n_real = int(getattr(instance, 'n_real_nodes', instance.graph.number_of_nodes()))
            empty = torch.empty((2, 0), dtype=torch.long, device=self.device)
            self._wrapper.set_orig_graph(
                empty, torch.empty((0, self._wrapper.edge_dim), device=self.device), n_real)

    def _finalize_edges(self, edge_index, n_real):
        # GINE classifier was trained with a self-loop on every node.
        sl = torch.arange(n_real, device=self.device)
        loops = torch.stack([sl, sl])
        return torch.cat([edge_index, loops], dim=1) if edge_index.numel() else loops


class GDCENci1Oracle(_GDCEOracleBase):
    """Wraps MutagGINLightning (plain GINConv; the NCI1_nofreq classifier)."""

    def __init__(self, id, ckpt_path, node_dim, oracle_store_path='', config_dict=None):
        super().__init__(id, ckpt_path, node_dim, oracle_store_path, config_dict)
        self._name = 'gdce_nci1_gin'

    def _build_wrapper(self, ckpt_path):
        from src.wrappers.nci1_gin_wrapper import NCI1GINWrapperC2
        return NCI1GINWrapperC2(ckpt_path)

    # _finalize_edges: identity -> NCI1GINWrapperC2.forward re-adds the full
    # self-loop diagonal it was trained on.
