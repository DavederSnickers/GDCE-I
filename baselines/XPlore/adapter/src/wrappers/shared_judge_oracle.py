"""GRETEL `Oracle` around the shared judge every method in the comparison targets.

The judge is a plain GCN over a binary adjacency: no bond types, no self-loop
diagonal. That makes it a much simpler oracle than the GDCE checkpoints wrapped
in `gdce_oracle.py` -- there is no bond lookup to maintain and no self-loop to
re-add -- so this class only has to translate GRETEL's `DataInstance` (an
adjacency matrix over padded nodes) into the `(x, edge_index)` the judge wants.

Node features come from the instance itself. RSGG-CE edits only the edge set on
a fixed node set, so the original's features are valid for every candidate
counterfactual, exactly as in `gdce_oracle`. During the GAN's training pass no
per-sample context is set and uniform features are used; that only scales the
discriminator loss, since every reported number is re-classified under the
correct context.
"""

import numpy as np
import torch
import torch.nn.functional as F

from src.oracle.oracle_base import Oracle


class SharedJudgeOracle(Oracle):

    def __init__(self, id, judge_prefix, node_dim, oracle_store_path='',
                 config_dict=None):
        super().__init__(id, oracle_store_path, config_dict)
        from shared_judge import load_judge   # GDCE's src/shared_judge, on sys.path

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.node_dim = int(node_dim)
        self.judge = load_judge(str(judge_prefix), device=self.device)
        self.judge.eval()
        for p in self.judge.parameters():
            p.requires_grad_(False)
        self._cache = None
        self._name = 'shared_judge'

    # ---- per-sample context (inference) ----
    def set_current_graph(self, instance) -> None:
        self._cache = {'x': instance.gdce_x.to(self.device).float(),
                       'n_real': int(instance.n_real_nodes)}

    def clear_current_graph(self) -> None:
        self._cache = None

    def _resolve(self, instance):
        if self._cache is not None:
            return self._cache['x'], self._cache['n_real']
        n_real = int(getattr(instance, 'n_real_nodes',
                             instance.graph.number_of_nodes()))
        x = torch.full((n_real, self.node_dim), 1.0 / self.node_dim,
                       dtype=torch.float, device=self.device)
        return x, n_real

    def _edges_from_instance(self, instance, n_real: int) -> torch.Tensor:
        adj = instance.to_numpy_array()[:n_real, :n_real]
        src, dst = np.nonzero(np.triu(adj, k=1))
        if len(src) == 0:
            return torch.empty((2, 0), dtype=torch.long, device=self.device)
        ei = np.stack([np.concatenate([src, dst]), np.concatenate([dst, src])])
        return torch.tensor(ei, dtype=torch.long, device=self.device)

    # ---- GRETEL Oracle interface ----
    def embedd(self, instance):
        x, n_real = self._resolve(instance)
        ei = self._edges_from_instance(instance, n_real)
        with torch.no_grad():
            # batch=None means "one graph"; the judge carries no edge features.
            logits = self.judge(x, ei, None, edge_attr=None, edge_weights=None)
        return logits

    def _real_predict(self, embedding):
        return int(embedding.argmax(dim=-1).item())

    def _real_predict_proba(self, embedding):
        return F.softmax(embedding, dim=-1).reshape(-1).detach().cpu().numpy()

    def fit(self, dataset, split_i=0):
        pass            # pretrained, frozen

    def read_oracle(self, oracle_name=None):
        pass

    def write_oracle(self):
        pass
