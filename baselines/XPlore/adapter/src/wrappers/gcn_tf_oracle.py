"""GRETEL's paper classifier (`gcn-tf`) as a GRETEL Oracle, trained on GDCE data.

This is the classifier RSGG-CE proposes in the GRETEL paper for the *molecular*
datasets (`config/optimus/gcountergan-bbbp` -> oracle `gcn-tf`, implemented in
`src/oracle/oracle_gcn_tf.py`): a 4-layer GCN (relu) -> mean-pool -> Dense(256)
-> Dense(1, sigmoid), trained with BCE / Adam.

Unlike GDCE's frozen GINE/GIN checkpoint, this oracle has NO pretrained weights:
it is trained here on GDCE's `mutagenicity_mol` / `nci1_nofreq` *train* split, so
the exact same graphs/labels/splits are used. We reuse GRETEL's exact
`GCNLayer`/`GRLayer` architecture, but feed node features from GDCE's per-node
one-hot (`gdce_x`) instead of GRETEL's uniform `to_numpy_arrays` path.

ENCODER/DECODER INVARIANT (Mutag SMILES):
    The classifier's node-feature columns, the SMILES `node_types`
    (= `gdce_x.argmax`) and `atom_decoder.json` all share ONE atom index space
    (GDCE's 13-type remap). Never introduce a second encoding (e.g. GRETEL's
    `GetAtomicNum`) -- otherwise the classifier and the SMILES validation would
    disagree on what an atom is. `rsgg_eval` asserts `len(atom_decoder) ==
    n_node_types` to keep this honest.

RSGG-CE is topology-only: it queries the oracle with adjacency-built
DataInstances. We cache the original molecule's `gdce_x` via `set_current_graph()`
before each `explain()`, so the original and every candidate counterfactual (same
fixed node set) classify with the true atom types. During the GAN's own training
(no per-sample context) features fall back to uniform.
"""

import os

import numpy as np
import tensorflow as tf

from src.oracle.oracle_base import Oracle


# GCNLayer / GRLayer copied verbatim from src/oracle/oracle_gcn_tf.py so we do NOT
# import that module (its top-level `import selfies` / `from rdkit.Chem.Draw import
# IPythonConsole` pull in heavy deps we never use). These only need tensorflow.
class GCNLayer(tf.keras.layers.Layer):
    """Implementation of GCN as layer (GRETEL paper)."""

    def __init__(self, activation=None, **kwargs):
        super(GCNLayer, self).__init__(**kwargs)
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        node_shape, adj_shape = input_shape
        self.w = self.add_weight(shape=(node_shape[2], node_shape[2]), name="w")

    def call(self, inputs):
        nodes, adj = inputs
        degree = tf.reduce_sum(adj, axis=-1)
        new_nodes = tf.einsum("bi,bij,bjk,kl->bil", 1 / degree, adj, nodes, self.w)
        out = self.activation(new_nodes)
        return out, adj


class GRLayer(tf.keras.layers.Layer):
    """Reduction layer: averages node features over the graph (GRETEL paper)."""

    def __init__(self, name="GRLayer", **kwargs):
        super(GRLayer, self).__init__(name=name, **kwargs)

    def call(self, inputs):
        nodes, adj = inputs
        reduction = tf.reduce_mean(nodes, axis=1)
        return reduction


class GCNTFOracle(Oracle):

    def __init__(self, id, n_node_types, max_n_nodes, oracle_store_path='',
                 epochs=30, config_dict=None):
        super().__init__(id, oracle_store_path, config_dict)
        self.name = 'gcn_tf'
        self.n_node_types = int(n_node_types)
        self.max_n_nodes = int(max_n_nodes)
        self.epochs = int(epochs)
        self._clf = None
        self._cache = None              # {'x': (n_real, k) float32, 'n_real': int}
        self._build_model()

    # ---- model (GRETEL paper architecture, src/oracle/oracle_gcn_tf.py) ----
    def _build_model(self):
        ninput = tf.keras.Input((None, self.n_node_types))
        ainput = tf.keras.Input((None, None))
        x = GCNLayer("relu")([ninput, ainput])
        x = GCNLayer("relu")(x)
        x = GCNLayer("relu")(x)
        x = GCNLayer("relu")(x)
        x = GRLayer()(x)
        x = tf.keras.layers.Dense(256)(x)
        x = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        self._clf = tf.keras.Model(inputs=(ninput, ainput), outputs=x)
        self._clf.compile(
            "adam",
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
            metrics=["accuracy"])

    # ---- per-sample context (inference) ----
    def set_current_graph(self, instance) -> None:
        self._cache = {
            'x': instance.gdce_x.detach().cpu().numpy().astype(np.float32),
            'n_real': int(instance.n_real_nodes),
        }

    def clear_current_graph(self) -> None:
        self._cache = None

    # ---- array builders (identical layout for train + inference) ----
    def _node_features(self, instance) -> np.ndarray:
        """(max_n_nodes, n_node_types). Real nodes carry gdce_x atom types; with no
        per-sample cache (GAN training) they fall back to uniform features."""
        N, k = self.max_n_nodes, self.n_node_types
        nodes = np.zeros((N, k), dtype=np.float32)
        if self._cache is not None:
            n_real = self._cache['n_real']
            nodes[:n_real] = self._cache['x'][:n_real]
        else:
            n_real = int(getattr(instance, 'n_real_nodes',
                                 instance.graph.number_of_nodes()))
            nodes[:n_real] = 1.0 / k
        return nodes

    def _adj(self, instance) -> np.ndarray:
        """(max_n_nodes, max_n_nodes) with self-loops (GCN needs degree >= 1)."""
        N = self.max_n_nodes
        adj = instance.to_numpy_array()[:N, :N].astype(np.float32)
        np.fill_diagonal(adj, 0.0)                 # strip any self-loops first
        return adj + np.eye(N, dtype=np.float32)

    # ---- GRETEL Oracle interface ----
    def embedd(self, instance):
        nodes = self._node_features(instance)
        adj = self._adj(instance)
        pred = self._clf((nodes[np.newaxis, ...], adj[np.newaxis, ...]),
                         training=False).numpy()
        return float(np.asarray(pred).reshape(-1)[0])   # P(class 1)

    def _real_predict(self, embedding):
        return int(float(embedding) > 0.5)

    def _real_predict_proba(self, embedding):
        p = float(embedding)
        return np.array([1.0 - p, p], dtype=np.float32)

    # ---- training on GDCE's train split (no pretrained checkpoint) ----
    def fit(self, dataset, split_i=0):
        oracle_name = f'{self.name}_fit_on_{dataset.name}'
        ckpt_dir = os.path.join(self._oracle_store_path, oracle_name)
        weights = os.path.join(ckpt_dir, 'oracle.ckpt')
        if os.path.isdir(ckpt_dir) and tf.io.gfile.glob(weights + '*'):
            self._clf.load_weights(weights)
            print(f"[gcn_tf] loaded cached oracle weights from {ckpt_dir}")
            return

        os.makedirs(ckpt_dir, exist_ok=True)
        train_ids = dataset.splits[split_i]['train']
        X_nodes, X_adj, Y = [], [], []
        for iid in train_ids:
            inst = dataset.instances[iid]
            self.set_current_graph(inst)            # gdce_x atom features
            X_nodes.append(self._node_features(inst))
            X_adj.append(self._adj(inst))
            Y.append(int(inst.graph_label))
        self.clear_current_graph()

        X_nodes = np.stack(X_nodes)
        X_adj = np.stack(X_adj)
        Y = np.asarray(Y, dtype=np.float32)
        print(f"[gcn_tf] training on {len(Y)} graphs "
              f"(pos={int(Y.sum())}, neg={int((1 - Y).sum())}) for {self.epochs} epochs...")
        self._clf.fit((X_nodes, X_adj), Y, batch_size=128, epochs=self.epochs, verbose=2)
        self._clf.save_weights(weights)
        print(f"[gcn_tf] saved oracle weights -> {ckpt_dir}")

    def read_oracle(self, oracle_name=None):
        name = oracle_name or self.name
        weights = os.path.join(self._oracle_store_path, name, 'oracle.ckpt')
        self._clf.load_weights(weights)

    def write_oracle(self):
        ckpt_dir = os.path.join(self._oracle_store_path, self.name)
        os.makedirs(ckpt_dir, exist_ok=True)
        self._clf.save_weights(os.path.join(ckpt_dir, 'oracle.ckpt'))
