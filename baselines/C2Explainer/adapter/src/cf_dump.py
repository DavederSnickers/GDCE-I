"""Canonical counterfactual dump — one JSONL line per counterfactual.

Every method writes the SAME format, so any judge can be pointed at any method's
output afterwards without re-running the method. This exists because the first
round of shared-judge runs persisted only metrics (and, for the molecular
datasets, a SMILES string): NCI1 counterfactuals turned out to be unrecoverable
altogether, and benzene lost ~70% of its flips because RDKit refuses to serialise
graphs with aromatic bonds that do not form a perceivable ring. A graph-level
dump has neither failure mode.

Deliberately stdlib-only (json, os) with duck-typed tensor handling, because this
file is copied verbatim into five repos spanning torch 1.11–2.5 and PyG 2.3–2.6.

Line schema:

    method        str   "gdce-i" | "ucexplainer" | "c2" | "cf2" | "d4"
    dataset       str   "mutagenicity_mol" | "benzene" | "nci1"
    judge         str   basename of the judge that decided `success`
    variant       str   free-form run label (e.g. "posterior_s3"), "" if unused
    run, seed     int   which of the 5 runs, and its seed
    sample_idx    int   index within the run
    data_idx      int   index into the dataset's test split
    source_class  int
    target_class  int
    n_nodes       int
    node_types    [int]        index into the SHARED node_type_map, after argmax
    edges         [[i,j,bond]] i < j, undirected, bond in 1..4
    bond_source   str   "predicted" | "inherited" | "binary"
    own_pred      int   what the method's own judge said (-1 if not recorded)
    projected     bool  node/bond rows are hard one-hot, not a relaxed mask

Conventions that are NOT negotiable, because a judge cannot detect a violation:
  * `node_types` indexes the shared vocabulary. A repo that carries its own atom
    ordering must remap BEFORE calling this.
  * Only projected counterfactuals. Handing a multi-hot `x` to a foreign
    classifier measures noise -- it never saw anything like it during training.
  * `bond_source="binary"` means the method has no bond types and every edge was
    written as a single bond. Recorded, not hidden, so a bond-aware judge's
    numbers can be read with that in mind.
"""

from __future__ import annotations

import json
import os

SCHEMA_VERSION = 1


def _to_list(t):
    """torch.Tensor | np.ndarray | list -> nested python list."""
    if t is None:
        return None
    if hasattr(t, "detach"):
        t = t.detach()
    if hasattr(t, "cpu"):
        t = t.cpu()
    if hasattr(t, "tolist"):
        return t.tolist()
    return list(t)


def node_types_from_x(x):
    """One-hot / logit node features (n, n_types) -> list of int indices."""
    if hasattr(x, "argmax"):
        return [int(v) for v in _to_list(x.argmax(-1))]
    return [int(v) for v in x]


def edges_from_edge_index(edge_index, edge_attr=None, default_bond=1,
                          zero_as_default=False):
    """(2, E) edge_index [+ (E, n_bond_types) one-hot] -> sorted [[i, j, bond]].

    Both directions of an undirected edge collapse into one entry with i < j;
    self-loops are dropped (the judges strip them anyway, and keeping them would
    make edge counts repo-dependent).

    A bond type of 0 means "no bond" in the dense encoding, so such rows are
    dropped by default. `zero_as_default=True` reads them as `default_bond`
    instead -- needed for methods that ADD edges without predicting a type and
    leave the edge_attr row all-zero (D4Explainer). Dropping those would silently
    delete exactly the edges the method contributed.
    """
    ei = _to_list(edge_index)
    if not ei or not ei[0]:
        return []
    src, dst = ei[0], ei[1]

    bonds = None
    if edge_attr is not None:
        ea = edge_attr
        if hasattr(ea, "argmax") and getattr(ea, "ndim", 1) > 1:
            bonds = [int(v) for v in _to_list(ea.argmax(-1))]
        else:
            bonds = [int(v) for v in _to_list(ea)]

    out = {}
    for k, (i, j) in enumerate(zip(src, dst)):
        i, j = int(i), int(j)
        if i == j:
            continue
        b = bonds[k] if bonds is not None else default_bond
        if b <= 0:
            if not zero_as_default:
                continue
            b = default_bond
        out[(min(i, j), max(i, j))] = int(b)
    return [[i, j, b] for (i, j), b in sorted(out.items())]


def edges_from_adj(adj, default_bond=1):
    """Dense (n, n) adjacency -> [[i, j, bond]] with i < j.

    For the datasets without bond types (nci1, proteins, twitter), where the
    methods carry an adjacency matrix rather than an edge_index.
    """
    a = _to_list(adj)
    out = []
    for i, row in enumerate(a):
        for j in range(i + 1, len(row)):
            if float(row[j]) > 0:
                out.append([i, j, int(default_bond)])
    return out


class CFDumpWriter:
    """Append-only JSONL writer. One instance per (method, dataset, judge) run."""

    def __init__(self, path, method, dataset, judge, bond_source, variant=""):
        if bond_source not in ("predicted", "inherited", "binary"):
            raise ValueError(f"bond_source must be predicted|inherited|binary, "
                             f"got {bond_source!r}")
        d = os.path.dirname(os.path.abspath(path))
        if d:
            os.makedirs(d, exist_ok=True)
        self.path = path
        self._fh = open(path, "w")
        self._const = dict(schema=SCHEMA_VERSION, method=method, dataset=dataset,
                           judge=os.path.basename(str(judge)), variant=variant,
                           bond_source=bond_source)
        self.n_written = 0

    def write(self, *, run, seed, sample_idx, data_idx, source_class, target_class,
              node_types, edges, own_pred=-1, projected=True, **extra):
        types = [int(v) for v in node_types]
        rec = dict(self._const)
        rec.update(run=int(run), seed=int(seed), sample_idx=int(sample_idx),
                   data_idx=int(data_idx), source_class=int(source_class),
                   target_class=int(target_class), n_nodes=len(types),
                   node_types=types,
                   edges=[[int(i), int(j), int(b)] for i, j, b in edges],
                   own_pred=int(own_pred), projected=bool(projected))
        if extra:
            rec.update(extra)
        self._fh.write(json.dumps(rec) + "\n")
        self._fh.flush()   # a job killed by the walltime keeps everything so far
        self.n_written += 1

    def write_from_tensors(self, *, x, edge_index, edge_attr=None,
                           default_bond=1, **kwargs):
        """Convenience wrapper for the usual (x, edge_index, edge_attr) triple."""
        self.write(node_types=node_types_from_x(x),
                   edges=edges_from_edge_index(edge_index, edge_attr, default_bond),
                   **kwargs)

    def close(self):
        if self._fh is not None and not self._fh.closed:
            self._fh.close()
            print(f"[cf_dump] {self.n_written} counterfactuals -> {self.path}")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def read_dump(path):
    """Read a dump back as a list of dicts. Tolerates a truncated final line."""
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                break
    return out
