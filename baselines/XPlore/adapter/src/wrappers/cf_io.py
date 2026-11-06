"""Persist paired (original, counterfactual) graphs from the RSGG-CE runs.

Each record stores the ORIGINAL graph next to its COUNTERFACTUAL so the two
can be compared / visualized later. RSGG-CE edits only the edge set on a fixed
node set, so node features are unchanged and stored once. For failed samples
the counterfactual edge_index is None (original is still kept).

Saved as <out_dir>/counterfactuals.pt (a list of dicts), loadable with
`torch.load(..., weights_only=False)`.
"""

from pathlib import Path

import torch


def _cpu(t):
    return t.detach().cpu() if torch.is_tensor(t) else None


def make_record(data, row, cf_edge_index=None, cf_edge_attr=None,
                orig_smiles=None, cf_smiles=None):
    """Build one paired original/counterfactual record (CPU tensors).

    `data` is the GDCE PyG Data of the original molecule (carries x, edge_index,
    edge_attr); `cf_edge_index` is the counterfactual edge set on the same nodes.
    """
    rec = {k: row.get(k) for k in (
        'run', 'seed', 'data_idx', 'source_class', 'target_class',
        'orig_pred', 'cf_pred', 'success', 'n_perturbs', 'ged')}
    rec['original'] = {
        'x': _cpu(data.x),
        'edge_index': _cpu(data.edge_index),
        'edge_attr': _cpu(getattr(data, 'edge_attr', None)),
        'smiles': orig_smiles,
    }
    rec['counterfactual'] = {
        'x': _cpu(data.x),  # RSGG-CE edits only edges; node features unchanged
        'edge_index': _cpu(cf_edge_index),
        'edge_attr': _cpu(cf_edge_attr),
        'smiles': cf_smiles,
    }
    return rec


def save_counterfactuals(records, out_dir):
    """Write the list of (original, counterfactual) records to disk."""
    path = Path(out_dir) / 'counterfactuals.pt'
    torch.save(records, path)
    print(f"Saved {len(records)} (original, counterfactual) pairs to: {path}")
    return path
