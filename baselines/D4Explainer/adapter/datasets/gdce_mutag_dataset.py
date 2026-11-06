"""
Dataset loader for GDCE's pre-processed Mutagenicity .pt files.

The GDCE SpectreGraphDataset saves train.pt / val.pt / test.pt in PyG's
InMemoryDataset (data, slices) format.  Each graph has:
  x          : [N, 13]  one-hot atom types  (13 types, not 14)
  edge_index : [2, E]   COO edge list
  edge_attr  : [E, 5]   one-hot bond types  (0=no bond, 1-4=bond type)
  guidance   : [1, 2]   one-hot class label  (used instead of y in GDCE)
  y          : [1, 0]   empty (GDCE leaves y empty for the diffusion model)

This loader fixes y ← guidance.argmax(-1) so that D4Explainer's pipeline
(which reads graph.y for classification) works correctly.

Usage:
  Copy the three .pt files to a directory, e.g. data/GDCE_MUTAG/
    data/GDCE_MUTAG/train.pt
    data/GDCE_MUTAG/val.pt
    data/GDCE_MUTAG/test.pt

  Then load with:
    from datasets.gdce_mutag_dataset import GDCEMutagDataset
    train = GDCEMutagDataset('data/GDCE_MUTAG', mode='training')
    val   = GDCEMutagDataset('data/GDCE_MUTAG', mode='evaluation')
    test  = GDCEMutagDataset('data/GDCE_MUTAG', mode='testing')
"""

import os
import pickle
import types
import torch
from torch_geometric.data import InMemoryDataset


def _load_gdce_pt(path):
    """
    Load a GDCE-saved .pt file whose pickled objects reference GDCE's 'src.*'
    modules (not available in the D4Explainer environment).

    GuidedInMemoryDataset (and other src.* classes) are subclasses of PyG's
    Data, so we stub them as Data subclasses.  This preserves __getitem__,
    __cat_dim__, and all tensor storage so _make_list works correctly.
    """
    from torch_geometric.data import Data

    class _SrcStub(Data):
        """Stand-in for any GDCE src.* class; inherits Data's full interface."""
        pass

    class _GDCEUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == 'src' or module.startswith('src.'):
                return type(name, (_SrcStub,), {})
            return super().find_class(module, name)

    # torch saves as a zip archive; we need to use torch's internal reader
    # but swap in our custom unpickler via the pickle_module hook.
    _pickle_mod = types.ModuleType('_gdce_pickle')
    _pickle_mod.Unpickler = _GDCEUnpickler
    _pickle_mod.dump = pickle.dump
    _pickle_mod.dumps = pickle.dumps
    _pickle_mod.load = pickle.load
    _pickle_mod.loads = pickle.loads

    return torch.load(path, map_location='cpu', pickle_module=_pickle_mod)


_FILE_MAP = {
    'training':   'train.pt',
    'evaluation': 'val.pt',
    'testing':    'test.pt',
}


class GDCEMutagDataset(InMemoryDataset):
    """
    Wraps GDCE's pre-processed Mutagenicity .pt files for use in D4Explainer.

    Node features: 13-dim one-hot atom types (GDCE encoding).
    Edge features:  5-dim one-hot bond types.
    Labels:         long tensor y with class index (0=Non-mutagenic, 1=Mutagenic).
    """

    splits = list(_FILE_MAP.keys())

    def __init__(self, root, mode='testing', transform=None,
                 pre_transform=None, pre_filter=None):
        assert mode in self.splits, f"mode must be one of {self.splits}"
        self.mode = mode
        self._pt_file = _FILE_MAP[mode]
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [self._pt_file]

    @property
    def processed_file_names(self):
        # Store as a separate processed file so we can fix the y field
        return [f'gdce_{self.mode}.pt']

    def download(self):
        # Expect files to be placed manually in self.raw_dir
        src = os.path.join(self.raw_dir, self._pt_file)
        if not os.path.exists(src):
            raise FileNotFoundError(
                f"Expected GDCE .pt file at: {src}\n"
                f"Copy the GDCE {self._pt_file} file there and re-run."
            )

    def process(self):
        src = os.path.join(self.raw_dir, self._pt_file)
        data, slices = _load_gdce_pt(src)

        # Reconstruct individual Data objects, fix y, re-collate
        dataset = self._make_list(data, slices)
        fixed = []
        for g in dataset:
            g = g.clone()
            if hasattr(g, 'guidance') and g.guidance is not None and g.guidance.numel() > 0:
                # guidance shape: [1, 2] one-hot → long scalar
                g.y = g.guidance.argmax(dim=-1).long().view(-1)
            elif g.y.numel() == 0:
                g.y = torch.zeros(1, dtype=torch.long)
            else:
                g.y = g.y.long().view(-1)
            fixed.append(g)

        torch.save(self.collate(fixed), self.processed_paths[0])

    # ------------------------------------------------------------------
    # Helper: reconstruct list of Data from (data, slices) without a
    # full DataLoader — mirrors PyG's InMemoryDataset.__getitem__ logic.
    # ------------------------------------------------------------------
    def _make_list(self, data, slices):
        from torch_geometric.data import Data
        keys = list(slices.keys())
        n = len(slices[keys[0]]) - 1
        out = []
        for i in range(n):
            d = Data()
            for key in keys:
                start, end = int(slices[key][i]), int(slices[key][i + 1])
                val = data[key]
                if val is None:
                    d[key] = None
                elif isinstance(val, torch.Tensor):
                    if val.dim() == 0:
                        d[key] = val
                    else:
                        # determine cat-dim: edge_index is cat along dim=1, rest along dim=0
                        cat_dim = data.__cat_dim__(key, val) if hasattr(data, '__cat_dim__') else 0
                        d[key] = val.narrow(cat_dim, start, end - start)
                else:
                    d[key] = val[start:end]
            out.append(d)
        return out
