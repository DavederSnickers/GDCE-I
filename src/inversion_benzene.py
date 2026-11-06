"""Inversion counterfactuals for the Benzene dataset.

Benzene uses exactly the same graph representation as mutagenicity_mol --
5-class bond types (0=no bond, 1=single, 2=double, 3=triple, 4=aromatic), a
GINEConv oracle, atom_decoder.json for SMILES reconstruction, and no self-loops
on the adjacency diagonal. The inversion logic is therefore identical, and this
entrypoint reuses `inversion_mutag_mol.main` instead of forking a fifth
near-duplicate of that ~1500-line script. The dataset-specific pieces (class
names, output directory) are selected there from `cfg.dataset.name`.

Usage:
    PYTHONPATH=<repo> apprun python src/inversion_benzene.py \
        +experiment=inversion_benzene dataset=benzene
"""
from inversion_mutag_mol import main

if __name__ == '__main__':
    main()
