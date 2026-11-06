"""XPlore (Beyond Edge Deletion, KDD'26), ported to the shared-judge protocol.

`XPlore.py` is the authors' file from https://anonymous.4open.science/r/XPlore
with only its import header rewritten (see the comment at the top of that file);
`graph_instance.py` and `data_instance.py` are GRETEL v2's instance classes,
which XPlore constructs its counterfactual in. Everything else in this package
is ours: `framework.py` replaces the GRETEL v2 base classes, and the driver
lives in `src/wrappers/xplore_eval.py`.
"""
