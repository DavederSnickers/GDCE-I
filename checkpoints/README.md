# Diffusion checkpoints

**No model weights are distributed with this repository.** This directory is
where the models you train land, under the flat names the inversion configs
expect:

```
mutag_mol_gcn.ckpt   mutag_mol_gine.ckpt
benzene_gcn.ckpt     benzene_gine.ckpt
proteins_gcn.ckpt    twitter_gcn.ckpt
```

The configs reach them through `${oc.env:GDCE_CKPT}/<name>.ckpt`, so pointing
`GDCE_CKPT` somewhere else is enough; nothing has to be edited.

```bash
reproduce/reproduce_gdce_i_results.sh train
```

**`docs/training.md`** is the instruction: what is pinned (the splits and the
classifier the model is conditioned on are both published, so a retrain changes
only the generator), what is not, every hyperparameter per model, the cost, and
four checks that tell you whether your retrained model is comparable to ours.

The shared classifiers themselves *are* in git, in `judges/` — they are 2.9 MB
and they are what makes the retraining comparable rather than merely similar.
