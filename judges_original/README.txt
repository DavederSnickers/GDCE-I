Original classifiers of the four baselines, re-implemented in GDCE src/shared_judge/shared_judge.py
(conv = c2gcn / cf2gcn / d4gcn / ucxgine) and trained with each baseline's own recipe on GDCE's
Mutagenicity split (CSV y labels) by src/shared_judge/train_original_judges.py, 2026-10-27.
Control experiment of Section S3 of the supplement.

Roles (attacked = the classifier the method optimises against; control = NA re-scoring):
  C2   attacked judge_mutagenicity_mol_c2gcn          (seed 0)  test 0.7311
       control  judge_mutagenicity_mol_c2gcn_seed1    (seed 1)  test 0.7424  agreement 0.7513
  CF2  attacked judge_mutagenicity_mol_cf2gcn_seed1   (seed 1)  test 0.7311
       control  judge_mutagenicity_mol_cf2gcn_seed2   (seed 2)  test 0.7058  agreement 0.8813
       seed 0 trained to a constant predictor (loss stuck at the label entropy 0.6808 from
       epoch 0, dead ReLU head) -> judge_mutagenicity_mol_cf2gcn_seed0_degenerate, not used.
       Rule: seeds in order, a run predicting a single class is discarded. seed 3 (0.7197) unused.
  D4   attacked judge_mutagenicity_mol_d4gcn          (seed 0)  test 0.8093
       control  judge_mutagenicity_mol_d4gcn_seed1    (seed 1)  test 0.8106  agreement 0.9558
  UCX  attacked judge_mutagenicity_mol_ucxgine        (seed 0)  test 0.7992
       control  judge_mutagenicity_mol_ucxgine_seed1  (seed 1)  test 0.8018  agreement 0.9192
(agreement = share of unmodified test graphs on which attacked and control predict the same class)

D4 in the paper ablation (decided 2026-10-28: no D4 tuning; the multi-day paper-default trainings
against judge_mutagenicity_mol_d4gcn were cancelled):
  D4   attacked judge_mutagenicity_mol_d4native       test 0.8207
       = D4Explainer's NATIVELY trained Mutag_GCN (param/gnns/gdce_mutag_gcn.pt, D4's own
         gnns/mutag_gnn.py recipe, 2026-06-16), exported by convert_d4_native.py
         (max logit deviation 0, 792/792 identical test predictions). The June D4Explainer model
         (results/gdce_mutag/best_model_flip_alpha_05.pt: alpha_cf 0.5, 800 epochs; NOT the
         paper's alpha_cf 0.001) was trained against it.
       control  judge_mutagenicity_mol_d4gcn_seed1    test 0.8106  agreement 0.9646
  judge_mutagenicity_mol_d4gcn (seed 0) and its GDCE-I run remain as a side result only.
