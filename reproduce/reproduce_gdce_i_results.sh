#!/usr/bin/env bash
# Every command that produced the numbers in the GDCE-I paper, in order.
#
#   reproduce/reproduce_gdce_i_results.sh                 # all stages
#   reproduce/reproduce_gdce_i_results.sh table3 table4   # only these
#   GDCE_DRYRUN=1 reproduce/reproduce_gdce_i_results.sh   # print, do not run
#   SMOKE=1 reproduce/reproduce_gdce_i_results.sh table2  # two graphs, checks the wiring
#
# Stages, and what each one needs:
#
#   data        -- TUDataset / Benzene download, filtering, split        (CPU, minutes)
#   judges      -- the 12 shared classifiers + control copies           (GPU, ~1 h total)
#   train       -- the 6 conditional diffusion models                   (GPU, ~2-4 d each)
#   table2      -- GDCE-I rows of Table 2, four datasets                (GPU, ~4-12 h each)
#   table3      -- noise-treatment ablation at the end of the search    (GPU, ~8 h)
#   table4      -- paired replay at fixed budgets tau in {10,50,100,200}(GPU, 5 x ~3 h)
#   table5      -- guidance-scale sweep s in {1,3,5}                    (GPU, ~8 h)
#   fig3        -- paired replay on the full budget grid of the search  (GPU, 2 x ~5 h)
#   tables      -- turn the run CSVs into the numbers in the paper      (CPU, seconds)
#
# Nothing but code and the shared classifiers ships with this repository: no
# datasets, no trained models, no result files.  `data` builds the datasets,
# `train` the six diffusion models (docs/training.md says what is pinned and
# what is not), the table stages write the per-run CSVs under $GDCE_OUT, and
# `tables` turns those into the numbers of the paper.  `judges` is the one stage
# you can skip, because those weights are in git.

. "$(dirname "${BASH_SOURCE[0]}")/common.sh"

ALL_STAGES="data judges train table2 table3 table4 table5 fig3 tables"
WANT="${*:-$ALL_STAGES}"
has() { case " $WANT " in *" $1 "*) return 0;; *) return 1;; esac; }

MUTAG=mutagenicity_mol
TW_LIST="[1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240]"
# the budgets Fig. 3b reports a greedy reversion for
GREEDY="[1,2,3,5,8,11,15,20,30,40,51,60,75,100,125,150,200]"

JUDGE_FLAGS="--batchnorm --cosine --dropout 0.3 --refit_train_val --epochs 300"

# Sample sizes of the two non-molecular datasets.  PROTEINS has only 174 test
# graphs, so the paper takes the complete test split in one run (and reports no
# spread); TWITTER draws 5 x 100 out of 13006, like the molecular datasets.
if [ "${SMOKE:-0}" = "1" ]; then
    PR_RUNS=1; PR_SAMPLES=2; TW_RUNS=1; TW_SAMPLES=2
else
    PR_RUNS=1; PR_SAMPLES=174; TW_RUNS=5; TW_SAMPLES=100
fi

# score_dump <dump> <dataset> <judge-stem> <label> <summary.csv>
# NAFR, NA and the SMILES column all come out of here: the base classifier has to
# reproduce the target class on the dumped graph, and the control copy (seed 1,
# never optimised against) is then asked the same question.
score_dump() {
    local dump="$1" ds="$2" stem="$3" label="$4" summary="$5"
    if [ ! -s "$dump" ] && [ "${GDCE_DRYRUN:-0}" != "1" ]; then
        echo "  !! no dump at $dump -- skipping $label" >&2; return 0
    fi
    mkdir -p "$(dirname "$summary")"
    pyrun src/shared_judge/adversarial_flip.py \
        --dump "$dump" --dataset "$ds" --data_root "$GDCE_DATA" \
        --judge_base "$GDCE_JUDGES/$stem" --judge_ctrl "$GDCE_JUDGES/${stem}_seed1" \
        --label "$label" \
        --out_csv "$(dirname "$summary")/$(echo "$label" | tr / _)_verdicts.csv" \
        --summary_csv "$summary"
}

# ===========================================================================
if has data; then
stage "data -- download, filter and split the four datasets"
# Each call runs download() (TUDataset or the Benzene archive -> raw/*.csv plus the
# node/edge marginals) and process() (csv -> PyG) and prints the shapes that
# SpectreDatasetInfos will later expect.
#
# NOTE ON THE SPLIT: process() draws the 60/20/20 split with an UNSEEDED
# np.random.permutation, so a fresh build does not reproduce the split the paper
# used.  data/*/raw/{train,val,test}.csv as shipped ARE that split; keep them and
# this stage is a no-op verification.  Delete them and every number below shifts
# by the split, not by the method.
for ds in "$MUTAG" benzene proteins twitter; do
    pyrun src/build_dataset.py --name "$ds" --datadir "$GDCE_DATA/$ds/"
done
fi

# ===========================================================================
if has judges; then
stage "judges -- the classifier every method targets, and its control copy"
mkdir -p "$GDCE_JUDGES"
# Molecular datasets: train on TRAIN, pick the epoch count on VAL, refit on
# TRAIN+VAL with that count, touch TEST once.  --seed 1 --tag _seed1 repeats the
# recipe with nothing changed but the seed; that copy is what NAFR is measured
# against.
for conv in gcn gine; do
    for ds in "$MUTAG" benzene; do
        # shellcheck disable=SC2086
        pyrun src/shared_judge/train_judge.py --dataset "$ds" --conv "$conv" \
            --data_root "$GDCE_DATA" --out_dir "$GDCE_JUDGES" $JUDGE_FLAGS
        # shellcheck disable=SC2086
        pyrun src/shared_judge/train_judge.py --dataset "$ds" --conv "$conv" \
            --data_root "$GDCE_DATA" --out_dir "$GDCE_JUDGES" $JUDGE_FLAGS --seed 1 --tag _seed1
    done
done
# PROTEINS and TWITTER: same architecture, their own capacity and epoch budget
# (PROTEINS 2 x 64, wd 1e-2, 300 epochs; TWITTER 60 epochs, batch 128), best
# validation epoch, no refit.  Both seeds in one call.
pyrun src/shared_judge/train_judges_nonmol.py

stage "judges -- agreement of base and control classifier on untouched test graphs"
# The floor for NA: two classifiers already disagree on unmodified graphs, so a
# transfer rate at or below this number says nothing about the counterfactuals.
for ds in "$MUTAG" benzene proteins twitter; do
    pyrun src/judge_agreement.py --name "$ds" \
        --judge-a "$GDCE_JUDGES/judge_${ds}_gcn" --judge-b "$GDCE_JUDGES/judge_${ds}_gcn_seed1"
done
fi

# ===========================================================================
if has train; then
stage "train -- one conditional diffusion model per dataset and classifier"
# The guidance condition is the label the SHARED classifier predicts
# (label_source: 'classifier'), which is why there is one model per classifier
# and not one per dataset.  Hyperparameters: see the configs and Section S4.
mkdir -p "$GDCE_CKPT"
train_one() {  # <experiment> <dataset> <run-name> <general.name>
    pyrun src/main.py "+experiment=$1" "dataset=$2" "hydra.run.dir=$GDCE_OUT/train/$3"
    # Lightning writes last.ckpt plus one file per ckpt_every_n_train_steps.
    # The last epoch is the one used; the epoch=*.ckpt fallback below only
    # matters if a run was cut short before Lightning wrote last.ckpt.
    local cdir="$GDCE_OUT/train/$3/checkpoints/$4" pick
    if [ "${GDCE_DRYRUN:-0}" = "1" ]; then
        pick="$cdir/last.ckpt"
    elif [ -f "$cdir/last.ckpt" ]; then
        pick="$cdir/last.ckpt"
    else
        pick=$(ls -t "$cdir"/epoch=*.ckpt 2>/dev/null | head -1)
        [ -n "$pick" ] || { echo "  !! no checkpoint under $cdir" >&2; return 1; }
    fi
    run cp "$pick" "$GDCE_CKPT/$3.ckpt"
}
train_one mutag_mol_gcn   "$MUTAG"  mutag_mol_gcn   mutag-mol-gcn
train_one mutag_mol_gine  "$MUTAG"  mutag_mol_gine  mutag-mol-gine
train_one benzene_gcn     benzene   benzene_gcn     benzene-gcn
train_one benzene_gine    benzene   benzene_gine    benzene-gine
train_one proteins_v1     proteins  proteins_gcn    proteins-v1
train_one twitter_v1      twitter   twitter_gcn     twitter-v1
# docs/training.md has the per-model table and the comparability checks.
fi

# ===========================================================================
if has table2; then
stage "table2 -- GDCE-I, four datasets, guidance scale s = 3"
# 5 runs x 100 test graphs, run seeds 42-46, stratified by the predicted class,
# drawn without replacement; budget grid tau = 1..100 and 110..200 step 10.  The
# baselines are evaluated on exactly these samples -- see baselines/README.md.
S2="$GDCE_OUT/results/table2_advflip.csv"

# shellcheck disable=SC2086
pyrun src/inversion_mutag_mol.py +experiment=inversion_mutag_mol_gcn_s3 "dataset=$MUTAG" \
    $SMOKE_SEARCH "hydra.run.dir=$GDCE_OUT/table2/mutag_gcn_s3"
score_dump "$GDCE_OUT/table2/mutag_gcn_s3/inversion_${MUTAG}_results_posterior_s3/cf_dump_${MUTAG}_posterior_s3.jsonl" \
    "$MUTAG" "judge_${MUTAG}_gcn" "gdce-i/mutag_gcn" "$S2"

# shellcheck disable=SC2086
pyrun src/inversion_mutag_mol.py +experiment=inversion_mutag_mol_gine_s3 "dataset=$MUTAG" \
    $SMOKE_SEARCH "hydra.run.dir=$GDCE_OUT/table2/mutag_gine_s3"
score_dump "$GDCE_OUT/table2/mutag_gine_s3/inversion_${MUTAG}_results_posterior_s3/cf_dump_${MUTAG}_posterior_s3.jsonl" \
    "$MUTAG" "judge_${MUTAG}_gine" "gdce-i/mutag_gine" "$S2"

# shellcheck disable=SC2086
pyrun src/inversion_benzene.py +experiment=inversion_benzene_gcn dataset=benzene \
    $SMOKE_SEARCH "hydra.run.dir=$GDCE_OUT/table2/benzene_gcn_s3"
score_dump "$GDCE_OUT/table2/benzene_gcn_s3/inversion_benzene_results_posterior_s3/cf_dump_benzene_posterior_s3.jsonl" \
    benzene judge_benzene_gcn "gdce-i/benzene_gcn" "$S2"

# shellcheck disable=SC2086
pyrun src/inversion_benzene.py +experiment=inversion_benzene_gine dataset=benzene \
    $SMOKE_SEARCH "hydra.run.dir=$GDCE_OUT/table2/benzene_gine_s3"
score_dump "$GDCE_OUT/table2/benzene_gine_s3/inversion_benzene_results_posterior_s3/cf_dump_benzene_posterior_s3.jsonl" \
    benzene judge_benzene_gine "gdce-i/benzene_gine" "$S2"

# PROTEINS has 174 test graphs, so the paper reports ONE run over the complete
# test split instead of five samples of 100 -- hence no standard deviation.
pyrun src/inversion_nci1_nofreq.py +experiment=inversion_proteins dataset=proteins \
    "++guidance.n_runs=$PR_RUNS" "++guidance.n_samples_per_run=$PR_SAMPLES" \
    ++guidance.t_step=10 ++guidance.t_max=240 ++guidance.dump_variant=s3 \
    "++guidance.transfer_classifier_path=$GDCE_JUDGES/judge_proteins_gcn_seed1" \
    "hydra.run.dir=$GDCE_OUT/table2/proteins_gcn_s3"
score_dump "$GDCE_OUT/table2/proteins_gcn_s3/inversion_proteins_results/cf_dump_proteins_s3.jsonl" \
    proteins judge_proteins_gcn "gdce-i/proteins_gcn" "$S2"

# TWITTER flips almost everything below tau = 30, so the grid is fine below 10.
pyrun src/inversion_nci1_nofreq.py +experiment=inversion_twitter dataset=twitter \
    "++guidance.n_runs=$TW_RUNS" "++guidance.n_samples_per_run=$TW_SAMPLES" \
    "++guidance.t_list=$TW_LIST" ++guidance.dump_variant=s3 \
    "++guidance.transfer_classifier_path=$GDCE_JUDGES/judge_twitter_gcn_seed1" \
    "hydra.run.dir=$GDCE_OUT/table2/twitter_gcn_s3"
score_dump "$GDCE_OUT/table2/twitter_gcn_s3/inversion_twitter_results/cf_dump_twitter_s3.jsonl" \
    twitter judge_twitter_gcn "gdce-i/twitter_gcn" "$S2"

echo "  -> $S2"
fi

# ===========================================================================
if has table3; then
stage "table3 -- noise treatment at the end of the budget search"
# Identical to the table2 Mutagenicity/GCN run except for guidance.inversion_method:
#   noinv      fresh noise at every reverse step (the conference version)
#   fresh      the noise drawn along the forward chain, reused (naive inversion)
#   posterior  the truncated-Gumbel posterior of GDCE-I
# Each variant stops at ITS OWN first flipping budget, which is why the rows are
# not measured at the same amount of regeneration -- Table 4 is the controlled
# comparison.
S3="$GDCE_OUT/results/table3_advflip.csv"
for v in noinv fresh; do
    # shellcheck disable=SC2086
    pyrun src/inversion_mutag_mol.py "+experiment=inversion_mutag_mol_gcn_s3_$v" "dataset=$MUTAG" \
        $SMOKE_SEARCH "hydra.run.dir=$GDCE_OUT/table3/${v}_s3"
    score_dump "$GDCE_OUT/table3/${v}_s3/inversion_${MUTAG}_results_${v}_s3/cf_dump_${MUTAG}_${v}_s3.jsonl" \
        "$MUTAG" "judge_${MUTAG}_gcn" "$v/mutag_gcn" "$S3"
done
# the posterior row is the table2 Mutagenicity/GCN run; score it into the same file
score_dump "$GDCE_OUT/table2/mutag_gcn_s3/inversion_${MUTAG}_results_posterior_s3/cf_dump_${MUTAG}_posterior_s3.jsonl" \
    "$MUTAG" "judge_${MUTAG}_gcn" "posterior/mutag_gcn" "$S3"
echo "  -> $S3"
fi

# ===========================================================================
if has table4; then
stage "table4 -- paired replay at fixed budgets"
# One forward reference trajectory per input graph, shared by all three noise
# treatments, replayed from the same G^tau for every tau in the grid, under the
# target condition AND under the unchanged source condition (the placebo).  One
# job per run id so the five runs can go in parallel.
for r in 0 1 2 3 4; do
    # shellcheck disable=SC2086
    pyrun src/review_paired_tau.py +experiment=inversion_mutag_mol_gcn_s3 "dataset=$MUTAG" \
        "+review.run_ids=[$r]" "+review.tau_grid=[10,50,100,200]" "+review.greedy_taus=[]" \
        $SMOKE_PAIRED "+review.out_dir=$GDCE_OUT/table4/fixedtau_r$r"
    [ "${SMOKE:-0}" = "1" ] && break
done
fi

# ===========================================================================
if has table5; then
stage "table5 -- guidance scale s in {1, 3, 5}"
# s = 3 is the Table 2 run; only s = 1 and s = 5 are added here.  s = 3 was
# chosen on this sweep and then used unchanged for every other dataset and
# classifier, which is the only hyperparameter of the method we vary.
S5="$GDCE_OUT/results/table5_advflip.csv"
for exp in inversion_mutag_mol_gcn_s1 inversion_mutag_mol_gcn; do
    case "$exp" in *_s1) tag=s1;; *) tag=s5;; esac
    # shellcheck disable=SC2086
    pyrun src/inversion_mutag_mol.py "+experiment=$exp" "dataset=$MUTAG" \
        $SMOKE_SEARCH "hydra.run.dir=$GDCE_OUT/table5/posterior_$tag"
    score_dump "$GDCE_OUT/table5/posterior_$tag/inversion_${MUTAG}_results_posterior_${tag}/cf_dump_${MUTAG}_posterior_${tag}.jsonl" \
        "$MUTAG" "judge_${MUTAG}_gcn" "posterior_$tag/mutag_gcn" "$S5"
done
echo "  -> $S5"
fi

# ===========================================================================
if has fig3; then
stage "fig3 -- paired replay on the full budget grid of the search"
# Same script as table4, but on the paper's grid (tau = 1..100, 110..200) so the
# figure shows the edit distance at every budget instead of binned statistics
# over successful outputs.  Split over two jobs by run id.
for ids in "0,1" "2,3,4"; do
    # shellcheck disable=SC2086
    pyrun src/review_paired_tau.py +experiment=inversion_mutag_mol_gcn_s3 "dataset=$MUTAG" \
        +review.grid=paper "+review.run_ids=[$ids]" "+review.greedy_taus=$GREEDY" \
        $SMOKE_PAIRED "+review.out_dir=$GDCE_OUT/fig3/paper_grid_r${ids//,/}"
    [ "${SMOKE:-0}" = "1" ] && break
done
fi

# ===========================================================================
if has tables; then
stage "tables -- run CSVs  ->  the numbers in the paper"
# Every script here reads the CSVs the stages above wrote under $GDCE_OUT.
cd "$GDCE_ROOT"

echo; echo "--- Table 3 (budget search): FR, NAFR, NA, NAFR & SMILES, median tau"
if compgen -G "$GDCE_OUT/table3/*/GDCE_*.csv" >/dev/null; then
    pyrun analysis/table3_search_smiles.py "$GDCE_OUT/table3"
else
    echo "    (nothing under $GDCE_OUT/table3 -- run the table3 stage first)"
fi

echo; echo "--- Table 4 (fixed budgets): the LaTeX body"
if compgen -G "$GDCE_OUT/table4/fixedtau_r*/paired_*.csv" >/dev/null; then
    # shellcheck disable=SC2086
    pyrun analysis/fixedtau_table3_tex.py "$GDCE_OUT/results" $GDCE_OUT/table4/fixedtau_r*/paired_*.csv
else
    echo "    (no paired CSVs under $GDCE_OUT/table4 -- run the table4 stage first)"
fi

echo; echo "--- Fig. 3 / paired analysis"
if compgen -G "$GDCE_OUT/fig3/paper_grid_r*/paired_*.csv" >/dev/null; then
    # shellcheck disable=SC2086
    pyrun analysis/analyze_paired.py $GDCE_OUT/fig3/paper_grid_r*/paired_*.csv
else
    echo "    (no paired CSVs under $GDCE_OUT/fig3 -- run the fig3 stage first)"
fi

echo; echo "--- Table 2: the collected flip/NAFR summary"
f="$GDCE_OUT/results/table2_advflip.csv"
if [ -f "$f" ]; then echo "  $f"; cat "$f" | sed 's/^/    /'
else echo "    (no $f -- run the table2 stage first)"; fi
fi

summary
