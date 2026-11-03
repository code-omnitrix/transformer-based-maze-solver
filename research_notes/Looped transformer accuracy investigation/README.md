# Reproducible experiment archive

This directory contains an audit of `../../aiml.ipynb`, extracted historical output,
and new controlled experiments. The original notebook is preserved byte for byte.

## What is historical and what is new

* `audit/original_histories.csv`: existing notebook output, not newly measured.
* `audit/cell_*.py`: exact source extracted from the saved notebook at audit time.
* `audit/notebook_structure.json`: cell execution counts, metadata, source hashes,
  and original notebook SHA-256.
* `audit/architecture_counts.json`, `audit/causal_checks.json`: fresh architecture
  instantiation counts and future-prefix perturbation diagnostic.
* `benchmark.json`: fresh GPU step benchmark, synthetic 36-token target prefixes,
  12 steps/model with final 10 used for reported medians. Not actual epoch timing.
* `runs/controlled12k_*/`: new controlled pilots on independently generated data.
* `independent_statistics*`: independent notebook generator replay and simple
  algorithmic baselines. These are separate from the controlled pilot datasets.

## Execution environment

Use the existing interpreter:

```powershell
$experimentPython = 'C:\Users\omnap\miniconda3\envs\llmdyn\python.exe'
$experimentRoot = 'D:\Projects\Loopedtrans\research_notes\Looped transformer accuracy investigation'
& $experimentPython "$experimentRoot\experiment_suite.py" --mode audit
```

Python 3.12.13, PyTorch 2.11.0+cu128, NumPy 2.4.6, Matplotlib 3.11.0;
NVIDIA GeForce RTX 5080 Laptop GPU (16 GB). The default Python 3.14 environment
has no torch. No dependency installs or paid remote compute are used.

## Controlled matrix

```powershell
& $experimentPython -u "$experimentRoot\experiment_suite.py" --mode train --label controlled12k --train-size 12000 --val-size 1000 --test-size 1000 --epochs 30 --seeds 42,43,44 --variants postln4,preln4,preln6,tied6,tied6_anchor,tied6_full,untied6_full --rollout-n 512
```

The suite refuses to overwrite existing run folders; use a new `--label` to rerun.
All three architecture seeds use the same fixed data, evaluation examples, and
per-seed shuffle order. Split generation seeds are 20261102, 20261103, 20261104.
The split files retain all grids and padded token sequences. Their byte hashes,
per-example grid hashes, path lengths, and exact train/val/test overlap are in
`data/manifest.json`. Statistical distribution matches the notebook problem, but
the NumPy RNG algorithm and start/goal sampling differ; these are not bit-for-bit
reproductions of the original unseen data or checkpoints.

| ID | Unique encoder/decoder layers | Effective encoder/decoder depth | Norm | Reinject input | Loop index |
|---|---|---|---|---|---|
| postln4 | 4 / 4 | 4 / 4 | Post-LN | no | no |
| preln4 | 4 / 4 | 4 / 4 | Pre-LN | no | no |
| preln6 | 6 / 6 | 6 / 6 | Pre-LN | no | no |
| tied6 | 1 / 1 | 6 / 6 | Pre-LN | no | no |
| tied6_anchor | 1 / 1 | 6 / 6 | Pre-LN | learned scalar, initially 1 | no |
| tied6_full | 1 / 1 | 6 / 6 | Pre-LN | learned scalar, initially 1 | sinusoidal |
| untied6_full | 6 / 6 | 6 / 6 | Pre-LN | learned scalar, initially 1 | sinusoidal |

All controlled models: width128, eight attention heads, FF256, dropout0.1,
max decoder positions128, final encoder/decoder layer norms, batch128,
AdamW lr0.001 weight_decay0.0001, gradient norm clipping1, float32, no AMP,
30epochs ×94 updates =2820 optimizer updates. Core attention/FF matrices use
Xavier uniform initialization consistently, matching the initialization *family*
of nn.Transformer; embeddings and output projection use PyTorch defaults.
This intentionally removes the original notebook's initialization confound.
Models use independent seeded initialization draws; seeds pair datasets/orders,
not identically initialized differently shaped parameter tensors.

## Run files and units

* `config.json`: requested arguments, architecture, split hashes, seed, optimizer.
* `environment.json`: executable/library versions, hardware, script/notebook hashes,
  UTC start, command line, determinism policy.
* `model.json`: exact string representation and trainable parameter count.
* `epochs.jsonl`: one record/epoch. Loss in natural-log units/token; accuracy as
  fraction0–1; elapsed times in seconds; counts integers. Validation excludes PAD,
  includes EOS; separate action-only and EOS counts are retained.
* `steps.jsonl`: one pre-update training record/update, dropout active. Per-step
  token NLL, count, LR, unclipped gradient norm, timing. Cumulative NLL field
  **resets each epoch**. This is repeated-data training loss, not an epiplexity
  or paper-equivalent prequential code-length estimate.
* `checkpoint.pt`: final model, optimizer, Python/NumPy/PyTorch CPU/CUDA RNG states,
  DataLoader shuffle generator state, completed epoch and config. No best-checkpoint
  selection is performed; final epoch is prespecified.
* `test_teacher_predictions.jsonl`: one record/test maze; target and predicted
  canonical tokens, target count, correct count, exact result, path length.
* `test_rollout_predictions.jsonl`: first512 fixed test mazes; generated tokens,
  target, maze hash, valid-path flag, EOS flag, goal status, solved status,
  shortest-path success, canonical exact-match, trajectory coordinates.
* `summary.json`: final teacher/rollout metrics and final checkpoint depth sweeps.
* `error.txt`: traceback if a run fails. Failed runs remain recorded, not erased.
* `run_registry.jsonl`: append-only started/complete/failed lifecycle records.
* `controlled12k_console.log`: exact streamed training console.

**Raw logging semantics corrected in derived analysis:**
`data/manifest.json`'s legacy `path_length_histogram` counts the full stored
sequence including BOS+EOS, so subtract2 for number of moves. Its adjacent
`max_sequence_length_including_BOS_EOS` is correctly labeled. Per-example
prediction files already store the correct action-pathlength. Raw `terminated`
means EOS was encountered *before tracing stopped at an illegal move*; it is
not the overall EOS-emission rate. `analyze_runs.py` reconstructs
`test_actual_eos_emitted` from the saved complete token sequence. Raw files are
preserved. `solved` and `optimal_solved` are unaffected by this logging caveat.

Teacher forcing tests prediction given true earlier actions. Rollout generates
the model's own prefix. `solved` requires a legal trajectory reaching the goal
and EOS. `optimal_solved` additionally requires the BFS-optimal number of moves;
it accepts alternative shortest paths. `rollout_exact` requires canonical BFS
tokens including EOS. Invalid PAD/BOS predictions count as invalid actions.
`goal_reached` is also logged without requiring EOS, so termination is auditable.
Autoregressive evaluation occurs only once per final checkpoint, not each epoch.

The sequence cap is64 predictions (at most63 moves plus EOS), sufficient for
any simple shortest path on64 cells. Ended sequences remain in the batch, but
their subsequent generated tokens are ignored.

## Reproducibility limits

Fixed seeds are set for Python, NumPy, CPU and CUDA. Deterministic algorithms
are not forced; GPU kernels and future library versions may introduce small
differences. Training runs share one GPU sequentially. Absolute timing is local
to this device and affected by thermal/power conditions; compare measured
cumulative training seconds as well as optimizer steps and epochs.

Variants run in a fixed order, so walltime differences are indicative and may
include laptop temperature or power-state effects. No energy-efficiency claim
is supported. The matched-depth/width sharing comparison deliberately has
different unique parameter counts. A parameter-matched narrower untied model
was not trained, so this study cannot establish the best architecture under a
fixed memory/parameter budget.

`determinism_environment_defaults.json` records fresh same-interpreter defaults:
deterministic algorithms disabled, cuDNN deterministic/benchmark false, CUDA
matmul TF32 false, cuDNN TF32 true, float32 precision highest. The frozen source
does not change these flags. This supplement is explicitly not an introspection
of the already-running process; per-run source and environment hashes document
what was executed. Fixed seeds do not imply bitwise CUDA reproducibility.

The three seeds measure initialization/dropout/shuffle variability on one fixed
generated dataset, not variation over independently sampled train datasets.
Bootstrap intervals over test examples measure this finite test set's sampling
uncertainty conditional on fitted models; they do not establish universal claims.

## Resuming and extending

Use `resume_run.py` (see its `--help`) to continue an existing final checkpoint
into a new directory. It restores optimizer and RNG state; it never overwrites
the source run. Source checkpoint, script and dataset hashes are carried forward.
Raw artifacts are the authoritative record; plots are regenerated from them.

```powershell
& $experimentPython "$experimentRoot\resume_run.py" --self-test
& $experimentPython "$experimentRoot\resume_run.py" --run "$experimentRoot\runs\controlled12k_tied6_full_seed42" --additional-epochs 10 --new-id continued_tied6_full_seed42
& $experimentPython "$experimentRoot\analyze_runs.py" --prefix controlled12k
& $experimentPython "$experimentRoot\archive_manifest.py"
```

The resume command creates an additional exploratory training continuation; it
does not alter or replace the fixed30epoch study endpoint. The implementation
was tested on a tiny CPU dropout+shuffled-data continuation: next-epoch loss and
all model state tensors match an uninterrupted trajectory exactly (see
`resume_validation.json`). CUDA bitwise resume is not established. Matrix runs
save only final checkpoints, so they cannot recover an interrupted mid-run;
full-size runs additionally save `checkpoint_latest.pt` every10epochs and the
originalbaseline saves fixed30and90 snapshots.

## Full-data follow-up

`STUDY_PLAN.md` and `fullscale_plan_record.json` record when the later extension
was specified. This is a local prospective plan, not public preregistration.

```powershell
& $experimentPython -u "$experimentRoot\fullscale_experiments.py" --rollout-n 1000
& $experimentPython "$experimentRoot\analyze_runs.py" --prefix full50k
& $experimentPython "$experimentRoot\plot_fullscale.py"
```

Four runs: exact notebook baseline90 (plusfixed30evaluation), exact notebook
loop30, controlledtiedfull30, controlleduntiedfull30. All use the exact original
NumPy/randomseed42 generatortrain50k thenval5k, plus a fresh5k test set. Each
model is freshly seeded42 and shares dedicatedshuffle seed1042; this is a
documented deviation from unknown historical globalRNG/execution state.
Original architecture definitions are extracted directly from the notebook.
Controlledmodels retain the matched initialization policy from the pilot.

`data/original50k_seed42/manifest.json` contains hashes, true actionlength
histograms, counts and overlap. The finaltest rollouts use the first1000fixed
test mazes. Evaluation atbaseline30 preserves RNGstate so added diagnostics
do not change subsequent baseline training. Training is sequential on one GPU.
`checkpoint_latest.pt` is overwritten within its own new run folder every10
epochs; fixed30/90 and final checkpoint files remain available.

`compare_loop_initializations.py` verifies that original_looped and controlled
tied6_full have the same43trainable tensors under a one-to-one mapping and
352,521parameters. Freshseed42 differs only in10core2Dattention/FFmatrices;
embeddings, outputprojection, all1Dnorm/bias/gate values match. The controlled
model applies an extraXavier reset tocore matrices. This changes initialization
draws as well as some distributions, so it is not a weightscale-only test.
CPU RNG advances differ; initialCUDA RNG hashes match; wholetraining dropout
masks were not independently traced. See
`original_loop_vs_controlled_initialization.json` for everymappedtensor.

`fullscale_gpu_telemetry.jsonl` records read-only nvidia-smi sensors every30seconds
for part of the full-data stage. It begins after baseline training already
started and has no matrix/early-baseline coverage. GPUtemperatureC, powerW,
utilizationpercent, memoryMiB, clocksMHz, driver andPstate are recorded as sensor
strings. It supports context for localtiming, not an energy-efficiency estimate.

Fullscale `summary.total_seconds` is sampled before finalevaluation; its registry
completion record includes finalevaluation. Matrix summary total includes final
testprediction/rollout but excludes laterdepthsweep. For compute comparisons use
`train_seconds` / `cumulative_train_seconds`, which measure training directly;
`val_seconds` is retained separately. These timing definitions are intentionally
made explicit instead of treating summary totals as identical across scripts.

After generic fullscale table generation, `plot_fullscale.py` replaces its two
generic accuracy figures with explicit single-seed versions and adds the fixed
baseline30 endpoint. No sampleSD is estimated from n=1. The replacement is
recorded in `analysis/full50k/full_plot_manifest.json`; numeric source tables
are unchanged. Run these two commands in the documented order for regeneration.

## Plot and analysis products

`analyze_runs.py` was exercised on complete partial runs while training continued,
then rerun after completion. It uses only saved logs and predictions, not hidden
notebook state. Outputs in `analysis/<prefix>/` include:

* `final_runs.csv`: each seed's counts, accuracies, parameters, trainingtime,
  firstactionaccuracy, learnedgates, clippingfrequency, firstepoch/time tofixed
  validation thresholds90/95/98/99 (blank means not reached).
* `fixed_intermediate_evaluations.csv`: originalbaseline fixed30epoch heldout
  teacher/rollout evaluation, explicitly separate from its90epoch final. It is
  never averaged with the final endpoint in architecture summaries.
* `architecture_summary.csv`: seedmean and sampleSD, with number of seeds.
* `all_epochs.csv`: flattened epoch histories for custom future plots.
* `*_length_metrics.csv`: counts and metrics by pathactionlength bins.
* `paired_comparisons.json`: primary/secondary paired effects and finite-test
  maze-bootstrap intervals; seed effects/SD reported separately. Ratios resample
  wholemazes with all their tokens, not independent tokens.

For multiple seeds, the bootstrap averages per-seed correctness differences on
each maze. It describes the mean of those fixed fitted models' metrics; it is
not an ensemble predictor formed by voting or averaging logits.
* `historical_curves.png`: original stored output, clearly labeled historical.
* `controlled_learning_curves.png`: fresh validation curves byepoch and measured
  cumulativetrainingtime; mean±seedSD and individualtimecurves.
* `test_token_vs_rollout.png`: tokenaccuracy versus autonomouscanonical and
  shortestpath success, percentunits and seedSDbars.
* `test_depth_sweep.png`: fixedcheckpoint teacherforced depthdiagnostic (trained6;
  evaluated1,3,6,9,12). Not proof of stabletesttime scaling.
* `gradient_clipping_and_gates.png`: clipping prevalence andlearnedinputgates;
  diagnostic observations, not isolated causal mechanisms.
* `analysis_manifest.json`: creationtime, source runs, analysiscodehash, caveats.

Firstactionaccuracy is derived from token0ofteacher predictions: onlyBOS precedes
it, so it is a useful planningdiagnostic without groundtruthactionprefix support.

## Exploratory harder mazes

`harder_eval_plan.json` was written before harder-set results. Two fixed8×8sets:
512mazes with wallprobability.40 (minpath4), and512mazes with wallprobability.28
conditioned on at least12moves. These are harderdistributions, not largergrid
extrapolation and not guaranteed to be outside training pathlength support.

```powershell
& $experimentPython "$experimentRoot\evaluate_harder_mazes.py" --prefix controlled12k --variants tied6_full,untied6_full
& $experimentPython "$experimentRoot\evaluate_harder_mazes.py" --prefix full50k --variants original_baseline,original_looped,tied6_full,untied6_full
```

Run aftertraining to avoid GPUtiming interference. These evaluate only fixed
final checkpoints with nofit/tuning. Per-examplepredictions, seeds, data and
checkpoint hashes are saved under `analysis/exploratory_harder/`. Preserve
exploratory labeling in future analysis; no OODresult is a formal epiplexity test.
