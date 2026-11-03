# Independent experiment harness review

Review date: 2 November 2026 (Asia/Calcutta). Reviewer scope: scientific and implementation checks of `experiment_suite.py`, read-only with respect to that file and its existing runs. New independent verification files are `harness_validation.py` and `harness_validation_results.json`. No GPU training was started or interrupted by this review.

## Verdict

No training-invalidating implementation defect was found in the seven-variant controlled matrix. The forward computations, target shift, masking, PAD handling, token/sequence aggregation, strict trajectory solve scoring, and recorded epoch losses passed the checks below. Two reporting fields require care: the manifest's `path_length_histogram` includes BOS/EOS, and the rollout's `terminated` field does not mean unconditional EOS emission. Neither changes the logged token accuracy or strict solve-rate outcomes.

The controlled family is not an exact re-run of the notebook's initialization or original data sequence. It is suitable for architecture contrasts within the controlled experiment, subject to finite seeds, reduced data/epochs, and comparable optimization budgets. It must not be described as reproducing the user's numerical results.

## Architecture and forward-path checks

- `experiment_suite.py:66–89`: each model uses the same embeddings, width, feedforward width, eight heads, dropout default, final norms, and consistent core Xavier matrix initialization. The Post-LN flag is explicitly `norm_first=False`; other variants use `True` (`:117–124`). The baseline `nn.Transformer` in notebook cell 2 includes final encoder/decoder norms, so the controlled Post-LN final norms are appropriate.
- `:96–113`: initial-state injection and sinusoidal iteration encoding are added **before** the shared block at every iteration, including iteration one. This matches notebook cell 5 exactly: when the gate is one, iteration one's block sees `2*h_init + step_encoding(1)`. The ordering and residual path were not accidentally changed.
- Independent CPU state-mapping check: copied every original notebook baseline embedding/core/final-norm/output parameter to `postln4`, and copied every original looped parameter to `tied6_full`. Evaluation forward outputs on two random grids and padded target prefixes matched exactly: maximum absolute difference **0.0** for each architecture. This directly checks computation rather than relying only on class-name similarity. It is an evaluation-mode equivalence check at mapped weights, not a claim of identical random initialization or identical training trajectories.
- Initialization caveat: the original `nn.Transformer` clones encoder/decoder prototype layers and resets matrix parameters; cloned one-dimensional biases initially match across layers. The controlled family independently constructs layers (`:75–76`) before resetting matrices (`:84–89`), so those biases are initialized independently. Controlled `postln4` also uses a 128-position action embedding rather than the original baseline's 64 (`:69,74`). Neither makes the comparison internally invalid; both prohibit calling that run a bit-identical original baseline.
- The original looped notebook uses default `TransformerEncoderLayer`/`TransformerDecoderLayer` initialization, whereas the controlled family deliberately applies the common Xavier convention. Thus original-versus-controlled scores cannot isolate architecture alone. Within the seven controlled variants the initialization convention is held consistent, although different parameter counts naturally consume random draws differently.

## Data, targets, masks, and teacher-forced metrics

- `:31–46`: grid cells follow the same Bernoulli-wall distribution, start/goal are distinct uniformly selected cells and overwrite walls, only solvable mazes with shortest path length at least four are accepted, and BFS neighbor order remains U/L/R/D. The harness changes the random-number generator implementation and combines random draws into NumPy `default_rng`; this preserves the intended sampling distribution, not the notebook's exact examples. The manifest already discloses this at `:280`.
- `:274–280`: train, validation and test use separate fixed generator seeds and hash overlap is checked explicitly. Identical saved data are reused across architecture and training-seed runs. The shuffled DataLoader uses its own generator reset to `training_seed + 1000` for each architecture (`:297–299`), so architecture construction does not alter minibatch order.
- `:306–307` and `:162–165`: decoder input is `seq[:,:-1]`, label is `seq[:,1:]`; cross entropy ignores PAD=0, with no label/input alignment error. BOS is an input, not a scored target. EOS is scored; action-only accuracy is additionally available (`:168–169`).
- `:108–112`: the boolean upper-triangular mask blocks future action inputs, while `seq.eq(0)` masks padded keys. Independent CPU future-input perturbation checks passed for all seven variants: prefix-logit maximum absolute change **0.0**. The first input is always BOS, so padded suffix queries are not left with all keys masked.
- `:166–173`: token accuracy is pooled over non-PAD target tokens; teacher-forced exact accuracy requires all scored tokens, including EOS, to be correct. The latter remains teacher forced and should not be renamed autoregressive sequence success.
- Cross-check of completed `controlled12k_postln4_seed42`: its 1,000 saved per-example teacher records sum exactly to summary token counts, correct counts, and exact-match counts. The 512 rollout records also reproduce all saved rollout summary fractions exactly.

## Rollout and path scoring

- `:191–209`: generation begins from BOS and repeatedly feeds model-generated tokens, with per-example output recording stopped at its first EOS. Sequences are evaluated independently even when completed batch members continue internal decoding. No cross-example attention exists, so completed members cannot contaminate active members.
- `:175–188`: an illegal move, PAD or BOS in an action sequence invalidates the trajectory; success requires all moves legal, final position at the goal, and EOS. `optimal_solved` compares the generated movement count with the BFS shortest reference length. Consequently alternative shortest paths count as optimal solves, while canonical exact match is reported separately.
- Seven independent handcrafted trajectory cases passed: canonical solution; different shortest route; legal longer route; goal reached without EOS; illegal move followed by EOS; PAD followed by EOS; premature EOS. This checks that shortest-path success is not conflated with canonical imitation.
- Generation is capped at 64 predicted tokens (`:198`). The run's 128-position embeddings comfortably cover this input length. Test summaries are therefore success within that fixed decoding budget, not success with unlimited rollout.
- **Reporting caveat — `terminated`:** evaluation breaks at the first illegal token (`:180–183`), before inspecting a later EOS. Thus the field means EOS was encountered before any earlier invalid action, rather than “the generator emitted EOS.” In the reviewed baseline run, **512/512** saved predictions contained EOS, while `terminated` counted **116/512**. `solved` and `optimal_solved` remain correct because an invalid path should fail regardless of later EOS. For termination plots, derive `emitted_eos = 2 in prediction` from saved records. Preserve the existing raw field and document its semantics.

## Losses, optimizer, logging, and recoverability

- `:296–309`: AdamW uses learning rate 0.001, weight decay 0.0001, and norm clipping at 1.0. The loss for each minibatch is the mean over non-PAD tokens. `train_token_mean_loss` weights each minibatch mean by its non-PAD count; `train_batch_mean_loss` separately preserves the unweighted minibatch-mean convention of the original notebook (`:309,317`). These are intentionally distinct estimands.
- For all **30 epochs / 2,820 optimizer steps** of the reviewed completed baseline run, recomputing epoch NLL from the saved per-step mean loss times target-token count reproduced the logged token-weighted epoch loss within 1e-10. Final step numbers matched each epoch record. At training size 12,000 and batch size 128 there are `ceil(12000/128)=94` steps per epoch, as recorded.
- `cumulative_observed_training_nll_nats` at `:312` resets every epoch; the adjacent note explicitly documents this. To plot the full-run cumulative observed NLL, cumulatively sum `pre_update_teacher_forced_token_mean_nll * target_tokens` across all saved step rows. This is dropout-active, repeated-data training NLL, **not** prequential MDL or an epiplexity estimator.
- Training and validation times are separated (`:315–320`); per-step host timings include loss/gradient synchronization and logging overhead. Final checkpoint/test diagnostics are included in total runtime but not cumulative training time (`:323–330`). Use the stated denominator consistently when comparing efficiency.
- Configurations, script/notebook hashes, dataset hashes, device/software versions, run-start/complete/failure registry, step/epoch rows, checkpoint, and per-example final predictions are saved (`:212–219,286–342`). Seeds are fixed but deterministic CUDA algorithms are explicitly not forced, so bit-identical GPU reproduction is not promised.
- **Recoverability limitation:** checkpoint saving occurs only after the final epoch (`:323–324`). There is no automatic resume CLI or periodic checkpoint in this version, and an interrupted active run could lose its trained weights despite retaining logs. The saved final checkpoint includes model, optimizer and Python/NumPy/Torch/CUDA/DataLoader RNG states, sufficient ingredients for a separately implemented careful continuation. Do not claim that automatic interruption recovery is already implemented.

## Other limitations for future plotting and reuse

- **Histogram labeling:** `path_length_histogram` at `:62` uses `(seqs != 0).sum(1)`, which includes both BOS and EOS. Convert histogram keys to movement lengths by subtracting two, or recompute from labels. The per-example evaluation `path_length` fields already use the correct movement length (`:172,208`).
- Dataset cache reload (`:49–51`) checks sample count but does not independently validate grid size, wall probability or generator seed against stored metadata. Current hard-coded filenames/seeds and defaults are consistent; future generator changes should produce new dataset paths or assert all metadata.
- Benchmark-mode peak CUDA memory (`:270`) is not reset between variants, so the saved maximum is cumulative across earlier benchmark allocations. Do not use that field for independent per-variant peak-memory comparisons. Training correctness is unaffected.
- Inference K sweeps (`:331–336`) evaluate teacher-forced metrics only. Any conclusion about additional-loop **maze solving** requires separate autoregressive rollouts at each K.
- The test split is reused across prespecified model contrasts. It should be described as a research evaluation set; a final untouched confirmation split would be preferable after adaptive architecture selection.

## Reproduce the independent checks

Run from PowerShell:

```powershell
& 'C:\Users\omnap\miniconda3\envs\llmdyn\python.exe' 'D:\Projects\Loopedtrans\research_notes\Looped transformer accuracy investigation\harness_validation.py'
```

This uses CPU with one thread, imports the harness without invoking its main training function, and writes only `harness_validation_results.json`. The saved baseline run is the aggregation-check fixture. The original notebook, running harness, datasets and run results are not modified.
