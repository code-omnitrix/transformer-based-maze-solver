# Independent review before original-scale training

**Outcome: no blocking correctness issue found.** This review executed only CPU diagnostics and wrote separate review artifacts. It did not edit the training scripts or notebook and did not use the GPU.

## Reviewed versions and scope

- `fullscale_experiments.py` SHA256:`e670b6b617d8e87e5d1f1183fad59cbe87dbc9eb222fb1f904f51857ce201d60`.
- Imported `experiment_suite.py` SHA256:`e855368e4a0ec9371330dd5e962d3ec30446c662e1f671769b6ae0c1ecb7cc0b`.
- Notebook SHA256:`fd187163c7ea47f268a1734f1e6ab3c6dc306906950bed1f0793b030743e4701`.
- Planned runs: original baseline90epochs, original loop30epochs, controlled tied6_full30epochs, controlled untied6_full30epochs. Baseline has an additional fixed epoch30 checkpoint/evaluation.
- Reviewer script:`fullscale_review_check.py`; machine-readable findings:`fullscale_review_checks.json`.

## Checks executed and passed

1. Set`CUDA_VISIBLE_DEVICES=-1` before importing PyTorch. No training was run. Original models were instantiated only on CPU.
2. Executed the actual full-scale `full_data()` function with its output ROOT redirected to the isolated `fullscale_review_scratch` directory. This generated original-code50k training,5k validation and fresh5k test examples. Production cache, runs and live scripts were not changed.
3. Compared its training/validation grids, normalized to uint8, against the earlier independently generated original-code seed42 replay. Both array hashes matched exactly; every shortest-path length histogram bin matched. Train has421,569 scored target tokens and28 sequence columns including BOS; validation41,903 target tokens and26 columns.
4. The original generator extraction includes the actual saved `generate_solvable_maze` and`MazeDataset`, fixed U,L,R,D order, min4 length rejection, NumPy legacy RNG and Python RNG seed42, with train generated before validation. This is separate from the controlled matrix's`default_rng` data.
5. Constructed factory and direct-notebook instances separately after reseeding42. For both original baseline and original loop, initial state dictionaries have identical keys and byte-identical tensor values. Counts are1,338,119 and352,521.
6. Compared direct saved-class forward paths against factory forward paths on the same CPU grids/prefixes. Maximum absolute logit difference is0 in eval mode and0 in training mode with synchronized dropout RNG. The factory encode/decode adapters also reproduce forward logits exactly in eval mode.
7. Read the training/checkpoint/evaluation code: optimizer defaults match the original AdamW invocation (lr0.001, weight decay0.0001), batch128, PAD ignore0, full teacher forcing, clip norm1, float32 and no AMP. Models reset into training mode every epoch. Dataset splits are shared by all variants.
8. Checkpoints preserve weights, optimizer and Python/NumPy/Torch/CUDA/loader RNG states. The baseline checkpoint30 evaluation saves and restores RNG around test evaluation, preventing evaluation-loader RNG draws from changing later dropout/shuffling trajectory. Latest checkpoints are written every10epochs, with fixed30/final checkpoints. No test-selected stopping or architecture selection is implemented in the script.

## Interpretation and documentation limits

- This is an original-scale fresh replication using saved definitions. It is **not historical bitwise reproduction**. Each architecture starts from a new seed42; a dedicated shuffle generator seed1042 supplies matched minibatch orders. The original notebook's global RNG/execution history is unknown, and its shuffle/dropout RNG coupling differs. CUDA deterministic algorithms are not forced. Exact original numerical results need not recur even when code/data match.
- The review verifies exact original constructor defaults, including the normal model's outer Xavier reset and loop model's default submodule initialization. The controlled models deliberately use their common initializer; they are a causal control family rather than exact copies of the historical comparison.
- Matching seed numbers across tied and untied models does not make their parameter initialization a paired tensor intervention. The within-seed sharing contrast pairs data order and seed label; normalization-only state-identity checks are a separate, stronger control. Report both accurately.
- Per-step NLL comes from dropout-active training on reused examples. It is suitable for optimization curves, not a validated prequential/epiplexity estimator. Code/logging should retain that distinction even though the full-scale step record lacks the matrix stream's explicit note.
- First-token accuracy is not a named summary field, but can be computed from saved per-maze teacher predictions/targets. Action accuracy and EOS counts are logged; strict rollout includes termination, invalid symbols/moves, goal success, optimality and canonical exact metrics.
- `summary.total_seconds` is captured before the final test evaluation, whereas the registry completion duration includes it. Baseline wall elapsed also includes its epoch30 test evaluation. Use cumulative training seconds for training-budget comparisons, and label end-to-end times precisely. This is a reporting issue, not a training correctness failure.
- Review-grid hashes use uint8 to match the independent audit; training manifests/prediction hashes use native int64 grid bytes. Their SHA strings will differ for the same logical maze unless dtype is normalized. Do not directly join files by these differently encoded grid hashes. Use run/split/example IDs or explicitly normalize and recompute.
- Predeclaration and production cache integrity are recorded when the full-scale script launches. A changed script hash after this review would require checking the change before treating this review as covering it. The reviewer did not test GPU training or bitwise GPU resume.

## Paper-version check

On2November2026 the primary [arXiv abstract page](https://arxiv.org/abs/2601.03220) lists v2,16March2026, as its latest revision. The analysis is explicitly pinned to [v2](https://arxiv.org/html/2601.03220v2); no newer version was listed by the retrieved metadata. This version check adds no new empirical confirmation of the paper.

## Review artifacts and rerun

Run `fullscale_review_check.py` using the documented llmdyn interpreter. It writes only its own resultJSON and scratch dataset/cache/manifest. Scratch contains independent duplicate datasets for checking; it is not a training run and must not be counted as a new neural experiment. Its hashes and machine-readable checks allow later verification without relying on this prose.
