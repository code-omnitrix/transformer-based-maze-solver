# Independent final controlled-matrix verification

**Outcome: numerical and metric checks passed.** The matrix contains21 completed runs:7 architectures ×3 initialization/shuffle seeds42,43,44, each at a fixed30-epoch endpoint. This review read raw logs, datasets and source snapshots; it imported no PyTorch, loaded no checkpoint, performed no training and used no GPU. It wrote only its own review/check files, leaving the active full-data runs untouched.

## Completeness and provenance

- Verified every expected architecture/seed pair exactly once, all configs at12,000 train/1,000 validation/1,000 test examples, batch128,30epochs,512 rollout examples, width128 rather than the small option, AdamW lr0.001, weight decay0.0001, clipping1.
- Every run has30 sequential epoch records,94 updates per epoch and2,820 sequential per-update records. Total checked:630 epoch records and59,220 update records. Per-epoch target-token counts agree with the actual training array. Token-weighted and batch-weighted epoch losses reproduce their step-log aggregates.
- Every run records the same harness SHA256`e855368e4a0ec9371330dd5e962d3ec30446c662e1f671769b6ae0c1ecb7cc0b`; its preserved source snapshot matches this hash. All notebook hashes match the unchanged original. The analysis manifest's script fingerprint matches the inspected analysis script.
- Exact NPZ hashes are common across the21 configs and match files on disk: train`cafaf4d7b50f9035a29353d688d96af8e9832a71530fc1a5a2d3ea4e542fa351`; validation`c2702be904b522eb47ba02c999c0a725bcd5c8bd0c98c446d975817e0cbc8b7e`; test`499991fdc37950fe43504550576018fea7325b9573653b654000798c16e861e8`.
- The test set contains8,505 scored tokens:7,505 actions plus1,000EOS. Mean action length is7.505. These are **controlled-matrix data**, distinct from the original-generator replay with7.3806 mean validation path length. Raw dataset histograms count BOS+actions+EOS; the derived action lengths correctly subtract2.

## Predictions and metric semantics

- Checked all21,000 teacher-prediction rows against actual test-array target sequences. Per-example IDs/order, token counts, canonical path lengths, correct counts and exact flags agree. Recomputed aggregate token/action/EOS/exact counts and token/action/exact accuracies match summaryJSON and finalCSV. First-action accuracy matches the saved first prediction, which conditions on BOS and the maze.
- Checked all10,752 autonomous prediction rows against actual maze grids using an independent trajectory simulator. Recomputed out-of-bounds and wall collisions, illegal action symbols, termination before the first illegal trajectory action, goal reach, strict solved, shortest-path solved, canonical exact, move counts and traces. Every raw flag matched.
- Reconstructed **actual EOS emission** from the whole generated token list independently of tracing. A prediction can emit EOS after a wall collision, while raw`terminated` remains false because tracing stopped at the illegal move. DerivedCSV correctly preserves this distinction. Max-step-truncation flags and each failure class agree with predictions. Strict solved/optimal labels were not changed by this diagnostic correction.
- Verified the per-run failure-countJSON and length-bin denominators/token ratios against raw predictions. Across all architecture/seed runs the outcome classification counts are5,709 wall collisions,28 out-of-bounds,127 EOS-at-wrong-goal,98 valid nonoptimal solutions,4,243 optimal canonical paths and547 optimal alternative paths. These are repeated evaluations of the same512 test mazes by21 models, **not10,752 independent mazes**.
- Aggregate held-out NLL cannot be reconstructed from argmax-only teacher-prediction rows. It would require reevaluating checkpoints or saved probabilities. This review did not do so while timed full-scale training was active. Its NLL check is limited to the consistency of recorded training step/epoch arithmetic, not independently recovered held-out likelihoods.

## Seed summaries and paired contrasts

Every architecture mean and sample standard deviation was independently recomputed from its three per-seed final rows. All reported values agree. SD uses the sample denominator`n−1`; it is not a confidence interval. Times are seconds, raw accuracy fields are0–1 fractions, and displayed percentages multiply both means and SD by100.

| Paired contrast (after−before) | Mean token effect, pp | Sample SD of paired seed effects, pp |
|---|---:|---:|
| tied6_full − untied6_full | −2.719969 | 4.490573 |
| preln4 − postln4 | +15.445816 | 1.451186 |
| preln6 − preln4 | −3.492063 | 5.852132 |
| tied6 − preln6 | −5.149912 | 8.538447 |
| tied6_anchor − tied6 | +1.297276 | 3.328521 |
| tied6_full − tied6_anchor | −1.477562 | 2.555158 |

The primary sharing contrast's individual seed effects are−7.901235,+0.047031 and−0.305703pp. The observed mean loss from tying in this finite study is not a universal tying disadvantage: its sign and size vary by seed and only one training dataset, learning rate, initialization family and budget were tested.

## Conditional bootstrap independently reproduced

All six teacher-token and six rollout-optimal conditional95% intervals were independently regenerated from raw per-maze outcomes using5,000 replicates and RNGseed20261105; they match the saved intervals to numerical tolerance.

For teacher-token effects, the analysis first averages each maze's correct-token difference across the **three fixed fitted models**, resamples1,000 maze IDs with replacement, and divides summed correct-token differences by summed target-token counts. All models use the same sampled maze IDs. For rollout-optimal effects it resamples512 paired maze IDs and averages the fixed models' binary success differences. This is neither a logit/voting ensemble nor bootstrap resampling of initialization seeds.

Thus these intervals quantify finite-test-maze variation **conditional on the fitted models and training data**. They do not quantify retraining variation, new training-data variation, or the uncertainty in an architecture's expected performance across future runs. In particular, the primary conditional token interval excludes zero even though its pairedseedSD is4.49pp and one seed has a small positive effect. Those statements answer different questions. Report them side by side rather than calling the conditional interval a general architecture confidence interval.

## Plot and table checks

Visually inspected all five PNGs and read their plotting code: learning curves, test metrics, depth sweep, gradient/gate dynamics and historical curves. The gradient/gate figure shows architecture means, without SD bands; its caption should state this explicitly.

- Learning curves correctly label n=3 per architecture, mean±seedSD,94updates/epoch, validation teacher-forced token accuracy in percent, and measured cumulative training seconds. The time panel displays individual runs.
- Test bars correctly separate teacher-token accuracy, autonomous canonical exact and autonomous shortest-path success; percent units and seedSD error bars agree with tables.
- The initially viewed standalone test-bar and depth images omitted n=3/test denominators. The owning agent corrected their captions and regenerated the analysis. Final images now explicitly state three seeds and1,000 teacher-forced test mazes; the bar image also states512 rollout mazes. I visually checked the revised images. Depth sweep measures teacher-forced accuracy at the same checkpoint and is not evidence of autonomous test-time planning gains.
- Historical curves are clearly labeled saved notebook output and unknown historical execution state. They are not mixed into fresh-run means or SD.
- The experiment notes correctly call whole-maze intervals conditional and acknowledge the limited number of seeds. Tables display3seeds and percent units. No statistical significance across retraining seeds follows from the maze bootstrap.

## Reproduce this audit

Run`final_matrix_review_check.py` with a Python environment containing NumPy. It reads the fixed matrix logs and overwrites only its own`final_matrix_review_checks.json`. That machine-readable file records counts, source/data fingerprints, independently calculated paired effects and scope limitations. The script and this review are separate from the active full-scale experiment sources.

During review, the analysis script changed for captions and separate full-scale epoch30 support, briefly making its source fingerprint newer than the matrix manifest. The owning agent finalized and froze the script at SHA256`899985b5fbf7a65efdb08c33381ddca4dcae6b6512159e0c9513ce9398245b4b`, archived an immutable source snapshot, and regenerated the matrix analysis. A complete independent recheck of this final state passed. No raw matrix numerical inconsistency was found before or after the edits.
