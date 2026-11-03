# Epiplexity and the maze experiments

## What does the requested paper actually claim about recurrence?

### Takeaway
**High confidence about paper contents; unconfirmed for these mazes:** the paper is directly relevant, but a looped model winning does not establish higher epiplexity. Its own looping experiment can produce the opposite direction.

### Cited findings

The following compact source summary and equations are the paper-derived portion of these notes (keep the final report's paper-derived paraphrase within200 words; the operational proposals below are our analysis).

Finzi, Qiu, Jiang, Izmailov, Kolter and Wilson, *From Entropy to Epiplexity*, arXiv2601.03220v2 (16March2026), separate model-program bits from residual code length at a fixed computational budget. Conditional definitions apply to label-given-input tasks. Prequential estimation uses cumulative pre-update loss above final-model loss; it is heuristic. Requential estimation uses cumulative teacher-student KL plus coding overhead. Both require optimizing the two-part code under a compute constraint. [Sections3–4, equations5–9](https://arxiv.org/html/2601.03220v2).

\[
P^*_{Y\mid X}=\arg\min_{P\in\mathcal P_T^X}\{|P|+\mathbb E[-\log_2P(Y\mid X)]\},\quad
S_T(Y\mid X)=|P^*|,\quad H_T(Y\mid X)=\mathbb E[-\log_2P^*(Y\mid X)].
\]

Their ECA experiment predicts intermediate states autoregressively. Beyond a compute threshold, looping improves MDL while **decreasing** epiplexity; AppendixC.8 reports benefit for16 loops, not2,4 or8. This differs from hidden-state weight tying. Higher epiplexity does not guarantee specific downstream generalization. [Section5.3.2, Figure6; C.8; Section6.1](https://arxiv.org/html/2601.03220v2).

The authors' repository logs cross-entropies, prequential `K_auc`, and requential `K_req`, and uses model-size/training-duration sweeps to construct compute frontiers. [Authors' repository](https://github.com/shikaiqiu/epiplexity).

### Inferences

- **Confidence high:** the notebook cannot currently confirm the epiplexity framework. It records rounded epoch-average training losses and token accuracy, with no explicit model code, fresh-example prequential stream, held-out NLL, or compute frontier. Architecture comparisons at different budgets do not identify a data property's epiplexity.
- **Confidence high:** fewer learned weights are not a measured shorter optimal probabilistic program. Parameter count omits precision, initialization/optimizer recipe, weight compressibility, recurrence execution cost, and alternative programs such as the known BFS solver.
- **Confidence high:** the maze task is conditional prediction`P(path | maze)`. The canonical BFS target is deterministic given the maze and its fixed tie order; ordinary conditional Shannon entropy is therefore zero in the idealized task. That does not mean a finite neural architecture can efficiently learn or execute the function. Any residual trained-model NLL is prediction error of the chosen model, not intrinsic randomness of the label generator.
- **Confidence medium:** a recurrent model might learn a short reusable update resembling wavefront propagation. If so, successful behavior could accompany a simpler effective description rather than more structure bits. Current hidden recurrence is not observed BFS expansion, and6 layers of global attention do not correspond automatically to6 local BFS hops.
- **Confidence high:** an accuracy gain does not violate data processing inequalities. Input stays the same; changing the observer/computation can improve accessible predictions without adding new maze facts.

### Gaps

- We have verified the paper's definitions and experimental design, not independently reproduced its ECA result or proofs. The paper is an arXiv preprint in the inspected metadata; do not invent a venue or review status.
- The true minimum over time-bounded programs is not computable by a small finite architecture sweep. Such a sweep yields a family-dependent proxy, not exact epiplexity.

## What would be a falsifiable later-stage investigation?

### Takeaway
Choose an explicit maze-data intervention, fix the observer family and budget, measure a coding proxy alongside genuinely held-out transfer, and preregister which outcomes count against the hypothesis. This is stronger than attaching the term epiplexity to the existing accuracy difference.

### Cited findings

The notebook sources for the task and confounds are [aiml.ipynb](../../aiml.ipynb), cells1–6; baseline/replay facts are in [independent_statistics.json](independent_statistics.json). The following design is proposed analysis, not paper-reported evidence or completed training.

**Proposed hypothesisE1: intermediate planning supervision makes reusable maze structure more accessible within a budget.**

1. Fix maze arrays, train/validation/test splits, model family, data-token/FLOP accounting, seed set and hyperparameter-search budget before training.
2. Compare final-path supervision with BFS-distance fields, BFS-frontier traces, or waypoint intermediates. Count all extra input/target tokens, teacher generation costs and test-time steps; compare at equal total compute as well as equal final-task examples. A larger target stream is not a free intervention.
3. Evaluate strict autonomous shortest-path success on unseen maze layouts, held-out long paths, changed wall density and prescribed detours. Larger grids require shared coordinate encodings or a fixed sufficient position table; the notebook's8×8 learned row/column embeddings alone do not support extrapolation.
4. Fit a prequential coding **proxy** with truly fresh samples before each update; use a separate final evaluation stream for final NLL and disclose approximation. Compare two-part-code frontiers across model width, training duration and intervention. Keep the final-task marginal consistent when auxiliaries change; joint trajectory likelihood and path-only likelihood are different objects.
5. Preregister support as repeatable transfer gains and a stable intervention ranking after budget controls. If rankings vary with seed/learner, depend on uncounted intermediate tokens, or disappear under normalization/init controls, report thatE1 is weakened. A failure ofE1 is not a disproof of the formal definition.

**Proposed hypothesisE2: recurrence trades execution time for a simpler reusable algorithm.**

- Hold task distribution and all non-tying components fixed; sweepK and training horizon. Compare matched-depth untied models, parameter-matched shallower models, and the explicit BFS algorithm as a correctness/complexity reference.
- Look for a compute-performance crossover and simpler estimated code under recurrence while maintaining strict rollout success. Agreement with the paper's qualitative scenario would be suggestive, not an independent confirmation of its theorem or ECA experiment.
- Falsifying signatures for this proposed mechanism include no crossover over the tested range, no robust rollout benefit, failure to transfer to harder detours, or the entire gain being explained by pre-LN or anchoring.

**Proposed negative controls.** Shuffle maze-to-path pairings using disjoint train/test maps; train with random target tokens of matched length; and use deterministic trivial path tasks. Log them as different distributions, and expect different attainable errors. These controls help detect memorization/leakage and estimator artifacts; do not infer epiplexity directly from poor accuracy or from random-data loss without modeling the baseline code.

### Inferences

- A fresh pre-update coding sum can be accumulated in bits as`sum(NLL_nats / ln(2))` with target-mask counts. A model-length proxy can subtract final-model NLL evaluated on the same encoded stream or an explicitly documented iid estimate. Both amounts must use identical token units, and all mask/EOS rules must be stored. On fixed-data multi-epoch training, repeated examples can be memorized; the notebook's epoch loss is not an iid prequential likelihood.
- Logging dropout-mode training loss is insufficient for a clean coding measurement. The synchronized probability model must have reproducible evaluation probabilities, or its randomness must be part of a specified shared code. Record the actual pre-update probabilities or deterministic eval-mode NLL before updates.
- Cumulative optimization work for tied models must include each block application; use measured/profiler-estimated FLOPs and wall-clock, and document attention/padding costs. Unique parameter count alone particularly misstates this comparison.

### Gaps

- E1/E2 are not yet full reproductions of the paper. An actual reproduction would use its ECA rule54 setup, model-size/depth/loop grid, model coding, all intermediate-token costs and budget frontier. This may be computationally much larger than the maze investigation.
- Requential coding needs a teacher/student protocol and KL measurements; no such data exists in the historical notebook. Do not retrospectively label an ordinary loss-curve AUC as requential coding.

## What should be logged now to make that later study possible?

### Takeaway
Preserve exact per-step likelihood and budget accounting now, so future plots can distinguish optimization, generalization and coding estimates without reconstructing missing state.

### Cited findings

These are documentation requirements for the proposed study, based on the above operational analysis:

| Stream | Required fields |
|---|---|
| Run manifest | source hashes, architecture, tie pattern, normalization, gate/init settings, train/testK, dtype, library/GPU versions, all RNG seeds/states, optimizer, clipping, LR schedule, split hashes, hardware |
| Per-update | step, epoch, batch sample IDs/hash, fresh/reused flag, valid target tokens, pre-update eval-mode summed NLL nats, loss used for gradient update, LR, grad norm, loop gate values, elapsed seconds, cumulative FLOPs estimate |
| Per-checkpoint | full weights, optimizer/RNG state, exact total train/eval FLOPs, held-out NLL sums and counts, action-only/EOS/first-token accuracy, selected checkpoint criterion |
| Per-maze prediction | maze ID/hash, true canonical actions and distance, generated actions including invalid symbols, goal success, shortest success, canonical exact, termination reason, invalid move count, steps, testK, difficulty stratum |
| Coding study | encoded sample order, probability-model mode, NLL conversion units, final-model NLL on encoded stream, code overhead assumptions, teacher/student checkpoint IDs and KL if used, hyperparameter search grid and frontier selection |

### Inferences

- Save both raw streams and derived summaries, with plots generated by versioned scripts. Keep source provenance flags so old persisted traces and new experimental runs cannot silently merge.
- Report maze-level paired uncertainty and training-seed uncertainty separately. Neither is supplied by a smooth learning curve.

### Gaps

- Historical missing logs cannot be repaired after the fact; the new study must state which measurements are unavailable for the old runs.
