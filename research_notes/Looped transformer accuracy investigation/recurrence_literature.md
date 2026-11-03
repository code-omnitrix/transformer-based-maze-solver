# Literature on recurrent transformer explanations

## What does the primary literature establish?

### Takeaway

There is credible primary evidence that repeated computation with shared parameters can help algorithmic tasks and reduce parameter requirements. That evidence does not identify the cause of this notebook's gain: normalization, effective depth, input reinjection, step encodings, and parameter count all change together. Confidence is high about this confounding; medium about recurrence being useful for this particular task; low about any one causal explanation before controlled experiments.

### Cited Findings

**L1 — Universal Transformers.** Dehghani et al., 2018/ICLR 2019; inspected arXiv v3 dated 5 March 2019, §§2.1, 3.4 and Table 4. The architecture repeats a shared transformation across depth and adds position/time encodings. On copying, reversal, and addition, models trained at length 40 were evaluated at length 400. UT improved over the conventional Transformer, but addition remained far from solved: character/sequence accuracy was 0.34/0.02. Thus recurrence can improve extrapolation without guaranteeing it. Their dynamic halting is absent from this notebook. [Primary full text](https://arxiv.org/html/1807.03819v3#S3.SS4)

**L2 — ALBERT.** Lan et al., 2019/ICLR 2020; inspected arXiv v6 dated 9 February 2020, §§3.1 and 4.5. Sharing controls parameter growth with depth. However, their direct sharing ablation found that sharing all layers reduced benchmark average by 1.5 points at embedding size 128 and 2.5 at size 768; much of the decline came from sharing FFNs. They observed oscillating, rather than convergent, representations across shared layers. This is useful counterevidence to both universal superiority and automatic fixed-point interpretations. Other ALBERT changes also contribute to its overall results. [Primary full text](https://arxiv.org/html/1909.11942v6#S4.SS5)

**L3 — Layer normalization placement.** Xiong et al., ICML 2020, §§3–4. Mean-field analysis associates Post-LN with large expected gradients near the output at initialization, motivating warmup. Pre-LN has better-behaved initialization gradients and their experiments remove warmup while retaining competitive results. This establishes normalization placement as a serious alternative explanation for the notebook, whose baseline is Post-LN and looped model Pre-LN, both trained with constant AdamW learning rate 0.001. It does not prove the baseline here is unstable. [Primary proceedings paper](https://proceedings.mlr.press/v119/xiong20b/xiong20b.pdf)

**L4 — Looped Transformers are Better at Learning Learning Algorithms.** Yang et al., 2023/ICLR 2024; inspected arXiv v3, §§4, 4.1–4.2. Their in-context regression model uses 0.79M parameters versus 9.48M for the 12-layer baseline, and reinjects the input each iteration. Training supervises a window of loop outputs; it is not simply final-only loss at one fixed depth. Without injection, performance deteriorates past trained depth, and the trained iteration budget influences extrapolation. The results concern learning data-fitting algorithms from prompts, not supervised maze-to-actions translation. [Primary full text](https://arxiv.org/html/2311.12424v3#S4)

**L5 — End-to-end Algorithm Synthesis with Recurrent Networks: Logical Extrapolation Without Overthinking.** Bansal et al., 2022; inspected arXiv v2, §§3.1–3.3 and 5. Their convolutional recurrent models address optimal maze paths, prefix sums, and chess. Additional recurrence can degrade solutions, termed overthinking. Recall concatenates the input to recurrent features; progressive training starts from partially computed states and trains further improvement. The maze connection is strong, but their fully convolutional, whole-output model differs from the notebook's autoregressive Transformer. Additive learned injection here is not identical to their recall operation. [Primary full text](https://arxiv.org/html/2202.05826v2#S3)

**L6 — Reasoning with Latent Thoughts: On the Power of Looped Transformers.** Saunshi et al., ICLR 2025; inspected arXiv v1 dated 24 February 2025, §§2–5 and Tables 1, 3. They explicitly separate equal-parameter shallow models from equal-effective-depth models. On addition and p-hop induction, looping can nearly match the latter using fewer parameters. Language-model results separate reasoning gains from weaker perplexity/memorization. Their general simulation theorem increases model dimensions, so it is not a guarantee that an unchanged small shared block can reproduce any deep model. They explicitly identify the mechanism of the reasoning bias as an open question (§3.3). [Primary full text](https://arxiv.org/html/2502.17416v1#S2)

**L7 — Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach.** Geiping et al., 2025; inspected arXiv v2 dated 17 February 2025, §§3.1–3.3, 4.3, 5. This large recurrent language-model study uses input injection, randomized training recurrence counts, truncated backpropagation, and specially chosen normalization. It demonstrates gains with additional inference recurrence on reasoning benchmarks. Its training procedure and scale differ substantially from the fixed-six-loop notebook. Consequently, the existence of this result supports testing inference scaling here, but does not establish that this notebook already learned stable iteration. [Primary full text](https://arxiv.org/html/2502.05171v2#S3)

**L8 — Can Looped Transformers Learn to Implement Multi-step Gradient Descent for In-context Learning?** Gatmiry et al., 2024; inspected arXiv v1 dated 10 October 2024, §§3.3–3.7, Theorem 3.2. For a restricted linear-attention model on Gaussian linear-regression prompts, the population-loss optimum implements preconditioned gradient descent, and gradient-flow convergence is studied. These are assumptions about a different architecture, objective, and data distribution. The theorem is evidence that iterative algorithm learning can be established in a controlled setting, not evidence that a nonlinear maze Transformer implements BFS or gradient descent. [Primary full text](https://arxiv.org/html/2410.08292v1#S3.SS7)

### Inferences

The strongest literature-based ranking for the notebook is: first test normalization and added depth as ordinary explanations; then isolate shared weights, input reinjection, and loop-index encodings. This is a proposed experimental priority, not a conclusion that normalization causes all the gain.

The function class distinction matters. A shared transition can encourage reuse of one operation, while independent layers can specialize. Either could be advantageous depending on the task and training regime. The notebook's step encoding lets the same shared block behave differently at different iterations, so shared weights alone do not imply an iteration-invariant learned algorithm.

### Gaps

- No paper reviewed measures the exact combination of 8×8 Bernoulli-wall mazes, BFS U/L/R/D tie-breaking, learned 2D positions, final-only six-loop supervision, and this notebook's optimizer.
- Literature scores above belong to their respective tasks and metrics. They must not be plotted against this notebook's token accuracy as if directly comparable.
- No reviewed result identifies the causal contributor to this specific notebook's reported improvement.

## Which hypotheses can the local experiments distinguish?

### Takeaway

Use factorial contrasts wherever practical, with the same data, width, masks, loss, minibatch order, training steps, and evaluation examples. Label results at equal epochs, equal effective depth, equal parameters, and equal measured wall time separately. They answer different questions.

### Cited Findings

The need for a normalization control follows from L3; separate parameter/depth controls follow from L6; injection and training-depth controls follow from L4–L5 and L7. These links establish why the controls are scientifically motivated, not what their outcomes will be. [Normalization](https://proceedings.mlr.press/v119/xiong20b/xiong20b.pdf), [depth/parameter distinction](https://arxiv.org/html/2502.17416v1#S2), [injection](https://arxiv.org/html/2311.12424v3#S4.SS1), [overthinking](https://arxiv.org/html/2202.05826v2#S5)

### Inferences

All entries below are original, falsifiable hypotheses for this notebook, rather than paper-reported results. Confidence describes plausibility before the new runs, not causal certainty.

| Hypothesis | Plausibility | Required contrast or measurement | Predicted signature | Evidence against it |
|---|---|---|---|---|
| Pre-LN improves optimization under the present learning rate | High | Fixed untied 4-layer Post-LN versus Pre-LN; then test warmup and a small learning-rate sweep | A substantial gap shrinks when only normalization or warmup changes | Several seeds show no benefit from normalization control and the original gap remains |
| Six transformations help more than four | High | Untied Pre-LN at 4 versus 6 layers; shared Pre-LN at K=4 versus K=6 | Gains follow effective depth before tying is introduced | Added untied depth fails while shared models consistently improve under comparable tuning |
| Input anchoring improves information availability or gradients | Medium | Exact matched 6-step architecture with gate retained versus set to zero; repeat for tied/untied | Reinjection helps either or both architectures; learned gates and hidden-state norms change | Removing injection preserves or improves outcomes across seeds |
| Sharing supplies a helpful algorithmic inductive bias | Medium | Exact 6-step matched computation and features, tied versus untied weights; identical data; tune both comparably | Shared model generalizes better particularly on longer/harder paths, with gains in real solve rate | Advantage disappears under feature/depth/normalization controls or appears only on teacher-forced accuracy |
| Fewer parameters reduce overfitting | Medium | Plot train and validation loss on identical deterministic evaluation batches; vary unique training examples and sharing | Larger untied model's generalization gap is larger; sharing benefit increases as data decrease | Shared improvement is chiefly lower training loss without reduced train/validation gap, or independent-layer model wins after fair regularization |
| Loop-step encoding supports specialization rather than generic refinement | Medium | Same shared model with/without step encoding; train variable K; sweep K at test time | Step encoding helps trained K while extrapolation deteriorates; iteration-specific states appear | No encoding dependence and robust improvements across unseen K |
| The reported gap benefits practical planning | Unknown | Greedy autoregressive evaluation with strict move legality, goal arrival, EOS, shortest valid length, and canonical match | Solve rate and valid shortest-path rate improve on the same held-out mazes | Token improvement does not yield a meaningful solve-rate gain |
| The model learned stable iterative computation | Low until measured | K=1,2,4,6,8,12,16 at inference; stratify by path difficulty; optionally random-K/progressive-training control | Harder cases benefit from more iterations; gains persist beyond trained K; predictions stabilize | Accuracy peaks sharply at K=6, solutions regress, or latent states drift while predictions worsen |

A more direct mechanism probe would train simple linear probes from each encoder loop to each cell's BFS distance-to-goal (on a separate probe split), reachable set, or membership of a shortest route. Improvement of these probe metrics over loops would suggest useful planning representations. It still would not prove the network literally runs BFS: all-to-all attention can implement other computations, and a probe can extract information not used by the decoder. Causal interventions on the candidate representation would strengthen the case.

### Gaps

- If only one seed is feasible, every contrast remains a pilot. Maze-level bootstrap intervals describe evaluation-set uncertainty, not variation across training seeds.
- A tiny width or reduced training dataset may reverse the ranking. Reduced pilots do not reproduce the user's 50,000-example, d_model=128, 90/30-epoch experiment.
- A narrow untied model matched in parameter count also changes width and head dimensions. That answers deployment resource tradeoffs, not a clean causal test of tying.

## What should be logged, and what claims should be avoided?

### Takeaway

Future analysis needs the individual maze and generated trajectory as the evaluation unit, plus full training histories and a stable provenance record. Avoid substituting parameter efficiency, token accuracy, and runtime efficiency for one another.

### Cited Findings

The reviewed papers distinguish metrics and budgets: L1 reports character and sequence accuracy separately; L6 contrasts parameter and effective-depth budgets; L4–L5 document failures under additional recurrence. These motivate the logging scheme and interpretation boundaries below. [Token/sequence distinction](https://arxiv.org/html/1807.03819v3#S3.SS4), [budget distinction](https://arxiv.org/html/2502.17416v1#S2), [recurrence failure](https://arxiv.org/html/2202.05826v2#S5)

### Inferences

Recommended plot-ready schema:

- Training row: run_id, architecture, seed, epoch, optimizer_step, examples_seen, nonpad_tokens_seen, train_loss_sum, train_token_count, validation_loss_sum, validation_token_count, token_correct, token_total, teacher_forced_exact_count, example_count, elapsed_training_seconds, evaluation_seconds, learning_rate, gradient_norm, encoder_gate, decoder_gate.
- Run metadata: full config, checkpoint-selection rule, source/script hashes, notebook hash, dataset/split hashes, package versions, device, precision, deterministic flags, timestamp, status and failure traceback.
- Per-example evaluation: immutable maze_id/hash, grid/start/goal, BFS canonical reference, shortest length, predicted action tokens including EOS status, illegal-move indicator, reached-goal flag, shortest-valid flag, exact-canonical-match flag, first-error index, inference K, and inference latency.
- Curves: validation token error rather than only near-ceiling accuracy; real solve rate versus steps/time; performance by shortest length and obstacle density; K-sweep solve rate; seed-level distributions; train/validation loss on comparable evaluation-mode batches.

Gradient nuance: when a parameter is reused, the chain rule sums its contributions from all uses. Contributions may reinforce or conflict, so weight sharing is not automatically equivalent to a larger effective batch or faster learning. A diagnostic is to unroll tied weights into temporary independent copies and inspect the cosine similarities and norms of their per-use gradient contributions; use the same forward state, data, dropout realization, and loss. This is an original diagnostic proposal, not an observed result or a mechanism established by these papers. Do not claim gradient interference was measured unless this diagnostic is actually executed and logged.

Stable latent-state norms or decreasing successive-state distances alone do not prove correct reasoning. A converged wrong answer is possible. Pair stability plots with per-example prediction correctness and path validity.

### Gaps

- Original notebook outputs do not supply all this provenance, and no new logging can reconstruct missing original checkpoints or exact execution order.
- FLOP estimates require an explicit counting convention and sequence lengths. Effective depth is a useful proxy, not a measurement of exact FLOPs or latency.
- Epiplexity is deliberately outside these notes; it is handled in the later-stage analysis requested by the user.
