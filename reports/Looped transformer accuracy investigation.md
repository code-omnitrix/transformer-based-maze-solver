# Looped transformer accuracy investigation

**Working draft — empirical study in progress.** Prepared on 2 November 2026 (Asia/Calcutta). The notebook audit, independent code checks and literature review are complete. The three-seed controlled matrix and full-size follow-up must finish before the empirical conclusions are final. “Historical” below means output already saved in the notebook; new runs are identified separately.

The saved notebook establishes a substantial observed improvement: **99.31% validation token accuracy after 30 looped epochs versus 98.09% after 90 original-transformer epochs**. It does not isolate weight sharing. The looped implementation also changes normalization, effective depth, repeated input injection, iteration encoding, initialization and parameter count. The strongest completed diagnostic so far concerns normalization; sharing effects vary with seed in the smaller-data regime. Teacher-forced token accuracy also differs from autonomous maze-solving success, so this study measures both. The aim is to reproduce the observation where possible, isolate its contributors, and distinguish observed effects from attractive algorithmic explanations. Source snapshots, datasets, individual predictions, training histories, independent checks and plotting scripts preserve the evidence for later analysis. [Notebook audit][audit], [study plan][plan].

## The notebook compares architectural bundles

### The apparent gain depends on the comparison point

The complete stored histories match the user's excerpt. At epoch 30 the original model records **90.91%** accuracy and training loss 0.2693; the looped model records **99.31%** and loss 0.0383. The original model later reaches 98.39% at epoch 89, then ends at 98.09% with loss 0.0843. These yield three different descriptive comparisons. Final-versus-final gives 1.22 percentage points, or **63.87% fewer token errors** as the error rate falls from 1.91% to 0.69%. Best-versus-best gives 0.92 points. Neither is a repeated-run effect estimate, and neighboring epochs are correlated observations from the same trajectory. [Extracted comparisons][endpoints], [historical curves][histories].

| Historical comparison | Original transformer | Looped transformer | Looped gain |
|---|---:|---:|---:|
| Both at epoch 30 | 90.91% | 99.31% | +8.40 pp |
| Final checkpoints, epochs 90 / 30 | 98.09% | 99.31% | +1.22 pp |
| Best saved validation scores, epochs 89 / 30 | 98.39% | 99.31% | +0.92 pp |

With the saved 50,000-example, batch-128 settings, one epoch contains 391 updates. Both epoch-30 models therefore receive 11,730 optimizer updates; original epoch 90 receives 35,190. Equal updates do not mean equal computation: the loop applies six encoder and six decoder blocks per prediction, compared with four of each. That is 1.5 times as many block applications, rather than an exact FLOP ratio. Padding, attention and implementation affect real cost. The notebook retains no historical runtime, so fewer epochs alone cannot establish faster wall-clock learning. [Endpoint accounting][endpoints], [original architecture][normal-source], [looped architecture][loop-source].

### Weight sharing is one of several interventions

All notebook cell indices here are zero based. Cells 2 and 5 match width 128, eight heads, feedforward width 256 and dropout 0.1. Both have final encoder and decoder layer norms. Their other differences are substantial. Parameter counts were freshly instantiated and independently cross-checked. [Architecture counts][counts], [independent audit][independent].

| Factor | Original transformer | Looped transformer |
|---|---|---|
| Distinct encoder / decoder blocks | 4 / 4 | 1 / 1 |
| Effective encoder / decoder depth | 4 / 4 | 6 / 6 |
| Normalization inside blocks | Post-LN | Pre-LN |
| Initial input added at each step | No | Learned scalar gate in encoder and decoder |
| Iteration encoding | None | Sinusoidal, added at every iteration |
| Decoder position table | 64 positions | 128 positions |
| Core initialization | Outer `nn.Transformer` Xavier reset | Individual layer defaults |
| Trainable parameters | 1,338,119 | 352,521 |
| AdamW / clipping | lr 0.001, weight decay 0.0001, norm cap 1 | Same |
| Epochs | 90 | 30 |

The looped model has **73.66% fewer unique parameters**, approximately a 3.80-fold reduction, while applying them more often. Its shared block receives the current hidden state plus a gated copy of the initial embedding and the iteration signal. The gate initially equals one, so the first iteration receives twice the original embedding plus its index. This reinjection changes computation independently of sharing. The iteration signal also lets the same weights implement different behaviors at different steps. Shared weights therefore do not by themselves establish one invariant algorithm. [Looped definition][loop-source].

Initialization differs measurably. Fresh independently seeded instances have first-encoder output-projection weight standard deviations 0.08811 and 0.05088 for the original and looped definitions; their second feedforward matrices have 0.07213 and 0.03623. These are **new initialization diagnostics, not recovered historical weights**. They establish different starting conditions without identifying their contribution to accuracy. [Initialization diagnostics][init].

### Historical execution state cannot be reconstructed

The notebook execution counts are `[21,22,23,25,13,null,18,20,null]`. The loop architecture cell has no count while its training output has an older count. Current code and historical text coexist, but exact executed source and kernel state are unknown. That is a provenance gap, rather than evidence that the result is invalid. Only cell 0 explicitly seeds Python, NumPy and Torch; neither training cell reseeds. The historical looped initialization cannot be described as an independent seed-42 run. [Notebook structure][structure], [independent audit][independent].

Original checkpoints, raw predictions, exact environment and data arrays were not retained. Fresh runs can faithfully instantiate the saved definitions and settings, but cannot retrospectively evaluate the missing weights. **Confidence is high** about the saved numbers and architectural differences, and **limited** about exact historical provenance. The new archive addresses this limitation prospectively.

## Controlled runs distinguish learning speed from maze-solving success

### The target is one canonical shortest path

The generator samples 8×8 walls at probability 0.28, places distinct start and goal cells, then rejects unsolvable mazes and paths shorter than four moves. BFS explores U, L, R, D in fixed order, selecting a canonical shortest sequence. The accepted distribution is conditional on these rejection rules. Targets contain BOS, actions and EOS. Training and validation receive shifted true prefixes. PAD labels are correctly excluded; EOS is scored. Future-prefix perturbation checks found zero influence on earlier logits in both original definitions and all seven controlled variants. There is no demonstrated padding-inflation or future-target leakage explanation. [Generator][generator], [causal checks][causal], [independent harness review][harness].

A fresh seed-42 replay generated 50,000 unique training and 5,000 unique validation mazes with no exact-grid overlap. Its validation shortest paths average 7.3806 moves, range 4–24, and contain 41,903 scored labels, including 5,000 EOS labels: **11.93% of the metric**. These facts concern a reproducible fresh replay, not recovered historical arrays. Crucially, **74.72% of its mazes have multiple shortest solutions**. Exact canonical matching can reject a valid optimal route. [Replay statistics][statistics], [individual maze records][baseline-records].

This study retains token accuracy, action-only accuracy, teacher-forced exact match and first-action accuracy, then evaluates autonomous generation separately. A solve requires legal moves, arrival at the goal and EOS. Optimal success also requires the BFS-optimal movement count, accepting alternative shortest routes. Canonical exact match is reported independently. Generated PAD/BOS and illegal moves fail. A cap of 64 predicted tokens covers any simple shortest path on the 64-cell grid plus EOS. [Metric definitions][readme].

A no-training wall-aware Manhattan heuristic demonstrates the distinction: **86.56% teacher-forced token accuracy**, but 58.32% autonomous goal success, 55.60% shortest-route success and 53.66% canonical exact match on the replay. This heuristic uses a 40-move cap, so it is an illustration rather than a budget-matched neural benchmark. Its outcomes cannot estimate the missing historical neural models' success rates. [Baseline implementation][independent-script], [aggregate results][statistics].

### Seven variants isolate components in an earlier-learning regime

The controlled matrix uses 12,000 training, 1,000 validation and 1,000 test mazes and architecture seeds 42, 43 and 44. Variants share saved splits and each seed's dedicated shuffled order. Width, heads, feedforward size, dropout, optimizer, clipping, final norms and position capacity are fixed. Core matrices use a common Xavier convention; embeddings and output projection use framework defaults. The seed-42 normalization pair's complete initial state tensors were checked **byte identical**. Differently sized models share an initialization family, not identical tensors. [Study plan][plan], [normalization check][norm-check].

| ID | Encoder / decoder depth | Sharing | Norm | Input injection | Step encoding | Contrast |
|---|---|---|---|---|---|---|
| `postln4` | 4 / 4 | No | Post-LN | No | No | Starting control |
| `preln4` | 4 / 4 | No | Pre-LN | No | No | Normalization |
| `preln6` | 6 / 6 | No | Pre-LN | No | No | Depth |
| `tied6` | 6 / 6 | Yes | Pre-LN | No | No | Bare sharing |
| `tied6_anchor` | 6 / 6 | Yes | Pre-LN | Yes | No | Anchoring |
| `tied6_full` | 6 / 6 | Yes | Pre-LN | Yes | Yes | Complete tied control |
| `untied6_full` | 6 / 6 | No | Pre-LN | Yes | Yes | Matched sharing comparison |

Each fixed final checkpoint follows 30 epochs, **2,820 updates**. This is much earlier than the historical 11,730 or 35,190 updates. The NumPy `default_rng` generator preserves the intended distribution rather than exact original arrays. The controlled four-layer model also has 128 decoder positions and independently initialized layer biases, unlike the original baseline. These are internally controlled experiments, not numerical replications. [Methods and deviations][readme], [computation-equivalence checks][harness-results].

The main sharing contrast is `tied6_full − untied6_full`, with matching depth, normalization, injection, index and initialization convention. Secondary contrasts test norm, depth, bare sharing, anchoring and index. Each effect is conditional on the other choices in that contrast. They cannot be summed into an additive decomposition of the historical 1.22-point gap: interactions and training regime differ. The seven variants and three seeds were fixed before their first test results. Primary/secondary labels were clarified after some early outcomes became visible; the local plan records that honestly and is not public preregistration. No architecture or stopping rule is selected by test score. [Decision chronology][plan].

**The completed three-seed result table will replace this draft paragraph.** Its authoritative inputs are the [individual run table][matrix-runs], [architecture mean/SD table][matrix-summary] and [paired comparisons][matrix-pairs]. Early seeds differ in the direction and magnitude of the matched sharing effect. A first-seed result must not become a general claim about sharing. The final report will include each seed, aggregate effects, learning curves and practical path success.

### Four full-size runs test fidelity and scale

The later extension restores the exact saved generator: 50,000 training, 5,000 validation and 5,000 fresh test mazes, with original seed-42 generation order. It trains the original baseline for 90 epochs, additionally evaluating a fixed epoch-30 checkpoint; the original looped definition for 30; and the controlled tied/untied full models for 30 each. Each new model starts at seed 42 with dedicated shuffle seed 1042. This is a reproducible deviation from unknown historical RNG state. Independent CPU checks reproduce original data hashes, initial tensors and synchronized-dropout forward computations exactly. [Full-size review][full-review], [checks][full-checks].

All four were specified after early smaller-study results but before their own test outcomes. They distinguish fresh replication of the original bundle from a full-scale sharing control. Final rollout uses 1,000 fixed test mazes; teacher forcing uses all 5,000. Baseline epoch-30 evaluation saves/restores RNG so added diagnostics do not alter later training. **Full-size numerical findings remain pending in this draft.** [Full-size plan][full-plan].

### Finite-test uncertainty differs from retraining uncertainty

The three controlled seeds vary initialization, dropout and order on one fixed dataset. Their individual results and sample SD describe that limited variation, rather than variation over newly generated training datasets. Paired maze bootstraps resample the shared test mazes with all their tokens together, conditional on fitted models. They cannot replace seed uncertainty. With three paired seeds, even three same-sign differences give a minimum two-sided sign-test p-value of 0.25. The one-seed full-size study remains a scale/fidelity diagnostic. [Analysis policy][plan], [analysis implementation][analysis-script].

## Literature motivates mechanisms without identifying the learned algorithm

Xiong et al. connect Post-LN with large initialization gradients and show competitive Pre-LN training without warmup. This makes normalization a serious alternative explanation under the notebook's constant learning rate. It motivates the norm-only comparison; it does not quantify its contribution here. Gradient/clipping logs can corroborate optimization differences, while warmup or learning-rate interventions would test their cause. [Xiong et al., ICML 2020, §§3–4](https://proceedings.mlr.press/v119/xiong20b/xiong20b.pdf).

Universal Transformers use shared depth transitions and position/time signals, improving several algorithmic extrapolation results without universally solving them. Their dynamic halting differs from this fixed-six-step model. The relevant inference is that recurrence is a credible bias to test, not that six network iterations equal six BFS expansions. [Dehghani et al., ICLR 2019, inspected v3, §§2.1, 3.4](https://arxiv.org/html/1807.03819v3#S3.SS4).

ALBERT's direct sharing ablation reports lower benchmark averages when all layers share weights; shared representations can oscillate rather than converge. Its task differs from maze navigation, but it supplies counterevidence to automatic superiority or fixed-point interpretations. [Lan et al., ICLR 2020, inspected v6, §4.5](https://arxiv.org/html/1909.11942v6#S4.SS5).

Yang et al.'s looped in-context regression model reinjects input and supervises a window of iterations; extrapolation depends on its training regime. Bansal et al. study recurrent maze reasoning, but with convolutional whole-output models and different recall/progressive-training operations; they also document degradation from excessive recurrence. Their findings motivate anchoring and inference-depth diagnostics without establishing this model's behavior. [Yang et al., ICLR 2024, inspected v3, §4](https://arxiv.org/html/2311.12424v3#S4), [Bansal et al., 2022, inspected v2, §§3, 5](https://arxiv.org/html/2202.05826v2#S3).

Saunshi et al. distinguish equal-parameter and equal-effective-depth comparisons and find task-dependent looping advantages, while leaving the reasoning-bias mechanism open. Geiping et al. demonstrate recurrent inference gains with randomized training depths, injection and specialized training at much larger scale. Their procedures differ from fixed-six-loop training, so additional inference loops here require direct testing. [Saunshi et al., ICLR 2025, inspected v1, §§2–5](https://arxiv.org/html/2502.17416v1#S2), [Geiping et al., 2025, inspected v2, §§3–5](https://arxiv.org/html/2502.05171v2#S3).

Gatmiry et al. establish a gradient-descent interpretation for restricted linear-attention regression under explicit assumptions. Those assumptions do not describe this nonlinear maze model. An algorithmic analogy becomes a mechanism claim only after an appropriate proof or measurement. [Gatmiry et al., 2024, inspected v1, §3](https://arxiv.org/html/2410.08292v1#S3.SS7).

For these mazes, reusable updates related to reachability or value propagation remain plausible **untested hypotheses**. Global attention differs from local BFS propagation, and true decoder prefixes simplify teacher-forced prediction. Attention maps or stable latent norms cannot establish BFS. A next probe would decode distance-to-goal, reachability and shortest-route membership at each encoder iteration on a separate probe split, then intervene on candidate representations. A probe shows available information, not necessarily the information used. [Mechanism proposals][literature].

Sharing also sums gradients through multiple uses. Contributions can reinforce or conflict; neither faster optimization nor a larger effective batch follows automatically. Unclipped norms, clipping rates and gate evolution are diagnostics. Per-use gradient comparisons or targeted optimizer controls would strengthen causal interpretation. The final evidence assessment will use the following hypotheses.

| Hypothesis | Measurement | Remaining boundary |
|---|---|---|
| Pre-LN improves learning under this optimizer | Exact-state norm comparison and clipping curves | Warmup/LR/clipping cause not isolated |
| Added depth helps | Untied Pre-LN four versus six | Tuning and norm interactions |
| Sharing helps beyond surrounding features | Full tied versus untied | Best parameter-budget allocation untested |
| Input anchoring helps | Bare versus anchored tied; gates | Interactions with initialization and width |
| Step encoding helps specialization | Anchored tied with/without index | Variable-depth or intermediate training |
| Token gains improve planning | Legal/goal/optimal rollout | Historical rollout unavailable |
| Extra test-time loops improve reasoning | Teacher-forced K=1,3,6,9,12 | Autonomous K-sweep not run |
| The model learned BFS | Difficulty-stratified behavior | Representation/causal probes not run |

## Epiplexity requires a different empirical test

Finzi et al. separate structure and residual coding costs under compute constraints. Their looping experiment generates intermediate cellular-automaton states; useful looping can lower epiplexity. Hidden-state weight tying differs, and maze accuracy alone cannot confirm the framework. [*From Entropy to Epiplexity*, v2, 16 March 2026, §§3–4, 5.3.2, C.8](https://arxiv.org/html/2601.03220v2).

**Epiplexity has not been measured here.** This generator maps each maze to a deterministic canonical path. Given the maze and tie-breaking rule, ideal conditional Shannon entropy is zero, yet a bounded learner can fail to learn or execute the mapping. Changing the learner's architecture does not add maze facts. Fewer weights also do not by themselves measure a shorter optimal probabilistic program: precision, compressibility, the algorithm description and execution cost matter. The known BFS solver already exactly implements the mapping. [Notebook-specific assessment][epiplexity].

The new step losses are dropout-active optimization measurements on repeated examples. Their cumulative field resets each epoch. Summing them is useful for training diagnostics, but does not supply a validated coding measurement. An actual coding proxy needs explicit probability-model, data-order and unit assumptions. [Logged-loss semantics][readme].

A useful later study would hold maze arrays and final-task targets fixed, then compare direct path supervision with explicit frontier, distance-field or waypoint intermediates. Count additional labels, teacher-generation work and inference steps. Use an untouched confirmation split and prescribed detours, then compare measured compute/performance frontiers across model families and training horizons. A fresh-example likelihood stream and explicit coding assumptions would support a family-dependent proxy. This tests whether intermediates make useful maze structure more accessible within a budget; it does not reproduce the paper's cellular-automaton experiment. [Saved protocol E1][epiplexity].

A second test would sweep recurrence depth and training horizon with non-sharing components fixed, looking for a repeatable compute/performance crossover with a simpler estimated description and preserved path success. Failure over the tested range would weaken this proposed maze mechanism, without refuting a formal definition. If the practical gain disappears under norm/init controls, a coding narrative cannot replace that simpler explanation. **Confidence is high** that the present notebook cannot confirm epiplexity; the operational maze hypotheses remain untested. [Protocol E2 and falsification conditions][epiplexity].

## The archive supports future analysis without hidden state

The original notebook remains unchanged. New files stay inside the research/report folders; no packages were installed and no paid remote compute was used. Training uses existing Python 3.12.13, PyTorch 2.11.0+cu128 and an RTX 5080 Laptop GPU with 16 GB memory. Seeds are fixed, but deterministic CUDA algorithms are not forced. Variants run sequentially in fixed order, so wall times are local, indicative measurements affected by thermal/power state and background work. No energy-efficiency conclusion follows. [Environment, commands and caveats][readme], [research ledger][ledger].

| Artifact | Preserved evidence | Future use |
|---|---|---|
| `audit/` and historical CSVs | Original source, metadata and stored curves | Keep historical and fresh results distinct |
| Data NPZs/manifests | Exact grids, targets, hashes, overlap | Evaluate identical cases and difficulty strata |
| Per-run config/environment/model JSON | Seeds, architecture, optimizer, software, source hashes | Reconstruct a plotted experiment |
| Step/epoch JSONL | Loss, counts, updates, timing, clipping, gates | Plot against updates, examples or measured time |
| Teacher/rollout JSONL | Targets, generated tokens, IDs, path outcomes | Recompute action/path/error metrics |
| Checkpoints/resume tool | Weights, optimizer and RNG states | Continue into a separate exploratory run |
| Analysis directories | CSVs, paired effects, figures and code manifest | Replot without retraining |
| Source/search/bibliography logs | Immutable code, exact URLs and versions | Audit provenance |

The [README][readme] defines units: raw accuracies are fractions, figure axes percent, loss natural-log units/token, counts integers and durations seconds. Two raw-field semantics require care. `path_length_histogram` includes BOS/EOS, so subtract two for moves. Raw `terminated` stops at the first illegal move; actual EOS emission is reconstructed from generated tokens. Raw records remain intact and strict solve scores are unaffected. Independent replay and training hashes use different grid dtypes; normalize before joining or use stable split/example IDs. [Harness review][harness], [full-size review][full-review].

Controlled runs save final checkpoints. Full-size runs additionally save latest states every ten epochs and the original baseline's fixed epoch-30/90 states. The resume CLI restores optimizer/RNG states into a new directory. A small CPU shuffled-data test with dropout matched an uninterrupted next epoch exactly. GPU bitwise resume is not established; interrupted controlled training before its final checkpoint cannot be recovered from logs alone. [Resume validation][resume-check].

The following PowerShell regenerates analyses without retraining, using the existing interpreter with PyTorch:

```powershell
$experimentPython = 'C:\Users\omnap\miniconda3\envs\llmdyn\python.exe'
$experimentRoot = 'D:\Projects\Loopedtrans\research_notes\Looped transformer accuracy investigation'
& $experimentPython "$experimentRoot\analyze_runs.py" --prefix controlled12k
& $experimentPython "$experimentRoot\analyze_runs.py" --prefix full50k
& $experimentPython "$experimentRoot\archive_manifest.py"
```

The harder-maze evaluation is explicitly exploratory: fixed 8×8 sets at wall probability 0.40, and at original density conditioned on at least 12 moves. Length support can overlap training; these are difficulty shifts, not guaranteed unseen-length or larger-grid extrapolation. Fixed final checkpoints are evaluated without tuning, with predictions retained. The recurrence sweep is teacher forced only. Neither establishes autonomous inference-depth scaling or an internal BFS mechanism. [Exploratory plan][harder-plan].

Substantive limits remain: one fixed controlled training dataset, three training seeds, one full-size seed, no parameter-matched narrow untied model, no warmup/clipping intervention, no causal representation probes, no larger-grid test and no independent epiplexity reproduction. These define the scope of a defensible final conclusion.

**The final interpretation, complete result tables and plots will be inserted after the specified runs finish.** The reliable current conclusion is the audit: the saved improvement concerns a bundle of interventions whose contributions need separate measurement. The study preserves enough evidence to revisit those explanations rather than relying on rounded endpoint logs.

[audit]: <D:/Projects/Loopedtrans/research_notes/Looped transformer accuracy investigation/notebook_audit.md>
[plan]: <D:/Projects/Loopedtrans/research_notes/Looped transformer accuracy investigation/STUDY_PLAN.md>
[endpoints]: <D:/Projects/Loopedtrans/research_notes/Looped transformer accuracy investigation/independent_original_comparison.csv>
[histories]: <D:/Projects/Loopedtrans/research_notes/Looped transformer accuracy investigation/audit/original_histories.csv>
[normal-source]: <D:/Projects/Loopedtrans/research_notes/Looped transformer accuracy investigation/audit/cell_02.py>
[loop-source]: <D:/Projects/Loopedtrans/research_notes/Looped transformer accuracy investigation/audit/cell_05.py>
[counts]: <D:/Projects/Loopedtrans/research_notes/Looped transformer accuracy investigation/audit/architecture_counts.json>
[independent]: <D:/Projects/Loopedtrans/research_notes/Looped transformer accuracy investigation/independent_audit.md>
[init]: <D:/Projects/Loopedtrans/research_notes/Looped transformer accuracy investigation/original_initialization_diagnostics.json>
[structure]: <D:/Projects/Loopedtrans/research_notes/Looped transformer accuracy investigation/audit/notebook_structure.json>
[generator]: <D:/Projects/Loopedtrans/research_notes/Looped transformer accuracy investigation/audit/cell_01.py>
[causal]: <D:/Projects/Loopedtrans/research_notes/Looped transformer accuracy investigation/audit/causal_checks.json>
[harness]: <D:/Projects/Loopedtrans/research_notes/Looped transformer accuracy investigation/experiment_harness_review.md>
[statistics]: <D:/Projects/Loopedtrans/research_notes/Looped transformer accuracy investigation/independent_statistics.json>
[baseline-records]: <D:/Projects/Loopedtrans/research_notes/Looped transformer accuracy investigation/independent_baselines.csv>
[independent-script]: <D:/Projects/Loopedtrans/research_notes/Looped transformer accuracy investigation/independent_statistics.py>
[readme]: <D:/Projects/Loopedtrans/research_notes/Looped transformer accuracy investigation/README.md>
[norm-check]: <D:/Projects/Loopedtrans/research_notes/Looped transformer accuracy investigation/normalization_control_validation.json>
[harness-results]: <D:/Projects/Loopedtrans/research_notes/Looped transformer accuracy investigation/harness_validation_results.json>
[matrix-runs]: <D:/Projects/Loopedtrans/research_notes/Looped transformer accuracy investigation/analysis/controlled12k/final_runs.csv>
[matrix-summary]: <D:/Projects/Loopedtrans/research_notes/Looped transformer accuracy investigation/analysis/controlled12k/architecture_summary.csv>
[matrix-pairs]: <D:/Projects/Loopedtrans/research_notes/Looped transformer accuracy investigation/analysis/controlled12k/paired_comparisons.json>
[full-review]: <D:/Projects/Loopedtrans/research_notes/Looped transformer accuracy investigation/fullscale_review.md>
[full-checks]: <D:/Projects/Loopedtrans/research_notes/Looped transformer accuracy investigation/fullscale_review_checks.json>
[full-plan]: <D:/Projects/Loopedtrans/research_notes/Looped transformer accuracy investigation/fullscale_plan_record.json>
[analysis-script]: <D:/Projects/Loopedtrans/research_notes/Looped transformer accuracy investigation/analyze_runs.py>
[literature]: <D:/Projects/Loopedtrans/research_notes/Looped transformer accuracy investigation/recurrence_literature.md>
[epiplexity]: <D:/Projects/Loopedtrans/research_notes/Looped transformer accuracy investigation/epiplexity.md>
[ledger]: <D:/Projects/Loopedtrans/research_notes/Looped transformer accuracy investigation/research_log.md>
[resume-check]: <D:/Projects/Loopedtrans/research_notes/Looped transformer accuracy investigation/resume_validation.json>
[harder-plan]: <D:/Projects/Loopedtrans/research_notes/Looped transformer accuracy investigation/harder_eval_plan.json>
