# Notebook audit and interpretation

## What do the saved results establish?

### Takeaway
KNOWN, high confidence: the saved notebook output contains the user's quoted
98.09% baseline endpoint and99.31% loop endpoint. This is a useful observation
but not a controlled estimate of the causal effect of weight sharing.

### Cited Findings

* 90baseline and30looped epochs are stored in cell3 andcell6. Baseline at30:
 90.91%; baselinebest98.39%at89; baselinefinal98.09%; loopfinal/best99.31%at30.
 Endpoint difference1.22percentagepoints. Token error drops from1.91%to0.69%,
 relative error reduction63.8743%. Comparingbest againstbest gives0.92points,
 not1.22. These are validation teacher-forced token accuracies. — Source:
 [historical extractedCSV](audit/original_histories.csv), [notebook](../../aiml.ipynb).
* Baseline at30 versus loopat30 is90.91%versus99.31% (8.40points), but their
 per-example computational graphs and all confounds below differ. — Same source.
* Execution counts: initialseeds21, data22, originalarchitecture23,
 baselinetrain25, earlier visualization13, looparchitecture null,
 looptrain18, loopvisualization20. Null can indicate unexecuted or reset/edited
 cell; it does not prove the code was never run. Neither historical kernel state
 nor exact executed code is recoverable from this file alone. —
 [structure/hash](audit/notebook_structure.json).

### Inferences

* The stored logs are observational evidence of a successful run associated with
 these notebook cells, not evidence of provenance sufficient for exact rerun.
* Lower looptrain loss is consistent with easier optimization or better learned
 prediction as well as a possible inductive bias. It does not isolate regularization.

### Gaps

Original trained state_dicts, raw heldout predictions, runtime/FLOPs, multiple
seeds, full data snapshots, original environmentversions and execution order are
not present. A fresh same-settings run is possible, historical recovery is not.

## Which architectural and training factors differ?

### Takeaway
KNOWN, high confidence: the notebook changes a bundle of factors simultaneously.
The primary alternative explanations are normalization placement, repeated input
anchoring and effective depth, in addition to sharing and initialization.

### Cited Findings

| Factor | Saved baseline definition | Saved loop definition |
|---|---|---|
| Encoder depth |4separateblocks|1sharedblock repeated6times|
| Decoder depth |4separateblocks|1sharedblock repeated6times|
| Norm placement |post-LN (`nn.Transformer`default)|pre-LN (`norm_first=True`)|
| Input injection |none|learnedscalar×initialembeddings everyiteration|
| Iteration index |none|sinusoidal stepencoding everyiteration|
| Final encoder/decoder norm |yes|yes|
| Decoder position capacity |64|128|
| Initialization |`nn.Transformer`resets core multidimweights Xavier|layer defaults; no modelwide Xavier reset|
| Trainable parameters |1,338,119|352,521|
| Width/heads/FF/dropout |128/8/256/.1|128/8/256/.1|
| Optimization |AdamW1e-3,wd1e-4,clip1|same|
| Epochs/updates |90×391=35,190|30×391=11,730|

Sources: [original architecture](audit/cell_02.py),
[loop architecture](audit/cell_05.py), [baseline train](audit/cell_03.py),
[loop train](audit/cell_06.py), [fresh parametercounts](audit/architecture_counts.json).

* Parameter reduction73.6555% (baseline/loop≈3.796). The loop has50%more block
 applications per forwardpass (6encoder+6decoder vs4+4), so fewer unique weights
 does not mean less forward/backward compute per example. No wallclock speedup
 follows from epoch counts alone. — Architecture source andbenchmark.json.
* Both optimizers are newly constructed in their cell; if executed as written,
 looptraining does not reuse baseline optimizer state or weights. But the global
 class name is redefined and seeds are set once, so execution order affects which
 model is constructed and RNGtrajectory. — Sourcecell0/2/3/5/6.
* The first loop input is `h_init + alpha*h_init + step_encoding`, initially
 `2*h_init + step_encoding`; reinjection is not just a skipconnection equivalent
 to the baseline's internal residual. Every encoder/decoder loop has its own
 learnable scalar across allitsiterations (2scalars total). — Sourcecell5.

### Inferences

* A repeatable algorithmic update may be a useful inductive bias for pathfinding,
 but the original comparison cannot separate it from norm, anchor, depth or init.
* Shared parameters receive gradients through multiple recurrent uses. Parameter
 reduction therefore does not imply that optimization is automatically equivalent
 to an untiednetwork with fewer training updates; gradient norms/clipping matter.
* Sinusoidal indices permit computing embeddings beyond6 but do not establish
 useful inference-depth extrapolation of the trained transition function.

### Gaps

The notebook has no norm-only, matched-depth, matched-initialization, anchor-only,
or iteration-encoding-only controls. New experiments supply these at a smaller
training-data budget, plus predeclared full-data comparisons.

## Are the dataset and accuracy metric appropriate?

### Takeaway
KNOWN, high confidence: the task is canonical shortest-path imitation in random
8×8grids; validation uses true prior actions. The code excludes PAD correctly,
but the reported number is not autonomous maze-solving success.

### Cited Findings

* Random wallprobability.28, random distinctstart/goal, rejection until solvable
 with at least4moves; deterministic BFS orderU,L,R,D selects onecanonical path.
 train50k andval5k are generated separately by calls sharing advanced RNGstate.
 They are not explicit train_test_split arrays but are separate draws. —
 [generator](audit/cell_01.py).
* Target sequencesBOS/actions/EOS padded to eachsplit's longest sequence.
 Training input removeslasttoken andlabel removesfirsttoken. Causal mask blocks
 futuredecoder positions; targetpaddingmask excludes PADkeys. Crossentropyloss
 and validation mask ignore PAD. No obvious label leakage is present in the
 written forwardpath. Fresh randomfutureprefix perturbation yielded maxdifference
0 at all earlier logits for both savedarchitecture definitions. —
 [causal diagnostic](audit/causal_checks.json), architecture/train sources.
* Validation includesEOS as onecorrect/incorrecttoken permaze; separate action
 accuracy and EOSaccuracy are not reported in notebook. Tokenaccuracy weights
 longerpaths more, rather than assigning eachmaze equalweight. — Traincells.
* Four/eight visualizednew mazes are qualitative samples, not reliable aggregate
 successestimates. The trace ignoresnonmove tokens in visualization, so new
 experiment evaluator treats BOS/PADoutputs as invalid to avoid leniency. —
 [visualization source](audit/cell_04.py), [loop visualizations](audit/cell_07.py).
* ModuleDict registeredmodules supportattribute access; `.transformer.decoder`
 againstloop ModuleDict is not itself a demonstrated bug. The notebook class
 forwardpath uses indexedaccess consistently. — Independent harnessdiagnostic.

### Inferences

* Groundtruthprefixes reset the actionstate eachstep; modelownmistakes may change
 laterinputs during rollout. Measure free-running legalgoalreach, EOStermination,
 shortest-pathsuccess and canonicalexact separately.
* An alternative shortest path should pass task-leveloptimality even when it
 differs from the canonical BFS tokens. Exactmatching alone penalizes valid
 alternative solutions; tokenaccuracyalone can obscure fullpathfailure.
* A highIIDscore does not establish generalBFS implementation or performance on
 largergrids, hardermazes, differentwall densities, or trainunseenlongpaths.

### Gaps

No originalcheckpoints allow retrospective rollout. Seed42fresh generatorreplay
describes a possible top-to-bottom data instantiation, not proof that historical
out-of-order runs used thoseexactarrays. Newexperiment logs retain data and
predictions so futureplotting and metricreanalysis do not require retraining.
