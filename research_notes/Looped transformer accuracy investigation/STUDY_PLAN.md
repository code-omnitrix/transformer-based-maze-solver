# Prospectively specified local study plan

Written UTC: 2026-11-02T02:43:21.709777+00:00

This is a local research record, **not** a formal public preregistration. The
chat records the original declarations. This file was written while training
was in progress, after seed42 pilot results became visible. It documents those
earlier decisions and the explicitly identified later extension honestly.

## Initial controlled study

Before the first heldout evaluation, the parent authorized:

* Seven variants: postln4, preln4, preln6, tied6, tied6_anchor, tied6_full,
  untied6_full; all three seeds42,43,44 run on a single fixed shared data split.
* Width128,8heads,FF256,dropout.1; batch128; AdamWlr.001,wd.0001,clip1;
  30epochs,12,000train,1,000validation,1,000heldouttest.
* Fixed final checkpoint at30, not the checkpoint selected by highest accuracy.
* Alltestmaze teacherforced predictions and first512testmaze rollouts retained.
* Sharedcore Xavier initialization family throughout controlled study.
* Final tiedmodel inference depth diagnostics atK1,3,6,9,12, without retraining.
* Save configurations, environment and code/data hashes, epoch/step histories,
  model/optimizer/RNG checkpoints and permaze predictions.

The immutable source snapshot is
`source/experiment_suite_e855368e4a0ec9371330dd5e962d3ec30446c662e1f671769b6ae0c1ecb7cc0b.py`.
Run environment timestamps and commandline establish actual execution timing.

Primary causal contrast: **tied6_full minusuntied6_full** (sharing changes,
effective depth6,preLN,anchoring,indexencoding and initializationfamily matched).
Secondary contrasts: preln4−postln4 (normalization); preln6−preln4 (depth);
tied6−preln6 (bare sharing); tied6_anchor−tied6 (anchor);
tied6_full−tied6_anchor (indexencoding).

The sevenvariants were already fixed before firsttestoutcomes. The primary/
secondary labels were clarified during training while some seed42 outcomes
were visible; no unobserved test results were used to select runs or stopping.
Seed42 postLN4/preLN4 tensorinitializations were checked identical, making the
normalization contrast especially well controlled. Different parameterizations
use the same initialization family, not identical parameter tensors.

## Later full-data extension

After some seed42 controlled pilot outcomes were visible, a full-data extension
was specified. Its motivation was fidelity to the user's original50k data scale
and controlling sharing at that scale. **All four following runs were fixed
before any of their own test results were observed:**

1. Original notebook baseline90epochs, with additional fixed30epoch evaluation.
2. Original notebook loop architecture30epochs.
3. Controlled tied6_full30epochs.
4. Controlled untied6_full30epochs.

Use original notebook generatorseed42, train50,000 thenval5,000, plusfreshheldout
test5,000. All four models are freshly seeded42; dedicated shuffle generator1042
gives shared order. This differs from unknown historical notebook RNG/execution
state. Original architecture, objective and optimizer settings are faithfully
represented, but historical outputs are not promised to reproduce bitforbit.

Fullstudy primary sharing contrast is controlledtiedfull−controlleduntiedfull
at30. Originalbaseline30−originalloop30 compares equalupdate budgets;
baseline90−originalloop30 compares the original reported training schedules.
Neither is alone a sharing-only comparison because originalarchitectures differ
in norm, anchoring, iteration encoding, effective depth and initialization.

Baseline30and90 checkpoint evaluations are explicitly specified; allothermodels
use final30. Latestcheckpoint every10epochs supports recovery. No paidcompute,
packageinstall, originalnotebookedit, or parallel GPUtraining is authorized here.

## Analysis and uncertainty

Primary taskmetrics: nonPADteacherforced tokenaccuracy and autonomous shortest
pathsuccess (legalmoves, goalreached, EOS, BFSoptimallength), with canonicalexact,
goalreach, actiononlyaccuracy, EOSaccuracy, and firstactionaccuracy also retained.
Firstaction is derived from savedpredictions; it uses BOSonly and no trueaction
prefix. Counts/denominators and pathlengthstrata are retained.

Report eachseed and mean±sampleSD. Three seeds do not support strong population
significance claims (even a consistent two-sided pairedsigntest has minp=.25).
Pairedmaze bootstrapintervals condition on the trainedmodels and fixedseeds;
they describe finite-test uncertainty, not uncertainty over retraining or new
trainingdatasets. Differentmodels share evaluationexamples, so use pairedmaze
resampling rather than treating individualtokens as independent observations.

Plots compare accuracy byepoch,optimizerupdate,cumulativemeasuredtrainingtime.
Epoch counts are not compute budgets: original loop uses6encoder+6decoderblock
applications; baseline uses4+4. Parametercount is a separate axis.
Timing is local to this laptop, power/thermal/backgroundconditions and software.

## Diagnostics and deviations

* Depthsweep and gradient/gateevolution are diagnostic, not proof of learned BFS
  or stable fixedpoint computation.
* Historicalnotebook and freshmeasurements remain distinct in all plots/tables.
* Two discovered rawloggingsemantics are corrected in **derived** analysis only:
  rawdataset histogram includes BOS+EOS; rawterminated means EOSbeforefirstinvalid
  traceaction. ActualEOSemission is reconstructed from generatedtokens.
* Any additional hardermaze evaluation is exploratory and must be labeled with
  its later specification date, data seed and selectionrule; no currenttestresult
  may be silently described as an independently preregistered confirmatory test.
* No wallclockpromise was made. Initialmatrix budget estimate35minutes andfull
  extensionabout25minutes are forecasts; actualtimes in logs are authoritative.

Study files and source hashes provide an auditable local record, not a guarantee
that all CUDA kernels are bitwise deterministic or that three seeds establish
universal architectural superiority.
