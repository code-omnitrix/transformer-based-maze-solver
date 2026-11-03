# Controlled experiments and fresh full-data runs


Generated UTC: 2026-11-02T03:04:11.279048+00:00


## What does the controlled matrix show?


### Takeaway
KNOWN, high confidence about recorded runs: these are fresh controlled-family experiments, not historical notebook outputs. All endpoints are fixed epoch30. Effect sizes apply to one fixed12k training sample and the specified optimizer/settings. Three architecture seeds are not three independently sampled datasets.


### Cited Findings
Accuracies below are percentage units; ± is sample SD across recorded architecture seeds, not a confidence interval. Teacher test uses1000 mazes; autonomous rollout uses first512 fixed test mazes.

| Variant | Seeds | Parameters | Teacher token % ±SD | First action % ±SD | Teacher exact % ±SD | Rollout optimal % ±SD | Training seconds mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| postln4 | 3 | 1346311 | 77.206 ± 0.401 | 59.567 ± 0.058 | 20.033 ± 0.814 | 21.549 ± 0.298 | 71.58 |
| preln4 | 3 | 1346311 | 92.651 ± 1.213 | 89.633 ± 1.801 | 64.567 ± 3.700 | 68.685 ± 2.983 | 71.97 |
| preln6 | 3 | 2008839 | 89.159 ± 5.389 | 82.400 ± 11.092 | 52.100 ± 19.050 | 56.641 ± 20.117 | 103.12 |
| tied6_anchor | 3 | 352521 | 85.307 ± 2.522 | 80.667 ± 4.717 | 38.333 ± 9.325 | 42.188 ± 9.117 | 106.00 |
| tied6_full | 3 | 352521 | 83.829 ± 0.686 | 79.000 ± 2.326 | 33.333 ± 3.163 | 36.458 ± 3.809 | 111.82 |
| tied6 | 3 | 352519 | 84.009 ± 3.952 | 77.767 ± 11.145 | 36.267 ± 13.136 | 40.690 ± 14.321 | 101.44 |
| untied6_full | 3 | 2008841 | 86.549 ± 3.886 | 80.033 ± 2.050 | 41.700 ± 9.896 | 45.638 ± 11.372 | 114.09 |


Sources: [architecture summaryCSV](analysis/controlled12k/architecture_summary.csv), [per-seed final CSV](analysis/controlled12k/final_runs.csv), [all epochsCSV](analysis/controlled12k/all_epochs.csv), [frozen study plan](STUDY_PLAN.md).


Primary contrast and secondary contrasts, paired on the same examples and architecture seeds:



| Contrast (after−before) | Paired seeds | Token effect pp | Seed effects pp | Conditional maze-bootstrap95% pp | Optimal rollout effect pp |
| --- | --- | --- | --- | --- | --- |
| tied6_full − untied6_full | 42,43,44 | -2.720 | -7.901, 0.047, -0.306 | -3.136, -2.307 | -9.180 |
| preln4 − postln4 | 42,43,44 | 15.446 | 15.838, 13.839, 16.661 | 14.765, 16.135 | 47.135 |
| preln6 − preln4 | 42,43,44 | -3.492 | 0.329, -0.576, -10.229 | -3.849, -3.138 | -12.044 |
| tied6 − preln6 | 42,43,44 | -5.150 | -14.168, -4.092, 2.810 | -5.621, -4.703 | -15.951 |
| tied6_anchor − tied6 | 42,43,44 | 1.297 | 4.444, 1.634, -2.187 | 0.885, 1.712 | 1.497 |
| tied6_full − tied6_anchor | 42,43,44 | -1.478 | -0.800, -4.303, 0.670 | -1.856, -1.104 | -5.729 |


Source: [paired comparisonsJSON](analysis/controlled12k/paired_comparisons.json). Wholemaze bootstrap conditions on the fitted seed ensemble; it does not quantify retraining or new-train-data uncertainty. With only3seeds, even all3paired differences sharing a sign gives minimum two-sided sign-testp=.25.


### Inferences
Normalization, anchoring, tying and iteration encodings must be interpreted through their own matched contrasts. The fixed-learning-rate/no-warmup policy may favor some parameterizations. A lower parametercount alone does not imply lower compute, and per-seed sharing effects can vary in sign. Gradient/gate curves describe optimization dynamics without isolating a causal mechanism.


### Gaps
No parameter-matched narrow untied model, independent training-dataset replications, model-specific optimizer tuning, larger-grid training/evaluation, or mechanistic BFS proof. Early-budget pilots cannot substitute for full50k near-convergence comparisons.


## Does the result persist at the original data scale?


### Takeaway
The fresh full-data extension is distinct from the controlled pilot and from historical output. The exact original generator is replayedseed42 fortrain50k/val5k, then a newtest5k. Per-model initializationseed42 and dedicatedshuffle1042 are explicit fresh-run choices; historical execution state is unknown.


### Cited Findings

Full-data runs not yet complete at this notes generation. Do not present a planned experiment as a result.


Sources: [full-data analysis](analysis/full50k/), individual `runs/full50k_*/summary.json`, `epochs.jsonl` and original baseline `epoch30_evaluation/summary.json`; [fullscale specification](fullscale_plan_record.json). Fulltest rollouts1000fixed mazes; teacher test5000; single modelseed pervariant.


### Inferences
Use the controlled tiedfull−untiedfull comparison to assess sharing at matched6depth/preLN/anchor/indexencoding/initializationfamily. Use originalbaseline30vsloop30 to assess the changed architecture bundle at equalupdates, and baseline90vsloop30 to compare the user's original training schedules. Distinguish validation from heldouttest in every numerical claim.


### Gaps
Onlyone full-data architectureseed, with fixeddata; inability to recover originalcheckpoint/RNGstate prevents claiming historical bitwise replication. Different initialization policies between original loop and controlledfamily are intentional and logged.


## How can future analysis reproduce and audit the results?


### Takeaway
Raw epoch/update histories, exact datasets, configs, source snapshots, checkpoints/RNG, and permaze teacher/rollout predictions are retained. Every plot is regenerated by a plain script without notebookglobalstate.


### Cited Findings
See [README](README.md), [research ledger](research_log.md), [study plan](STUDY_PLAN.md), [run registry](run_registry.jsonl), [resume validation](resume_validation.json), [harness review](experiment_harness_review.md), and archive_manifest.json when finalized. Accuracies inrawCSV/JSON are fractions0–1; plots/thisnotes table usepercent. Loss is natural-log crossentropy pernonPADtargettoken; timesseconds.


### Inferences
Saved predictions permit correcting metric semantics, evaluating firstaction/BOS-only accuracy, classifying failure modes, accepting alternative shortestpaths, and bootstrapping wholemaze effects without retraining. The savedonline training NLL is dropout-active repeated-data fitting loss, explicitly not an epiplexity estimate.


### Gaps
Matrixcheckpoints are finalonly; fullruns additionally snapshot latest every10epochs. CPUresume matches exactly; CUDA bitwise reproducibility is not guaranteed. Localwalltime reflects fixedvariantorder, laptopthermal/power/backgroundconditions, and does not measureenergy. Raw dataset `path_length_histogram` includesBOS/EOS; derivedactionlength subtracts2. Raw `terminated` means EOSbeforefirstillegaltraceaction; actualEOSemission is reconstructed fromtokens. Neither caveat changes strict solved/optimalpathlabels.
