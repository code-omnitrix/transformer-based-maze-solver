# Full-data training-loop fidelity review

**Outcome: no material mismatch or implementation bug found in the audited training settings/loop. The historical score difference remains unexplained.** Matching the recorded code does not identify which difference in random trajectory, runtime or saved notebook provenance caused a numerical divergence.

This review read the notebook, frozen training source, run config/environment, a few completed epoch-log lines, and installed optimizer/loss source. It imported no PyTorch, loaded no checkpoint, allocated no GPU memory, ran no training, and changed no live source/run file. It writes only this review.

## Exact code/settings comparisons

| Component | Saved notebook | Fresh full-data runner | Assessment |
|---|---|---|---|
| Model | `MazeTransformerSolver()` from baseline cell2 or looped cell5 | Factory executes those saved definitions, default dimensions | Exact saved definitions; CPU state/forward equivalence independently passed in fullscale_review.md |
| Optimizer | `torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)` | Same invocation | Exact explicit arguments |
| Adam defaults in installed PyTorch2.11 | Implicit | Implicit | Betas(0.9,0.999),eps1e-8,amsgradFalse,maximizeFalse,foreachNone,capturableFalse,differentiableFalse,fusedNone; historical installed version/backend choice not logged |
| Labels | `seqs[:,1:]` | `s[:,1:]` | Exact shift |
| Decoder input | `seqs[:,:-1]` | `s[:,:-1]` | Exact teacher forcing |
| Loss | `nn.CrossEntropyLoss(ignore_index=PAD)`; PAD0; flattened logits/labels | `nn.functional.cross_entropy(...,ignore_index=0)` | Same default reductionmean over nonignored targets, no class weights, label smoothing0 |
| Flattening | `reshape(-1,7)`, labels`reshape(-1)` | logits`flatten(0,1)`, labels`flatten()` | Equivalent B×T row order; copies for noncontiguous slices do not change values |
| Gradient clearing | `optimizer.zero_grad()` | `opt.zero_grad(set_to_none=True)` | Equivalent under installed optimizer defaultTrue; no accumulation |
| Update order | zero→forward→loss→backward→clip→step | Same | Exact sequence |
| Clipping | `clip_grad_norm_(parameters,1.0)` | `clip_grad_norm_(parameters,1)` | Equivalent float/int threshold; same default L2 norm and backend options |
| Modes | `model.train()` every epoch; `model.eval()` plus`no_grad` for validation | `m.train()` every epoch; decorated`teacher_eval` calls`m.eval()` | Correct dropout/autograd toggling; no accidental eval-mode training |
| Dropout | Model defaults0.1 | Original extracted model defaults0.1 | Same specified probability |
| Batch |128,shuffleTrue,drop_last defaultFalse |128,shuffleTrue,drop_last defaultFalse |391 updates/epoch, last batch80 |
| Data loading | Custom dataset materializes long tensors on access; default workers0/pinningFalse | TensorDataset holds same int64 arrays; default workers0/pinningFalse | Storage/collation changed, data values/dtype/padding verified equivalent |
| LR schedule | None; constant1e-3 | None; constant1e-3 | No warmup, scheduler or hidden LR change in inspected source |
| Precision | No autocast/scaler/cast settings in notebook | No autocast/scaler; new config float32/AMPFalse | Same written code policy; historical global flags/runtime unknown |
| Epochs | Baseline90, loop30 | Baseline90, loop30 | Same intended endpoint schedules; fresh runner adds baseline fixed30 evaluation |
| Validation metric | Correct nonPAD argmax tokens / nonPAD count, includesEOS | Same, plus extra diagnostics | Same primary accuracy definition |

Installed-source references: `C:\Users\omnap\miniconda3\envs\llmdyn\Lib\site-packages\torch\optim\adamw.py` constructor, `torch\optim\optimizer.py` zero_grad default, and `torch\nn\functional.py` cross_entropy defaults. Local task sources are original notebook cells0–3,5–6, `fullscale_experiments.py` main loop, and `experiment_suite.py` teacher_eval/model factory. These references establish the inspected current implementation, not the unrecorded historical package state.

## Intentional differences and runtime unknowns

1. **Data-loader random trajectory differs.** The original train loader uses global Torch RNG with`shuffle=True` and no explicit generator. The full-data loader uses a dedicated `torch.Generator` seeded1042 for matched per-model minibatch orders. This changes the optimization trajectory, even with identical initial weights. It is an actual documented design difference; this audit does not establish that it quantitatively explains the score gap.
2. **Fresh independent model seeds differ from unobserved notebook state.** The full runner reseeds Python/NumPy/Torch/CUDA42 before each model. The notebook seeds only in cell0 and does not reseed in either training cell. Its out-of-order execution counts and null loop-definition execution count prevent reconstructing historical initialization/dropout/shuffle states. Do not describe the historical looped model as independently seed42-initialized.
3. **Exact replay is a fresh data reconstruction.** Full train50k/val5k replay matches the separately reconstructed seed42 saved generator byte-for-byte after dtype normalization. The notebook did not retain its historical arrays, so their identity with the fresh reconstruction remains unverified. Additional fresh test5k is intentionally new and does not enter optimization/validation.
4. **Extra evaluation/checkpointing/logging are present.** Baseline30 test evaluation saves/restores global Python/NumPy/Torch/CUDA RNG; test evaluation never updates weights. The dedicated train-loader generator is not consumed by it. Subsequent epochs explicitly restore training mode. Checkpoint writes and logged gradient norms do not alter model gradients or optimizer settings. They add timing overhead.
5. **Hardware/runtime are known only for fresh runs.** Fresh run records Python3.12.13,NumPy2.4.6,PyTorch2.11.0+cu128,CUDA12.8 and RTX5080 Laptop GPU. Historical notebook records Python/kernel/deviceCUDA but no full package/backend/precision/driver metadata.
6. **Determinism and TF32 are not retrospectively known.** Training source does not enable autocast, change TF32/matmul settings, force deterministic algorithms, or set cuDNN benchmark flags. `determinism_environment_defaults.json` explicitly describes a separate fresh llmdyn interpreter: deterministic algorithmsFalse, cuDNN deterministicFalse/benchmarkFalse, CUDA matmul allow_tf32False,cuDNN allow_tf32True,float32 matmul precisionhighest. It is not introspection of the already-running training process or historical kernel. Do not overstate these as measured historical settings.
7. **CPU thread setting differs from unspecified historical default.** Full runner sets4Torch CPU threads; original notebook does not set a count. GPU kernels perform training. This may affect timing and is a runtime distinction, without evidence that it explains the accuracy difference.

## Loss reporting correction and observed endpoint

The notebook prints **mean of batch-mean losses**: sum of each minibatch's nonPAD mean CE divided by number of batches. The new runner records both that exact aggregation (`train_batch_mean_loss`) and a target-token-weighted epoch mean (`train_token_mean_loss`), but prints the latter in console progress.

At newly completed baseline epoch90:

- New token-weighted training loss:`0.1469308558316885`.
- New batch-mean training loss:`0.14657939663704703`.
- Difference from aggregation choice:`0.00035145919464147` nats, far smaller than the historical/new loss difference.
- Saved historical epoch90 batch-mean loss:`0.0843` (display-rounded).
- New validation token accuracy:`40435/41903 = 96.49667088275302%`; historical saved epoch90:`98.09%`.

Therefore compare`train_batch_mean_loss` when discussing fidelity to the notebook's printed training loss. The aggregation choice does **not** explain the substantial observed training-loss gap. These values are validation results, not held-out test metrics, and the newly observed poorer last epoch must not be silently replaced with a best epoch.

## Source integrity and conclusion

The source hashes still match the frozen run environment: full runner`e670b6b617d8e87e5d1f1183fad59cbe87dbc9eb222fb1f904f51857ce201d60`; imported suite`e855368e4a0ec9371330dd5e962d3ec30446c662e1f671769b6ae0c1ecb7cc0b`. No training-source change occurred during the reviewed run.

This audit rules out the inspected common implementation errors—wrong optimizer arguments, target shift, PAD reduction, loss order, clipping, train/eval mode, epoch budget, scheduler or an AMP switch—as visible explanations. It does not prove that RNG caused the discrepancy, nor that all historical settings matched. The correct scientific description is **a fresh original-scale run with matching saved architecture/generator/training code and explicitly changed random-order protocol, which did not reproduce the exact historical numerical trace; the causal reason for that difference remains unresolved**.
