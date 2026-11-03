# Research and experiment ledger

Date: 2026-11-02 (user timezone Asia/Calcutta). Environment files use UTC timestamps.

1. Read saved `aiml.ipynb` as JSON; never executed notebook training cells directly.
   Original SHA256: `fd187163c7ea47f268a1734f1e6ab3c6dc306906950bed1f0793b030743e4701`.
2. Found9cells, historical baseline90epoch output and loop30epoch output matching
   the user's pasted values. Saved execution counts are nonmonotonic and the
   loop architecture cell has null execution_count. Historical executed source,
   checkpoints and runtime are unknown.
3. Found existing llmdyn Python3.12 environment with PyTorchCUDA and RTX5080Laptop.
   DefaultPython3.14 has no torch. No software installation or remote service.
4. Extracted source and historical training curves into audit/. Instantiated
   original classes, counted1,338,119 vs352,521parameters; tested future-prefix
   perturbation (no influence on earlier logits at eval for either architecture).
5. Benchmarked synthetic batch128×target36 float32 warmup and10steady steps.
   Baseline median24.41ms, loop39.90ms. Actual epochs are shorter because actual
   padded sequences have a different length; measured run times are authoritative.
6. Predeclared controlled pilot: fixedtrain12k/val1k/test1k,30epochs,7architectures,
   seeds42/43/44, originalwidth128, batch128. Same fixed data and per-seed order
   across architectures. Primary sharing contrast tied6_full minusuntied6_full;
   normalization, depth, bare sharing, anchoring, loop index are secondary.
7. Training source frozen at hash
   `e855368e4a0ec9371330dd5e962d3ec30446c662e1f671769b6ae0c1ecb7cc0b`
   in `source/`. Running Python process is independent of later source-file edits;
   source was not edited after launch. New analysis/resume/fullscale files are
   separate from the frozen training program.
8. Independent harness review validated original-versus-controlled forward paths
   at copiedweights, all7causal masks, and legal/optimal/exact trajectory cases.
   Two raw logging semantics need derived corrections, without touching raw data:
   `path_length_histogram` counts BOS+actions+EOS (subtract2);
   `terminated` means EOS was encountered before tracing stopped at an illegal
   action, not whether the raw generation eventually emitted EOS. Plot/analysis
   scripts reconstruct actual EOS and action-length bins from saved arrays.
9. Implemented explicit resume CLI restoring model/optimizer/CPU+CUDA/Python/NumPy
   RNG and shuffle-generator state. CPU two-epoch dropout+shuffle test gives exact
   next-epoch loss and state equality; GPU bitwise equality is not asserted.
10. Predeclared four full-sizefollowups BEFORE examining their heldout results:
    original_baseline90, original_looped30, controlledtied6_full30,
    controlleduntied6_full30; originaltrain50k/val5k generatorseed42, freshtest5k,
    seed42architecture. Baseline also saved/evaluated at30. These are new fresh
    runs, not recoveries of unknown historical checkpoints.

All pilots use fixed final checkpoints. Validation is observed each epoch;
heldout test is evaluated only at the prespecified final checkpoint (baseline
additional fixed30 checkpoint). No architecture or stopping rule is selected by
test accuracy. Extra inference-depth sweeps and any harder-maze evaluation are
exploratory diagnostics. Timing is local device evidence, not hardware-invariant.

The effect of the new files is confined to this research directory. Removing it
undoes the file additions; the notebook and environment packages are unchanged.
Compute is local and sequential per GPU. Runtime and disk consumption are logged.
