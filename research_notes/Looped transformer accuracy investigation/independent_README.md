# Independent audit artifacts

These files belong to the independent notebook/statistical audit. They are separate from the GPU training suite. The original notebook was read only and has SHA256`fd187163c7ea47f268a1734f1e6ab3c6dc306906950bed1f0793b030743e4701`.

| File | Contents/provenance |
|---|---|
| `independent_audit.md` | Saved-code comparison, metric/provenance caveats, new replay/heuristic results, pre-results interpretation rules |
| `independent_statistics.py` | Replays the original maze generator in isolation at seed42, checks uniqueness/overlap, instantiates saved model classes on CPU for exact parameter counts, evaluates a nonlearned heuristic, extracts saved training histories |
| `independent_statistics.json` | Aggregate replay statistics, hashes, parameter counts, execution counts, environment versions and elapsed runtime |
| `independent_original_histories.csv` | 120 rows extracted from persisted notebook stdout; **not new training**; loss and accuracy retain only original displayed precision |
| `independent_original_comparison.py` / `.csv` | Generates three endpoint contrasts: equal epoch30, final versus final, and post hoc best versus best; update/example/block-call counts explicitly assume saved50k/batch128 settings |
| `independent_baselines.csv` | 5,000 rows for newly replayed validation mazes: IDs/grid hashes, shortest length and path count, heuristic teacher-forced correct/total tokens, autonomous solved/shortest/canonical-exact flags |
| `epiplexity.md` | Requested paper's verified scope, no-confirmation assessment, later-stage falsifiable study and logging protocol |
| `epiplexity_sources.json` | Versioned primary-source bibliography and notebook provenance |
| `epiplexity_search_log.jsonl` | Ordered literature retrieval/search record; secondary results not used for conclusions |

## Reproduce diagnostics

From PowerShell, run:

```powershell
& 'C:\Users\omnap\miniconda3\envs\llmdyn\python.exe' 'D:\Projects\Loopedtrans\research_notes\Looped transformer accuracy investigation\independent_statistics.py'
```

The checked environment is Python3.12.13, NumPy2.4.6, PyTorch2.11.0+cu128. The script runs on CPU, including model instantiation; it does not train any model. It writes the three independent JSON/CSV outputs beside itself, replacing its own previous diagnostic outputs. Archive/copy those outputs before rerunning if preserving every replay timestamp/runtime matters. It does not install packages, change configuration, use external compute, or modify the notebook. The final checked execution completed in4.13seconds excluding interpreter/import startup; timing varies across machines.

The script was executed twice during preparation: first for dataset/count/history diagnostics, then after adding the heuristic/shortest-path-count diagnostic. The second execution supersedes the first for timing and adds metrics; shared deterministic data/count/hash results matched. These are diagnostic development executions, **not repeated independent statistical replicates**.

## Units and plotting

- Original history column `val_token_accuracy_rounded_percent` uses0–100 percentage units; JSON heuristic accuracy/success fields use0–1 fractions. Convert explicitly before comparing plots.
- `train_loss_rounded_nats` is the saved original minibatch-average displayed cross-entropy. Do not treat it as exact summed token NLL or a validated prequential code.
- Replay hashes cover uint8 grid contents; they are not historical dataset IDs. NumPy sampling is the original legacy `np.random.choice`, Python seed42, with training generated before validation. This differs from the controlled suite's new `default_rng` datasets.
- `n_shortest_paths` counts all shortest grid routes; canonical correctness refers specifically to BFS tie order U,L,R,D. Greedy rollout caps at40 moves.
- Plot token accuracy against epoch only as a data-exposure view. It is not a matched compute view across different effective depths. Use measured training-suite timings for new compute comparisons.
- For path-length-conditioned heuristic plots, group `independent_baselines.csv` by `path_length`; weight teacher-forced token accuracy by summed `tf_total`, while success rates average per-maze binary flags. Do not average per-maze token ratios if intending the notebook's micro-average metric.
- No neural model predictions or historical checkpoints were available to this audit. Do not infer their autonomous success or paired statistical significance from the heuristic CSV.
