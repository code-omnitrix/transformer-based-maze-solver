"""Read-only notebook audit; deterministic replay is NOT recovered historical data."""
import ast
import csv
import hashlib
import json
import math
import platform
import random
import re
import time
from collections import Counter, deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT.parents[1] / 'aiml.ipynb'
started = time.perf_counter()
raw = NOTEBOOK.read_bytes()
nb = json.loads(raw)
source = lambda i: ''.join(nb['cells'][i]['source'])
namespace = dict(random=random, np=np, deque=deque, GRID_SIZE=8, WALL_PROB=.28,
                 MAZE_VOCAB={'FREE':0,'WALL':1,'START':2,'GOAL':3},
                 MOVE_DELTAS={'U':(-1,0),'L':(0,-1),'R':(0,1),'D':(1,0)})
func = next(x for x in ast.parse(source(1)).body if isinstance(x, ast.FunctionDef))
exec(compile(ast.Module(body=[func], type_ignores=[]), str(NOTEBOOK), 'exec'), namespace)
random.seed(42)
np.random.seed(42)
arrays = {}
hashes = {}
stats = {}
for split, size in [('train',50000),('validation',5000)]:
    grids, lengths = [], []
    for _ in range(size):
        grid, path, start, goal = namespace['generate_solvable_maze']()
        grids.append(grid.astype(np.uint8))
        lengths.append(len(path))
    arr = np.stack(grids)
    arrays[split] = arr
    hashes[split] = [hashlib.sha256(g.tobytes()).hexdigest() for g in arr]
    steps = sum(lengths)
    target_tokens = steps + size
    max_target = max(lengths)+1
    stats[split] = dict(n=size, unique_grids=len(set(hashes[split])),
                        array_sha256=hashlib.sha256(arr.tobytes()).hexdigest(),
                        min_path=min(lengths), max_path=max(lengths),
                        mean_path=float(np.mean(lengths)), path_histogram=dict(sorted(Counter(lengths).items())),
                        nonpad_target_tokens=target_tokens, eos_target_tokens=size,
                        eos_fraction=size/target_tokens,
                        hypothetical_pad_fraction_if_counted=(size*max_target-target_tokens)/(size*max_target),
                        note='replayed seed42 code with fresh NumPy and Python RNG; not historical dataset recovery')
stats['train_validation_grid_overlap'] = len(set(hashes['train']) & set(hashes['validation']))
def bfs_oracle(grid):
    start = tuple(np.argwhere(grid == 2)[0])
    goal = tuple(np.argwhere(grid == 3)[0])
    queue = deque([start])
    distances, counts, paths = {start:0}, {start:1}, {start:[]}
    while queue:
        pos = queue.popleft()
        for act, (dr, dc) in namespace['MOVE_DELTAS'].items():
            nxt = (pos[0]+dr,pos[1]+dc)
            if 0 <= nxt[0] < 8 and 0 <= nxt[1] < 8 and grid[nxt] != 1:
                if nxt not in distances:
                    distances[nxt] = distances[pos]+1
                    counts[nxt] = counts[pos]
                    paths[nxt] = paths[pos]+[act]
                    queue.append(nxt)
                elif distances[nxt] == distances[pos]+1:
                    counts[nxt] += counts[pos]
    return start,goal,paths[goal],counts[goal]

def greedy_next(grid,pos,goal):
    if pos == goal:
        return 'EOS'
    candidates = []
    for act,(dr,dc) in namespace['MOVE_DELTAS'].items():
        nxt = (pos[0]+dr,pos[1]+dc)
        if 0<=nxt[0]<8 and 0<=nxt[1]<8 and grid[nxt] != 1:
            candidates.append((abs(nxt[0]-goal[0])+abs(nxt[1]-goal[1]),act))
    # Python stable sort respects canonical U,L,R,D on equal Manhattan distance.
    return min(candidates,key=lambda x:x[0])[1] if candidates else 'EOS'

baseline_rows = []
for maze_id,grid in enumerate(arrays['validation']):
    start,goal,truth,n_shortest = bfs_oracle(grid)
    pos, correct = start,0
    for act in truth+['EOS']:
        correct += greedy_next(grid,pos,goal)==act
        if act != 'EOS':
            dr,dc = namespace['MOVE_DELTAS'][act]
            pos = (pos[0]+dr,pos[1]+dc)
    pos,rollout = start,[]
    for _ in range(40):
        act = greedy_next(grid,pos,goal)
        if act == 'EOS':
            break
        rollout.append(act)
        dr,dc = namespace['MOVE_DELTAS'][act]
        pos = (pos[0]+dr,pos[1]+dc)
    solved = pos==goal
    baseline_rows.append(dict(maze_id=maze_id,grid_sha256=hashes['validation'][maze_id],
                             path_length=len(truth),n_shortest_paths=n_shortest,
                             greedy_tf_correct=correct,tf_total=len(truth)+1,
                             greedy_rollout_solved=int(solved),
                             greedy_rollout_shortest=int(solved and len(rollout)==len(truth)),
                             greedy_canonical_exact=int(rollout==truth)))
with (ROOT/'independent_baselines.csv').open('w',newline='',encoding='utf8') as handle:
    writer = csv.DictWriter(handle, fieldnames=list(baseline_rows[0])); writer.writeheader(); writer.writerows(baseline_rows)
stats['validation']['multiple_shortest_path_maze_fraction'] = sum(r['n_shortest_paths']>1 for r in baseline_rows)/5000
stats['greedy_wall_aware_manhattan_baseline'] = dict(
    teacher_forced_token_accuracy=sum(r['greedy_tf_correct'] for r in baseline_rows)/sum(r['tf_total'] for r in baseline_rows),
    rollout_goal_success=sum(r['greedy_rollout_solved'] for r in baseline_rows)/5000,
    rollout_shortest_success=sum(r['greedy_rollout_shortest'] for r in baseline_rows)/5000,
    rollout_canonical_exact=sum(r['greedy_canonical_exact'] for r in baseline_rows)/5000,
    max_rollout_steps=40,description='no learned parameters; choose valid neighbor minimizing Manhattan distance, tie U,L,R,D; stop at goal')
stats['bfs_oracle_baseline'] = dict(canonical_exact=1.0,shortest_success=1.0,
                                  note='tautological correctness baseline because BFS generates labels; not learned')
stats['notebook_sha256'] = hashlib.sha256(raw).hexdigest()
stats['environment'] = dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__)
stats['execution_counts'] = [c.get('execution_count') for c in nb['cells']]
namespace.update(torch=torch, nn=nn, math=math,
                 ACTION_VOCAB={'PAD':0,'BOS':1,'EOS':2,'U':3,'L':4,'R':5,'D':6})
exec(source(2), namespace)
normal = namespace['MazeTransformerSolver']()
normal_params = sum(p.numel() for p in normal.parameters())
exec(source(5), namespace)
looped = namespace['MazeTransformerSolver']()
looped_params = sum(p.numel() for p in looped.parameters())
stats['parameters'] = dict(normal=normal_params, looped=looped_params,
                           normal_over_looped=normal_params/looped_params,
                           looped_module_dict_supports_decoder_attribute=hasattr(looped.transformer,'decoder'))
stats['optim_steps'] = dict(per_epoch=math.ceil(50000/128), normal=90*math.ceil(50000/128), looped=30*math.ceil(50000/128))
history = []
for cell, name in [(3,'normal_persisted'),(6,'looped_persisted')]:
    output = ''.join(''.join(o.get('text', [])) for o in nb['cells'][cell].get('outputs', []))
    for epoch, total, loss, accuracy in re.findall(r'Epoch (\d+)/(\d+) \| Train Loss: ([\d.]+) \| Val Token Acc: ([\d.]+)%',output):
        history.append(dict(source='saved_notebook_output_not_new_run', model=name, epoch=int(epoch),total_epochs=int(total),
                            train_loss_rounded_nats=float(loss),val_token_accuracy_rounded_percent=float(accuracy)))
with (ROOT/'independent_original_histories.csv').open('w',newline='',encoding='utf8') as handle:
    writer = csv.DictWriter(handle, fieldnames=list(history[0])); writer.writeheader(); writer.writerows(history)
stats['elapsed_seconds'] = time.perf_counter()-started
(ROOT/'independent_statistics.json').write_text(json.dumps(stats,indent=2),encoding='utf8')
print(json.dumps(stats,indent=2))
