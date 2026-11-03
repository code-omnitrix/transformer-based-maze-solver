"""Summarize exploratory harder sets and correct diagnostic labels from raw tokens."""
import collections,csv,json
import numpy as np
from experiment_suite import ROOT,write_json
from analyze_runs import read_lines,save_csv,derive_rollout,by_length

def main():
    root=ROOT/'analysis'/'exploratory_harder';rows=[]
    if not root.exists():return
    for p in sorted(root.glob('*/*/summary.json')):
        d=p.parent;s=json.loads(p.read_text());teacher=read_lines(d/'teacher_predictions.jsonl');roll=read_lines(d/'rollout_predictions.jsonl')
        grids=np.load(s['data']['file'])['grids'];derived=[derive_rollout(r,grids[r['example_id']]) for r in roll]
        save_csv(d/'derived_rollout_diagnostics.csv',[{k:v for k,v in r.items() if k not in ['trace','target','prediction']} for r in derived])
        save_csv(d/'length_metrics.csv',by_length(teacher,derived))
        run=ROOT/'runs'/s['run_id'];cfg=json.loads((run/'config.json').read_text());trainfile=cfg.get('dataset_files',{}).get('train')
        if trainfile is None:trainfile=str(ROOT/'data'/f'train_n{cfg["train_size"]}_s20261102.npz')
        training_grids=np.load(trainfile)['grids'];import hashlib
        training_hashes={hashlib.sha256(g.tobytes()).hexdigest() for g in training_grids}
        rows.append({'run_id':s['run_id'],'variant':cfg['variant'],'seed':cfg['seed'],'dataset':s['data']['name'],
            'teacher_n':s['teacher_test']['examples'],'teacher_tokens':s['teacher_test']['tokens'],'teacher_correct':s['teacher_test']['correct'],
            'teacher_token_accuracy':s['teacher_test']['token_accuracy'],'teacher_exact':s['teacher_test']['exact_accuracy'],
            'first_action_accuracy':s['first_action_accuracy'],'rollout_n':len(roll),**s['rollout_test'],
            'actual_eos_emitted':np.mean([r['actual_eos_emitted'] for r in derived]),'max_step_truncation':sum(r['max_step_truncation'] for r in derived),
            'failure_counts_json':json.dumps(dict(collections.Counter(r['outcome_reason'] for r in derived))),
            'training_grid_overlap':len(training_hashes&set(s['data']['grid_hashes'])),
            'note':'Exploratory harder-size8 distribution, not guaranteed novel length support; fixed final checkpoint; no tuning'})
    save_csv(root/'final_runs.csv',rows);write_json(root/'summary_manifest.json',{'complete_evaluations':len(rows),'scope':'exploratory; fixed final checkpoints; no train/val changes','units':'accuracy fractions0to1, counts integer, elapsedseconds'})
    print(json.dumps(rows),flush=True)

if __name__=='__main__':main()
