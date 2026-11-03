"""Read-only CPU/NumPy audit of completed controlled matrix; no Torch import."""
import collections
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'analysis'/'controlled12k'
def j(path):return json.loads(path.read_text(encoding='utf8'))
def lines(path):return [json.loads(x) for x in path.read_text(encoding='utf8').splitlines() if x.strip()]
def csvrows(path):
    with path.open(encoding='utf8',newline='') as f:return list(csv.DictReader(f))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def near(a,b):assert math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=1e-10),(a,b)
def flag(s):return s if isinstance(s,bool) else s=='True'

variants={'postln4','preln4','preln6','tied6','tied6_anchor','tied6_full','untied6_full'}
runs=sorted((ROOT/'runs').glob('controlled12k_*'))
assert len(runs)==21
datasetmeta=j(ROOT/'data'/'manifest.json')
data={}
for item in datasetmeta['splits']:
    p=Path(item['file']);assert sha(p)==item['sha256']
    z=np.load(p);grids,seqs=z['grids'],z['seqs']
    key='train' if len(grids)==12000 else ('val' if 'val_' in p.name else 'test')
    assert item['path_length_histogram']=={str(k):v for k,v in collections.Counter(np.count_nonzero(seqs,axis=1).tolist()).items()}
    data[key]=(grids,seqs,item)
assert set(data)=={'train','val','test'}
train_tokens=int(np.count_nonzero(data['train'][1][:,1:]))
testgrids,testseqs,_=data['test']
summaries=csvrows(OUT/'final_runs.csv');byid={r['run_id']:r for r in summaries}
all_epochs=csvrows(OUT/'all_epochs.csv');assert len(all_epochs)==630
sources=set();results={};combinations=set();failurecounts=collections.Counter()
DELTAS={3:(-1,0),4:(0,-1),5:(0,1),6:(1,0)}

def recompute_roll(row,grid):
    pos=tuple(np.argwhere(grid==2)[0]);goal=tuple(np.argwhere(grid==3)[0])
    moves=0;valid=True;eos_before_illegal=False;reason=None;trace=[list(map(int,pos))]
    for token in row['prediction']:
        if token==2:eos_before_illegal=True;break
        if token not in DELTAS:valid=False;reason='invalid_action_token';break
        dr,dc=DELTAS[token];nxt=(pos[0]+dr,pos[1]+dc)
        if not (0<=nxt[0]<8 and 0<=nxt[1]<8):valid=False;reason='out_of_bounds';break
        if grid[nxt]==1:valid=False;reason='wall_collision';break
        pos=nxt;moves+=1;trace.append(list(map(int,pos)))
    goal_reached=valid and pos==goal
    solved=goal_reached and eos_before_illegal
    exact=row['prediction']==row['target']
    optimal=solved and moves==len(row['target'])-1
    emitted=2 in row['prediction'];truncated=len(row['prediction'])>=64 and not emitted
    if optimal:reason='optimal_canonical' if exact else 'optimal_alternative'
    elif solved:reason='valid_nonoptimal_solution'
    elif valid:reason='EOS_at_wrong_goal' if emitted else ('decode_cap_without_EOS' if truncated else 'unterminated_other')
    return dict(rollout_exact=exact,valid_path=valid,terminated=eos_before_illegal,goal_reached=goal_reached,
                solved=solved,optimal_solved=optimal,moves=moves,trace=trace,
                actual_eos_emitted=emitted,max_step_truncation=truncated,outcome_reason=reason)

for d in runs:
    cfg,env,summary=j(d/'config.json'),j(d/'environment.json'),j(d/'summary.json')
    name,seed=cfg['variant'],cfg['seed'];combinations.add((name,seed));sources.add(env['script_sha256'])
    assert name in variants and seed in [42,43,44]
    assert (cfg['epochs'],cfg['train_size'],cfg['val_size'],cfg['test_size'],cfg['batch_size'],cfg['rollout_n'])==(30,12000,1000,1000,128,512)
    assert not cfg['small'] and cfg['lr']==.001 and cfg['weight_decay']==.0001 and cfg['clip_grad_norm']==1
    assert cfg['dataset_sha256']=={k:v[2]['sha256'] for k,v in data.items()}
    assert env['notebook_sha256']==sha(ROOT.parents[1]/'aiml.ipynb')
    assert sha(ROOT/'source'/('experiment_suite_'+env['script_sha256']+'.py'))==env['script_sha256']
    ep=lines(d/'epochs.jsonl');steps=lines(d/'steps.jsonl')
    assert len(ep)==30 and [r['epoch'] for r in ep]==list(range(1,31))
    assert [r['optimizer_steps'] for r in ep]==[94*i for i in range(1,31)]
    assert len(steps)==2820 and [r['optimizer_step'] for r in steps]==list(range(1,2821))
    for e in ep:
        st=steps[(e['epoch']-1)*94:e['epoch']*94]
        assert all(r['epoch']==e['epoch'] for r in st)
        assert sum(r['target_tokens'] for r in st)==train_tokens==e['train_tokens']
        near(e['train_token_mean_loss'],sum(r['pre_update_teacher_forced_token_mean_nll']*r['target_tokens'] for r in st)/train_tokens)
        near(e['train_batch_mean_loss'],statistics.mean(r['pre_update_teacher_forced_token_mean_nll'] for r in st))
    near(summary['train_seconds'],sum(r['train_seconds'] for r in ep))
    teacher,roll=lines(d/'test_teacher_predictions.jsonl'),lines(d/'test_rollout_predictions.jsonl')
    assert len(teacher)==1000 and len(roll)==512
    assert [r['example_id'] for r in teacher]==list(range(1000))
    assert [r['example_id'] for r in roll]==list(range(512))
    counts=collections.Counter()
    for i,row in enumerate(teacher):
        target=testseqs[i,1:];target=target[target!=0].tolist();assert row['target']==target
        assert row['tokens']==len(target) and row['path_length']==len(target)-1
        correct=sum(a==b for a,b in zip(row['teacher_pred'],target))
        assert len(row['teacher_pred'])==len(target) and row['correct']==correct and row['exact']==(correct==len(target))
        counts.update(tokens=len(target),correct=correct,examples=1,exact=int(correct==len(target)),
                      actions=sum(t>2 for t in target),actions_correct=sum(a==b and b>2 for a,b in zip(row['teacher_pred'],target)),
                      eos=target.count(2),eos_correct=sum(a==b==2 for a,b in zip(row['teacher_pred'],target)))
    for k,v in counts.items():assert summary['teacher_test'][k]==v
    near(summary['teacher_test']['token_accuracy'],counts['correct']/counts['tokens'])
    near(summary['teacher_test']['action_accuracy'],counts['actions_correct']/counts['actions'])
    near(summary['teacher_test']['exact_accuracy'],counts['exact']/1000)
    derived=csvrows(OUT/(d.name+'_rollout_diagnostics.csv'));assert len(derived)==512
    reasons=collections.Counter()
    for i,row in enumerate(roll):
        assert row['target']==teacher[i]['target'] and row['path_length']==len(row['target'])-1
        assert row['grid_sha256']==hashlib.sha256(testgrids[i].tobytes()).hexdigest()
        expected=recompute_roll(row,testgrids[i])
        for k in ['rollout_exact','valid_path','terminated','goal_reached','solved','optimal_solved','moves','trace']:assert row[k]==expected[k],(d.name,i,k)
        for k in ['rollout_exact','valid_path','terminated','goal_reached','solved','optimal_solved','actual_eos_emitted','max_step_truncation']:assert flag(derived[i][k])==expected[k]
        assert derived[i]['outcome_reason']==expected['outcome_reason']
        reasons[expected['outcome_reason']]+=1
    for k,v in summary['rollout_test'].items():near(v,statistics.mean(r[k] for r in roll))
    final=byid[d.name]
    for c,k in [('test_token_accuracy','token_accuracy'),('test_action_accuracy','action_accuracy'),('test_teacher_exact','exact_accuracy')]:near(final[c],summary['teacher_test'][k])
    for k,v in summary['rollout_test'].items():near(final['test_'+k],v)
    near(final['test_first_action_accuracy'],statistics.mean(r['teacher_pred'][0]==r['target'][0] for r in teacher))
    near(final['test_actual_eos_emitted'],statistics.mean(2 in r['prediction'] for r in roll))
    near(final['test_max_step_truncation_fraction'],statistics.mean(len(r['prediction'])>=64 and 2 not in r['prediction'] for r in roll))
    assert json.loads(final['test_rollout_failure_counts_json'])==dict(reasons)
    for binrow in csvrows(OUT/(d.name+'_length_metrics.csv')):
        lo,hi=int(binrow['min_path_length']),int(binrow['max_path_length'])
        ts=[r for r in teacher if lo<=r['path_length']<=hi];rs=[r for r in roll if lo<=r['path_length']<=hi]
        assert len(ts)==int(binrow['teacher_n']) and len(rs)==int(binrow['rollout_n'])
        assert sum(r['tokens'] for r in ts)==int(binrow['teacher_tokens'])
        if ts:near(binrow['teacher_token_accuracy'],sum(r['correct'] for r in ts)/sum(r['tokens'] for r in ts))
    failurecounts.update(reasons)
    results[(name,seed)]={'teacher':teacher,'roll':roll,'final':final}

assert combinations=={(v,s) for v in variants for s in [42,43,44]}
aggregates=csvrows(OUT/'architecture_summary.csv')
for row in aggregates:
    assert int(row['n_seeds'])==3 and row['seeds']=='42,43,44'
    for col,value in row.items():
        if col.endswith('_mean') or col.endswith('_sd'):
            metric,kind=col.rsplit('_',1)
            vals=[float(results[(row['variant'],s)]['final'][metric]) for s in [42,43,44]]
            near(value,statistics.mean(vals) if kind=='mean' else statistics.stdev(vals))

def boot(delta,denom):
    rng=np.random.default_rng(20261105);values=[]
    for batch in range(50):
        ix=rng.integers(0,len(delta),size=(100,len(delta)))
        values.extend((np.sum(delta[ix],axis=1)/np.sum(denom[ix],axis=1)).tolist())
    return (float(np.sum(delta)/np.sum(denom)),np.quantile(values,[.025,.975]).tolist())

paired=j(OUT/'paired_comparisons.json');paired_result=[]
for contrast in paired:
    bd,ad=contrast['before'],contrast['after'];tdeltas=[];rdeltas=[];effects=[]
    for seed in [42,43,44]:
        b,a=results[(bd,seed)],results[(ad,seed)]
        den=np.array([r['tokens'] for r in b['teacher']]);delta=np.array([a['teacher'][i]['correct']-r['correct'] for i,r in enumerate(b['teacher'])])
        tdeltas.append(delta);effects.append(float(delta.sum()/den.sum()))
        rdeltas.append(np.array([int(a['roll'][i]['optimal_solved'])-int(r['optimal_solved']) for i,r in enumerate(b['roll'])]))
    assert np.allclose(effects,contrast['seed_token_effects'])
    near(statistics.mean(effects),contrast['paired_seed_difference_mean']);near(statistics.stdev(effects),contrast['paired_seed_difference_sd'])
    for key,delta,denom in [('teacher_token',np.mean(tdeltas,axis=0),den),('rollout_optimal',np.mean(rdeltas,axis=0),np.ones(512))]:
        estimate,ci=boot(delta,denom);near(estimate,contrast[key]['difference'])
        assert np.allclose(ci,contrast[key]['conditional_maze_bootstrap_95pct'],rtol=1e-12,atol=1e-12)
    paired_result.append({'contrast':ad+' minus '+bd,'mean_token_effect_pp':100*statistics.mean(effects),'seed_sd_pp':100*statistics.stdev(effects)})

manifest=j(OUT/'analysis_manifest.json')
assert manifest['complete_runs']==21 and manifest['analysis_script_sha256']==sha(ROOT/'analyze_runs.py')
report={'status':'all_matrix_numeric_checks_passed','runs':21,'variants':7,'seeds_per_variant':3,
        'epochs_per_run':30,'updates_per_run':2820,'total_epoch_records':630,'total_update_records':21*2820,
        'teacher_rows_checked':21000,'rollout_rows_checked':10752,'source_hashes':sorted(sources),
        'analysis_script_sha256':manifest['analysis_script_sha256'],'analysis_generated_utc':manifest['generated_utc'],
        'dataset_hashes':{k:v[2]['sha256'] for k,v in data.items()},
        'test_target_tokens':int(np.count_nonzero(testseqs[:,1:])),'test_action_tokens':int(np.sum(testseqs>2)),
        'derived_failures_across_all_models':dict(failurecounts),'paired_contrasts':paired_result,
        'bootstrap':'5000 whole-maze resamples with RNG20261105; common sampled maze IDs across all3 fixed fitted seed models; token-weighted ratio; no retraining or training-data-resampling uncertainty',
        'nll_limit':'Teacher NLL probabilities not retained per example; aggregate NLL was not independently recalculated without model re-evaluation.',
        'plot_review':'Viewed all5 PNGs; percentages and seed SD correct. Final bar/depth captions confirm3seeds,teacher n1000 and bar rollout n512; gradient plot shows means without SD.'}
(ROOT/'final_matrix_review_checks.json').write_text(json.dumps(report,indent=2),encoding='utf8')
print(json.dumps(report,indent=2))
