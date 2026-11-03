"""Regenerate tables, plots and paired finite-test uncertainty from raw logs.

No training or checkpoint selection occurs here. Fixed-seed conditional bootstrap
resamples mazes, not individual tokens. Seed variability is reported separately.
"""
import argparse, collections, csv, datetime, hashlib, json, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiment_suite import ROOT, write_json, DELTAS

def read_lines(path):
    return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]

def save_csv(path,rows):
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def bootstrap_ratio(delta,denom,n=5000,seed=20261105):
    rng=np.random.default_rng(seed);out=[]
    for _ in range(0,n,100):
        ix=rng.integers(0,len(delta),size=(min(100,n-len(out)),len(delta)))
        out.extend((delta[ix].sum(1)/denom[ix].sum(1)).tolist())
    return {'difference':float(delta.sum()/denom.sum()),'conditional_maze_bootstrap_95pct':np.quantile(out,[.025,.975]).tolist(),
            'replicates':n,'seed':seed,'unit':'paired maze, mean of fixed fitted model metrics; no voting/logit ensemble, no train-dataset or retraining uncertainty'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--prefix',default='controlled12k');a=p.parse_args()
    out=ROOT/'analysis'/a.prefix;out.mkdir(parents=True,exist_ok=True)
    rows=[];runs=[];epochs=[]
    for d in sorted((ROOT/'runs').glob(a.prefix+'_*')):
        if not (d/'summary.json').exists():continue
        cfg=json.loads((d/'config.json').read_text());s=json.loads((d/'summary.json').read_text());hist=read_lines(d/'epochs.jsonl')
        teacher=read_lines(d/'test_teacher_predictions.jsonl');roll=read_lines(d/'test_rollout_predictions.jsonl')
        # Preserve raw files; add corrected diagnostic flags in a derived file.
        testfile=Path(cfg['dataset_files']['test']) if 'dataset_files' in cfg else ROOT/'data'/f'test_n{cfg["test_size"]}_s20261104.npz'
        testgrids=np.load(testfile)['grids']
        derived=[derive_rollout(r,testgrids[r['example_id']]) for r in roll]
        save_csv(out/(d.name+'_rollout_diagnostics.csv'),[{k:v for k,v in r.items() if k not in ['trace','prediction','target']} for r in derived])
        save_csv(out/(d.name+'_length_metrics.csv'),by_length(teacher,derived))
        row={'run_id':d.name,'variant':cfg['variant'],'seed':cfg['seed'],'epochs':hist[-1]['epoch'],'optimizer_steps':hist[-1]['optimizer_steps'],
             'parameters':s['parameters'],'train_seconds':s['train_seconds'],'val_seconds':s['val_seconds'],
             'test_teacher_tokens':s['teacher_test']['tokens'],'test_teacher_correct':s['teacher_test']['correct'],
             'test_token_accuracy':s['teacher_test']['token_accuracy'],'test_action_accuracy':s['teacher_test']['action_accuracy'],
             'test_teacher_exact':s['teacher_test']['exact_accuracy'],'rollout_n':len(roll),
             **{'test_'+k:v for k,v in s['rollout_test'].items()},'test_actual_eos_emitted':np.mean([r['actual_eos_emitted'] for r in derived])}
        row['test_max_step_truncation_fraction']=float(np.mean([r['max_step_truncation'] for r in derived]))
        row['test_rollout_failure_counts_json']=json.dumps(dict(collections.Counter(r['outcome_reason'] for r in derived)))
        row['test_first_action_accuracy']=float(np.mean([r['teacher_pred'][0]==r['target'][0] for r in teacher]))
        row['teacher_rollout_exact_disagreement_count']=sum(bool(teacher[r['example_id']]['exact'])!=bool(r['rollout_exact']) for r in roll)
        norms=[e['gradient_norm_mean_before_clip'] for e in hist if 'gradient_norm_mean_before_clip' in e]
        clips=[e['gradient_clip_fraction'] for e in hist if 'gradient_clip_fraction' in e]
        row.update(epoch_mean_gradient_norm_before_clip=float(np.mean(norms)) if norms else None,
                   epoch_mean_gradient_clip_fraction=float(np.mean(clips)) if clips else None)
        if 'encoder_gate' in hist[-1]:row.update(final_encoder_gate=hist[-1]['encoder_gate'],final_decoder_gate=hist[-1]['decoder_gate'])
        for threshold in [.90,.95,.98,.99]:
            reached=[e for e in hist if e['val_token_accuracy']>=threshold]
            row[f'first_epoch_val{int(threshold*100)}']=reached[0]['epoch'] if reached else None
            row[f'first_training_seconds_val{int(threshold*100)}']=reached[0]['cumulative_train_seconds'] if reached else None
        rows.append(row);runs.append({'directory':d,'config':cfg,'summary':s,'epochs':hist,'teacher':teacher,'rollout':roll});epochs+=hist
    intermediate_rows=analyze_intermediates(out,a.prefix)
    if not rows:print(json.dumps({'complete_final_runs':0,'fixed_intermediate_evaluations':len(intermediate_rows),'analysis':str(out)}));return
    save_csv(out/'final_runs.csv',rows);save_csv(out/'all_epochs.csv',epochs)
    groups=collections.defaultdict(list)
    for r in rows:groups[r['variant']].append(r)
    agg=[]
    metrics=['test_token_accuracy','test_action_accuracy','test_first_action_accuracy','test_teacher_exact','test_rollout_exact','test_solved','test_optimal_solved','train_seconds']
    for key,rs in groups.items():
        x={'variant':key,'seeds':','.join(str(r['seed']) for r in rs),'n_seeds':len(rs),'parameters':rs[0]['parameters']}
        for m in metrics:
            vals=[r[m] for r in rs];x[m+'_mean']=float(np.mean(vals));x[m+'_sd']=float(np.std(vals,ddof=1)) if len(vals)>1 else None
        agg.append(x)
    save_csv(out/'architecture_summary.csv',agg)
    comparisons=[]
    pairs=[('untied6_full','tied6_full'),('postln4','preln4'),('preln4','preln6'),('preln6','tied6'),('tied6','tied6_anchor'),('tied6_anchor','tied6_full')]
    for before,after in pairs:
        bs={r['config']['seed']:r for r in runs if r['config']['variant']==before};ats={r['config']['seed']:r for r in runs if r['config']['variant']==after}
        seeds=sorted(set(bs)&set(ats))
        if not seeds:continue
        deltas=[];seed_effects=[];roll_deltas=[]
        for seed in seeds:
            b=bs[seed];t=ats[seed]
            bd=np.array([r['correct'] for r in b['teacher']]);td=np.array([r['correct'] for r in t['teacher']]);den=np.array([r['tokens'] for r in b['teacher']])
            assert len(bd)==len(td) and np.array_equal(den,[r['tokens'] for r in t['teacher']])
            deltas.append(td-bd);roll_deltas.append(np.array([r['optimal_solved'] for r in t['rollout']],dtype=float)-np.array([r['optimal_solved'] for r in b['rollout']],dtype=float))
            seed_effects.append(float((td-bd).sum()/den.sum()))
        comparisons.append({'before':before,'after':after,'primary_contrast':(before,after)==('untied6_full','tied6_full'),'paired_seeds':seeds,'seed_token_effects':seed_effects,
            'paired_seed_difference_mean':float(np.mean(seed_effects)),'paired_seed_difference_sd':float(np.std(seed_effects,ddof=1)) if len(seeds)>1 else None,
            'teacher_token':bootstrap_ratio(np.mean(deltas,axis=0),den),
            'rollout_optimal':bootstrap_ratio(np.mean(roll_deltas,axis=0),np.ones(len(roll_deltas[0])))})
    write_json(out/'paired_comparisons.json',comparisons)
    style();plot_historical(out);plot_controlled(out,runs,groups);plot_rollout(out,agg,runs);plot_depth(out,runs);plot_gradients(out,runs)
    manifest={'generated_utc':datetime.datetime.now(datetime.UTC).isoformat(),'prefix':a.prefix,
              'complete_runs':len(rows),'source_runs':[str(r['directory']) for r in runs],
              'analysis_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'notes':['All accuracies fractions0to1 in CSV; plotted percent.','Conditional paired maze bootstrap is not a confidence interval over retraining seeds.','Only fixed final checkpoint used; no best-checkpoint selection.','Raw path_length_histogram in data/manifest.json includes BOS+EOS; subtract2 for action length.']}
    write_json(out/'analysis_manifest.json',manifest);print(json.dumps({'analysis':str(out),'complete_runs':len(rows),'architecture_summary':agg}),flush=True)

def analyze_intermediates(out,prefix):
    rows=[]
    for p in sorted((ROOT/'runs').glob(prefix+'_*'+'/epoch*_evaluation/summary.json')):
        d=p.parent;run=d.parent;cfg=json.loads((run/'config.json').read_text());s=json.loads(p.read_text())
        epoch=int(d.name.split('_')[0].replace('epoch',''));h=read_lines(run/'epochs.jsonl');e=next(x for x in h if x['epoch']==epoch)
        teacher=read_lines(d/'test_teacher_predictions.jsonl');roll=read_lines(d/'test_rollout_predictions.jsonl')
        grids=np.load(cfg['dataset_files']['test'])['grids'];derived=[derive_rollout(r,grids[r['example_id']]) for r in roll]
        tag=run.name+f'_fixed_epoch{epoch}'
        save_csv(out/(tag+'_length_metrics.csv'),by_length(teacher,derived))
        save_csv(out/(tag+'_rollout_diagnostics.csv'),[{k:v for k,v in r.items() if k not in ['trace','target','prediction']} for r in derived])
        rows.append({'run_id':tag,'parent_run':run.name,'variant':cfg['variant'],'seed':cfg['seed'],'epoch':epoch,'optimizer_steps':e['optimizer_steps'],
            'parameters':s['parameters'],'train_seconds':s['train_seconds'],'val_token_accuracy':e['val_token_accuracy'],
            **{'test_teacher_'+k:v for k,v in s['teacher_test'].items()},
            **{'test_'+k:v for k,v in s['rollout_test'].items()},'rollout_n':len(roll),
            'test_first_action_accuracy':np.mean([r['teacher_pred'][0]==r['target'][0] for r in teacher]),
            'max_step_truncation_count':sum(r['max_step_truncation'] for r in derived),
            'note':'Prespecified intermediate endpoint, separate from final90; not averaged with final endpoint'})
    save_csv(out/'fixed_intermediate_evaluations.csv',rows)
    return rows

def by_length(teacher,roll):
    allrows=[]
    for lo,hi in [(4,5),(6,7),(8,9),(10,11),(12,15),(16,63)]:
        ts=[r for r in teacher if lo<=r['path_length']<=hi];rs=[r for r in roll if lo<=r['path_length']<=hi]
        row={'min_path_length':lo,'max_path_length':hi,'teacher_n':len(ts),'teacher_tokens':sum(r['tokens'] for r in ts),'teacher_correct':sum(r['correct'] for r in ts),'rollout_n':len(rs)}
        if ts:row.update(teacher_token_accuracy=sum(r['correct'] for r in ts)/sum(r['tokens'] for r in ts),teacher_exact=np.mean([r['exact'] for r in ts]))
        if rs:row.update({k:np.mean([r[k] for r in rs]) for k in ['rollout_exact','solved','optimal_solved','actual_eos_emitted','max_step_truncation']})
        allrows.append(row)
    return allrows

def derive_rollout(r,grid):
    emitted=2 in r['prediction'];truncated=len(r['prediction'])>=64 and not emitted
    if r['optimal_solved']:reason='optimal_canonical' if r['rollout_exact'] else 'optimal_alternative'
    elif r['solved']:reason='valid_nonoptimal_solution'
    elif not r['valid_path']:
        token=r['prediction'][r['moves']]
        if token not in DELTAS:reason='invalid_action_token'
        else:
            pos=r['trace'][-1];dr,dc=DELTAS[token];nr,nc=pos[0]+dr,pos[1]+dc
            reason='out_of_bounds' if not(0<=nr<len(grid) and 0<=nc<len(grid)) else 'wall_collision'
    elif emitted:reason='EOS_at_wrong_goal'
    elif truncated:reason='decode_cap_without_EOS'
    else:reason='unterminated_other'
    return {**r,'actual_eos_emitted':emitted,'max_decode_steps':64,'max_step_truncation':truncated,
            'outcome_reason':reason,'raw_terminated_semantics':'EOS encountered before first invalid trajectory action'}

def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':150,'savefig.bbox':'tight'})

def plot_historical(out):
    rows=list(csv.DictReader((ROOT/'audit'/'original_histories.csv').open(encoding='utf-8')))
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for name,color in [('baseline','#3c6e9a'),('looped','#bd572b')]:
        rs=[r for r in rows if r['model']==name];ep=[int(r['epoch']) for r in rs]
        axs[0].plot(ep,[float(r['val_token_accuracy_pct']) for r in rs],label=name,color=color)
        axs[1].plot(ep,[float(r['train_loss']) for r in rs],label=name,color=color)
    axs[0].set(ylabel='Teacher-forced validation token accuracy (%)',xlabel='Epoch',title='Saved notebook output: observational comparison')
    axs[1].set(ylabel='Reported batch-mean training cross entropy',xlabel='Epoch',yscale='log',title='Saved history; historical execution state unknown')
    for ax in axs:ax.legend();ax.grid(alpha=.15)
    fig.tight_layout();fig.savefig(out/'historical_curves.png');plt.close(fig)

def plot_controlled(out,runs,groups):
    fig,axs=plt.subplots(1,2,figsize=(12,4.6))
    colors=plt.get_cmap('tab10')
    for i,name in enumerate(groups):
        rs=[r for r in runs if r['config']['variant']==name];color=colors(i)
        ys=np.array([[e['val_token_accuracy'] for e in r['epochs']] for r in rs])*100
        xs=np.array([e['epoch'] for e in rs[0]['epochs']]);mean=ys.mean(0);sd=ys.std(0,ddof=1) if len(rs)>1 else ys[0]*0
        axs[0].plot(xs,mean,label=f'{name} (n={len(rs)})',color=color);axs[0].fill_between(xs,mean-sd,mean+sd,color=color,alpha=.12)
        for r in rs:axs[1].plot([e['cumulative_train_seconds'] for e in r['epochs']],[100*e['val_token_accuracy'] for e in r['epochs']],color=color,alpha=.45,linewidth=1)
    updates=math.ceil(runs[0]['config']['train_size']/runs[0]['config']['batch_size'])
    axs[0].set(xlabel=f'Epoch ({updates} updates/epoch)',ylabel='Validation teacher-forced token accuracy (%)',title='Fresh runs: mean ± seed SD')
    axs[1].set(xlabel='Cumulative measured training seconds',ylabel='Validation teacher-forced token accuracy (%)',title='Time to accuracy: individual runs')
    axs[0].legend(fontsize=8);axs[0].grid(alpha=.15);axs[1].grid(alpha=.15)
    fig.tight_layout();fig.savefig(out/'controlled_learning_curves.png');plt.close(fig)

def plot_rollout(out,agg,runs):
    names=[r['variant'] for r in agg];x=np.arange(len(names));fig,ax=plt.subplots(figsize=(12,4.8))
    for j,(key,label,color) in enumerate([('test_token_accuracy','Teacher-forced token','#477598'),('test_rollout_exact','Rollout canonical exact','#a9b6c2'),('test_optimal_solved','Rollout shortest-path success','#bd572b')]):
        means=[100*r[key+'_mean'] for r in agg];sd=[100*(r[key+'_sd'] or 0) for r in agg]
        ax.bar(x+(j-1)*.25,means,.24,label=label,yerr=sd,capsize=3,color=color)
    ax.set_xticks(x,names,rotation=25,ha='right');ax.set(ylabel='Test accuracy / success (%)',ylim=(0,100))
    seed_n='/'.join(map(str,sorted(set(r['n_seeds'] for r in agg))));test_n='/'.join(map(str,sorted(set(r['config']['test_size'] for r in runs))));roll_n='/'.join(map(str,sorted(set(r['summary']['rollout_n'] for r in runs))))
    ax.set_title(f'Held-out metrics: mean ± sample SD across {seed_n} seeds\nTeacher-forced n={test_n} mazes; rollout n={roll_n} mazes',pad=35)
    ax.legend(fontsize=8,loc='lower center',bbox_to_anchor=(.5,1.01),ncol=3,frameon=False);ax.grid(axis='y',alpha=.15);fig.tight_layout();fig.savefig(out/'test_token_vs_rollout.png');plt.close(fig)

def plot_depth(out,runs):
    fig,ax=plt.subplots(figsize=(8,4.8));groups=collections.defaultdict(list)
    for r in runs:
        if 'depth_sweep' in r['summary']:groups[r['config']['variant']].append(r['summary']['depth_sweep'])
    if not groups:plt.close(fig);return
    for name,ss in groups.items():
        ks=sorted(map(int,ss[0]));ys=np.array([[s[str(k)]['token_accuracy']*100 for k in ks] for s in ss]);mean=ys.mean(0);sd=ys.std(0,ddof=1) if len(ss)>1 else mean*0
        ax.errorbar(ks,mean,yerr=sd,marker='o',capsize=3,label=name)
    seed_n='/'.join(map(str,sorted(set(len(ss) for ss in groups.values()))));test_n='/'.join(map(str,sorted(set(r['config']['test_size'] for r in runs))))
    ax.axvline(6,color='gray',ls=':',label='trained depth K=6');ax.set(xlabel='Inference iterations (same final checkpoint)',ylabel='Test teacher-forced token accuracy (%)',title=f'Exploratory depth diagnostic: no retraining\nTeacher-forced n={test_n} mazes; {seed_n} seeds, mean ± sample SD');ax.legend();ax.grid(alpha=.15)
    fig.tight_layout();fig.savefig(out/'test_depth_sweep.png');plt.close(fig)

def plot_gradients(out,runs):
    runs=[r for r in runs if all('gradient_clip_fraction' in e for e in r['epochs'])]
    if not runs:return
    fig,axs=plt.subplots(1,2,figsize=(11,4.5));colors=plt.get_cmap('tab10');names=list(dict.fromkeys(r['config']['variant'] for r in runs))
    for i,name in enumerate(names):
        rs=[r for r in runs if r['config']['variant']==name];ys=np.array([[e['gradient_clip_fraction'] for e in r['epochs']] for r in rs]);ep=np.arange(1,ys.shape[1]+1)
        axs[0].plot(ep,ys.mean(0)*100,label=name,color=colors(i))
        if 'encoder_gate' in rs[0]['epochs'][-1]:
            for key,ls in [('encoder_gate','-'),('decoder_gate','--')]:
                gate=np.array([[e[key] for e in r['epochs']] for r in rs]);axs[1].plot(ep,gate.mean(0),ls,label=f'{name} {key.split("_")[0]}',color=colors(i))
    axs[0].set(xlabel='Epoch',ylabel='Updates with pre-clip norm > 1 (%)',title='Clipping prevalence (diagnostic, no causal isolation)')
    axs[1].set(xlabel='Epoch',ylabel='Learned input-injection scalar',title='Anchor evolution: encoder solid, decoder dashed')
    for ax in axs:ax.legend(fontsize=7);ax.grid(alpha=.15)
    fig.tight_layout();fig.savefig(out/'gradient_clipping_and_gates.png');plt.close(fig)

if __name__=='__main__':main()
