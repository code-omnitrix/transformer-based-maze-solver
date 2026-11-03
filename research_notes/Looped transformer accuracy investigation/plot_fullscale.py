"""Single-seed full-data figures, including fixed baseline30 endpoint.

Run after analyze_runs.py --prefix full50k. Replaces two generic figure files
with explicitly single-seed figures; no sample SD is estimated from n=1.
"""
import datetime,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiment_suite import ROOT,write_json
from analyze_runs import read_lines,style,save_csv

def main():
    out=ROOT/'analysis'/'full50k';out.mkdir(parents=True,exist_ok=True);style()
    names=['original_baseline','original_looped','tied6_full','untied6_full'];labels={
      'original_baseline':'Original baseline','original_looped':'Original loop',
      'tied6_full':'Controlled tied full','untied6_full':'Controlled untied full'}
    runs=[]
    for name in names:
        d=ROOT/'runs'/f'full50k_{name}_seed42'
        if not(d/'summary.json').exists():continue
        runs.append((name,d,json.loads((d/'summary.json').read_text()),read_lines(d/'epochs.jsonl')))
    thresholds=[]
    for name,d,s,hist in runs:
        for threshold in [.90,.95,.98,.99]:
            met=next((e for e in hist if e['val_token_accuracy']>=threshold),None)
            thresholds.append({'variant':name,'seed':42,'validation_token_threshold':threshold,
                'first_observed_epoch':met['epoch'] if met else None,'first_observed_optimizer_update':met['optimizer_steps'] if met else None,
                'first_observed_cumulative_train_seconds':met['cumulative_train_seconds'] if met else None,
                'target_tokens_seen_including_repeated_epochs':sum(e['train_tokens'] for e in hist if met and e['epoch']<=met['epoch']) if met else None,
                'scope':'Descriptive first epoch-level validation crossing; no stable-crossing guarantee, no checkpoint selection or hypothesis test'})
    save_csv(out/'validation_thresholds.csv',thresholds)
    fig,axs=plt.subplots(1,2,figsize=(12,4.8));colors=plt.get_cmap('tab10')
    for i,(name,d,s,hist) in enumerate(runs):
        for ax,key in [(axs[0],'optimizer_steps'),(axs[1],'cumulative_train_seconds')]:
            ax.plot([e[key] for e in hist],[100*e['val_token_accuracy'] for e in hist],label=labels[name],color=colors(i))
            if name=='original_baseline':
                e=hist[29];ax.scatter(e[key],100*e['val_token_accuracy'],marker='s',s=40,color=colors(i),zorder=4)
    axs[0].set(xlabel='Optimizer updates',ylabel='Validation teacher-forced token accuracy (%)',title='Fresh full-size runs: one architecture seed each')
    axs[1].set(xlabel='Cumulative measured training seconds',ylabel='Validation teacher-forced token accuracy (%)',title='50k train / 5k validation; fixed checkpoints')
    for ax in axs:ax.legend(fontsize=8);ax.grid(alpha=.15)
    fig.tight_layout();fig.savefig(out/'controlled_learning_curves.png');plt.close(fig)
    points=[]
    for name,d,s,hist in runs:
        if name=='original_baseline':
            ep=json.loads((d/'epoch30_evaluation'/'summary.json').read_text());points.append(('Baseline30',ep));points.append(('Baseline90',s))
        else:points.append((labels[name]+'30',s))
    fig,ax=plt.subplots(figsize=(12,5));x=np.arange(len(points))
    for i,(key,label,color) in enumerate([('token','Teacher-forced token','#477598'),('canonical','Rollout canonical exact','#a9b6c2'),('optimal','Rollout shortest-path success','#bd572b')]):
        vals=[]
        for _,s in points:
            vals.append(100*(s['teacher_test']['token_accuracy'] if key=='token' else s['rollout_test']['rollout_exact' if key=='canonical' else 'optimal_solved']))
        ax.bar(x+(i-1)*.25,vals,.24,color=color,label=label)
    ax.set_xticks(x,[a for a,_ in points],rotation=18,ha='right');ax.set(ylabel='Held-out accuracy / success (%)',ylim=(0,100))
    ax.set_title('Fresh full-size results: one seed per model; no between-seed uncertainty estimate\nTeacher-forced n=5000 mazes; rollout n=1000 mazes',pad=35)
    ax.legend(loc='lower center',bbox_to_anchor=(.5,1.01),ncol=3,frameon=False,fontsize=8);ax.grid(axis='y',alpha=.15)
    fig.tight_layout();fig.savefig(out/'test_token_vs_rollout.png');plt.close(fig)
    write_json(out/'full_plot_manifest.json',{'generated_utc':datetime.datetime.now(datetime.UTC).isoformat(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      'source_runs':[str(d) for _,d,_,_ in runs],'scope':'Explicit n=1 full-data figures; no seed SD; originalbaseline30 separately included',
      'replaced_generic_figures':['controlled_learning_curves.png','test_token_vs_rollout.png'],'no_numeric_source_changes':True})
    print(json.dumps({'full_run_count':len(runs),'fixed_endpoints':len(points),'output':str(out)}))

if __name__=='__main__':main()
