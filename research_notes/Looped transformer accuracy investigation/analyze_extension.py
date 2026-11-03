"""Combined full-size table and figures: prespecified full50k runs + exploratory ext50k runs.

Reads only saved run artifacts (epochs.jsonl, summary.json). Writes analysis/combined50k/.
Usage: python analyze_extension.py
"""
import csv, datetime, hashlib, json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent; OUT=ROOT/'analysis'/'combined50k'; OUT.mkdir(parents=True,exist_ok=True)
UPDATES_PER_EPOCH=391
# Fixed display order and colour per model (validated categorical palette, light mode).
MODELS=[('original_baseline','Notebook baseline (Post-LN, 4L)','#2a78d6','prespecified'),
        ('original_baseline_warmup','Baseline + LR warmup','#eda100','exploratory'),
        ('original_baseline_preln','Baseline, Pre-LN (one flag)','#eb6834','exploratory'),
        ('original_baseline_preln6','Baseline, Pre-LN, 6L','#1baf7a','exploratory'),
        ('untied6_full','Untied 6L + reinjection + step enc.','#e87ba4','prespecified'),
        ('tied6_full','Looped, controlled init','#6250d6','prespecified'),
        ('original_looped','Notebook looped model','#008300','prespecified')]

def lines(p): return [json.loads(l) for l in p.read_text(encoding='utf-8').splitlines() if l.strip()]

LN2=0.6931471805599453

def loss_area_bits(run_dir,final_nll,max_epoch):
    """Heuristic analogue of Finzi et al. (2026) Sec. 4.1 prequential estimate: sum_i (loss_i - final)
    over training tokens, in bits. Final = final model's eval-mode test NLL per token (iid stand-in for
    re-scoring the same samples). Epoch 1 is the only fresh-sample stream; later epochs repeat data and
    use dropout, so the 'all' value is a loss-curve area, not an epiplexity estimate."""
    fresh=total=0.
    for s in lines(run_dir/'steps.jsonl'):
        if s['epoch']>max_epoch:break
        a=(s['pre_update_teacher_forced_token_mean_nll']-final_nll)*s['target_tokens']/LN2
        total+=a
        if s['epoch']==1:fresh+=a
    return fresh,total

def load():
    rows=[]
    for d in sorted((ROOT/'runs').iterdir()):
        if not (d.name.startswith('full50k_') or d.name.startswith('ext50k_')) or not (d/'summary.json').exists():continue
        cfg=json.loads((d/'config.json').read_text());ep=lines(d/'epochs.jsonl')
        evals=[(d/'summary.json',ep[-1]['epoch'])]
        if (d/'epoch30_evaluation'/'summary.json').exists():evals.append((d/'epoch30_evaluation'/'summary.json',30))
        for path,epoch in evals:
            s=json.loads(path.read_text());t=s['teacher_test'];r=s['rollout_test'];e=ep[epoch-1]
            fresh,total=loss_area_bits(d,t['mean_nll'],epoch)
            rows.append({'loss_area_bits_epoch1_fresh':fresh,'loss_area_bits_all_steps':total,'run_id':d.name,'variant':cfg['variant'],'seed':cfg['seed'],'stage':'prespecified' if d.name.startswith('full50k_') else 'exploratory',
                'epoch':epoch,'optimizer_steps':e['optimizer_steps'],'parameters':s['parameters'],'warmup_steps':cfg.get('warmup_steps',0),
                'cumulative_train_seconds':round(e['cumulative_train_seconds'],1),'val_token_accuracy':e['val_token_accuracy'],
                'test_token_accuracy':t['token_accuracy'],'test_token_error':1-t['token_accuracy'],'test_teacher_exact':t['exact_accuracy'],
                'test_mean_nll':t['mean_nll'],'rollout_n':s['rollout_n'],'rollout_solved':r['solved'],'rollout_optimal_solved':r['optimal_solved'],
                'rollout_canonical_exact':r['rollout_exact'],'rollout_valid_path':r['valid_path']})
    return rows

def curves():
    out={}
    for d in sorted((ROOT/'runs').iterdir()):
        if (d.name.startswith('full50k_') or d.name.startswith('ext50k_')) and (d/'epochs.jsonl').exists():
            cfg=json.loads((d/'config.json').read_text());out[(cfg['variant'],cfg['seed'])]=lines(d/'epochs.jsonl')
    return out

def style(ax):
    for s in ['top','right']:ax.spines[s].set_visible(False)
    ax.spines['left'].set_color('#9a9993');ax.spines['bottom'].set_color('#9a9993');ax.tick_params(colors='#52514e')
    ax.grid(axis='y',color='#e6e5e0',lw=.8);ax.set_axisbelow(True)

def plot_curves(cv):
    fig,ax=plt.subplots(figsize=(10.5,6));fig.patch.set_facecolor('#fcfcfb');ax.set_facecolor('#fcfcfb')
    hist=list(csv.DictReader((ROOT/'audit'/'original_histories.csv').open()))
    for model,ls,label in [('baseline',(0,(1,2)),'Historical notebook baseline (saved output)'),('looped',(0,(4,2)),'Historical notebook looped (saved output)')]:
        h=[r for r in hist if r['model']==model]
        ax.plot([int(r['epoch'])*UPDATES_PER_EPOCH for r in h],[100-float(r['val_token_accuracy_pct']) for r in h],color='#8a8983',lw=1.4,ls=ls,label=label)
    ends=[]
    for v,label,c,_ in MODELS:
        for seed,lw,alpha in [(42,2,1),(43,1.2,.55)]:
            if (v,seed) not in cv:continue
            e=cv[(v,seed)];x=[r['optimizer_steps'] for r in e];y=[100*(1-r['val_token_accuracy']) for r in e]
            ax.plot(x,y,color=c,lw=lw,alpha=alpha,label=label if seed==42 else None)
            if seed==42:ends.append((x[-1],y[-1],label,c))
    ax.set_yscale('log');ax.set_ylim(.4,40)
    ax.set_yticks([.5,1,2,5,10,20]);ax.set_yticklabels(['0.5%','1%','2%','5%','10%','20%'])
    ax.axvline(30*UPDATES_PER_EPOCH,color='#c3c2b7',lw=1,ls=':');ax.text(30*UPDATES_PER_EPOCH,33,' 30 epochs\n (11,730 updates)',color='#52514e',fontsize=8,va='top')
    ax.set_xlabel('Optimizer updates (391 per epoch, batch 128)',color='#52514e');ax.set_ylabel('Validation token error rate (log scale)',color='#52514e')
    ax.set_title('Validation token error by training updates, full-size data\nThick line: seed 42. Thin line: seed 43 where run. Dotted/dashed gray: your notebook\'s saved history',loc='left',fontsize=11,color='#0b0b0b')
    style(ax);ax.legend(frameon=False,fontsize=8.5,loc='upper right',ncol=1)
    fig.tight_layout();fig.savefig(OUT/'validation_error_vs_updates.png',dpi=160);plt.close(fig)

def plot_bars(rows):
    end=[r for r in rows if r['epoch']==30 and r['seed']==42]
    order=[(v,l,c) for v,l,c,_ in MODELS if any(r['variant']==v for r in end)]
    for metric,fname,xlabel,title in [('test_token_error','test_token_error_at_30_epochs.png','Test token error rate (%), teacher forced, 5,000 mazes','Token errors at equal updates (11,730), seed 42'),
                                      ('rollout_optimal_solved','test_optimal_failure_at_30_epochs.png','Mazes not solved optimally (%), model generates its own path, 1,000 mazes','Autonomous maze-solving failures at equal updates (11,730), seed 42')]:
        fig,ax=plt.subplots(figsize=(9.5,.55*len(order)+1.6));fig.patch.set_facecolor('#fcfcfb');ax.set_facecolor('#fcfcfb')
        vals=[]
        for v,l,c in order:
            r=next(r for r in end if r['variant']==v);val=100*r[metric] if metric=='test_token_error' else 100*(1-r[metric]);vals.append((l,c,val))
        for i,(l,c,val) in enumerate(vals):
            ax.barh(i,val,color=c,height=.62);ax.text(val,i,f'  {val:.2f}%' if metric=='test_token_error' else f'  {val:.1f}%',va='center',fontsize=9,color='#0b0b0b')
        ax.set_yticks(range(len(vals)));ax.set_yticklabels([l for l,_,_ in vals],color='#0b0b0b');ax.invert_yaxis()
        ax.set_xlabel(xlabel,color='#52514e');ax.set_title(title,loc='left',fontsize=11);style(ax);ax.grid(axis='x',color='#e6e5e0');ax.grid(axis='y',visible=False)
        ax.set_xlim(0,max(v for _,_,v in vals)*1.18);fig.tight_layout();fig.savefig(OUT/fname,dpi=160);plt.close(fig)

def main():
    rows=load()
    with (OUT/'combined_endpoints.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    plot_curves(curves());plot_bars(rows)
    json.dump({'created_utc':datetime.datetime.now(datetime.UTC).isoformat(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'runs':sorted({r['run_id'] for r in rows}),
               'caveats':['ext50k runs were specified after full50k test outcomes were seen (exploratory).','Seed 42 only for most variants; seed 43 for four.',
                          'Historical curves are saved notebook output, not re-measured.','Error rates on log axis; bars are fixed final checkpoints, no selection.']},
              (OUT/'manifest.json').open('w'),indent=2)
    for r in rows:print(f"{r['run_id']:<45} ep{r['epoch']:>3} tok {100*r['test_token_accuracy']:6.2f}  opt {100*r['rollout_optimal_solved']:5.1f}  nll {r['test_mean_nll']:.4f}  s {r['cumulative_train_seconds']}")

if __name__=='__main__':main()
