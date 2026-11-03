"""Figures and phase-alignment summary from analysis/residual/metrics.jsonl (written by residual_analysis.py).

Outputs in analysis/residual/:
  phase_dynamics.png        per-epoch validation error next to residual-stream measurements (dyn50k runs)
  per_iteration.png         probe/PCA/attention measurements by encoder layer or loop iteration (final checkpoints)
  wavefront_heatmap.png     distance readout (within +/-1) by iteration and true distance-to-goal
  pca_snapshots.png         2-D PCA of final-encoder cell residuals over training, coloured by distance
  decoder_layers.png        logit-lens accuracy and agent-cell probe by decoder layer
  phase_alignment.json      epoch of steepest validation-error drop vs steepest change of each residual metric
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'analysis'/'residual'
INK='#0b0b0b';INK2='#52514e';GRID='#e6e5e0';SURF='#fcfcfb'
DYN=[('dyn50k_original_baseline_seed42','Post-LN baseline (notebook)','#2a78d6'),
     ('dyn50k_preln_original_baseline_preln_seed42','Pre-LN baseline (one flag)','#eb6834'),
     ('dyn50k_original_looped_seed42','Looped model (notebook)','#008300')]
FINAL=[('full50k_original_baseline_seed42',90,'Post-LN baseline, 90 ep','#2a78d6','-'),
       ('full50k_original_baseline_seed42',30,'Post-LN baseline, 30 ep','#2a78d6',':'),
       ('ext50k_original_baseline_preln_seed42',30,'Pre-LN baseline','#eb6834','-'),
       ('ext50k_original_baseline_preln6_seed42',30,'Pre-LN baseline, 6L','#1baf7a','-'),
       ('full50k_untied6_full_seed42',30,'Untied 6L + reinjection','#e87ba4','-'),
       ('full50k_tied6_full_seed42',30,'Looped, controlled init','#6250d6','-'),
       ('full50k_original_looped_seed42',30,'Looped (notebook)','#008300','-'),
       ('random_init_original_looped',0,'Looped, random weights','#9a9993','--')]

def style(ax,title=None):
    ax.set_facecolor(SURF)
    for s in['top','right']:ax.spines[s].set_visible(False)
    for s in['left','bottom']:ax.spines[s].set_color('#9a9993')
    ax.tick_params(colors=INK2,labelsize=8);ax.grid(color=GRID,lw=.7);ax.set_axisbelow(True)
    if title:ax.set_title(title,loc='left',fontsize=9.5,color=INK)

def load():
    rows=[json.loads(l) for l in (OUT/'metrics.jsonl').read_text().splitlines() if l.strip()]
    for r in rows:r['key']=(r['run'],r['epoch'])
    return rows

def epochs_file(run):
    p=ROOT/'runs'/run/'epochs.jsonl'
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []

def enc_final(rows,run,epoch=None,trained=True):
    """Encoder row at the trained depth (last non-'beyond' layer) for each epoch of a run."""
    out={}
    for r in rows:
        if r['run']==run and r['part']=='encoder' and not r['beyond_trained'] and (epoch is None or r['epoch']==epoch):
            if r['epoch'] not in out or r['layer']>out[r['epoch']]['layer']:out[r['epoch']]=r
    return out

def mean_over_layers(rows,run,metric):
    acc={}
    for r in rows:
        if r['run']==run and r['part']=='encoder' and not r['beyond_trained'] and metric in r:acc.setdefault(r['epoch'],[]).append(r[metric])
    return {e:float(np.mean(v)) for e,v in acc.items()}

def part(rows,run,p):
    return {r['epoch']:r for r in rows if r['run']==run and r['part']==p}

def plot_phase(rows):
    panels=[('Validation token error (log)',None),('Final encoder: shortest-move probe accuracy','probe_policy_acc'),
            ('Final encoder: distance-to-goal probe R²','probe_dist_r2'),('Final encoder: PCA participation ratio','participation_ratio'),
            ('Attention mass on 4 grid neighbours (mean over layers)','attn_neighbour_mass'),('Predicted moves equal to greedy Manhattan move','frac_greedy'),
            ('Predicted moves that are shortest-path moves','frac_optimal'),('Decoder: agent-cell probe accuracy (last layer)','probe_agent_cell_acc')]
    fig,axs=plt.subplots(2,4,figsize=(16,7.2),sharex=True);fig.patch.set_facecolor(SURF)
    for ax,(title,metric) in zip(axs.flat,panels):
        style(ax,title)
        for run,label,c in DYN:
            if metric is None:
                e=epochs_file(run);
                if not e:continue
                ax.plot([x['epoch'] for x in e],[100*(1-x['val_token_accuracy']) for x in e],color=c,lw=2,label=label);ax.set_yscale('log')
                continue
            if metric=='attn_neighbour_mass':d=mean_over_layers(rows,run,metric)
            elif metric.startswith('frac_'):d={e:r[metric] for e,r in part(rows,run,'behaviour').items()}
            elif metric=='probe_agent_cell_acc':
                d={}
                for r in rows:
                    if r['run']==run and r['part']=='decoder' and (r['epoch'] not in d or r['layer']>d[r['epoch']][0]):d[r['epoch']]=(r['layer'],r[metric])
                d={e:v for e,(_,v) in d.items()}
            else:d={e:r[metric] for e,r in enc_final(rows,run).items()}
            if d:ks=sorted(d);ax.plot(ks,[d[k] for k in ks],color=c,lw=2,marker='o',ms=3,label=label)
    for ax in axs[1]:ax.set_xlabel('Epoch (391 updates each)',color=INK2,fontsize=8.5)
    axs[0,0].legend(frameon=False,fontsize=8)
    fig.suptitle('Training dynamics: when validation error drops, what changes in the residual stream? (seed 42, full-size data, 1,000 held-out probe mazes)',x=.01,ha='left',fontsize=11,color=INK)
    fig.tight_layout();fig.savefig(OUT/'phase_dynamics.png',dpi=150);plt.close(fig)

def plot_iterations(rows):
    panels=[('Distance readout within ±1 move','probe_dist_within1'),('Shortest-move probe accuracy','probe_policy_acc'),
            ('Relative update ‖h_k − h_{k−1}‖ / ‖h_{k−1}‖','relative_update'),('Attention mass on 4 grid neighbours','attn_neighbour_mass'),
            ('PCA participation ratio','participation_ratio'),('Canonical-path membership (balanced acc.)','probe_onpath_balanced_acc')]
    fig,axs=plt.subplots(2,3,figsize=(15,7.5));fig.patch.set_facecolor(SURF)
    for ax,(title,metric) in zip(axs.flat,panels):
        style(ax,title)
        for run,ep,label,c,ls in FINAL:
            rs=sorted([r for r in rows if r['run']==run and r['epoch']==ep and r['part']=='encoder' and metric in r],key=lambda r:r['layer'])
            if not rs:continue
            x=[r['layer'] for r in rs];y=[r[metric] for r in rs]
            ax.plot(x,y,color=c,lw=1.8,ls=ls,label=label,marker='o',ms=2.5)
        ax.axvline(6,color='#c3c2b7',lw=1,ls=':');ax.set_xlabel('Encoder layer / loop iteration (0 = embedding; tied models unrolled past trained K=6)',color=INK2,fontsize=7.5)
    axs[0,2].set_yscale('log');axs[0,0].legend(frameon=False,fontsize=7.5)
    fig.suptitle('Encoder residual stream by layer or loop iteration, final checkpoints (linear probes: fit 4,000 mazes, scored on 1,000)',x=.01,ha='left',fontsize=11,color=INK)
    fig.tight_layout();fig.savefig(OUT/'per_iteration.png',dpi=150);plt.close(fig)

def plot_wavefront(rows):
    picks=[('random_init_original_looped',0,'Looped, random weights'),('full50k_original_baseline_seed42',30,'Post-LN baseline, 30 ep'),
           ('full50k_original_baseline_seed42',90,'Post-LN baseline, 90 ep'),('ext50k_original_baseline_preln_seed42',30,'Pre-LN baseline, 30 ep'),
           ('full50k_untied6_full_seed42',30,'Untied 6L + reinjection'),('full50k_original_looped_seed42',30,'Looped (notebook)')]
    fig,axs=plt.subplots(1,len(picks),figsize=(18,3.9),sharey=False);fig.patch.set_facecolor(SURF);im=None
    for ax,(run,ep,label) in zip(axs,picks):
        rs=sorted([r for r in rows if r['run']==run and r['epoch']==ep and r['part']=='encoder'],key=lambda r:r['layer'])
        if not rs:ax.axis('off');continue
        D=list(range(0,15));M=np.array([[r['probe_dist_within1_by_true_dist'].get(str(d),r['probe_dist_within1_by_true_dist'].get(d,np.nan)) for d in D] for r in rs])
        im=ax.imshow(M,aspect='auto',origin='lower',cmap='Blues',vmin=0,vmax=1);ax.set_title(label,fontsize=9,loc='left')
        ax.set_xlabel('True distance to goal',fontsize=8,color=INK2);ax.set_ylabel('Layer / iteration',fontsize=8,color=INK2);ax.tick_params(labelsize=7)
    if im is not None:fig.colorbar(im,ax=axs,fraction=.015,label='Linear readout within ±1 move')
    fig.suptitle('Does distance-to-goal become readable one hop further per iteration? (wavefront test)',x=.01,ha='left',fontsize=11)
    fig.savefig(OUT/'wavefront_heatmap.png',dpi=150,bbox_inches='tight');plt.close(fig)

def plot_pca():
    p=OUT/'pca_samples.npz'
    if not p.exists():return
    z=np.load(p);runs=[('dyn50k_original_baseline_seed42','Post-LN baseline',4),('dyn50k_original_looped_seed42','Looped (notebook)',6),
                        ('dyn50k_preln_original_baseline_preln_seed42','Pre-LN baseline',4)]
    eps=[1,3,5,8,12,18,30]
    fig,axs=plt.subplots(len(runs),len(eps),figsize=(2.3*len(eps),2.4*len(runs)));fig.patch.set_facecolor(SURF)
    for i,(run,label,L) in enumerate(runs):
        for j,ep in enumerate(eps):
            ax=axs[i,j];key=f'{run}@ep{ep}|enc{L}'
            ax.set_xticks([]);ax.set_yticks([])
            if key not in z:ax.axis('off');continue
            P=z[key];d=z[key+'|dist'];ok=d>=0
            ax.scatter(P[~ok,0],P[~ok,1],s=2,c='#e34948',alpha=.5,lw=0)
            ax.scatter(P[ok,0],P[ok,1],s=2,c=np.minimum(d[ok],14),cmap='Blues_r',vmin=0,vmax=16,lw=0)
            if i==0:ax.set_title(f'epoch {ep}',fontsize=9)
            if j==0:ax.set_ylabel(label,fontsize=9)
    fig.suptitle('PCA (PC1 vs PC2) of final-encoder cell residuals over training. Dark = near goal, light = far, red = unreachable',x=.01,ha='left',fontsize=10.5)
    fig.tight_layout();fig.savefig(OUT/'pca_snapshots.png',dpi=150);plt.close(fig)

def plot_decoder(rows):
    fig,axs=plt.subplots(1,2,figsize=(11,4));fig.patch.set_facecolor(SURF)
    for ax,(metric,title) in zip(axs,[('logit_lens_token_acc','Logit lens: token accuracy if read out at this layer'),('probe_agent_cell_acc','Agent current-cell probe accuracy (64-way)')]):
        style(ax,title)
        for run,ep,label,c,ls in FINAL:
            rs=sorted([r for r in rows if r['run']==run and r['epoch']==ep and r['part']=='decoder'],key=lambda r:r['layer'])
            if rs:ax.plot([r['layer'] for r in rs],[r[metric] for r in rs],color=c,ls=ls,lw=1.8,marker='o',ms=3,label=label)
        ax.set_xlabel('Decoder layer / iteration',color=INK2,fontsize=8.5)
    axs[0].legend(frameon=False,fontsize=7.5);fig.tight_layout();fig.savefig(OUT/'decoder_layers.png',dpi=150);plt.close(fig)

def alignment(rows):
    """Epoch of steepest validation-error drop (log scale) vs steepest rise of each residual metric."""
    out={}
    for run,label,_ in DYN:
        e=epochs_file(run)
        if not e:continue
        err=np.log([1-x['val_token_accuracy'] for x in e]);ep=np.array([x['epoch'] for x in e])
        res={'label':label,'steepest_val_error_drop_epoch':int(ep[1:][np.argmin(np.diff(err))])}
        for metric in['probe_policy_acc','probe_dist_r2','participation_ratio']:
            d=enc_final(rows,run);ks=sorted(d)
            if len(ks)>2:
                v=np.array([d[k][metric] for k in ks]);dv=np.diff(v);i=int(np.argmax(np.abs(dv)))
                res[f'{metric}_steepest_change_epoch']=int(ks[i+1]);res[f'{metric}_steepest_change']=float(dv[i])
                res[f'{metric}_corr_with_log_val_error']=float(np.corrcoef(v,[err[k-1] for k in ks])[0,1])
        nb=mean_over_layers(rows,run,'attn_neighbour_mass');ks=sorted(nb)
        if len(ks)>2:res['attn_neighbour_mass_corr_with_log_val_error']=float(np.corrcoef([nb[k] for k in ks],[err[k-1] for k in ks])[0,1])
        out[run]=res
    (OUT/'phase_alignment.json').write_text(json.dumps(out,indent=2));return out

def main():
    rows=load();plot_phase(rows);plot_iterations(rows);plot_wavefront(rows);plot_pca();plot_decoder(rows)
    print(json.dumps(alignment(rows),indent=1))

if __name__=='__main__':main()
