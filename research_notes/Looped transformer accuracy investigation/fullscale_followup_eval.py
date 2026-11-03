"""Exploratory fixed-checkpoint evaluation of all full-size final checkpoints (full50k_*, ext50k_*).

1. Harder 8x8 sets (same two prespecified sets as evaluate_harder_mazes.py), teacher + rollout.
2. Inference-depth sweep for weight-tied models: teacher-forced token accuracy on 5,000 test mazes
   and autonomous rollout on the first 1,000, at K loops different from the trained K=6.
No training, tuning or checkpoint selection. Run after GPU training finishes.
Writes analysis/combined50k/{harder_sets.csv,depth_sweep.csv}.
"""
import csv, json, time
import torch
from torch.utils.data import DataLoader
from experiment_suite import ROOT, make_model, teacher_eval, rollout
from evaluate_harder_mazes import SETS, data as harder_data
from fullscale_experiments import full_data
from extension_experiments import build

OUT=ROOT/'analysis'/'combined50k';OUT.mkdir(parents=True,exist_ok=True)
DEPTHS=[1,2,3,4,5,6,7,8,10,12,16]
TIED={'original_looped','tied6_full'}

def load(d):
    cfg=json.loads((d/'config.json').read_text());snap=torch.load(d/'checkpoint.pt',map_location='cpu',weights_only=False)
    m=build(cfg['variant'])[0] if d.name.startswith('ext50k_') else make_model(cfg['variant'])
    m.load_state_dict(snap['state_dict']);return cfg,m.cuda().eval()

def main():
    torch.set_num_threads(4);data,_=full_data();test=data['test'];tl=DataLoader(test,batch_size=128)
    harder=[(cfg,*harder_data(cfg)) for cfg in SETS]
    hrows=[];drows=[]
    for d in sorted((ROOT/'runs').iterdir()):
        if not (d.name.startswith('full50k_') or d.name.startswith('ext50k_')) or not (d/'checkpoint.pt').exists():continue
        cfg,m=load(d);t=time.perf_counter()
        for spec,ds,_ in harder:
            met,_=teacher_eval(m,DataLoader(ds,batch_size=128),torch.device('cuda'));roll,_=rollout(m,ds,torch.device('cuda'),len(ds))
            hrows.append({'run_id':d.name,'variant':cfg['variant'],'seed':cfg['seed'],'set':spec['name'],'n':len(ds),
                          'token_accuracy':met['token_accuracy'],'teacher_exact':met['exact_accuracy'],'solved':roll['solved'],'optimal_solved':roll['optimal_solved']})
        if cfg['variant'] in TIED:
            for k in DEPTHS:
                met,_=teacher_eval(m,tl,torch.device('cuda'),depth=k);roll,_=rollout(m,test,torch.device('cuda'),1000,depth=k)
                drows.append({'run_id':d.name,'variant':cfg['variant'],'seed':cfg['seed'],'trained_depth':6,'inference_depth':k,
                              'token_accuracy':met['token_accuracy'],'mean_nll':met['mean_nll'],'optimal_solved':roll['optimal_solved'],'solved':roll['solved']})
        print(d.name,f'{time.perf_counter()-t:.0f}s',flush=True);del m
    for name,rows in [('harder_sets.csv',hrows),('depth_sweep.csv',drows)]:
        with (OUT/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

if __name__=='__main__':main()
