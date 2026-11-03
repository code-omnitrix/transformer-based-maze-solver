"""Exploratory fixed-checkpoint evaluation on two prespecified harder 8x8 sets.

Run only after training jobs finish to keep GPU timing isolated. No fitting,
checkpoint selection or hyperparameter tuning. Distribution shift is explicit.
"""
import argparse,collections,datetime,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import TensorDataset,DataLoader
from experiment_suite import ROOT,make_model,generate_one,teacher_eval,rollout,write_json,append_json,environment

SETS=[{'name':'walls40','wall_probability':.40,'min_path_actions':4,'seed':20261106,'n':512},
      {'name':'longpaths12','wall_probability':.28,'min_path_actions':12,'seed':20261107,'n':512}]

def data(cfg):
    root=ROOT/'data'/'exploratory_harder';root.mkdir(exist_ok=True);p=root/(cfg['name']+'.npz')
    if not p.exists():
        rng=np.random.default_rng(cfg['seed']);samples=[]
        while len(samples)<cfg['n']:
            g,s=generate_one(rng,p=cfg['wall_probability'])
            if len(s)-2>=cfg['min_path_actions']:samples.append((g,s))
        grids=np.stack([x[0] for x in samples]);seqs=np.zeros((len(samples),max(len(x[1]) for x in samples)),dtype=np.int64)
        for i,(_,s) in enumerate(samples):seqs[i,:len(s)]=s
        np.savez_compressed(p,grids=grids,seqs=seqs)
    z=np.load(p);grids,seqs=z['grids'],z['seqs']
    meta={**cfg,'file':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),
          'action_length_histogram':dict(collections.Counter(((seqs!=0).sum(1)-2).tolist())),
          'grid_hashes':[hashlib.sha256(g.tobytes()).hexdigest() for g in grids],
          'interpretation':'Harder 8x8 distribution; length-conditioned set is not guaranteed outside training length support.'}
    write_json(root/(cfg['name']+'_manifest.json'),meta)
    return TensorDataset(torch.from_numpy(grids),torch.from_numpy(seqs)),meta

def main():
    p=argparse.ArgumentParser();p.add_argument('--prefix',default='controlled12k');p.add_argument('--variants',default='tied6_full,untied6_full');a=p.parse_args()
    torch.set_num_threads(4);device=torch.device('cuda');datasets=[(cfg,*data(cfg)) for cfg in SETS]
    for d in sorted((ROOT/'runs').glob(a.prefix+'_*')):
        if not(d/'checkpoint.pt').exists():continue
        cfg=json.loads((d/'config.json').read_text())
        if cfg['variant'] not in a.variants.split(','):continue
        checkpoint=d/'checkpoint.pt';snap=torch.load(checkpoint,map_location='cpu',weights_only=False)
        m=make_model(cfg['variant'],cfg.get('small',False)).to(device);m.load_state_dict(snap['state_dict']);m.eval()
        for spec,ds,meta in datasets:
            out=ROOT/'analysis'/'exploratory_harder'/d.name/spec['name']
            if out.exists():raise RuntimeError(f'Refusing to overwrite {out}')
            out.mkdir(parents=True)
            t=time.perf_counter();metrics,ts=teacher_eval(m,DataLoader(ds,batch_size=128),device,per_example=True);rm,rs=rollout(m,ds,device,len(ds))
            for i,r in enumerate(ts):append_json(out/'teacher_predictions.jsonl',{'example_id':i,**r})
            for r in rs:append_json(out/'rollout_predictions.jsonl',{**r,'actual_eos_emitted':2 in r['prediction']})
            write_json(out/'summary.json',{'scope':'Exploratory harder 8x8 sets; no tuning or training','run_id':d.name,'data':meta,
                'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'checkpoint_epoch':snap['epoch'],
                'teacher_test':metrics,'rollout_test':rm,'first_action_accuracy':np.mean([r['teacher_pred'][0]==r['target'][0] for r in ts]),
                'evaluation_seconds':time.perf_counter()-t,'environment':environment(),
                'evaluator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
            print(json.dumps({'run':d.name,'dataset':spec['name'],'teacher_token':metrics['token_accuracy'],'rollout_optimal':rm['optimal_solved']}),flush=True)
        del m

if __name__=='__main__':main()
