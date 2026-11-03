"""Exploratory full-size extension, specified AFTER the four full50k test results were seen.

Question: the matched untied6_full equalled original_looped at full scale, so sharing is not
the source of the notebook gap. Which non-sharing ingredient is? Each run is a minimal edit:

  original_baseline_preln   exact notebook baseline, single flag norm_first=True
  original_baseline_warmup  exact notebook baseline (Post-LN), linear LR warmup 2000 steps
  original_baseline_preln6  exact notebook baseline, norm_first=True and num_layers=6
  + second seeds (43) for tied6_full, untied6_full, original_looped, original_baseline_preln

Same data, optimizer, clipping, batch and evaluation as fullscale_experiments.py.
Fixed final epoch 30 (11,730 updates); no test-selected choices. Labelled exploratory.
"""
import argparse, datetime, hashlib, json, math, time, traceback
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import experiment_suite as suite
from experiment_suite import ROOT, NOTEBOOK, write_json, append_json, seed_all, make_model, teacher_eval, environment
from fullscale_experiments import full_data, save_checkpoint, evaluation

DECLARED=[('original_baseline_preln',42),('original_baseline_warmup',42),('original_baseline_preln6',42),
          ('tied6_full',43),('untied6_full',43),('original_looped',43),('original_baseline_preln',43)]
WARMUP_STEPS=2000

def notebook_baseline(norm_first=False,num_layers=4):
    """Exact notebook cell 2 class with at most one source edit (norm_first) and a layer-count argument."""
    n=json.loads(NOTEBOOK.read_text(encoding='utf-8'));src=''.join(n['cells'][2]['source'])
    if norm_first:
        assert src.count('batch_first=True')==1
        src=src.replace('batch_first=True','batch_first=True, norm_first=True')
    ns={'torch':torch,'nn':nn,'math':math,'GRID_SIZE':8,
        'MAZE_VOCAB':{'FREE':0,'WALL':1,'START':2,'GOAL':3},
        'ACTION_VOCAB':{'PAD':0,'BOS':1,'EOS':2,'U':3,'L':4,'R':5,'D':6}}
    exec(src,ns);m=ns['MazeTransformerSolver'](num_layers=num_layers)
    # Same adapters as experiment_suite.make_model, used only by rollout.
    m.encode=m.encode_maze
    def decode(seq,memory,depth=None):
        x=m.action_token_emb(seq)+m.pos_emb(torch.arange(seq.shape[1],device=seq.device))[None]
        mask=torch.ones(seq.shape[1],seq.shape[1],device=seq.device,dtype=torch.bool).triu(1)
        return m.fc_out(m.transformer.decoder(x,memory,tgt_mask=mask,tgt_key_padding_mask=seq.eq(0)))
    m.decode=decode
    return m,src

def build(variant):
    if variant=='original_baseline_preln':return notebook_baseline(True,4)
    if variant=='original_baseline_warmup':return notebook_baseline(False,4)
    if variant=='original_baseline_preln6':return notebook_baseline(True,6)
    return make_model(variant),None

def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',default=','.join(f'{v}@{s}' for v,s in DECLARED))
    p.add_argument('--epochs',type=int,default=30);p.add_argument('--rollout-n',type=int,default=1000)
    p.add_argument('--prefix',default='ext50k');p.add_argument('--snapshots',action='store_true',help='save model weights after every epoch (snapshots/epNN.pt) for training-dynamics probes')
    args=p.parse_args();torch.set_num_threads(4);device=torch.device('cuda')
    script_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    write_json(ROOT/f'{args.prefix}_predeclaration.json' if args.prefix!='ext50k' else ROOT/'extension_predeclaration.json',{'prefix':args.prefix,'snapshots':args.snapshots,'declared_utc':datetime.datetime.now(datetime.UTC).isoformat(),'runs':DECLARED,
      'status':'EXPLORATORY: specified after full50k test outcomes were visible (untied6_full matched original_looped).',
      'epochs':args.epochs,'warmup_steps':WARMUP_STEPS,'checkpoint_rule':'Fixed final epoch; no test-selected stopping.',
      'shuffle_seed':'1000+architecture seed (1042 matches full50k)','script_sha256':script_hash})
    data,meta=full_data()
    for item in args.runs.split(','):
        variant,seed=item.split('@');seed=int(seed)
        run_id=f'{args.prefix}_{variant}_seed{seed}';out=ROOT/'runs'/run_id
        if out.exists():raise RuntimeError(f'Refusing to overwrite {out}')
        out.mkdir(parents=True);seed_all(seed)
        warmup=WARMUP_STEPS if variant.endswith('_warmup') else 0
        cfg={'mode':'train','variant':variant,'seed':seed,'epochs':args.epochs,'batch_size':128,'train_size':50000,'val_size':5000,'test_size':5000,'small':False,
             'optimizer':'AdamW','lr':.001,'weight_decay':.0001,'clip_grad_norm':1.,'warmup_steps':warmup,'dtype':'float32','amp':False,
             'dataset_files':{k:str(ROOT/'data'/'original50k_seed42'/f'{k}.npz') for k in data},'dataset_sha256':{k:v['sha256'] for k,v in meta.items()},
             'scope':'Exploratory extension at full data scale.','architecture':suite.VARIANTS.get(variant,'notebook cell 2 definition with edit named by variant'),
             'shuffle':f'Dedicated torch.Generator seed{1000+seed}.'}
        start=time.perf_counter()
        try:
            m,src=build(variant);m=m.to(device)
            if src is not None:cfg['edited_source_sha256']=hashlib.sha256(src.encode()).hexdigest();(out/'model_source.py').write_text(src,encoding='utf-8')
            write_json(out/'config.json',cfg);env=environment();env['extension_script_sha256']=script_hash;write_json(out/'environment.json',env)
            append_json(ROOT/'run_registry.jsonl',{'run_id':run_id,'status':'started','config':str(out/'config.json'),'utc':datetime.datetime.now(datetime.UTC).isoformat()})
            opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001)
            sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda s:min(1.,(s+1)/warmup) if warmup else 1.)
            write_json(out/'model.json',{'parameters':sum(q.numel() for q in m.parameters()),'structure':str(m)})
            gen=torch.Generator().manual_seed(1000+seed)
            tr=DataLoader(data['train'],batch_size=128,shuffle=True,generator=gen)
            vl=DataLoader(data['val'],batch_size=128);tl=DataLoader(data['test'],batch_size=128)
            train_total=0.;eval_total=0.;steps=0
            for epoch in range(1,args.epochs+1):
                m.train();nll=0.;tokens=0;batch_mean=0.;norms=[];t=time.perf_counter()
                for g,s in tr:
                    st=time.perf_counter();g,s=g.to(device),s.to(device);labels=s[:,1:];opt.zero_grad(set_to_none=True)
                    logits=m(g,s[:,:-1]);loss=nn.functional.cross_entropy(logits.flatten(0,1),labels.flatten(),ignore_index=0)
                    loss.backward();gn=nn.utils.clip_grad_norm_(m.parameters(),1);lr=sched.get_last_lr()[0];opt.step();sched.step()
                    nt=int(labels.ne(0).sum());nll+=loss.item()*nt;tokens+=nt;batch_mean+=loss.item();norms.append(float(gn));steps+=1
                    append_json(out/'steps.jsonl',{'run_id':run_id,'epoch':epoch,'optimizer_step':steps,'pre_update_teacher_forced_token_mean_nll':loss.item(),
                                'target_tokens':nt,'lr':lr,'gradient_norm_before_clip':float(gn),'step_wall_seconds':time.perf_counter()-st})
                torch.cuda.synchronize();train_sec=time.perf_counter()-t;train_total+=train_sec;t=time.perf_counter()
                metrics,_=teacher_eval(m,vl,device);val_sec=time.perf_counter()-t;eval_total+=val_sec
                row={'run_id':run_id,'epoch':epoch,'optimizer_steps':steps,'train_token_mean_loss':nll/tokens,'train_batch_mean_loss':batch_mean/len(tr),
                     'train_tokens':tokens,'train_seconds':train_sec,'cumulative_train_seconds':train_total,'val_seconds':val_sec,
                     'gradient_norm_mean_before_clip':float(np.mean(norms)),'gradient_clip_fraction':float(np.mean(np.array(norms)>1)),
                     'parameters':sum(q.numel() for q in m.parameters()),**{'val_'+k:v for k,v in metrics.items()}}
                if hasattr(m,'encoder_gate'):row.update(encoder_gate=float(m.encoder_gate.detach()),decoder_gate=float(m.decoder_gate.detach()))
                append_json(out/'epochs.jsonl',row)
                print(json.dumps({k:row[k] for k in ['run_id','epoch','train_token_mean_loss','val_token_accuracy','val_exact_accuracy','train_seconds']}),flush=True)
                if epoch%10==0:save_checkpoint(out/'checkpoint_latest.pt',m,opt,cfg,epoch,gen)
                if args.snapshots:(out/'snapshots').mkdir(exist_ok=True);torch.save(m.state_dict(),out/'snapshots'/f'ep{epoch:02d}.pt')
            save_checkpoint(out/'checkpoint.pt',m,opt,cfg,args.epochs,gen)
            evaluation(out,m,tl,data['test'],device,args.rollout_n,train_total,eval_total,time.perf_counter()-start,run_id)
            append_json(ROOT/'run_registry.jsonl',{'run_id':run_id,'status':'complete','summary':str(out/'summary.json'),'total_seconds':time.perf_counter()-start})
            del m,opt
        except Exception as e:
            (out/'error.txt').write_text(traceback.format_exc(),encoding='utf-8');append_json(ROOT/'run_registry.jsonl',{'run_id':run_id,'status':'failed','error':str(e)});raise
    write_json(ROOT/f'{args.prefix}_complete.json',{'completed_utc':datetime.datetime.now(datetime.UTC).isoformat(),'runs':args.runs})

if __name__=='__main__':main()
