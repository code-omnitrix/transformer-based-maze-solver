"""Predeclared original-size follow-up; no choices based on held-out test results.

Original50k train/5k val generator and exact saved architecture definitions.
Separate matched controlled sharing comparison, plus baseline30/90 checkpoints.
"""
import argparse, collections, datetime, hashlib, json, math, random, sys, time, traceback
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
import experiment_suite as suite
from experiment_suite import ROOT, NOTEBOOK, write_json, append_json, seed_all, make_model, teacher_eval, rollout, environment

DECLARED=[('original_baseline',90),('original_looped',30),('tied6_full',30),('untied6_full',30)]

def full_data():
    out=ROOT/'data'/'original50k_seed42';out.mkdir(exist_ok=True)
    paths={k:out/f'{k}.npz' for k in ['train','val','test']}
    if not all(p.exists() for p in paths.values()):
        n=json.loads(NOTEBOOK.read_text(encoding='utf-8'));seed_all(42)
        ns={'np':np,'random':random,'torch':torch,'Dataset':Dataset,'deque':collections.deque,
            'GRID_SIZE':8,'WALL_PROB':.28,'MAZE_VOCAB':{'FREE':0,'WALL':1,'START':2,'GOAL':3},
            'ACTION_VOCAB':{'PAD':0,'BOS':1,'EOS':2,'U':3,'L':4,'R':5,'D':6},
            'MOVE_DELTAS':{'U':(-1,0),'L':(0,-1),'R':(0,1),'D':(1,0)}}
        exec(''.join(n['cells'][1]['source']).split('# Create Training & Validation Datasets')[0],ns)
        for split,n in [('train',50000),('val',5000),('test',5000)]:
            d=ns['MazeDataset'](num_samples=n)
            grids=np.stack([s[0] for s in d.samples]);seqs=np.zeros((n,d.max_len),dtype=np.int64)
            for i,(_,seq) in enumerate(d.samples):seqs[i,:len(seq)]=seq
            np.savez_compressed(paths[split],grids=grids,seqs=seqs)
    ds={};meta={}
    for name,p in paths.items():
        z=np.load(p);grids,seqs=z['grids'],z['seqs'];ds[name]=TensorDataset(torch.from_numpy(grids),torch.from_numpy(seqs))
        meta[name]={'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'n':len(grids),'max_seq_len':seqs.shape[1],
                    'mean_path_actions':float(np.mean((seqs!=0).sum(1)-2)),
                    'action_length_histogram':dict(collections.Counter(((seqs!=0).sum(1)-2).tolist())),
                    'grid_hashes':[hashlib.sha256(g.tobytes()).hexdigest() for g in grids]}
    hashes={k:set(v['grid_hashes']) for k,v in meta.items()}
    write_json(out/'manifest.json',{'generator':'Exact notebook cell1, np.random.seed42 and random.seed42, train50k thenval5k thenfresh test5k',
              'splits':meta,'overlap':{a+'_'+b:len(hashes[a]&hashes[b]) for a,b in [('train','val'),('train','test'),('val','test')]}})
    return ds,meta

def save_checkpoint(path,m,opt,cfg,epoch,loader_gen):
    torch.save({'state_dict':m.state_dict(),'optimizer_state_dict':opt.state_dict(),'config':cfg,'seed':42,'epoch':epoch,
                'rng_python':random.getstate(),'rng_numpy':np.random.get_state(),'rng_torch':torch.get_rng_state(),
                'rng_cuda':torch.cuda.get_rng_state_all(),'loader_rng':loader_gen.get_state()},path)

def preserve_rng():
    return (random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all())

def restore_rng(s):
    random.setstate(s[0]);np.random.set_state(s[1]);torch.set_rng_state(s[2]);torch.cuda.set_rng_state_all(s[3])

def evaluation(out,m,tl,test,device,n,train_sec,val_sec,total_sec,run_id):
    rng=preserve_rng()
    try:
        met,ts=teacher_eval(m,tl,device,per_example=True);roll,rs=rollout(m,test,device,n)
        for i,r in enumerate(ts):append_json(out/'test_teacher_predictions.jsonl',{'example_id':i,**r})
        for r in rs:append_json(out/'test_rollout_predictions.jsonl',r)
        summary={'run_id':run_id,'parameters':sum(p.numel() for p in m.parameters()),'teacher_test':met,'rollout_test':roll,
                 'rollout_n':len(rs),'train_seconds':train_sec,'val_seconds':val_sec,'total_seconds':total_sec}
        write_json(out/'summary.json',summary)
        print('EVALUATION '+json.dumps(summary),flush=True)
    finally:restore_rng(rng)

def main():
    p=argparse.ArgumentParser();p.add_argument('--variants',default=','.join(v for v,_ in DECLARED));p.add_argument('--rollout-n',type=int,default=1000)
    args=p.parse_args();torch.set_num_threads(4);device=torch.device('cuda')
    write_json(ROOT/'fullscale_predeclaration.json',{'declared_utc':datetime.datetime.now(datetime.UTC).isoformat(),'all_runs':DECLARED,
      'baseline_extra_checkpoint_epoch':30,'architecture_seed':42,'data':'Exact original generator train50k val5k thenfresh test5k',
      'primary_sharing_contrast':'controlled tied6_full minus untied6_full at30epochs',
      'not_historical_bitwise_replication':'Fresh each-model seed42; original notebook execution order/state is unknown. Same problem, generator, original model code/optimizer/settings for original_*; new data/cache/evaluation/logging added.',
      'checkpoint_rule':'Fixed final epoch; baseline alsofixed30. No test-selected architecture or stopping.',
      'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    data,meta=full_data()
    for variant,epochs in DECLARED:
        if variant not in args.variants.split(','):continue
        run_id=f'full50k_{variant}_seed42';out=ROOT/'runs'/run_id
        if out.exists():raise RuntimeError(f'Refusing to overwrite {out}')
        out.mkdir(parents=True);seed_all(42)
        cfg={'mode':'train','variant':variant,'seed':42,'epochs':epochs,'batch_size':128,'train_size':50000,'val_size':5000,'test_size':5000,'small':False,
             'optimizer':'AdamW','lr':.001,'weight_decay':.0001,'clip_grad_norm':1.,'dtype':'float32','amp':False,
             'dataset_files':{k:str(ROOT/'data'/'original50k_seed42'/f'{k}.npz') for k in data},'dataset_sha256':{k:v['sha256'] for k,v in meta.items()},
             'scope':'Full original data scale. Original_* exact notebook architecture, controlled_* matched family, fresh per-model seed42 initialization.',
             'architecture':suite.VARIANTS.get(variant,'exact saved notebook definition'),
             'shuffle':'Dedicated torch.Generator seed1042, identical orders per model; differs original global-generator historical trajectory.'}
        write_json(out/'config.json',cfg);env=environment();env['fullscale_script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();write_json(out/'environment.json',env)
        append_json(ROOT/'run_registry.jsonl',{'run_id':run_id,'status':'started','config':str(out/'config.json'),'utc':datetime.datetime.now(datetime.UTC).isoformat()})
        start=time.perf_counter()
        try:
            m=make_model(variant).to(device);opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001)
            write_json(out/'model.json',{'parameters':sum(p.numel() for p in m.parameters()),'structure':str(m)})
            gen=torch.Generator().manual_seed(1042)
            tr=DataLoader(data['train'],batch_size=128,shuffle=True,generator=gen)
            vl=DataLoader(data['val'],batch_size=128);tl=DataLoader(data['test'],batch_size=128)
            train_total=0.;eval_total=0.;steps=0
            for epoch in range(1,epochs+1):
                m.train();nll=0.;tokens=0;batch_mean=0.;norms=[];t=time.perf_counter()
                for g,s in tr:
                    st=time.perf_counter();g,s=g.to(device),s.to(device);labels=s[:,1:];opt.zero_grad(set_to_none=True)
                    logits=m(g,s[:,:-1]);loss=nn.functional.cross_entropy(logits.flatten(0,1),labels.flatten(),ignore_index=0)
                    loss.backward();gn=nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
                    nt=int(labels.ne(0).sum());nll+=loss.item()*nt;tokens+=nt;batch_mean+=loss.item();norms.append(float(gn));steps+=1
                    append_json(out/'steps.jsonl',{'run_id':run_id,'epoch':epoch,'optimizer_step':steps,'pre_update_teacher_forced_token_mean_nll':loss.item(),
                                'target_tokens':nt,'lr':.001,'gradient_norm_before_clip':float(gn),'step_wall_seconds':time.perf_counter()-st})
                torch.cuda.synchronize();train_sec=time.perf_counter()-t;train_total+=train_sec;t=time.perf_counter()
                metrics,_=teacher_eval(m,vl,device);val_sec=time.perf_counter()-t;eval_total+=val_sec
                row={'run_id':run_id,'epoch':epoch,'optimizer_steps':steps,'train_token_mean_loss':nll/tokens,'train_batch_mean_loss':batch_mean/len(tr),
                     'train_tokens':tokens,'train_seconds':train_sec,'cumulative_train_seconds':train_total,'val_seconds':val_sec,
                     'gradient_norm_mean_before_clip':float(np.mean(norms)),'gradient_clip_fraction':float(np.mean(np.array(norms)>1)),
                     'parameters':sum(p.numel() for p in m.parameters()),**{'val_'+k:v for k,v in metrics.items()}}
                if hasattr(m,'encoder_gate'):row.update(encoder_gate=float(m.encoder_gate.detach()),decoder_gate=float(m.decoder_gate.detach()))
                append_json(out/'epochs.jsonl',row)
                print(json.dumps({k:row[k] for k in ['run_id','epoch','train_token_mean_loss','val_token_accuracy','val_exact_accuracy','train_seconds']}),flush=True)
                if epoch % 10 == 0:
                    save_checkpoint(out/'checkpoint_latest.pt',m,opt,cfg,epoch,gen)
                if epoch==30 or epoch==epochs:
                    save_checkpoint(out/f'checkpoint_epoch{epoch}.pt',m,opt,cfg,epoch,gen)
                    if epoch==epochs:save_checkpoint(out/'checkpoint.pt',m,opt,cfg,epoch,gen)
                    if epoch==30 and epochs>30:
                        epout=out/'epoch30_evaluation';epout.mkdir()
                        evaluation(epout,m,tl,data['test'],device,args.rollout_n,train_total,eval_total,time.perf_counter()-start,run_id+'_epoch30')
            evaluation(out,m,tl,data['test'],device,args.rollout_n,train_total,eval_total,time.perf_counter()-start,run_id)
            append_json(ROOT/'run_registry.jsonl',{'run_id':run_id,'status':'complete','summary':str(out/'summary.json'),'total_seconds':time.perf_counter()-start})
            del m,opt
        except Exception as e:
            (out/'error.txt').write_text(traceback.format_exc(),encoding='utf-8');append_json(ROOT/'run_registry.jsonl',{'run_id':run_id,'status':'failed','error':str(e)});raise
    write_json(ROOT/'fullscale_complete.json',{'completed_utc':datetime.datetime.now(datetime.UTC).isoformat(),'variants':args.variants})

if __name__=='__main__':main()
