"""Continue a trusted local run checkpoint into a new folder, restoring RNG.

Example: python resume_run.py --run runs/controlled12k_tied6_full_seed42
         --additional-epochs 10 --new-id continued_tied6_full_seed42
"""
import argparse, copy, hashlib, json, random, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset,DataLoader
from experiment_suite import ROOT,make_model,seed_all,write_json,append_json,environment,teacher_eval,rollout

def capture(m,opt,generator):
    return {'state_dict':copy.deepcopy(m.state_dict()),'optimizer_state_dict':copy.deepcopy(opt.state_dict()),
            'rng_python':random.getstate(),'rng_numpy':np.random.get_state(),'rng_torch':torch.get_rng_state(),
            'rng_cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],'loader_rng':generator.get_state()}

def restore(snapshot,m,opt,generator):
    m.load_state_dict(snapshot['state_dict']);opt.load_state_dict(snapshot['optimizer_state_dict'])
    random.setstate(snapshot['rng_python']);np.random.set_state(snapshot['rng_numpy']);torch.set_rng_state(snapshot['rng_torch'])
    if snapshot['rng_cuda']:torch.cuda.set_rng_state_all(snapshot['rng_cuda'])
    generator.set_state(snapshot['loader_rng'])

def epoch(m,opt,loader,device,callback=None):
    m.train();nll=0.;tokens=0
    for g,s in loader:
        step_start=time.perf_counter()
        g,s=g.to(device),s.to(device);opt.zero_grad(set_to_none=True)
        y=s[:,1:];loss=nn.functional.cross_entropy(m(g,s[:,:-1]).flatten(0,1),y.flatten(),ignore_index=0)
        loss.backward();gn=nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
        n=int(y.ne(0).sum());nll+=loss.item()*n;tokens+=n
        if callback:callback({'pre_update_teacher_forced_token_mean_nll':loss.item(),'target_tokens':n,
            'gradient_norm_before_clip':float(gn),'lr':opt.param_groups[0]['lr'],'step_wall_seconds':time.perf_counter()-step_start})
    return nll/tokens

def self_test():
    # CPU determinism, dropout and shuffled minibatches, independently reconstructed
    # model/optimizer/loader. Compare the next complete epoch, not a mirrored formula.
    torch.set_num_threads(1);seed_all(91)
    g=torch.randint(0,4,(8,8,8));s=torch.tensor([[1,3,4,6,2,0]]*8);ds=TensorDataset(g,s)
    model=make_model('tied6_full',True);opt=torch.optim.AdamW(model.parameters(),lr=.001)
    gen=torch.Generator().manual_seed(19);loader=DataLoader(ds,batch_size=4,shuffle=True,generator=gen)
    epoch(model,opt,loader,'cpu');snap=capture(model,opt,gen);loss_a=epoch(model,opt,loader,'cpu')
    final_a=copy.deepcopy(model.state_dict());m2=make_model('tied6_full',True);o2=torch.optim.AdamW(m2.parameters(),lr=.001)
    gen2=torch.Generator();loader2=DataLoader(ds,batch_size=4,shuffle=True,generator=gen2);restore(snap,m2,o2,gen2)
    loss_b=epoch(m2,o2,loader2,'cpu');diff=max(float((final_a[k]-m2.state_dict()[k]).abs().max()) for k in final_a)
    out={'scope':'CPU two-epoch tiny dropout+shuffle continuation diagnostic','uninterrupted_next_epoch_loss':loss_a,
         'resumed_next_epoch_loss':loss_b,'maximum_state_difference':diff,'exact_match':diff==0 and loss_a==loss_b,
         'gpu_bitwise_resume_not_claimed':True}
    write_json(ROOT/'resume_validation.json',out);print(json.dumps(out));assert out['exact_match']

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path);p.add_argument('--checkpoint',default='checkpoint.pt')
    p.add_argument('--additional-epochs',type=int,default=10);p.add_argument('--new-id');p.add_argument('--self-test',action='store_true');a=p.parse_args()
    if a.self_test:self_test();return
    if not a.run or not a.new_id:p.error('--run and --new-id required')
    torch.set_num_threads(4);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    run=a.run if a.run.is_absolute() else ROOT/a.run
    cp=run/a.checkpoint;snapshot=torch.load(cp,map_location='cpu',weights_only=False);cfg=snapshot['config'];start_epoch=snapshot['epoch']
    out=ROOT/'runs'/a.new_id
    if out.exists():raise RuntimeError(f'Refusing to overwrite {out}')
    out.mkdir(parents=True)
    cfg={**cfg,'parent_run':str(run),'parent_checkpoint':str(cp),'parent_checkpoint_sha256':hashlib.sha256(cp.read_bytes()).hexdigest(),
         'resume_start_epoch':start_epoch,'additional_epochs':a.additional_epochs,'epochs':start_epoch+a.additional_epochs}
    write_json(out/'config.json',cfg);write_json(out/'environment.json',environment())
    append_json(ROOT/'run_registry.jsonl',{'run_id':a.new_id,'status':'started','parent_run':str(run),'parent_checkpoint_sha256':cfg['parent_checkpoint_sha256'],'config':str(out/'config.json')})
    datasets={}
    for split in ['train','val','test']:
        if 'dataset_files' in cfg:path=Path(cfg['dataset_files'][split])
        else:path=ROOT/'data'/f'{split}_n{cfg[split+"_size"]}_s{dict(train=20261102,val=20261103,test=20261104)[split]}.npz'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==cfg['dataset_sha256'][split]
        z=np.load(path);datasets[split]=TensorDataset(torch.from_numpy(z['grids']),torch.from_numpy(z['seqs']))
    m=make_model(cfg['variant'],cfg.get('small',False)).to(device);opt=torch.optim.AdamW(m.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    gen=torch.Generator();tr=DataLoader(datasets['train'],batch_size=cfg['batch_size'],shuffle=True,generator=gen)
    vl=DataLoader(datasets['val'],batch_size=cfg['batch_size']);tl=DataLoader(datasets['test'],batch_size=cfg['batch_size'])
    restore(snapshot,m,opt,gen);start=time.perf_counter();train_total=0.;val_total=0.;counter=[start_epoch*len(tr)]
    for ep in range(start_epoch+1,start_epoch+a.additional_epochs+1):
        norm_values=[]
        def record_step(record):
            counter[0]+=1;norm_values.append(record['gradient_norm_before_clip']);append_json(out/'steps.jsonl',{'run_id':a.new_id,'epoch':ep,'optimizer_step':counter[0],**record})
        t=time.perf_counter();loss=epoch(m,opt,tr,device,record_step);trainsec=time.perf_counter()-t;train_total+=trainsec;t=time.perf_counter();met,_=teacher_eval(m,vl,device);valsec=time.perf_counter()-t;val_total+=valsec
        row={'run_id':a.new_id,'epoch':ep,'optimizer_steps':ep*len(tr),'train_token_mean_loss':loss,'train_seconds':trainsec,'cumulative_train_seconds':train_total,
             'val_seconds':valsec,'parameters':sum(p.numel() for p in m.parameters()),'gradient_norm_mean_before_clip':float(np.mean(norm_values)),
             'gradient_clip_fraction':float(np.mean(np.array(norm_values)>1)),**{'val_'+k:v for k,v in met.items()}}
        append_json(out/'epochs.jsonl',row);print(json.dumps(row),flush=True)
        state=capture(m,opt,gen);state.update(config=cfg,seed=cfg['seed'],epoch=ep);torch.save(state,out/'checkpoint.pt')
    final,ts=teacher_eval(m,tl,device,per_example=True);roll,rs=rollout(m,datasets['test'],device,512)
    for i,r in enumerate(ts):append_json(out/'test_teacher_predictions.jsonl',{'example_id':i,**r})
    for r in rs:append_json(out/'test_rollout_predictions.jsonl',r)
    summary={'run_id':a.new_id,'teacher_test':final,'rollout_test':roll,'rollout_n':len(rs),'parameters':sum(p.numel() for p in m.parameters()),'train_seconds':train_total,'val_seconds':val_total,'total_seconds':time.perf_counter()-start}
    write_json(out/'summary.json',summary);append_json(ROOT/'run_registry.jsonl',{'run_id':a.new_id,'status':'complete','summary':str(out/'summary.json'),'total_seconds':summary['total_seconds']})

if __name__=='__main__':main()
