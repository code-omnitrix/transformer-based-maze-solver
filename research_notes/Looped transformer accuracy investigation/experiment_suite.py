"""Reproducible maze-transformer audits and controlled pilots. No notebook state.

Run with the llmdyn interpreter documented in README.md. Every run writes its
configuration, dataset fingerprints, epochs.jsonl, checkpoint, predictions.jsonl,
evaluation summaries and runtime details. Original notebook is read, never edited.
"""
from __future__ import annotations
import argparse, collections, copy, csv, datetime, hashlib, json, math, os
from pathlib import Path
import platform, random, sys, time, traceback
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
NOTEBOOK = ROOT.parents[1] / 'aiml.ipynb'
DELTAS = {3: (-1, 0), 4: (0, -1), 5: (0, 1), 6: (1, 0)}

def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str), encoding='utf-8')

def append_json(path, value):
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(value, default=str) + '\n')

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def generate_one(rng, size=8, p=.28):
    while True:
        grid = rng.choice([0, 1], size=(size, size), p=[1-p, p])
        start, goal = tuple(rng.integers(size, size=2)), tuple(rng.integers(size, size=2))
        if start == goal: continue
        grid[start], grid[goal] = 2, 3
        queue = collections.deque([(start, [])]); seen = {start}
        while queue:
            pos, path = queue.popleft()
            if pos == goal:
                if len(path) >= 4: return grid.astype(np.int64), [1] + path + [2]
                break
            for token, (dr, dc) in DELTAS.items():
                nxt = (pos[0]+dr, pos[1]+dc)
                if 0 <= nxt[0] < size and 0 <= nxt[1] < size and grid[nxt] != 1 and nxt not in seen:
                    seen.add(nxt); queue.append((nxt, path+[token]))

def dataset(path, n, seed, size=8, p=.28):
    if path.exists():
        z = np.load(path); grids, seqs = z['grids'], z['seqs']
        assert len(grids) == n
    else:
        rng = np.random.default_rng(seed)
        samples = [generate_one(rng, size, p) for _ in range(n)]
        grids = np.stack([s[0] for s in samples])
        seqs = np.zeros((n, max(len(s[1]) for s in samples)), dtype=np.int64)
        for i, (_, seq) in enumerate(samples): seqs[i,:len(seq)] = seq
        np.savez_compressed(path, grids=grids, seqs=seqs)
    return TensorDataset(torch.from_numpy(grids), torch.from_numpy(seqs)), {
        'file': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'n': n, 'seed': seed, 'grid_size': size, 'wall_probability': p,
        'path_length_histogram': dict(collections.Counter((seqs != 0).sum(1).tolist())),
        'max_sequence_length_including_BOS_EOS': seqs.shape[1],
        'grid_hashes': [hashlib.sha256(g.tobytes()).hexdigest() for g in grids]}

class Solver(nn.Module):
    def __init__(self, *, depth=6, tied=False, preln=True, injection=False,
                 loop_encoding=False, d_model=128, ff=256, dropout=.1, size=8,
                 max_seq=128, initialization='xavier'):
        super().__init__()
        self.depth, self.tied, self.injection, self.loop_encoding = depth,tied,injection,loop_encoding
        self.maze_token_emb=nn.Embedding(4,d_model)
        self.row_emb=nn.Embedding(size,d_model); self.col_emb=nn.Embedding(size,d_model)
        self.action_token_emb=nn.Embedding(7,d_model); self.pos_emb=nn.Embedding(max_seq,d_model)
        self.encoder=nn.ModuleList([nn.TransformerEncoderLayer(d_model,8,ff,dropout,batch_first=True,norm_first=preln) for _ in range(1 if tied else depth)])
        self.decoder=nn.ModuleList([nn.TransformerDecoderLayer(d_model,8,ff,dropout,batch_first=True,norm_first=preln) for _ in range(1 if tied else depth)])
        self.encoder_norm=nn.LayerNorm(d_model); self.decoder_norm=nn.LayerNorm(d_model)
        if injection:
            self.encoder_gate=nn.Parameter(torch.ones(1)); self.decoder_gate=nn.Parameter(torch.ones(1))
        self.fc_out=nn.Linear(d_model,7)
        self.register_buffer('rows',torch.arange(size).repeat_interleave(size))
        self.register_buffer('cols',torch.arange(size).repeat(size))
        self.register_buffer('div',torch.exp(torch.arange(0,d_model,2)*(-math.log(10000.)/d_model)))
        if initialization == 'xavier':
            # Matches nn.Transformer core matrix initialization, applied consistently
            # across the controlled family. Embeddings/output retain their defaults.
            for blocks in [self.encoder,self.decoder]:
                for p in blocks.parameters():
                    if p.dim()>1: nn.init.xavier_uniform_(p)

    def step_encoding(self,k,x):
        v=torch.zeros((1,1,x.shape[-1]),device=x.device)
        v[...,0::2]=torch.sin(k*self.div); v[...,1::2]=torch.cos(k*self.div)
        return v

    def encode(self,grid,depth=None):
        init=self.maze_token_emb(grid.flatten(1))+self.row_emb(self.rows)[None]+self.col_emb(self.cols)[None]
        x=init
        for k in range(self.depth if depth is None else depth):
            if self.injection: x=x+self.encoder_gate*init
            if self.loop_encoding: x=x+self.step_encoding(k+1,x)
            x=self.encoder[0 if self.tied else k](x)
        return self.encoder_norm(x)

    def decode(self,seq,memory,depth=None):
        init=self.action_token_emb(seq)+self.pos_emb(torch.arange(seq.shape[1],device=seq.device))[None]
        x=init
        mask=torch.ones(seq.shape[1],seq.shape[1],device=seq.device,dtype=torch.bool).triu(1)
        for k in range(self.depth if depth is None else depth):
            if self.injection: x=x+self.decoder_gate*init
            if self.loop_encoding: x=x+self.step_encoding(k+1,x)
            x=self.decoder[0 if self.tied else k](x,memory,tgt_mask=mask,tgt_key_padding_mask=seq.eq(0))
        return self.fc_out(self.decoder_norm(x))

    def forward(self,grid,seq,depth=None): return self.decode(seq,self.encode(grid,depth),depth)

VARIANTS={
 'postln4': dict(depth=4,preln=False),
 'preln4': dict(depth=4),
 'preln6': dict(depth=6),
 'tied6': dict(depth=6,tied=True),
 'tied6_anchor': dict(depth=6,tied=True,injection=True),
 'tied6_full': dict(depth=6,tied=True,injection=True,loop_encoding=True),
 'untied6_full': dict(depth=6,injection=True,loop_encoding=True),
}

def original_classes():
    # Extract only definitions; skip all notebook dataset/training/visualization code.
    n=json.loads(NOTEBOOK.read_text(encoding='utf-8'))
    ns={'torch':torch,'nn':nn,'math':math,'GRID_SIZE':8,
        'MAZE_VOCAB':{'FREE':0,'WALL':1,'START':2,'GOAL':3},
        'ACTION_VOCAB':{'PAD':0,'BOS':1,'EOS':2,'U':3,'L':4,'R':5,'D':6}}
    exec(''.join(n['cells'][2]['source']),ns)
    base=ns['MazeTransformerSolver']
    exec(''.join(n['cells'][5]['source']),ns)
    return base,ns['MazeTransformerSolver']

def make_model(name,small=False):
    if name.startswith('original_'):
        base,loop=original_classes()
        m=(base if name=='original_baseline' else loop)(**({'d_model':64,'dim_feedforward':128} if small else {}))
        # Adapter functions preserve exact forward path; encode/decode used in rollout.
        m.encode=m.encode_maze
        if name=='original_baseline':
            def decode(seq,memory,depth=None):
                x=m.action_token_emb(seq)+m.pos_emb(torch.arange(seq.shape[1],device=seq.device))[None]
                mask=torch.ones(seq.shape[1],seq.shape[1],device=seq.device,dtype=torch.bool).triu(1)
                return m.fc_out(m.transformer.decoder(x,memory,tgt_mask=mask,tgt_key_padding_mask=seq.eq(0)))
        else:
            def decode(seq,memory,depth=None):
                x=m.action_token_emb(seq)+m.pos_emb(torch.arange(seq.shape[1],device=seq.device))[None]
                mask=torch.ones(seq.shape[1],seq.shape[1],device=seq.device,dtype=torch.bool).triu(1)
                return m.fc_out(m.transformer['decoder'](x,memory,tgt_mask=mask,tgt_key_padding_mask=seq.eq(0),num_loops=depth))
        m.decode=decode
        return m
    return Solver(**VARIANTS[name],**({'d_model':64,'ff':128} if small else {}))

@torch.no_grad()
def teacher_eval(model,loader,device,depth=None,per_example=False):
    model.eval(); totals=collections.Counter(); rows=[]
    for grid,seq in loader:
        grid,seq=grid.to(device),seq.to(device); labels=seq[:,1:]; mask=labels.ne(0)
        logits=model(grid,seq[:,:-1]) if depth is None else model(grid,seq[:,:-1],depth)
        pred=logits.argmax(-1); corr=pred.eq(labels)&mask
        loss=nn.functional.cross_entropy(logits.flatten(0,1),labels.flatten(),ignore_index=0,reduction='sum')
        totals.update(tokens=mask.sum().item(),correct=corr.sum().item(),examples=len(grid),
                      exact=((corr|~mask).all(1)).sum().item(),nll=loss.item(),
                      actions=labels.gt(2).sum().item(),actions_correct=(corr&labels.gt(2)).sum().item(),
                      eos=labels.eq(2).sum().item(),eos_correct=(corr&labels.eq(2)).sum().item())
        if per_example:
            for i in range(len(grid)):
                l=int(mask[i].sum()); rows.append({'path_length':l-1,'tokens':l,'correct':int(corr[i].sum()),'exact':bool((corr[i]|~mask[i]).all()),'teacher_pred':pred[i,:l].cpu().tolist(),'target':labels[i,:l].cpu().tolist()})
    return {**totals,'token_accuracy':totals['correct']/totals['tokens'],'action_accuracy':totals['actions_correct']/totals['actions'],'exact_accuracy':totals['exact']/totals['examples'],'mean_nll':totals['nll']/totals['tokens']},rows

def path_metrics(grid,pred,target):
    pos=tuple(np.argwhere(grid==2)[0]); goal=tuple(np.argwhere(grid==3)[0])
    valid=True; terminated=False; moves=0; trace=[list(map(int,pos))]
    for token in pred:
        if token==2: terminated=True; break
        if token not in DELTAS: valid=False; break
        dr,dc=DELTAS[token]; nxt=(pos[0]+dr,pos[1]+dc)
        if not(0<=nxt[0]<len(grid) and 0<=nxt[1]<len(grid)) or grid[nxt]==1:
            valid=False;break
        pos=nxt;moves+=1;trace.append(list(map(int,pos)))
    solved=valid and terminated and pos==goal
    return {'rollout_exact':pred==target,'valid_path':valid,'terminated':terminated,
            'goal_reached':valid and pos==goal,'solved':solved,
            'optimal_solved':solved and moves==len(target)-1,'moves':moves,'trace':trace}

@torch.no_grad()
def rollout(model,data,device,n=256,depth=None):
    model.eval(); grids,seqs=data.tensors; result=[]
    for offset in range(0,min(n,len(grids)),64):
        gs=grids[offset:offset+min(64,n-offset)].to(device)
        memory=model.encode(gs) if depth is None else model.encode(gs,depth)
        gen=torch.ones((len(gs),1),dtype=torch.long,device=device); done=torch.zeros(len(gs),dtype=torch.bool,device=device)
        outputs=[[] for _ in range(len(gs))]
        for _ in range(64):
            token=model.decode(gen,memory,depth)[:,-1].argmax(-1)
            for j,t in enumerate(token.cpu().tolist()):
                if not done[j]: outputs[j].append(t)
            done|=token.eq(2)
            if done.all():break
            gen=torch.cat([gen,token[:,None]],1)
        for j,pred in enumerate(outputs):
            idx=offset+j; target=seqs[idx,1:]; target=target[target.ne(0)].tolist()
            result.append({'example_id':idx,'prediction':pred,'target':target,
                           'path_length':len(target)-1,'grid_sha256':hashlib.sha256(grids[idx].numpy().tobytes()).hexdigest(),
                           **path_metrics(grids[idx].numpy(),pred,target)})
    return {k:sum(r[k] for r in result)/len(result) for k in ['rollout_exact','valid_path','terminated','goal_reached','solved','optimal_solved']},result

def environment():
    return {'python':sys.version,'executable':sys.executable,'platform':platform.platform(),
            'torch':torch.__version__,'numpy':np.__version__,'cuda_available':torch.cuda.is_available(),
            'cuda_version':torch.version.cuda,'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'threads':torch.get_num_threads(),'notebook_sha256':hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest(),
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'argv':sys.argv,'utc':datetime.datetime.now(datetime.UTC).isoformat(),
            'determinism':'Fixed RNG seeds; deterministic algorithms not forced; CUDA kernels may vary.'}

def audit():
    out=ROOT/'audit';out.mkdir(exist_ok=True)
    notebook=json.loads(NOTEBOOK.read_text(encoding='utf-8'))
    hist=[]
    import re
    for i,c in enumerate(notebook['cells']):
        (out/f'cell_{i:02d}.py').write_text(''.join(c.get('source',[])),encoding='utf-8')
        for o in c.get('outputs',[]):
            for epoch,total,loss,acc in re.findall(r'Epoch (\d+)/(\d+) \| Train Loss: ([\d.]+) \| Val Token Acc: ([\d.]+)%',''.join(o.get('text',[]))):
                hist.append({'source':'saved_notebook_output_not_new_run','model':'baseline' if i==3 else 'looped','cell':i,'execution_count':c.get('execution_count'),'epoch':int(epoch),'total_epochs':int(total),'train_loss':float(loss),'val_token_accuracy_pct':float(acc)})
    with (out/'original_histories.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(hist[0]));w.writeheader();w.writerows(hist)
    seed_all(42); base,loop=original_classes(); models={'baseline':base(),'looped':loop()}
    counts={k:{'parameters':sum(p.numel() for p in m.parameters()),'trainable_parameters':sum(p.numel() for p in m.parameters() if p.requires_grad),
               'modules':str(m)} for k,m in models.items()}
    write_json(out/'architecture_counts.json',counts)
    write_json(out/'notebook_structure.json',{'sha256':hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest(),'metadata':notebook['metadata'],'cells':[{'index':i,'execution_count':c.get('execution_count'),'source_sha256':hashlib.sha256(''.join(c.get('source',[])).encode()).hexdigest(),'outputs':len(c.get('outputs',[]))} for i,c in enumerate(notebook['cells'])]})
    # Direct mask diagnostic: future-input perturbation must not change prefix logits.
    checks={}
    for name,m in models.items():
        m.eval();g=torch.randint(0,4,(2,8,8));s=torch.tensor([[1,3,4,5,6,2],[1,4,3,6,5,2]])
        changed=s.clone();changed[:,3:]=torch.tensor([6,3,4])
        with torch.no_grad():
            a=m(g,s);b=m(g,changed)
        checks[name]={'future_prefix_max_abs_difference':float((a[:,:3]-b[:,:3]).abs().max()),'finite':bool(torch.isfinite(a).all())}
    write_json(out/'causal_checks.json',checks)
    print(json.dumps({'counts':{k:v['parameters'] for k,v in counts.items()},'causal':checks}),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['audit','benchmark','train'],default='audit')
    p.add_argument('--variants',default='postln4,preln6,tied6,tied6_anchor,tied6_full')
    p.add_argument('--seeds',default='42,43,44');p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--train-size',type=int,default=6000);p.add_argument('--val-size',type=int,default=1000)
    p.add_argument('--test-size',type=int,default=1000);p.add_argument('--rollout-n',type=int,default=256)
    p.add_argument('--batch-size',type=int,default=128);p.add_argument('--small',action='store_true')
    p.add_argument('--label',default='pilot');p.add_argument('--threads',type=int,default=4)
    args=p.parse_args();torch.set_num_threads(args.threads)
    if args.mode=='audit':audit();return
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.mode=='benchmark':
        result=[]
        for name in args.variants.split(','):
            seed_all(42);m=make_model(name,args.small).to(device);opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001)
            g=torch.randint(0,4,(args.batch_size,8,8),device=device);s=torch.randint(1,7,(args.batch_size,36),device=device)
            times=[]
            for step in range(12):
                torch.cuda.synchronize();t=time.perf_counter();opt.zero_grad(set_to_none=True)
                loss=nn.functional.cross_entropy(m(g,s).flatten(0,1),s.flatten());loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
                torch.cuda.synchronize();times.append(time.perf_counter()-t)
            result.append({'name':name,'parameters':sum(p.numel() for p in m.parameters()),'step_seconds_median_last10':float(np.median(times[2:])),'all_step_seconds':times,'peak_cuda_memory':torch.cuda.max_memory_allocated()})
            print(json.dumps(result[-1]),flush=True);del m,opt
        write_json(ROOT/'benchmark.json',{'environment':environment(),'results':result});return
    data_dir=ROOT/'data';data_dir.mkdir(exist_ok=True)
    train,tm=dataset(data_dir/f'train_n{args.train_size}_s20261102.npz',args.train_size,20261102)
    val,vm=dataset(data_dir/f'val_n{args.val_size}_s20261103.npz',args.val_size,20261103)
    test,sm=dataset(data_dir/f'test_n{args.test_size}_s20261104.npz',args.test_size,20261104)
    hashes={k:set(v['grid_hashes']) for k,v in [('train',tm),('val',vm),('test',sm)]}
    overlap={f'{a}_{b}':len(hashes[a]&hashes[b]) for a,b in [('train','val'),('train','test'),('val','test')]}
    write_json(data_dir/'manifest.json',{'splits':[tm,vm,sm],'exact_grid_overlap':overlap,
              'generator':'NumPy default_rng; statistically same problem distribution, NOT bit-identical original notebook RNG sequence.'})
    for seed in map(int,args.seeds.split(',')):
        for name in args.variants.split(','):
            run_id=f'{args.label}_{name}_seed{seed}';out=ROOT/'runs'/run_id
            if out.exists():raise RuntimeError(f'Refusing to overwrite {out}')
            out.mkdir(parents=True);seed_all(seed)
            cfg={**vars(args),'variant':name,'seed':seed,'optimizer':'AdamW','lr':.001,'weight_decay':.0001,'clip_grad_norm':1.,
                 'dtype':'float32','amp':False,'dataset_sha256':{x:y['sha256'] for x,y in [('train',tm),('val',vm),('test',sm)]},
                 'scope':'Reduced data/epoch pilot, not replication of original numerical results',
                 'architecture':VARIANTS.get(name,'exact extracted notebook definition')}
            write_json(out/'config.json',cfg);write_json(out/'environment.json',environment())
            append_json(ROOT/'run_registry.jsonl',{'run_id':run_id,'status':'started','config':str(out/'config.json'),'utc':datetime.datetime.now(datetime.UTC).isoformat()})
            started=time.perf_counter()
            try:
                m=make_model(name,args.small).to(device);parameters=sum(p.numel() for p in m.parameters())
                write_json(out/'model.json',{'parameters':parameters,'structure':str(m)})
                opt=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001)
                generator=torch.Generator().manual_seed(seed+1000)
                tr=DataLoader(train,batch_size=args.batch_size,shuffle=True,generator=generator)
                vl=DataLoader(val,batch_size=args.batch_size);tl=DataLoader(test,batch_size=args.batch_size)
                train_total=0.;eval_total=0.;steps=0
                for epoch in range(1,args.epochs+1):
                    m.train();nll=0.;tokens=0;batch_mean_loss=0;norms=[];t=time.perf_counter()
                    for g,s in tr:
                        step_t=time.perf_counter()
                        g,s=g.to(device),s.to(device);opt.zero_grad(set_to_none=True)
                        logits=m(g,s[:,:-1]);labels=s[:,1:]
                        loss=nn.functional.cross_entropy(logits.flatten(0,1),labels.flatten(),ignore_index=0)
                        loss.backward();gn=nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
                        nt=int(labels.ne(0).sum());nll+=loss.item()*nt;tokens+=nt;batch_mean_loss+=loss.item();norms.append(float(gn));steps+=1
                        append_json(out/'steps.jsonl',{'run_id':run_id,'epoch':epoch,'optimizer_step':steps,
                            'pre_update_teacher_forced_token_mean_nll':loss.item(),'target_tokens':nt,
                            'cumulative_observed_training_nll_nats':nll,'lr':opt.param_groups[0]['lr'],
                            'gradient_norm_before_clip':float(gn),'step_wall_seconds':time.perf_counter()-step_t,
                            'note':'Dropout-active repeated-data training NLL, not an epiplexity estimator; cumulative NLL resets each epoch.'})
                    torch.cuda.synchronize();train_sec=time.perf_counter()-t;train_total+=train_sec;t=time.perf_counter()
                    metrics,_=teacher_eval(m,vl,device);eval_sec=time.perf_counter()-t;eval_total+=eval_sec
                    row={'run_id':run_id,'epoch':epoch,'optimizer_steps':steps,'train_token_mean_loss':nll/tokens,'train_batch_mean_loss':batch_mean_loss/len(tr),
                         'train_tokens':tokens,'train_seconds':train_sec,'cumulative_train_seconds':train_total,'val_seconds':eval_sec,
                         'gradient_norm_mean_before_clip':float(np.mean(norms)), 'gradient_clip_fraction':float(np.mean(np.array(norms)>1)),
                         'parameters':parameters,**{'val_'+k:v for k,v in metrics.items()}}
                    if hasattr(m,'encoder_gate'):row.update(encoder_gate=float(m.encoder_gate.detach()),decoder_gate=float(m.decoder_gate.detach()))
                    append_json(out/'epochs.jsonl',row);print(json.dumps({k:row[k] for k in ['run_id','epoch','train_token_mean_loss','val_token_accuracy','val_exact_accuracy','train_seconds']}),flush=True)
                torch.save({'state_dict':m.state_dict(),'optimizer_state_dict':opt.state_dict(),'config':cfg,'seed':seed,'epoch':args.epochs,
                            'rng_python':random.getstate(),'rng_numpy':np.random.get_state(),'rng_torch':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state_all(),'loader_rng':generator.get_state()},out/'checkpoint.pt')
                final,teacher_rows=teacher_eval(m,tl,device,per_example=True)
                roll,roll_rows=rollout(m,test,device,args.rollout_n)
                for i,r in enumerate(teacher_rows):append_json(out/'test_teacher_predictions.jsonl',{'example_id':i,**r})
                for r in roll_rows:append_json(out/'test_rollout_predictions.jsonl',r)
                summary={'run_id':run_id,'parameters':parameters,'teacher_test':final,'rollout_test':roll,'rollout_n':len(roll_rows),
                         'train_seconds':train_total,'val_seconds':eval_total,'total_seconds':time.perf_counter()-started}
                # Preregistered diagnostic sweep: same final checkpoint, no retraining.
                if name in ['tied6','tied6_full','tied6_anchor']:
                    summary['depth_sweep']={}
                    for d in [1,3,6,9,12]:
                        met,_=teacher_eval(m,tl,device,depth=d)
                        summary['depth_sweep'][d]=met
                write_json(out/'summary.json',summary)
                append_json(ROOT/'run_registry.jsonl',{'run_id':run_id,'status':'complete','summary':str(out/'summary.json'),'total_seconds':summary['total_seconds']})
                print('COMPLETE '+json.dumps(summary),flush=True);del m,opt
            except Exception as exc:
                (out/'error.txt').write_text(traceback.format_exc(),encoding='utf-8')
                append_json(ROOT/'run_registry.jsonl',{'run_id':run_id,'status':'failed','error':str(exc)})
                raise

if __name__=='__main__':main()
