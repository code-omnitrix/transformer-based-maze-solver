"""Residual-stream analysis of full-size checkpoints and per-epoch training snapshots.

For each checkpoint (eval mode, 5,000 validation mazes of the full50k split; probes fit on the
first 4,000 mazes and scored on the last 1,000):

Encoder, at the embedding (layer 0) and after every layer / loop iteration (tied models are also
unrolled past the trained K=6, up to K=16):
  * linear ridge probes for BFS distance-to-goal (0..19, 20+, unreachable), the shortest-path move
    from each cell (first optimal of U,L,R,D), and canonical-path membership; distance accuracy is
    also kept per true distance, which shows whether information spreads one hop per iteration;
  * PCA of cell residuals: participation ratio, top-1/top-5 explained variance;
  * residual norm, relative update ||h_k-h_{k-1}||/||h_{k-1}|| and cosine to the previous state;
  * attention mass on the 4 grid neighbours, on the goal cell and on the start cell.
Decoder, after every layer / iteration:
  * logit lens (final norm + output layer) teacher-forced token accuracy;
  * linear probe for the agent's current cell (64-way) at each prefix position.
Behaviour (teacher-forced argmax at each prefix position): fraction legal, fraction any optimal
move, fraction equal to the greedy Manhattan move, fraction equal to the canonical target.

Linear probes show linearly available information, not what the network uses. Random-init models
are included as controls. Writes analysis/residual/metrics.jsonl and pca_samples.npz.
Usage: python residual_analysis.py [--only-prefix dyn50k]
"""
import argparse, collections, json, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from experiment_suite import ROOT, make_model, seed_all
from fullscale_experiments import full_data
from extension_experiments import build

OUT=ROOT/'analysis'/'residual';OUT.mkdir(parents=True,exist_ok=True)
DELTAS=[(-1,0),(0,-1),(0,1),(1,0)]  # U L R D, token ids 3..6
N_TRAIN=4000;DIST_CAP=20;UNREACH=21;MAX_K=16

# ---------- targets ----------
def targets(grids,seqs):
    n=len(grids);dist=np.full((n,64),-1,np.int64);policy=np.full((n,64),-1,np.int64);onpath=np.zeros((n,64),np.int64)
    greedy=np.full((n,64),-1,np.int64);optimal=np.zeros((n,64,4),bool);path_cells=[]
    for i,g in enumerate(grids):
        goal=tuple(np.argwhere(g==3)[0]);d=np.full((8,8),-1);d[goal]=0;q=collections.deque([goal])
        while q:
            r,c=q.popleft()
            for dr,dc in DELTAS:
                a,b=r+dr,c+dc
                if 0<=a<8 and 0<=b<8 and g[a,b]!=1 and d[a,b]<0:d[a,b]=d[r,c]+1;q.append((a,b))
        dist[i]=d.flatten()
        for r in range(8):
            for c in range(8):
                if g[r,c]==1:continue
                best=None
                for k,(dr,dc) in enumerate(DELTAS):
                    a,b=r+dr,c+dc
                    if 0<=a<8 and 0<=b<8 and g[a,b]!=1:
                        if d[r,c]>0 and d[a,b]==d[r,c]-1:
                            optimal[i,r*8+c,k]=True;best=k if best is None else best
                        if greedy[i,r*8+c]<0 and abs(a-goal[0])+abs(b-goal[1])<abs(r-goal[0])+abs(c-goal[1]):greedy[i,r*8+c]=k
                if best is not None:policy[i,r*8+c]=best
        pos=tuple(np.argwhere(g==2)[0]);cells=[pos[0]*8+pos[1]]
        for t in seqs[i,1:]:
            if t<3:break
            dr,dc=DELTAS[t-3];pos=(pos[0]+dr,pos[1]+dc);cells.append(pos[0]*8+pos[1])
        onpath[i,cells]=1;path_cells.append(cells)
    dclass=np.where(dist<0,UNREACH,np.minimum(dist,DIST_CAP))
    return {'free':grids.reshape(n,64)!=1,'dist':dist,'dclass':dclass,'policy':policy,'onpath':onpath,'greedy':greedy,'optimal':optimal,'path_cells':path_cells}

# ---------- tracing (re-implements each forward pass; checked against model output) ----------
def kind(m):
    if hasattr(m,'transformer') and isinstance(m.transformer,nn.Transformer):return 'nb'
    if hasattr(m,'transformer'):return 'nl'
    return 'solver'

def embed(m,k,grid):
    B=grid.shape[0]
    if k=='solver':return m.maze_token_emb(grid.flatten(1))+m.row_emb(m.rows)[None]+m.col_emb(m.cols)[None]
    return m.maze_token_emb(grid.view(B,-1))+m.row_emb(m.grid_rows)[None]+m.col_emb(m.grid_cols)[None]

def tied(m,k): return k=='nl' or (k=='solver' and m.tied)

def enc_trace(m,k,grid,K):
    src=embed(m,k,grid);h=src;states=[src];attn=[]
    for i in range(K):
        if k=='nb':layer=m.transformer.encoder.layers[i];inp=h
        elif k=='nl':
            e=m.transformer['encoder'];layer=e.shared_block;inp=h+e.input_gate*src+e.loop_emb(i+1,h.device)
        else:
            layer=m.encoder[0 if m.tied else i];inp=h+(m.encoder_gate*src if m.injection else 0)
            if m.loop_encoding:inp=inp+m.step_encoding(i+1,h)
        a=layer.norm1(inp) if layer.norm_first else inp
        attn.append(layer.self_attn(a,a,a,need_weights=True,average_attn_weights=False)[1])
        h=layer(inp);states.append(h)
    final={'nb':lambda:m.transformer.encoder.norm,'nl':lambda:m.transformer['encoder'].final_norm,'solver':lambda:m.encoder_norm}[k]()
    return states,attn,final(h)

def dec_trace(m,k,seq,memory,K):
    T=seq.shape[1];x0=m.action_token_emb(seq)+m.pos_emb(torch.arange(T,device=seq.device))[None]
    mask=torch.ones(T,T,device=seq.device,dtype=torch.bool).triu(1);pad=seq.eq(0);h=x0;states=[]
    for i in range(K):
        if k=='nb':layer=m.transformer.decoder.layers[i];inp=h
        elif k=='nl':
            d=m.transformer['decoder'];layer=d.shared_block;inp=h+d.input_gate*x0+d.loop_emb(i+1,h.device)
        else:
            layer=m.decoder[0 if m.tied else i];inp=h+(m.decoder_gate*x0 if m.injection else 0)
            if m.loop_encoding:inp=inp+m.step_encoding(i+1,h)
        h=layer(inp,memory,tgt_mask=mask,tgt_key_padding_mask=pad);states.append(h)
    norm={'nb':lambda:m.transformer.decoder.norm,'nl':lambda:m.transformer['decoder'].final_norm,'solver':lambda:m.decoder_norm}[k]()
    return states,lambda z:m.fc_out(norm(z))

def trained_depth(m,k):
    return len(m.transformer.encoder.layers) if k=='nb' else (m.transformer['encoder'].num_loops if k=='nl' else m.depth)

# ---------- probes ----------
def ridge_probe(X,y,ntr,ncls,lam=1e-3):
    """Least-squares one-hot linear classifier with bias; X (n,d) on GPU, y (n,) ints. Returns test predictions."""
    mu=X[:ntr].mean(0);sd=X[:ntr].std(0)+1e-6;Z=torch.cat([(X-mu)/sd,torch.ones(len(X),1,device=X.device)],1)
    Y=torch.nn.functional.one_hot(y[:ntr],ncls).float();A=Z[:ntr].T@Z[:ntr];A+=lam*ntr*torch.eye(A.shape[0],device=X.device)
    W=torch.linalg.solve(A,Z[:ntr].T@Y);return (Z[ntr:]@W).argmax(1)

def ridge_regress(S,target,select,lam=1e-3):
    """Linear ridge regression of a per-cell scalar; returns (test prediction, test target)."""
    tr=select[:N_TRAIN];te=select[N_TRAIN:];X=torch.cat([S[:N_TRAIN][tr],S[N_TRAIN:][te]]);y=torch.cat([target[:N_TRAIN][tr],target[N_TRAIN:][te]]).float();n=int(tr.sum())
    mu=X[:n].mean(0);sd=X[:n].std(0)+1e-6;Z=torch.cat([(X-mu)/sd,torch.ones(len(X),1,device=X.device)],1)
    A=Z[:n].T@Z[:n];A+=lam*n*torch.eye(A.shape[0],device=X.device);w=torch.linalg.solve(A,Z[:n].T@y[:n])
    return Z[n:]@w,y[n:]

def cell_probe(S,target,select,ncls):
    """S (N,64,d); select (N,64) bool. Train on mazes < N_TRAIN."""
    tr=select[:N_TRAIN];te=select[N_TRAIN:]
    X=torch.cat([S[:N_TRAIN][tr],S[N_TRAIN:][te]]);y=torch.cat([target[:N_TRAIN][tr],target[N_TRAIN:][te]])
    pred=ridge_probe(X,y,int(tr.sum()),ncls);return pred,y[int(tr.sum()):]

def pca_stats(X):
    X=X-X.mean(0);ev=torch.linalg.eigvalsh(X.T@X/len(X)).flip(0).clamp(min=0);tot=ev.sum()
    return {'participation_ratio':float(tot**2/(ev**2).sum()),'pc1_var':float(ev[0]/tot),'top5_var':float(ev[:5].sum()/tot)}

# ---------- per-checkpoint analysis ----------
@torch.no_grad()
def analyse(m,label,meta,grids,seqs,tg,pca_store):
    m=m.cuda().eval();k=kind(m);Kt=trained_depth(m,k);K=MAX_K if tied(m,k) else Kt;dev='cuda'
    G=torch.from_numpy(grids).cuda();Sq=torch.from_numpy(seqs).cuda();N=len(G)
    enc=[[] for _ in range(K+1)];loc=np.zeros((K,3));logits_layers=[[] for _ in range(Kt)];dec=[[] for _ in range(Kt)];check=0.
    free=torch.from_numpy(tg['free']).cuda()
    nbr=torch.zeros(64,64,device=dev)
    for c in range(64):
        r,cc=divmod(c,8)
        for dr,dc in DELTAS:
            if 0<=r+dr<8 and 0<=cc+dc<8:nbr[c,(r+dr)*8+cc+dc]=1
    for s in range(0,N,250):
        g=G[s:s+250];q=Sq[s:s+250];states,attn,mem=enc_trace(m,k,g,K)
        for i,h in enumerate(states):enc[i].append(h.float())
        goal=(g.view(len(g),64)==3).float();start=(g.view(len(g),64)==2).float();fq=free[s:s+250].float()
        for i,a in enumerate(attn):
            a=a.mean(1)  # heads averaged: (B,64,64)
            w=fq/fq.sum()
            loc[i]+=[float(((a*nbr).sum(-1)*w).sum()),float(((a*goal[:,None,:]).sum(-1)*w).sum()),float(((a*start[:,None,:]).sum(-1)*w).sum())]
        if K!=Kt:_,_,mem=enc_trace(m,k,g,Kt)
        dstates,lens=dec_trace(m,k,q[:,:-1],mem,Kt)
        for i,h in enumerate(dstates):dec[i].append(h.float());logits_layers[i].append(lens(h).argmax(-1))
        ref=m(g,q[:,:-1]);check=max(check,float((lens(dstates[-1])-ref).abs().max()))
    assert check<1e-3,f'trace mismatch {check}'
    loc/=int(np.ceil(N/250))
    rows=[];dclass=torch.from_numpy(tg['dclass']).cuda();dist=torch.from_numpy(tg['dist']).cuda()
    pol=torch.from_numpy(tg['policy']).cuda();onp=torch.from_numpy(tg['onpath']).cuda()
    for i in range(K+1):
        S=torch.cat(enc[i]);row={**meta,'part':'encoder','layer':i,'beyond_trained':i>Kt}
        pd,yd=cell_probe(S,dclass,free,UNREACH+1);row['probe_dist_acc']=float((pd==yd).float().mean())
        te_dist=dist[N_TRAIN:][free[N_TRAIN:]]
        row['probe_dist_acc_by_true_dist']={int(d):float((pd[te_dist==d]==yd[te_dist==d]).float().mean()) for d in range(0,15) if (te_dist==d).any()}
        rp,ry=ridge_regress(S,dist,dist>=0);row['probe_dist_r2']=float(1-((rp-ry)**2).sum()/((ry-ry.mean())**2).sum())
        row['probe_dist_within1']=float(((rp-ry).abs()<=1).float().mean())
        row['probe_dist_within1_by_true_dist']={int(d):float(((rp-ry).abs()<=1)[ry==d].float().mean()) for d in range(0,15) if (ry==d).any()}
        pp,yp=cell_probe(S,pol,pol>=0,4);row['probe_policy_acc']=float((pp==yp).float().mean())
        po,yo=cell_probe(S,onp,free,2);row['probe_onpath_acc']=float((po==yo).float().mean())
        row['probe_onpath_balanced_acc']=float(0.5*((po[yo==1]==1).float().mean()+(po[yo==0]==0).float().mean()))
        Xf=S[N_TRAIN:][free[N_TRAIN:]];row.update(pca_stats(Xf));row['residual_norm']=float(S.norm(dim=-1).mean())
        if i>0:
            P=torch.cat(enc[i-1]);row['relative_update']=float(((S-P).norm(dim=-1)/P.norm(dim=-1)).mean())
            row['cosine_to_previous']=float(nn.functional.cosine_similarity(S,P,dim=-1).mean())
            row.update(attn_neighbour_mass=loc[i-1,0],attn_goal_mass=loc[i-1,1],attn_start_mass=loc[i-1,2])
        rows.append(row)
        if pca_store is not None:
            Xc=Xf-Xf.mean(0);_,_,V=torch.linalg.svd(Xc[:20000],full_matrices=False);proj=(Xc[:3000]@V[:2].T).cpu().numpy()
            pca_store[f'{label}|enc{i}']=proj;pca_store[f'{label}|enc{i}|dist']=te_dist[:3000].cpu().numpy()
    labels=Sq[:,1:];vmask=labels.ne(0)
    T=seqs.shape[1]-1;pos=np.full((N,T),-1,np.int64)
    for i,cells in enumerate(tg['path_cells']):pos[i,:len(cells)]=cells
    pos=torch.from_numpy(pos).cuda()
    for i in range(Kt):
        L=torch.cat(logits_layers[i]);S=torch.cat(dec[i]);row={**meta,'part':'decoder','layer':i+1}
        row['logit_lens_token_acc']=float((L.eq(labels)&vmask).sum()/vmask.sum())
        sel=pos>=0;tr=sel[:N_TRAIN];X=torch.cat([S[:N_TRAIN][tr],S[N_TRAIN:][sel[N_TRAIN:]]]);y=torch.cat([pos[:N_TRAIN][tr],pos[N_TRAIN:][sel[N_TRAIN:]]])
        row['probe_agent_cell_acc']=float((ridge_probe(X,y,int(tr.sum()),64)==y[int(tr.sum()):]).float().mean())
        rows.append(row)
    # behaviour of the final model at every teacher-forced prefix position (action targets only)
    pred=torch.cat(logits_layers[-1]).cpu().numpy();lab=labels.cpu().numpy();b=collections.Counter()
    for i in range(N):
        for t,cell in enumerate(tg['path_cells'][i][:-1]):
            p=pred[i,t];b['n']+=1
            if p<3 or p>6:continue
            r,c=divmod(cell,8);dr,dc=DELTAS[p-3];a,bb=r+dr,c+dc
            b['legal']+=int(0<=a<8 and 0<=bb<8 and grids[i,a,bb]!=1);b['optimal']+=int(tg['optimal'][i,cell,p-3])
            b['greedy']+=int(tg['greedy'][i,cell]==p-3);b['canonical']+=int(lab[i,t]==p)
    rows.append({**meta,'part':'behaviour',**{f'frac_{x}':b[x]/b['n'] for x in ['legal','optimal','greedy','canonical']},
                 'greedy_is_optimal_rate':float(np.mean([tg['optimal'][i,c,tg['greedy'][i,c]] for i in range(N) for c in tg['path_cells'][i][:-1] if tg['greedy'][i,c]>=0]))})
    return rows

def checkpoints(only):
    items=[]
    for d in sorted((ROOT/'runs').iterdir()):
        pre=d.name.split('_')[0]
        if pre not in('full50k','ext50k','dyn50k') or (only and pre!=only):continue
        cfg=json.loads((d/'config.json').read_text());v=cfg['variant'];factory=(lambda v=v:build(v)[0]) if pre in('ext50k','dyn50k') else (lambda v=v:make_model(v))
        if pre=='dyn50k':
            for p in sorted((d/'snapshots').glob('ep*.pt')):items.append((d.name,v,cfg['seed'],int(p.stem[2:]),p,factory,int(p.stem[2:]) in(1,3,5,8,12,18,24,30)))
        else:
            if (d/'checkpoint.pt').exists():items.append((d.name,v,cfg['seed'],cfg['epochs'],d/'checkpoint.pt',factory,cfg['seed']==42))
            if (d/'checkpoint_epoch30.pt').exists() and cfg['epochs']!=30:items.append((d.name,v,cfg['seed'],30,d/'checkpoint_epoch30.pt',factory,True))
    if not only:
        for v in['original_baseline','original_looped']:items.append((f'random_init_{v}',v,0,0,None,lambda v=v:make_model(v),True))
    return items

def main():
    p=argparse.ArgumentParser();p.add_argument('--only-prefix',default='');a=p.parse_args()
    torch.set_num_threads(4);data,_=full_data();g,s=data['val'].tensors;grids=g.numpy();seqs=s.numpy();tg=targets(grids,seqs)
    name=f'metrics_{a.only_prefix}.jsonl' if a.only_prefix else 'metrics.jsonl';out=OUT/name;out.write_text('')
    pca={}
    for run,v,seed,epoch,path,factory,keep_pca in checkpoints(a.only_prefix):
        t=time.perf_counter();seed_all(0);m=factory()
        if path is not None:
            sd=torch.load(path,map_location='cpu',weights_only=False);m.load_state_dict(sd['state_dict'] if 'state_dict' in sd else sd)
        label=f'{run}@ep{epoch}';meta={'run':run,'variant':v,'seed':seed,'epoch':epoch,'checkpoint':str(path) if path else 'random_init'}
        rows=analyse(m,label,meta,grids,seqs,tg,pca if keep_pca else None)
        with out.open('a') as f:
            for r in rows:f.write(json.dumps(r)+'\n')
        print(label,f'{time.perf_counter()-t:.0f}s',flush=True);del m;torch.cuda.empty_cache()
    if pca:np.savez_compressed(OUT/(f'pca_samples_{a.only_prefix}.npz' if a.only_prefix else 'pca_samples.npz'),**pca)

if __name__=='__main__':main()
