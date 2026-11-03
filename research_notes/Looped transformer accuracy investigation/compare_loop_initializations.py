"""CPU parameter mapping: original loop vs matched controlled full loop.

Only inspects freshly initialized models; does not change fitted checkpoints.
Initial CUDA RNG hashes are captured, but dropout masks are not traced through
the entire GPU training trajectory.
"""
import hashlib,json
import torch
from experiment_suite import ROOT,make_model,seed_all,write_json

def rng_hash(state):return hashlib.sha256(state.numpy().tobytes()).hexdigest()

def main():
    seed_all(42);a=make_model('original_looped');cpu_a=rng_hash(torch.get_rng_state());cuda_a=[rng_hash(x) for x in torch.cuda.get_rng_state_all()]
    seed_all(42);b=make_model('tied6_full');cpu_b=rng_hash(torch.get_rng_state());cuda_b=[rng_hash(x) for x in torch.cuda.get_rng_state_all()]
    mapping={k:k for k in a.state_dict() if k in b.state_dict()}
    for old,new in [('transformer.encoder.shared_block.','encoder.0.'),('transformer.decoder.shared_block.','decoder.0.'),
                    ('transformer.encoder.final_norm.','encoder_norm.'),('transformer.decoder.final_norm.','decoder_norm.')]:
        for k in a.state_dict():
            if k.startswith(old):mapping[k]=new+k[len(old):]
    mapping.update({'transformer.encoder.input_gate':'encoder_gate','transformer.decoder.input_gate':'decoder_gate',
                    'grid_rows':'rows','grid_cols':'cols','transformer.encoder.loop_emb.div_term':'div',
                    'transformer.decoder.loop_emb.div_term':'div'})
    rows=[]
    for old,new in mapping.items():
        x,y=a.state_dict()[old],b.state_dict()[new]
        rows.append({'original_key':old,'controlled_key':new,'shape':list(x.shape),'identical':torch.equal(x,y),
                     'max_abs_difference':float((x-y).abs().max()),'original_std':float(x.float().std()) if x.numel()>1 else None,
                     'controlled_std':float(y.float().std()) if y.numel()>1 else None})
    params=set(dict(a.named_parameters()));mapped_params=[r for r in rows if r['original_key'] in params]
    missing=params-{r['original_key'] for r in mapped_params}
    result={'scope':'Fresh per-modelseed42 original_looped vs controlledtied6_full; same shapes/forward graph under mapped weights',
            'trainable_parameter_count_original':sum(p.numel() for p in a.parameters()),'trainable_parameter_count_controlled':sum(p.numel() for p in b.parameters()),
            'mapped_trainable_tensors':len(mapped_params),'missing_trainable_keys':sorted(missing),
            'different_trainable_tensors':[r for r in mapped_params if not r['identical']],
            'identical_noncore_tensors':[r['original_key'] for r in mapped_params if r['identical'] and 'shared_block' not in r['original_key']],
            'cpu_rng_after_model_creation_equal':cpu_a==cpu_b,'cuda_rng_after_model_creation_equal':cuda_a==cuda_b,
            'cpu_rng_hashes':[cpu_a,cpu_b],'cuda_rng_hashes':[cuda_a,cuda_b],
            'interpretation':'Core matrix initialization policy/draws differ; embeddings/output/all1Dparameters match atseed42. CPU RNG advances differently; shared dedicatedshuffle seed1042 and sameinitial CUDA RNG documented. Per-step CUDA dropout mask identity not independently traced; singlemodelseed does not isolate which initialization component matters.',
            'all_mapped_tensors':rows}
    assert not missing
    assert all(r['original_key'].startswith('transformer.') and 'shared_block' in r['original_key'] and len(r['shape'])>1 for r in mapped_params if not r['identical'])
    write_json(ROOT/'original_loop_vs_controlled_initialization.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['all_mapped_tensors','different_trainable_tensors']}))
    print('differing matrices:',len(result['different_trainable_tensors']))

if __name__=='__main__':main()
