"""Regenerate original initialization stats and exact norm-only control check."""
from experiment_suite import ROOT,original_classes,seed_all,make_model,write_json

def main():
    result={};base,loop=original_classes()
    for name,cls in [('baseline',base),('looped',loop)]:
        seed_all(42);m=cls()
        enc=m.transformer.encoder.layers[0] if name=='baseline' else m.transformer['encoder'].shared_block
        result[name]={k:{'shape':list(v.shape),'mean':float(v.detach().mean()),'std':float(v.detach().std())} for k,v in enc.named_parameters() if v.ndim>1}
    write_json(ROOT/'original_initialization_diagnostics.json',{'scope':'Fresh independently seed42-instantiated saved definitions; not historical fitted weights','encoder_first_block':result})
    seed_all(42);a=make_model('postln4');seed_all(42);b=make_model('preln4')
    diffs={k:float((v-b.state_dict()[k]).abs().max()) for k,v in a.state_dict().items()}
    write_json(ROOT/'normalization_control_validation.json',{'same_seed_postln4_preln4_max_state_difference':max(diffs.values()),
               'different_keys':[k for k,v in diffs.items() if v!=0],
               'known_difference':'norm_first boolean affects forward placement only; trainable parameters and all initialized tensors identical'})

if __name__=='__main__':main()
