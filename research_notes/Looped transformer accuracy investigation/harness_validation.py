"""Independent CPU checks. Does not modify experiment_suite.py or its runs."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('reviewed_suite', ROOT/'experiment_suite.py')
suite = importlib.util.module_from_spec(spec)
spec.loader.exec_module(suite)
torch.set_num_threads(1)

@torch.no_grad()
def mapped_forward_check(which):
    suite.seed_all(987)
    original = suite.make_model('original_baseline' if which=='postln4' else 'original_looped')
    controlled = suite.make_model(which)
    for name in ['maze_token_emb','row_emb','col_emb','action_token_emb','fc_out']:
        getattr(controlled,name).load_state_dict(getattr(original,name).state_dict())
    controlled.pos_emb.weight[:len(original.pos_emb.weight)].copy_(original.pos_emb.weight)
    if which=='postln4':
        for i in range(4):
            controlled.encoder[i].load_state_dict(original.transformer.encoder.layers[i].state_dict())
            controlled.decoder[i].load_state_dict(original.transformer.decoder.layers[i].state_dict())
        controlled.encoder_norm.load_state_dict(original.transformer.encoder.norm.state_dict())
        controlled.decoder_norm.load_state_dict(original.transformer.decoder.norm.state_dict())
    else:
        enc,dec=original.transformer['encoder'],original.transformer['decoder']
        controlled.encoder[0].load_state_dict(enc.shared_block.state_dict())
        controlled.decoder[0].load_state_dict(dec.shared_block.state_dict())
        controlled.encoder_norm.load_state_dict(enc.final_norm.state_dict())
        controlled.decoder_norm.load_state_dict(dec.final_norm.state_dict())
        controlled.encoder_gate.copy_(enc.input_gate)
        controlled.decoder_gate.copy_(dec.input_gate)
    original.eval();controlled.eval()
    grid=torch.randint(0,4,(2,8,8))
    seq=torch.tensor([[1,3,4,5,6,2,0],[1,4,5,6,2,0,0]])
    err=(original(grid,seq)-controlled(grid,seq)).abs().max().item()
    assert err < 1e-5, (which,err)
    return err

@torch.no_grad()
def causal_check():
    results={}
    grid=torch.randint(0,4,(2,8,8))
    seq=torch.tensor([[1,3,4,5,6,2],[1,4,5,3,6,2]])
    changed=seq.clone();changed[:,3:]=torch.tensor([6,3,4])
    for name in suite.VARIANTS:
        suite.seed_all(42)
        model=suite.make_model(name).eval()
        difference=(model(grid,seq)[:,:3]-model(grid,changed)[:,:3]).abs().max().item()
        assert difference < 1e-5,(name,difference)
        results[name]=difference
    return results

def trajectory_check():
    grid=np.array([[2,0],[0,3]])
    target=[5,6,2]
    cases={
        'canonical':([5,6,2],True,True),
        'alternative_shortest':([6,5,2],True,True),
        'legal_longer':([5,4,6,5,2],True,False),
        'no_eos':([5,6],False,False),
        'illegal_then_eos':([3,2],False,False),
        'pad_then_eos':([0,2],False,False),
        'premature_eos':([2],False,False),
    }
    output={}
    for name,(pred,solved,optimal) in cases.items():
        value=suite.path_metrics(grid,pred,target)
        assert value['solved']==solved and value['optimal_solved']==optimal,(name,value)
        output[name]=value
    assert not output['alternative_shortest']['rollout_exact']
    # Explicitly preserve the reporting caveat rather than silently fix the harness.
    assert output['illegal_then_eos']['terminated'] is False
    return output

def saved_metrics_check():
    out=ROOT/'runs'/'controlled12k_postln4_seed42'
    summary=json.loads((out/'summary.json').read_text())
    teacher=[json.loads(s) for s in (out/'test_teacher_predictions.jsonl').read_text().splitlines()]
    assert sum(r['tokens'] for r in teacher)==summary['teacher_test']['tokens']
    assert sum(r['correct'] for r in teacher)==summary['teacher_test']['correct']
    assert sum(r['exact'] for r in teacher)==summary['teacher_test']['exact']
    rollout=[json.loads(s) for s in (out/'test_rollout_predictions.jsonl').read_text().splitlines()]
    emitted_eos=sum(2 in r['prediction'] for r in rollout)
    counted_termination=sum(r['terminated'] for r in rollout)
    for key in ['rollout_exact','valid_path','terminated','goal_reached','solved','optimal_solved']:
        assert abs(sum(r[key] for r in rollout)/len(rollout)-summary['rollout_test'][key]) < 1e-12
    steps=[json.loads(s) for s in (out/'steps.jsonl').read_text().splitlines()]
    epochs=[json.loads(s) for s in (out/'epochs.jsonl').read_text().splitlines()]
    for epoch in epochs:
        subset=[r for r in steps if r['epoch']==epoch['epoch']]
        nll=sum(r['pre_update_teacher_forced_token_mean_nll']*r['target_tokens'] for r in subset)
        assert abs(nll/epoch['train_tokens']-epoch['train_token_mean_loss']) < 1e-10
        assert subset[-1]['optimizer_step']==epoch['optimizer_steps']
    return {'teacher_rows':len(teacher),'rollout_rows':len(rollout),
            'generated_EOS_count':emitted_eos,'harness_terminated_count':counted_termination,
            'training_steps':len(steps),'epochs':len(epochs),
            'all_summary_aggregations_match_saved_rows':True,
            'token_weighted_epoch_NLL_matches_steps':True}

result={'scope':'CPU read-only validation of running harness; no training or existing-file changes',
        'mapped_original_forward_max_abs_difference':{v:mapped_forward_check(v) for v in ['postln4','tied6_full']},
        'controlled_future_perturbation_prefix_max_abs_difference':causal_check(),
        'trajectory_cases':trajectory_check(),
        'saved_completed_run_crosschecks':saved_metrics_check()}
(ROOT/'harness_validation_results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
