"""Independent CPU-only review. Writes only its own scratch and result files."""
import collections
import hashlib
import json
import os
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import torch
import numpy as np
import experiment_suite as suite
import fullscale_experiments as full

ROOT=Path(__file__).resolve().parent
torch.set_num_threads(1)
result={'cuda_visible_devices':os.environ['CUDA_VISIBLE_DEVICES'],
        'fullscale_script_sha256':hashlib.sha256((ROOT/'fullscale_experiments.py').read_bytes()).hexdigest(),
        'suite_script_sha256':hashlib.sha256((ROOT/'experiment_suite.py').read_bytes()).hexdigest()}
scratch=ROOT/'fullscale_review_scratch'
(scratch/'data').mkdir(parents=True,exist_ok=True)
full.ROOT=scratch
datasets,metadata=full.full_data()
independent=json.loads((ROOT/'independent_statistics.json').read_text())
result['data_checks']={}
for current,prior in [('train','train'),('val','validation')]:
    grids,seqs=datasets[current].tensors
    gridhash=hashlib.sha256(grids.numpy().astype(np.uint8).tobytes()).hexdigest()
    hist={str(k):v for k,v in collections.Counter(((seqs!=0).sum(1)-2).tolist()).items()}
    checks={'grid_uint8_array_sha256':gridhash,
            'grid_hash_matches_independent':gridhash==independent[prior]['array_sha256'],
            'path_histogram_matches_independent':hist==independent[prior]['path_histogram'],
            'sample_count':len(grids),'target_tokens':int((seqs[:,1:]!=0).sum()),
            'sequence_columns_including_BOS':int(seqs.shape[1])}
    result['data_checks'][current]=checks
    assert checks['grid_hash_matches_independent'] and checks['path_histogram_matches_independent']
result['model_checks']={}
base,loop=suite.original_classes()
grid=torch.tensor(datasets['val'].tensors[0][:2].numpy(),dtype=torch.long)
seq=datasets['val'].tensors[1][:2,:8]
for variant,cls in [('original_baseline',base),('original_looped',loop)]:
    suite.seed_all(42);factory=suite.make_model(variant)
    suite.seed_all(42);direct=cls()
    exact_state=(factory.state_dict().keys()==direct.state_dict().keys() and
                 all(torch.equal(v,direct.state_dict()[k]) for k,v in factory.state_dict().items()))
    factory.eval();direct.eval()
    with torch.no_grad():
        a=factory(grid,seq);b=direct(grid,seq)
        adapted=factory.decode(seq,factory.encode(grid))
    evaldiff=float((a-b).abs().max());adapterdiff=float((a-adapted).abs().max())
    factory.train();direct.train()
    suite.seed_all(77)
    with torch.no_grad():a=factory(grid,seq)
    suite.seed_all(77)
    with torch.no_grad():b=direct(grid,seq)
    traindiff=float((a-b).abs().max())
    checks={'initial_state_identical':exact_state,'eval_forward_max_abs_difference':evaldiff,
            'adapter_decode_max_abs_difference':adapterdiff,'train_forward_synced_dropout_max_abs_difference':traindiff,
            'parameters':sum(p.numel() for p in factory.parameters())}
    result['model_checks'][variant]=checks
    assert exact_state and evaldiff==0 and adapterdiff==0 and traindiff==0
result['declarations']=full.DECLARED
result['status']='all_cpu_checks_passed'
(ROOT/'fullscale_review_checks.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps(result,indent=2))
