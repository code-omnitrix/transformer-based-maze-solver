"""Endpoint contrasts from rounded persisted outputs, not new experimental runs."""
import csv
from pathlib import Path

ROOT=Path(__file__).resolve().parent
with (ROOT/'independent_original_histories.csv').open(encoding='utf8') as f:
    history=list(csv.DictReader(f))
baseline={int(r['epoch']):r for r in history if r['model']=='normal_persisted'}
looped={int(r['epoch']):r for r in history if r['model']=='looped_persisted'}
best_a=max(baseline.values(),key=lambda r:float(r['val_token_accuracy_rounded_percent']))
best_b=max(looped.values(),key=lambda r:float(r['val_token_accuracy_rounded_percent']))
rows=[]
for contrast,a,b in [('equal_epoch30',baseline[30],looped[30]),
                     ('final_vs_final',baseline[max(baseline)],looped[max(looped)]),
                     ('best_vs_best_posthoc',best_a,best_b)]:
    ae,be=int(a['epoch']),int(b['epoch'])
    aa,ba=float(a['val_token_accuracy_rounded_percent']),float(b['val_token_accuracy_rounded_percent'])
    al,bl=float(a['train_loss_rounded_nats']),float(b['train_loss_rounded_nats'])
    rows.append(dict(contrast=contrast,source='saved_notebook_output_not_new_run',
                     baseline_epoch=ae,looped_epoch=be,
                     baseline_accuracy_percent=aa,looped_accuracy_percent=ba,
                     accuracy_gain_percentage_points=ba-aa,
                     baseline_error_percent=100-aa,looped_error_percent=100-ba,
                     relative_token_error_reduction_percent=100*(ba-aa)/(100-aa),
                     baseline_train_loss_rounded_nats=al,looped_train_loss_rounded_nats=bl,
                     train_loss_reduction_nats=al-bl,
                     baseline_updates_if50k_batch128=ae*391,looped_updates_if50k_batch128=be*391,
                     baseline_updates_normalized_to_epoch30=ae/30,looped_updates_normalized_to_epoch30=be/30,
                     baseline_example_presentations_if50k=ae*50000,looped_example_presentations_if50k=be*50000,
                     baseline_summed_block_calls_proxy=ae*391*8,looped_summed_block_calls_proxy=be*391*12,
                     block_calls_note='8 baseline vs12 looped encoder+decoder calls/update; NOT FLOPs or timing',
                     historical_rng_note='Only cell0 seeds42; neither training cell resets RNG; loop initialization seed/state historically unknown'))
with (ROOT/'independent_original_comparison.csv').open('w',newline='',encoding='utf8') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
for row in rows:print(row)
