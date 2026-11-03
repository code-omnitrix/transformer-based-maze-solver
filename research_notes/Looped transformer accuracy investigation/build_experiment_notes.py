"""Regenerate experiment evidence notes after analyses; never selects checkpoints."""
import csv,datetime,json
from experiment_suite import ROOT

def table(rows,cols):
    return '\n'.join(['| '+' | '.join(h for h,_ in cols)+' |','| '+' | '.join('---' for _ in cols)+' |']+
        ['| '+' | '.join(str(fn(r)) for _,fn in cols)+' |' for r in rows])

def main():
    ap=ROOT/'analysis'/'controlled12k';fp=ROOT/'analysis'/'full50k'
    rows=list(csv.DictReader((ap/'architecture_summary.csv').open(encoding='utf-8')))
    fmt=lambda r,k:f"{100*float(r[k+'_mean']):.3f} ± {100*float(r[k+'_sd'] or 0):.3f}"
    parts=['# Controlled experiments and fresh full-data runs',
        f"\nGenerated UTC: {datetime.datetime.now(datetime.UTC).isoformat()}",
        '\n## What does the controlled matrix show?',
        '\n### Takeaway\nKNOWN, high confidence about recorded runs: these are fresh controlled-family experiments, not historical notebook outputs. All endpoints are fixed epoch30. Effect sizes apply to one fixed12k training sample and the specified optimizer/settings. Three architecture seeds are not three independently sampled datasets.',
        '\n### Cited Findings\nAccuracies below are percentage units; ± is sample SD across recorded architecture seeds, not a confidence interval. Teacher test uses1000 mazes; autonomous rollout uses first512 fixed test mazes.',
        table(rows,[('Variant',lambda r:r['variant']),('Seeds',lambda r:r['n_seeds']),('Parameters',lambda r:r['parameters']),
              ('Teacher token % ±SD',lambda r:fmt(r,'test_token_accuracy')),('First action % ±SD',lambda r:fmt(r,'test_first_action_accuracy')),
              ('Teacher exact % ±SD',lambda r:fmt(r,'test_teacher_exact')),('Rollout optimal % ±SD',lambda r:fmt(r,'test_optimal_solved')),
              ('Training seconds mean',lambda r:f"{float(r['train_seconds_mean']):.2f}")]),
        '\nSources: [architecture summaryCSV](analysis/controlled12k/architecture_summary.csv), [per-seed final CSV](analysis/controlled12k/final_runs.csv), [all epochsCSV](analysis/controlled12k/all_epochs.csv), [frozen study plan](STUDY_PLAN.md).',
        '\nPrimary contrast and secondary contrasts, paired on the same examples and architecture seeds:',
        '']
    comparisons=json.loads((ap/'paired_comparisons.json').read_text())
    parts.append(table(comparisons,[('Contrast (after−before)',lambda r:r['after']+' − '+r['before']),('Paired seeds',lambda r:','.join(map(str,r['paired_seeds']))),
        ('Token effect pp',lambda r:f"{100*r['teacher_token']['difference']:.3f}"),
        ('Seed effects pp',lambda r:', '.join(f'{100*x:.3f}' for x in r['seed_token_effects'])),
        ('Conditional maze-bootstrap95% pp',lambda r:', '.join(f'{100*x:.3f}' for x in r['teacher_token']['conditional_maze_bootstrap_95pct'])),
        ('Optimal rollout effect pp',lambda r:f"{100*r['rollout_optimal']['difference']:.3f}")]))
    parts+=['\nSource: [paired comparisonsJSON](analysis/controlled12k/paired_comparisons.json). Wholemaze bootstrap conditions on the fitted seed ensemble; it does not quantify retraining or new-train-data uncertainty. With only3seeds, even all3paired differences sharing a sign gives minimum two-sided sign-testp=.25.',
        '\n### Inferences\nNormalization, anchoring, tying and iteration encodings must be interpreted through their own matched contrasts. The fixed-learning-rate/no-warmup policy may favor some parameterizations. A lower parametercount alone does not imply lower compute, and per-seed sharing effects can vary in sign. Gradient/gate curves describe optimization dynamics without isolating a causal mechanism.',
        '\n### Gaps\nNo parameter-matched narrow untied model, independent training-dataset replications, model-specific optimizer tuning, larger-grid training/evaluation, or mechanistic BFS proof. Early-budget pilots cannot substitute for full50k near-convergence comparisons.',
        '\n## Does the result persist at the original data scale?',
        '\n### Takeaway\nThe fresh full-data extension is distinct from the controlled pilot and from historical output. The exact original generator is replayedseed42 fortrain50k/val5k, then a newtest5k. Per-model initializationseed42 and dedicatedshuffle1042 are explicit fresh-run choices; historical execution state is unknown.',
        '\n### Cited Findings']
    full=[]
    for d in sorted((ROOT/'runs').glob('full50k_*')):
        if not(d/'summary.json').exists():continue
        s=json.loads((d/'summary.json').read_text());h=[json.loads(x) for x in (d/'epochs.jsonl').read_text().splitlines()]
        full.append({'variant':d.name,'epoch':h[-1]['epoch'],'val':h[-1]['val_token_accuracy'],'summary':s})
        e=d/'epoch30_evaluation'/'summary.json'
        if e.exists():full.append({'variant':d.name+'_fixed30','epoch':30,'val':h[29]['val_token_accuracy'],'summary':json.loads(e.read_text())})
    if full:
        parts.append(table(full,[('Fresh run',lambda r:r['variant']),('Epoch',lambda r:r['epoch']),
              ('Validation token %',lambda r:f"{100*r['val']:.3f}"),('Test token %',lambda r:f"{100*r['summary']['teacher_test']['token_accuracy']:.3f}"),
              ('Teacher exact %',lambda r:f"{100*r['summary']['teacher_test']['exact_accuracy']:.3f}"),
              ('Rollout optimal %',lambda r:f"{100*r['summary']['rollout_test']['optimal_solved']:.3f}"),
              ('Rollout canonical exact %',lambda r:f"{100*r['summary']['rollout_test']['rollout_exact']:.3f}"),
              ('Training seconds',lambda r:f"{r['summary']['train_seconds']:.2f}")]))
    else:parts.append('Full-data runs not yet complete at this notes generation. Do not present a planned experiment as a result.')
    parts+=['\nSources: [full-data analysis](analysis/full50k/), individual `runs/full50k_*/summary.json`, `epochs.jsonl` and original baseline `epoch30_evaluation/summary.json`; [fullscale specification](fullscale_plan_record.json). Fulltest rollouts1000fixed mazes; teacher test5000; single modelseed pervariant.',
        '\n### Inferences\nUse the controlled tiedfull−untiedfull comparison to assess sharing at matched6depth/preLN/anchor/indexencoding/initializationfamily. Use originalbaseline30vsloop30 to assess the changed architecture bundle at equalupdates, and baseline90vsloop30 to compare the user\'s original training schedules. Distinguish validation from heldouttest in every numerical claim.',
        '\n### Gaps\nOnlyone full-data architectureseed, with fixeddata; inability to recover originalcheckpoint/RNGstate prevents claiming historical bitwise replication. Different initialization policies between original loop and controlledfamily are intentional and logged.',
        '\n## How can future analysis reproduce and audit the results?',
        '\n### Takeaway\nRaw epoch/update histories, exact datasets, configs, source snapshots, checkpoints/RNG, and permaze teacher/rollout predictions are retained. Every plot is regenerated by a plain script without notebookglobalstate.',
        '\n### Cited Findings\nSee [README](README.md), [research ledger](research_log.md), [study plan](STUDY_PLAN.md), [run registry](run_registry.jsonl), [resume validation](resume_validation.json), [harness review](experiment_harness_review.md), and archive_manifest.json when finalized. Accuracies inrawCSV/JSON are fractions0–1; plots/thisnotes table usepercent. Loss is natural-log crossentropy pernonPADtargettoken; timesseconds.',
        '\n### Inferences\nSaved predictions permit correcting metric semantics, evaluating firstaction/BOS-only accuracy, classifying failure modes, accepting alternative shortestpaths, and bootstrapping wholemaze effects without retraining. The savedonline training NLL is dropout-active repeated-data fitting loss, explicitly not an epiplexity estimate.',
        '\n### Gaps\nMatrixcheckpoints are finalonly; fullruns additionally snapshot latest every10epochs. CPUresume matches exactly; CUDA bitwise reproducibility is not guaranteed. Localwalltime reflects fixedvariantorder, laptopthermal/power/backgroundconditions, and does not measureenergy. Raw dataset `path_length_histogram` includesBOS/EOS; derivedactionlength subtracts2. Raw `terminated` means EOSbeforefirstillegaltraceaction; actualEOSemission is reconstructed fromtokens. Neither caveat changes strict solved/optimalpathlabels.']
    (ROOT/'experiments.md').write_text('\n\n'.join(parts)+'\n',encoding='utf-8')
    print(json.dumps({'path':str(ROOT/'experiments.md'),'matrix_architectures':len(rows),'matrix_seeds':[r['n_seeds'] for r in rows],'full_run_records':len(full)}))

if __name__=='__main__':main()
