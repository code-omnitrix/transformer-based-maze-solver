"""Flatten existing teacher-forced depth sweeps; no new model evaluation."""
import argparse,csv,json
from experiment_suite import ROOT

def main():
    p=argparse.ArgumentParser();p.add_argument('--prefix',default='controlled12k');a=p.parse_args();rows=[]
    for p in sorted((ROOT/'runs').glob(a.prefix+'_*'+'/summary.json')):
        s=json.loads(p.read_text())
        for k,v in s.get('depth_sweep',{}).items():
            rows.append({'run_id':p.parent.name,'inference_depth':int(k),'trained_depth':6,
                **v,'eos_accuracy':v['eos_correct']/v['eos'],
                'scope':'Exploratory teacher-forced same checkpoint; encoder and decoder depth both change; no rollout or adaptive halting'})
    path=ROOT/'analysis'/a.prefix/'depth_sweep_metrics.csv'
    if rows:
        with path.open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps({'rows':len(rows),'path':str(path)}))

if __name__=='__main__':main()
