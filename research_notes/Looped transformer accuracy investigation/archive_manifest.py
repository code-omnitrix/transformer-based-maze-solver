"""Build an auditable file inventory after analyses finish. Does not alter raw data."""
import datetime,hashlib,json
from pathlib import Path
from experiment_suite import ROOT,NOTEBOOK,write_json

def main():
    files=[]
    for p in sorted(ROOT.rglob('*')):
        if not p.is_file() or '__pycache__' in p.parts or p.name=='archive_manifest.json':continue
        rel=p.relative_to(ROOT).as_posix()
        category='new_research_documentation'
        if rel.startswith('audit/cell_') or rel=='audit/original_histories.csv':category='historical_notebook_extraction'
        elif rel.startswith('runs/'):category='new_training_and_evaluation'
        elif rel.startswith('data/'):category='new_saved_datasets'
        elif rel.startswith('analysis/'):category='derived_analysis'
        elif rel.startswith('source/'):category='immutable_training_source'
        sha=hashlib.sha256()
        with p.open('rb') as f:
            for chunk in iter(lambda:f.read(8*1024*1024),b''):sha.update(chunk)
        files.append({'path':rel,'size_bytes':p.stat().st_size,'sha256':sha.hexdigest(),'category':category})
    out={'generated_utc':datetime.datetime.now(datetime.UTC).isoformat(),'root':str(ROOT),
         'notebook_path':str(NOTEBOOK),'notebook_sha256':hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest(),
         'files':files,'total_size_bytes':sum(x['size_bytes'] for x in files)}
    write_json(ROOT/'archive_manifest.json',out);print(json.dumps({'files':len(files),'total_size_bytes':out['total_size_bytes'],'path':str(ROOT/'archive_manifest.json')}))

if __name__=='__main__':main()
