"""Read-only GPU telemetry during remaining full-data runs. No GPU kernels.

Starts after full-data baseline already began, so it is incomplete for early
epochs and absent for the controlled matrix. Stops on fullscale_complete.json.
"""
import argparse,csv,datetime,json,subprocess,time
from experiment_suite import ROOT,append_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--interval',type=float,default=30);a=p.parse_args()
    keys=['name','driver_version','temperature.gpu','power.draw','power.limit','utilization.gpu','memory.used','memory.total','clocks.sm','clocks.mem','pstate']
    path=ROOT/'fullscale_gpu_telemetry.jsonl'
    append_json(path,{'event':'logger_started','utc':datetime.datetime.now(datetime.UTC).isoformat(),
       'scope':'Partial full-data telemetry, absent for matrix and early full-data epochs; sensors only, no GPU computation','units':'temperature C; power W; utilization percent; memory MiB; clocks MHz'})
    while not(ROOT/'fullscale_complete.json').exists():
        t=datetime.datetime.now(datetime.UTC).isoformat()
        cp=subprocess.run(['nvidia-smi','--query-gpu='+','.join(keys),'--format=csv,noheader,nounits'],capture_output=True,text=True)
        if cp.returncode==0:
            for vals in csv.reader(cp.stdout.splitlines()):append_json(path,{'utc':t,**dict(zip(keys,[v.strip() for v in vals]))})
        else:append_json(path,{'utc':t,'error':cp.stderr.strip(),'returncode':cp.returncode})
        time.sleep(a.interval)
    append_json(path,{'event':'logger_stopped','utc':datetime.datetime.now(datetime.UTC).isoformat(),'reason':'fullscale_complete.json observed'})

if __name__=='__main__':main()
