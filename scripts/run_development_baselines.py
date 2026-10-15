"""Sequential, bounded experiment queue after the complete data and split audits."""
from pathlib import Path
from datetime import datetime,timezone
import json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/development_queue';OUT.mkdir(parents=True,exist_ok=True)
LOCK=OUT/'queue.lock'
def status(stage,**extra):
    value={'stage':stage,'pid':os.getpid(),'utc':datetime.now(timezone.utc).isoformat(),**extra}
    (OUT/'status.json').write_text(json.dumps(value,indent=2));print(json.dumps(value),flush=True)

if __name__=='__main__':
    with LOCK.open('x') as handle:handle.write(str(os.getpid()))
    try:
        start=time.monotonic();marker=ROOT/'data/processed/sen2gf3/split_audit_passed.json'
        while not marker.exists():
            if time.monotonic()-start>10800:raise TimeoutError('Data audit incomplete')
            status('waiting_for_complete_data_and_spatial_audit');time.sleep(30)
        for modality in ['sar','fusion','rgb_fusion']:
            run_id='dev_'+modality+'_seed17';status('training',run_id=run_id)
            with (OUT/(run_id+'.log')).open('a',encoding='utf-8') as log:
                subprocess.run([sys.executable,'-s',str(ROOT/'experiments/train_baseline.py'),'--run-id',run_id,'--modality',modality,
                                '--epochs','20','--patience','6','--batch-size','16'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        status('paired_development_evaluation')
        with (OUT/'paired_evaluation.log').open('w',encoding='utf-8') as log:
            subprocess.run([sys.executable,'-s',str(ROOT/'experiments/evaluate_development_pair.py')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        status('completed',core_test_evaluated=False)
    except BaseException as exc:
        status('failed',error=repr(exc));raise
    finally:LOCK.unlink(missing_ok=True)
