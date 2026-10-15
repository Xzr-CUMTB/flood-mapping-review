"""Serialize the pinned foundation comparison after the nested development queue."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, json, os, shutil, subprocess, sys, time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(Path(__file__).resolve().parent))
from project_inventory import count_bytes
OUT = ROOT/'runs/foundation_queue'


def status(stage, **extra):
    result = {'stage':stage,'pid':os.getpid(),'utc':datetime.now(timezone.utc).isoformat(),**extra}
    temporary = OUT/'status.tmp';temporary.write_text(json.dumps(result,indent=2));temporary.replace(OUT/'status.json')
    print(json.dumps(result),flush=True)


def main(wait_for_nested):
    OUT.mkdir(parents=True,exist_ok=True)
    lock = OUT/'queue.lock'
    with lock.open('x') as stream:stream.write(str(os.getpid()))
    try:
        if wait_for_nested:
            status('waiting_for_nested_queue',gpu_training_started=False)
            start=time.monotonic()
            while True:
                nested=json.loads((ROOT/'runs/nested_queue/status.json').read_text())
                if nested['stage']=='completed':break
                if nested['stage']=='failed':raise ValueError('Nested queue failed; preserve the GPU for diagnosis')
                if time.monotonic()-start>12*3600:raise TimeoutError('Nested queue did not complete within 12 hours')
                time.sleep(20)
        plan=json.loads((ROOT/'configs/foundation_experiments_s17.json').read_text())
        environment=json.loads((ROOT/'runs/foundation_setup/status.json').read_text())
        if environment['stage']!='completed':raise ValueError('Foundation runtime not ready')
        for index,job in enumerate(plan['jobs'],1):
            if (ROOT/'runs/gpu_training.lock').exists():raise ValueError('Another experiment holds the GPU lock')
            occupied=count_bytes(ROOT)
            if occupied>78_000_000_000 or shutil.disk_usage(ROOT).free<8_000_000_000:
                raise ValueError('Foundation queue storage reserve reached')
            status('running',job_index=index,job_count=len(plan['jobs']),run_id=job['run_id'],observed_workspace_bytes=occupied)
            cmd=[str(ROOT/'.venv-foundation/Scripts/python.exe'),'-s',str(ROOT/'experiments/train_foundation.py'),
                 '--run-id',job['run_id'],'--train-groups',job['train_groups'],'--val-groups',job['val_groups'],
                 '--epochs',str(plan['epochs']),'--batch-size',str(plan['batch_size']),'--seed',str(plan['seed']),
                 '--backbone-lr',str(plan['backbone_lr']),'--adaptation-lr',str(plan['adaptation_lr'])]
            if job['no_validation']:cmd.append('--no-validation')
            if job['random_init']:cmd.append('--random-init')
            with (OUT/(job['run_id']+'.log')).open('a',encoding='utf-8') as stream:
                subprocess.run(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
            result=json.loads((ROOT/'runs'/job['run_id']/'complete.json').read_text())
            if result['status']!='complete':raise ValueError('Missing completion evidence')
        status('completed',job_count=len(plan['jobs']),core_test_evaluated=False)
    except BaseException as exc:
        status('failed',error=repr(exc));raise
    finally:lock.unlink(missing_ok=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--wait-for-nested',action='store_true')
    main(parser.parse_args().wait_for_nested)
