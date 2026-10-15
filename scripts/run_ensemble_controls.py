"""Fit two additional prescribed fusion seeds after earlier GPU queues finish."""
from pathlib import Path
from datetime import datetime,timezone
import json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/ensemble_queue'


def status(stage,**extra):
    value={'stage':stage,'pid':os.getpid(),'utc':datetime.now(timezone.utc).isoformat(),**extra}
    temp=OUT/'status.tmp';temp.write_text(json.dumps(value,indent=2));temp.replace(OUT/'status.json');print(json.dumps(value),flush=True)


def main():
    OUT.mkdir(parents=True,exist_ok=True);lock=OUT/'queue.lock'
    with lock.open('x') as stream:stream.write(str(os.getpid()))
    try:
        status('waiting_for_post_training_queue');start=time.monotonic()
        while True:
            dependency=json.loads((ROOT/'runs/post_training_queue/status.json').read_text())
            if dependency['stage']=='completed':break
            if dependency['stage']=='failed':raise ValueError('Post-training dependency failed')
            if time.monotonic()-start>12*3600:raise TimeoutError('Preceding queues did not finish')
            time.sleep(20)
        plan=json.loads((ROOT/'configs/ensemble_controls.json').read_text())
        for seed,run_id in zip(plan['seeds'][1:],plan['new_members']):
            if (ROOT/'runs/gpu_training.lock').exists():raise ValueError('Another process holds the GPU lock')
            status('running',run_id=run_id,seed=seed)
            command=[str(ROOT/'.venv-experiments/Scripts/python.exe'),'-s',str(ROOT/'experiments/train_baseline.py'),
                '--run-id',run_id,'--modality','fusion','--train-groups',plan['train_groups'],'--val-groups','',
                '--no-validation','--selection','fixed_last','--epochs',str(plan['epochs']),
                '--batch-size',str(plan['batch_size']),'--lr',str(plan['learning_rate']),'--seed',str(seed)]
            with (OUT/(run_id+'.log')).open('a',encoding='utf-8') as stream:
                subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
            if json.loads((ROOT/'runs'/run_id/'complete.json').read_text())['status']!='complete':raise ValueError('Missing completion record')
        status('completed',core_test_evaluated=False,additional_training_runs=2)
    except BaseException as exc:
        status('failed',error=repr(exc));raise
    finally:lock.unlink(missing_ok=True)


if __name__=='__main__':main()
