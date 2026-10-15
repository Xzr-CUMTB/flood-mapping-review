"""Complete development calibration and the matched fusion-review control, without test access."""
from pathlib import Path
from datetime import datetime, timezone
import json, os, subprocess, sys, time
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/post_training_queue'


def status(stage,**extra):
    value={'stage':stage,'pid':os.getpid(),'utc':datetime.now(timezone.utc).isoformat(),**extra}
    temp=OUT/'status.tmp';temp.write_text(json.dumps(value,indent=2));temp.replace(OUT/'status.json')
    print(json.dumps(value),flush=True)


def await_completion(directory):
    start=time.monotonic()
    while True:
        state=json.loads((ROOT/'runs'/directory/'status.json').read_text())
        if state['stage']=='completed':return
        if state['stage']=='failed':raise ValueError(directory+' failed; dependent analysis stopped')
        if time.monotonic()-start>12*3600:raise TimeoutError(directory+' did not complete within 12 hours')
        time.sleep(20)


def execute(script,log):
    with (OUT/log).open('a',encoding='utf-8') as stream:
        subprocess.run([str(ROOT/'.venv-experiments/Scripts/python.exe'),'-s',str(ROOT/script)],cwd=ROOT,
                       stdout=stream,stderr=subprocess.STDOUT,check=True)


def main():
    OUT.mkdir(parents=True,exist_ok=True);lock=OUT/'queue.lock'
    with lock.open('x') as stream:stream.write(str(os.getpid()))
    try:
        status('waiting_for_nested_queue');await_completion('nested_queue')
        status('calibrating_development_oof',core_test_evaluated=False)
        execute('experiments/calibrate_oof.py','calibration.log')
        status('waiting_for_foundation_queue');await_completion('foundation_queue')
        if (ROOT/'runs/gpu_training.lock').exists():raise ValueError('Another process holds the GPU lock')
        status('fitting_matched_fusion_omission_control',core_test_evaluated=False)
        execute('experiments/fit_fusion_risk.py','fusion_risk.log')
        status('completed',core_test_evaluated=False,next_stage='Review all development evidence before freezing core release')
    except BaseException as exc:
        status('failed',error=repr(exc));raise
    finally:lock.unlink(missing_ok=True)


if __name__=='__main__':main()
