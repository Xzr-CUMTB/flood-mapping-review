"""One finite local experiment continuation: dependencies, audit, frozen inference, results."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.project_inventory import count_bytes
OUT=ROOT/'runs/core_completion_queue'
PY=ROOT/'.venv-experiments/Scripts/python.exe'
FPY=ROOT/'.venv-foundation/Scripts/python.exe'


def read(path):return json.loads(path.read_text())


def write(path,value):
    temp=path.with_suffix('.json.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)


def status(stage,**extra):
    value={'stage':stage,'pid':os.getpid(),'utc':datetime.now(timezone.utc).isoformat(),**extra}
    write(OUT/'status.json',value);print(json.dumps(value),flush=True)


def execute(script,args=(),python=PY):
    name=Path(script).stem
    with (OUT/(name+'.log')).open('a',encoding='utf-8') as stream:
        subprocess.run([str(python),'-s',str(ROOT/script),*args],cwd=ROOT,
                       stdout=stream,stderr=subprocess.STDOUT,check=True)


def await_training():
    names=['nested_queue','foundation_queue','post_training_queue','ensemble_queue']
    start=time.monotonic()
    while True:
        states={name:read(ROOT/'runs'/name/'status.json') for name in names}
        failures={name:value for name,value in states.items() if value['stage']=='failed'}
        if failures:raise ValueError('Preceding experiment queue failed: '+json.dumps(failures))
        if all(value['stage']=='completed' for value in states.values()):return
        if time.monotonic()-start>18*3600:raise TimeoutError('Training queues did not finish within 18 hours')
        time.sleep(20)


def prepare_review():
    policy=read(ROOT/'configs/core_execution_policy_s17.json')
    assert policy['core_predictions_inspected_when_policy_written'] is False
    audit=read(ROOT/'reports/training_artifact_audit.json')
    assert audit['passed'] and audit['verified_jobs']==audit['planned_jobs']==38
    folds={str(k):read(ROOT/'runs'/f'nested_development_fold{k}'/'complete.json') for k in range(4)}
    assert all(v['status']=='complete' and v['core_test_evaluated'] is False for v in folds.values())
    calibration=read(ROOT/'runs/oof_calibration_s17/complete.json')
    assert calibration['status']=='complete' and calibration['core_test_evaluated'] is False
    assert calibration['samples']==8786 and set(calibration['operating_points'])=={'matched_fpr','matched_recall'}
    for name,temperature in calibration['temperatures'].items():
        assert 0<temperature['temperature']<100,name
    review={'passed':True,'reviewed_at_utc':datetime.now(timezone.utc).isoformat(),
        'core_test_predictions_inspected':False,'review_type':policy['review_type'],
        'execution_policy':policy,'completed_training_artifacts':audit['verified_jobs'],
        'nested_development_folds':folds,'calibration_temperatures':calibration['temperatures'],
        'acceptance_criterion':'Completion, provenance and arithmetic validity; no development-performance gate',
        'additional_frozen_files':['configs/core_execution_policy_s17.json','reports/label_semantics_audit.json',
            'reports/retained_support_audit.json','scripts/run_core_completion.py','scripts/summarize_core_results.py'],
        'additional_comparisons':['TerraMind pretrained/random controls','Matched fusion omission head',
            'Three-member fusion uncertainty ensemble','Development-matched FPR/recall operating points']}
    write(ROOT/'reports/core_release_review_s17.json',review)


def main():
    OUT.mkdir(parents=True,exist_ok=True);lock=OUT/'queue.lock'
    with lock.open('x') as stream:stream.write(str(os.getpid()))
    try:
        status('waiting_for_all_prescribed_training');await_training()
        if (ROOT/'runs/gpu_training.lock').exists():raise ValueError('GPU still locked after training completion')
        status('auditing_completed_artifacts',core_test_evaluated=False)
        execute('scripts/audit_completed_training.py',['--require-complete'])
        execute('experiments/evaluate_nested_development.py')
        execute('scripts/summarize_nested_progress.py')
        if not (ROOT/'configs/core_release_s17.json').exists():prepare_review()
        execute('scripts/freeze_core_release.py')
        status('predicting_frozen_core',workspace_bytes=count_bytes(ROOT))
        execute('experiments/infer_frozen_core.py')
        status('predicting_foundation_core',workspace_bytes=count_bytes(ROOT))
        execute('experiments/infer_foundation_core.py',python=FPY)
        status('evaluating_frozen_predictions',workspace_bytes=count_bytes(ROOT))
        execute('experiments/evaluate_frozen_core.py')
        execute('scripts/summarize_core_results.py')
        execute('scripts/summarize_nested_progress.py')
        status('completed',workspace_bytes=count_bytes(ROOT),core_test_evaluated=True,
               result_report='reports/13_core_computational_results.md',figure_generation=False)
    except BaseException as exc:
        status('failed',error=repr(exc));raise
    finally:lock.unlink(missing_ok=True)


if __name__=='__main__':main()
