"""Execute the reviewed nested-OOF plan on one GPU, with restart-safe run IDs."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,os,shutil,subprocess,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parent))
from project_inventory import count_bytes

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/nested_queue'


def ordered_jobs(plan):
    by_id={job['run_id']:job for job in plan['jobs']};done=set();ordered=[]
    for job in plan['jobs']:
        if job['type']!='head':continue
        spec=json.loads((ROOT/job['spec']).read_text())
        for pair in spec.get('pairs',[])+spec.get('predict_pairs',[]):
            for key in ['sar_run','fusion_run']:
                name=pair[key]
                if name not in done:ordered.append(by_id[name]);done.add(name)
        for name in spec.get('system_runs',[]):
            if name not in done:raise ValueError('Whole-system OOF dependency not scheduled')
        ordered.append(job);done.add(job['run_id'])
    for job in plan['jobs']:
        if job['run_id'] not in done:ordered.append(job);done.add(job['run_id'])
    return ordered


def write_status(stage,**extra):
    value={'stage':stage,'pid':os.getpid(),'utc':datetime.now(timezone.utc).isoformat(),**extra}
    temp=OUT/'status.tmp';temp.write_text(json.dumps(value,indent=2));temp.replace(OUT/'status.json')
    print(json.dumps(value),flush=True)


def run(args):
    plan=json.loads(Path(args.plan).read_text())
    manifest=ROOT/'data/processed/sen2gf3/manifest.csv'
    if hashlib.sha256(manifest.read_bytes()).hexdigest()!=plan['manifest_sha256']:
        raise ValueError('Frozen data manifest changed')
    review=ROOT/'runs/development_pair/reviewed_for_crossfit.json'
    if not review.exists():raise ValueError('Initial real-data baseline review is required before this queue')
    reviewed=json.loads(review.read_text())
    if not reviewed.get('passed') or reviewed['manifest_sha256']!=plan['manifest_sha256']:
        raise ValueError('Baseline review does not match this dataset')
    OUT.mkdir(parents=True,exist_ok=True);lock=OUT/'queue.lock'
    with lock.open('x') as stream:stream.write(str(os.getpid()))
    try:
        jobs=ordered_jobs(plan)
        (OUT/'ordered_jobs.json').write_text(json.dumps(jobs,indent=2))
        runs_base=count_bytes(ROOT/'runs')
        workspace=count_bytes(ROOT)
        static_bytes=workspace-runs_base
        reserve=plan['estimated_expert_probability_bytes']+plan['estimated_outer_gate_arrays_bytes']+2_000_000_000
        if static_bytes+reserve>80_000_000_000:raise ValueError('Nested plan would exceed the project storage reserve')
        for i,job in enumerate(jobs,1):
            if (ROOT/'runs/gpu_training.lock').exists():raise ValueError('Another training process holds the GPU lock')
            used=count_bytes(ROOT)
            if used>76_000_000_000 or shutil.disk_usage(ROOT).free<10_000_000_000:
                raise ValueError('Storage reserve reached; completed outputs are preserved')
            write_status('running',job_index=i,job_count=len(jobs),run_id=job['run_id'],observed_workspace_bytes=used)
            if job['type']=='expert':
                cmd=[sys.executable,'-s',str(ROOT/'experiments/train_baseline.py'),'--run-id',job['run_id'],
                     '--modality',job['modality'],'--train-groups',','.join(job['train_groups']),
                     '--val-groups',','.join(job['val_groups']),'--epochs',str(job['epochs']),
                     '--selection','fixed_last','--seed',str(job['seed']),'--batch-size','16']
                if job.get('no_validation'):cmd.append('--no-validation')
            else:
                cmd=[sys.executable,'-s',str(ROOT/'experiments/fit_review_head.py'),'--run-id',job['run_id'],
                     '--kind',job['kind'],'--spec',str(ROOT/job['spec']),'--epochs',str(job['epochs']),
                     '--seed',str(job['seed']),'--temperature',str(job['temperature']),'--batch-size','16']
            with (OUT/(job['run_id']+'.log')).open('a',encoding='utf-8') as stream:
                subprocess.run(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
            result=json.loads((ROOT/'runs'/job['run_id']/'complete.json').read_text())
            if result.get('status')!='complete':raise ValueError('Child did not produce a completion record')
        write_status('completed',job_count=len(jobs),core_test_evaluated=False,
                     next_stage='Freeze final evaluation and calibrations, then perform locked-core-test inference')
    except BaseException as exc:
        write_status('failed',error=repr(exc));raise
    finally:lock.unlink(missing_ok=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--plan',default=str(ROOT/'configs/crossfit_seed17/plan.json'))
    run(parser.parse_args())
