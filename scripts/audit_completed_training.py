"""Audit actual saved training artifacts before a frozen test release, without test predictions."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json, sys
import numpy as np
import pandas as pd
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.train_baseline import atomic_json


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def planned_jobs():
    plan=read(ROOT/'configs/crossfit_seed17/plan.json')
    jobs=list(plan['jobs'])
    foundation=read(ROOT/'configs/foundation_experiments_s17.json')
    jobs += [{**job,'type':'foundation','epochs':foundation['epochs'],'seed':foundation['seed']}
             for job in foundation['jobs']]
    jobs += [{'run_id':'final_s17_fusion_risk','type':'head','epochs':10,'seed':17}]
    ensemble=read(ROOT/'configs/ensemble_controls.json')
    jobs += [{'run_id':run_id,'type':'expert','modality':'fusion','epochs':ensemble['epochs'],'seed':seed,
              'train_groups':ensemble['train_groups'].split(','),'val_groups':[],'no_validation':True}
             for seed,run_id in zip(ensemble['seeds'][1:],ensemble['new_members'])]
    return plan,jobs


def audit(job, frame):
    directory=ROOT/'runs'/job['run_id']
    config,complete=read(directory/'config.json'),read(directory/'complete.json')
    assert complete['status']=='complete' and complete['core_test_evaluated'] is False
    assert config['seed']==job['seed'] and config['epochs']==job['epochs']
    if job['type'] in ['expert','foundation']:
        assert complete['selected_epoch']==job['epochs']
        assert config.get('validation_used_for_selection',False) is False
        if job['type']=='expert':assert config['checkpoint_selection_mode']=='fixed_last'
        else:assert config['random_init']==job['random_init']
        for split,key in [('train','train_groups'),('validation','val_groups')]:
            declared=job[key].split(',') if isinstance(job[key],str) and job[key] else job[key]
            if not declared:declared=[]
            assert set(declared).isdisjoint({'2023_holdout','unassigned'})
            expected=set(frame.loc[frame.event_group.isin(declared),'sample_id'])
            observed=pd.read_csv(directory/(split+'_manifest.csv'))
            assert not observed.sample_id.duplicated().any()
            assert set(observed.sample_id)==expected,(job['run_id'],split,'sample mismatch')
            assert set(observed.event_group)==set(declared)
        expected_hash=sha(ROOT/'data/processed/sen2gf3/manifest.csv')
        assert config['manifest_sha256']==expected_hash
    else:
        records=pd.read_csv(directory/'training_records.csv')
        assert not records.cache_index.duplicated().any()
        groups=config['source_provenance']['training_groups']
        expected=set(frame.loc[frame.event_group.isin(groups),'cache_index'])
        assert set(records.cache_index)==expected
        assert set(records.event_group)==set(groups)
        assert set(groups).isdisjoint({'2023_holdout','unassigned'})
        if job.get('spec'):
            assert config['spec_sha256']==sha(ROOT/job['spec'])
            assert sha(directory/'input_spec.json')==config['spec_sha256']
    for relative,expected in config['source_sha256'].items():
        assert sha(directory/'source_snapshot'/relative)==expected
        assert sha(ROOT/relative)==expected,(job['run_id'],relative,'training source changed')
    history=pd.read_csv(directory/'history.csv')
    assert history.epoch.tolist()==list(range(1,job['epochs']+1))
    loss=history['train_bce'] if 'train_bce' in history else history['loss']
    assert np.isfinite(loss.to_numpy()).all() and (loss>=0).all()
    checkpoint=torch.load(directory/'last.pt',map_location='cpu',weights_only=False)
    assert checkpoint['epoch']==job['epochs']
    assert checkpoint['model'] and all(torch.isfinite(t).all().item() for t in checkpoint['model'].values())
    return {'run_id':job['run_id'],'seed':config['seed'],'epochs':job['epochs'],
            'checkpoint_tensors':len(checkpoint['model']),
            'config_sha256':sha(directory/'config.json'),'complete_sha256':sha(directory/'complete.json')}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--require-complete',action='store_true');args=parser.parse_args()
    torch.set_num_threads(2)
    plan,jobs=planned_jobs();manifest=ROOT/'data/processed/sen2gf3/manifest.csv'
    assert sha(manifest)==plan['manifest_sha256']
    frame=pd.read_csv(manifest)
    frame=frame[frame.eligible & frame.spatial_nonoverlap & frame.role.eq('development')]
    pending=[];verified=[]
    for job in jobs:
        if not (ROOT/'runs'/job['run_id']/'complete.json').exists():pending.append(job['run_id']);continue
        verified.append(audit(job,frame))
    report={'passed':not pending,'completed_artifacts_valid':True,'checked_at_utc':datetime.now(timezone.utc).isoformat(),
            'planned_jobs':len(jobs),'verified_jobs':len(verified),'pending':pending,'verified':verified,
            'manifest_sha256':plan['manifest_sha256'],'core_test_predictions_inspected':False,
            'checks':['exact declared development sample sets','no core or quarantine supervision',
                      'complete fixed epoch histories','source snapshot and current training code identity',
                      'finite saved CPU checkpoint tensors','configuration and completion fingerprints'],
            'auditor_sha256':sha(Path(__file__))}
    atomic_json(ROOT/'reports/training_artifact_audit.json',report)
    print(json.dumps({k:report[k] for k in ['passed','completed_artifacts_valid','planned_jobs','verified_jobs','pending']}),flush=True)
    if args.require_complete and pending:raise ValueError('Training jobs remain incomplete')


if __name__=='__main__':main()
