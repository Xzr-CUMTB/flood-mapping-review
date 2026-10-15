"""Freeze event folds and complete-pipeline nested OOF dependencies before running them."""
from pathlib import Path
from itertools import combinations
from datetime import datetime,timezone
import hashlib,json
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'configs/crossfit_seed17'


def main():
    manifest=ROOT/'data/processed/sen2gf3/manifest.csv'
    frame=pd.read_csv(manifest)
    dev=frame[frame.eligible&frame.spatial_nonoverlap&frame.role.eq('development')]
    groups=sorted(dev.event_group.unique())
    if len(groups)<4:raise ValueError('This nested plan requires four verified development groups')
    if len(groups)>4:raise ValueError('Review the plan for additional event groups')
    OUT.mkdir(parents=True,exist_ok=True)
    jobs=[];runs={};seed=17
    for count in [2,3]:
        for train_ids in combinations(range(4),count):
            train_groups=[groups[i] for i in train_ids];val_groups=[g for g in groups if g not in train_groups]
            key=''.join(map(str,train_ids));pair={}
            for modality in ['sar','fusion']:
                run_id=f'oof_s17_g{key}_{modality}';pair[modality+'_run']=run_id
                jobs.append({'type':'expert','run_id':run_id,'modality':modality,'train_groups':train_groups,
                             'val_groups':val_groups,'epochs':20,'selection':'fixed_last','seed':seed})
            runs[tuple(train_ids)]=pair
    outer_runs=[];outer_pairs=[]
    for outer in range(4):
        remaining=[i for i in range(4) if i!=outer]
        inner_pairs=[]
        for inner_target in remaining:
            training=tuple(i for i in remaining if i!=inner_target)
            inner_pairs.append({**runs[training],'groups':[groups[inner_target]]})
        outer_pair={**runs[tuple(remaining)],'groups':[groups[outer]]}
        outer_pairs.append(outer_pair)
        spec={'pairs':inner_pairs,'excluded_groups':[groups[outer]],'predict_pairs':[outer_pair]}
        path=OUT/f'outer_{outer}_value.json';path.write_text(json.dumps(spec,indent=2))
        run_id=f'nested_s17_outer{outer}_value';outer_runs.append(run_id)
        jobs.append({'type':'head','kind':'value','run_id':run_id,'spec':str(path.relative_to(ROOT)),
                     'epochs':10,'seed':seed,'temperature':.05})
    for kind in ['value','ordinary_gate','stacker']:
        path=OUT/f'final_{kind}.json';path.write_text(json.dumps({'pairs':outer_pairs,'excluded_groups':[]},indent=2))
        jobs.append({'type':'head','kind':kind,'run_id':f'final_s17_{kind}','spec':str(path.relative_to(ROOT)),
                     'epochs':10,'seed':seed,'temperature':.05})
    path=OUT/'final_risk.json';path.write_text(json.dumps({'system_runs':outer_runs},indent=2))
    jobs.append({'type':'head','kind':'risk','run_id':'final_s17_risk','spec':str(path.relative_to(ROOT)),
                 'epochs':10,'seed':seed,'temperature':.05})
    for modality in ['sar','fusion','optical','rgb_fusion']:
        jobs.append({'type':'expert','run_id':f'final_s17_{modality}','modality':modality,'train_groups':groups,
                     'val_groups':[],'epochs':20,'selection':'fixed_last','no_validation':True,'seed':seed})
    # Pair-trained experts predict two omitted groups; triple-trained experts predict one.
    probability_bytes=8*len(dev)*256*256*4
    gate_bytes=3*len(dev)*256*256*4
    plan={'version':1,'created_at_utc':datetime.now(timezone.utc).isoformat(),'seed':seed,'development_groups':groups,
          'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),'training_samples':len(dev),
          'event_counts':dev.event_group.value_counts().sort_index().to_dict(),'jobs':jobs,
          'estimated_expert_probability_bytes':probability_bytes,'estimated_outer_gate_arrays_bytes':gate_bytes,
          'core_test_evaluated_by_queue':False,'figure_generation':False,
          'selection':'Fixed 20 expert epochs and 10 head epochs; no outer-event early stopping',
          'value_target':'OOF expert Brier difference; pure squared regression loss',
          'risk_target':'Flood label on predicted-negative pixels from complete-system nested OOF predictions',
          'unverified_scene_age_used':False,'calibration_claim':False,
          'queued_after':'Manual review of completed initial baseline outputs and provenance'}
    path=OUT/'plan.json'
    if path.exists():
        old=json.loads(path.read_text())
        if old['manifest_sha256']!=plan['manifest_sha256']:raise ValueError('An existing plan targets a different data manifest')
    path.write_text(json.dumps(plan,indent=2))
    print(json.dumps({k:v for k,v in plan.items() if k!='jobs'},indent=2));print('Jobs:',len(jobs))


if __name__=='__main__':main()
