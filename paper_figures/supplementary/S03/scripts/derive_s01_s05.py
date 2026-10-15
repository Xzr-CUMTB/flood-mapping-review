"""Read-only scientific audits and compact figure sources for S01--S05."""
import os,sys,json,hashlib,shutil,time
os.environ.setdefault('MKL_THREADING_LAYER','SEQUENTIAL')
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd
ROOT=next(p for p in Path(os.path.abspath(__file__)).parents if (p/'configs/core_release_s17.json').is_file());OUT=ROOT/'figures/supplementary';sys.path.insert(0,str(ROOT))
EVENTS=['2020_flood_season','2021_Henan','2024_April','2024_June'];GROUPS=EVENTS+['2023_holdout']
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as ff:
        for b in iter(lambda:ff.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(path,obj):path.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')
def sd(n):return OUT/f'S{n:02d}'/'source_data'
def table(n,name,data):data.to_csv(sd(n)/name,index=False)
def main():
    start=time.time()
    for n in range(1,6):
        for sub in ['source_data','scripts','assets']:(OUT/f'S{n:02d}'/sub).mkdir(parents=True,exist_ok=True)
    release_path=ROOT/'configs/core_release_s17.json';release=read(release_path);release_sha=sha(release_path);proof=read(ROOT/'reports/completed_batch_verification.json');assert proof['passed'] and proof['release_sha256']==release_sha
    print('Checking frozen release and main-figure preservation',flush=True)
    frozen=[]
    for name,digest in release['sha256'].items():
        p=ROOT/name;observed=sha(p);assert observed==digest,name;frozen.append(dict(path=name,bytes=p.stat().st_size,sha256=observed))
    assert len(frozen)==142
    preserved=[]
    for n in range(1,7):
        name=f'Fig{n:02d}';p=ROOT/'figures/main'/name;v=read(p/(name+'_validation.json'));assert v['passed']
        for r in v['sha256_manifest']:assert sha(p/r['path'])==r['sha256'],(name,r['path'])
        preserved.append(dict(figure=name,validation_sha256=sha(p/(name+'_validation.json')),files=len(v['sha256_manifest'])))
    p=ROOT/'data/processed/sen2gf3';frame=pd.read_csv(p/'manifest.csv');keep=frame.eligible&frame.spatial_nonoverlap;retained=frame[keep].copy();assert len(frame)==21483 and len(retained)==17019
    assert retained.groupby('event_group').size().to_dict()==dict(zip(GROUPS,[2949,5022,476,339,8233]))
    # Reproduce declared eligibility and deterministic non-overlap selection from saved pairs.
    eligibility=(frame.common_valid_pixels>=6554)&~frame.duplicate_bounds&~frame.duplicate_input&frame.event_group.ne('unassigned');assert (eligibility==frame.eligible).all()
    overlap=pd.read_csv(ROOT/'reports/spatial_overlap_pairs.csv');assert len(overlap)==3386;conflicts={int(i):set() for i in frame.sample_id}
    for r in overlap.itertuples():conflicts[r.sample_a].add(r.sample_b);conflicts[r.sample_b].add(r.sample_a)
    accepted=set()
    for r in frame.sort_values('sample_id').itertuples():
        if r.eligible and not (conflicts[r.sample_id]&accepted):accepted.add(r.sample_id)
    assert accepted==set(retained.sample_id);assert not any(a in accepted and b in accepted for a,b in overlap[['sample_a','sample_b']].itertuples(index=False,name=None))
    frame['retained']=keep;frame['offset_px']=frame[['hh_grid_offset_px','hv_grid_offset_px','optical_grid_offset_px']].max(axis=1);frame['coverage_fraction']=frame.common_valid_pixels/65536
    assert abs(frame.loc[keep,'offset_px'].max()-.9658286031061382)<1e-12
    print('Reading cached reference/support arrays; no model inference',flush=True)
    y=np.load(p/'labels_uint8.npy',mmap_mode='r');support=np.load(p/'support_uint8.npy',mmap_mode='r');area=np.load(p/'row_area_m2.npy',mmap_mode='r');x=np.load(p/'x_uint8.npy',mmap_mode='r')
    exact_area=[];exact_flood=[];full_area=[];counts=[]
    for row in frame.itertuples():
        idx=row.cache_index;yy=y[idx];valid=support[idx]==7;weights=area[idx].astype('float64');a=float(np.sum(valid.sum(1)*weights)/1e6);f=float(np.sum(((yy==1)&valid).sum(1)*weights)/1e6)
        exact_area.append(a);exact_flood.append(f);full_area.append(float(weights.sum()*256/1e6));counts.append([int((yy==v).sum()) for v in [0,1,255]])
        assert int(valid.sum())==row.common_valid_pixels and int(((yy==1)&valid).sum())==row.common_positive_pixels
    frame['valid_area_km2']=exact_area;frame['flood_area_km2']=exact_flood;frame['footprint_area_km2']=full_area;frame['valid_area_fraction']=frame.valid_area_km2/frame.footprint_area_km2;frame['flood_fraction']=frame.flood_area_km2/frame.valid_area_km2.replace(0,np.nan)
    for j,v in enumerate([0,1,255]):frame['label_count_'+str(v)]=np.array(counts)[:,j]
    table(2,'tile_audit.csv.gz',frame);table(2,'overlap_pairs.csv',overlap)
    # A transparent sequential decomposition of existing Boolean filters, not inferred reasons.
    named=frame.event_group.ne('unassigned');valid_unique=named&eligibility
    membership=pd.DataFrame(dict(sample_id=frame.sample_id,raw=True,metadata_named=named,valid_unique_eligible=valid_unique,nonoverlap_retained=keep,development=keep&frame.role.eq('development'),core=keep&frame.role.eq('locked_test')))
    table(2,'inclusion_membership.csv.gz',membership)
    stages=pd.DataFrame([dict(stage=k,count=int(membership[k].sum())) for k in membership.columns[1:]]);table(2,'inclusion_stages.csv',stages)
    groups=[]
    for event in GROUPS:
        q=frame[keep&frame.event_group.eq(event)];a=q.valid_area_km2.sum();f=q.flood_area_km2.sum()
        groups.append(dict(event_group=event,tiles=len(q),valid_area_km2=a,flood_area_km2=f,flood_fraction=f/a,zero_flood_tiles=int(q.flood_area_km2.eq(0).sum()),zero_flood_tile_fraction=float(q.flood_area_km2.eq(0).mean()),zero_flood_area_fraction=float(q.loc[q.flood_area_km2.eq(0),'valid_area_km2'].sum()/a),lon_min=q.left.min(),lon_max=q.right.max(),lat_min=q.bottom.min(),lat_max=q.top.max(),common_valid_pixels=int(q.common_valid_pixels.sum()),common_positive_pixels=int(q.common_positive_pixels.sum())))
    groups=pd.DataFrame(groups);table(2,'event_composition.csv',groups);table(1,'event_composition.csv',groups)
    degree=pd.DataFrame(dict(sample_id=frame.sample_id,degree=[len(conflicts[i]) for i in frame.sample_id],retained=keep));table(2,'overlap_degree.csv',degree)
    # A fixed deterministic representative overlap component, independent of predictions/labels.
    components=[];visited=set()
    for i in sorted(conflicts):
        if not conflicts[i] or i in visited:continue
        todo=[i];component=set()
        while todo:
            j=todo.pop()
            if j in component:continue
            component.add(j);todo.extend(conflicts[j]-component)
        visited|=component;components.append(sorted(component))
    chosen=sorted(components,key=lambda q:(-len(q),q[0]))[0];table(2,'displayed_overlap_footprints.csv',frame[frame.sample_id.isin(chosen)])
    # Real co-registered example: closest to median positive flood fraction in BB among high-coverage tiles.
    cand=frame[keep&frame.provisional_site.isin(['Beijing_region','Baoding_region'])&frame.valid_area_fraction.ge(.95)&frame.flood_area_km2.gt(0)].copy();median=float(cand.flood_fraction.median());cand['distance_from_median']=(cand.flood_fraction-median).abs();case=cand.sort_values(['distance_from_median','sample_id']).iloc[0];idx=int(case.cache_index);sid=int(case.sample_id)
    np.savez_compressed(sd(2)/'displayed_scene.npz',x=np.array(x[idx]),label=np.array(y[idx]),support=np.array(support[idx]),row_area_m2=np.array(area[idx]),extent=case[['left','right','bottom','top']].to_numpy(dtype=float))
    raw=ROOT/'data/extracted/sen2gf3/Sen2GF3Floods';rawfiles=[raw/'label'/f'label_{sid}.tif',raw/'gaofen3'/f'after_gf3_hh_{sid}.TIF',raw/'gaofen3'/f'after_gf3_hv_{sid}.TIF',raw/'sentinel2'/f'before_s2_{sid}.TIF']
    save(sd(2)/'scene_provenance.json',dict(sample_id=sid,cache_index=idx,site=case.provisional_site,selection_rule='Median positive flood fraction among retained Beijing/Baoding tiles with >=95% area coverage; sample ID tie-break',candidate_count=len(cand),median_flood_fraction=median,crop='entire native 256 by 256 label-grid window',display='RGB/HH/HV/NIR raw aligned uint8 divided by 255, no contrast stretch; label and valid masks discrete; RGB is never alpha-masked by NIR',alignment='bilinear sources; nearest support; original label grid unchanged',raw_files=[dict(path=str(z.relative_to(ROOT)),sha256=sha(z)) for z in rawfiles]))
    labelrows=[]
    for name,mask in [('All original tiles',np.ones(len(frame),bool)),('Retained tiles',keep.to_numpy())]:
        q=frame[mask]
        for v in [0,1,255]:labelrows.append(dict(population=name,label_value=v,pixels=int(q['label_count_'+str(v)].sum())))
    for v,val in [(0,int(frame.loc[keep,'common_valid_pixels'].sum()-frame.loc[keep,'common_positive_pixels'].sum())),(1,int(frame.loc[keep,'common_positive_pixels'].sum())),(255,0)]:labelrows.append(dict(population='Common-valid retained',label_value=v,pixels=val))
    table(2,'label_values.csv',pd.DataFrame(labelrows));shutil.copyfile(ROOT/'reports/label_semantics_audit.json',sd(2)/'label_semantics.json');shutil.copyfile(ROOT/'reports/sen2gf3_data_audit.json',sd(2)/'original_audit.json')
    # S01: site dates and nine-site metadata, without assigning independent scene dates to tiles.
    dates=pd.read_csv(ROOT/'data/acquisition_dates.csv');assert len(dates)==9
    for c in ['optical_date','sar_date','paper_flood_date']:dates[c]=pd.to_datetime(dates[c])
    dates['optical_age_days']=(dates.sar_date-dates.optical_date).dt.days;dates['sar_delay_days']=(dates.sar_date-dates.paper_flood_date).dt.days;dates['optical_relative_reference_days']=(dates.optical_date-dates.paper_flood_date).dt.days
    dates['membership']=np.where(dates.conservative_group.eq('2023_holdout'),'frozen core','development')
    for i,r in dates.iterrows():
        q=frame[keep&frame.provisional_site.str.replace('_region','',regex=False).eq(r.site)];dates.loc[i,'retained_tiles']=len(q)
        for col,val in [('valid_area_km2',q.valid_area_km2.sum()),('lon_min',q.left.min()),('lon_max',q.right.max()),('lat_min',q.bottom.min()),('lat_max',q.top.max())]:dates.loc[i,col]=val
    dates['geographic_metadata_source']='retained EPSG:4326 label-grid footprints; provisional geographic site association';table(1,'site_dates.csv',dates);table(1,'retained_footprints.csv.gz',frame[keep][['sample_id','cache_index','provisional_site','event_group','role','left','bottom','right','top']]);assert dates.retained_tiles.sum()==17019
    roles=[]
    for e in GROUPS:
        for j,target in enumerate(EVENTS):roles.append(dict(event_group=e,stage='Outer '+str(j+1),role='unused' if e=='2023_holdout' else 'held out' if e==target else 'training'))
        roles.append(dict(event_group=e,stage='Final fit',role='unused' if e=='2023_holdout' else 'training'));roles.append(dict(event_group=e,stage='Core test',role='evaluation' if e=='2023_holdout' else 'unused'))
    table(1,'event_roles.csv',pd.DataFrame(roles))
    # S03: reuse the existing read-only audit function without calling its file-writing main.
    print('Rechecking 38 completed jobs and nested event isolation',flush=True)
    import torch
    torch.set_num_threads(2)
    from scripts.audit_completed_training import planned_jobs,audit
    plan,jobs=planned_jobs();dev=frame[keep&frame.role.eq('development')];verified=[]
    for job in jobs:
        v=audit(job,dev);v['checkpoint_sha256']=sha(ROOT/'runs'/job['run_id']/'last.pt');verified.append(v)
    assert len(verified)==38;table(3,'training_rechecks.csv',pd.DataFrame(verified));save(sd(3)/'frozen_files_verified.json',frozen)
    provenance=[];matrix=[];oof_ids=[]
    for fold,event in enumerate(EVENTS):
        run=f'nested_s17_outer{fold}_value';d=ROOT/'runs'/run;cfg=read(d/'config.json');spec=read(d/'input_spec.json');assert spec==read(ROOT/f'configs/crossfit_seed17/outer_{fold}_value.json');train=pd.read_csv(d/'training_records.csv');pred=pd.read_csv(d/'prediction_records.csv');assert event not in set(train.event_group) and set(pred.event_group)=={event}
        for role,pairs in [('inner',spec['pairs']),('outer',spec['predict_pairs'])]:
            for pair in pairs:
                target=pair['groups'][0]
                for modality in ['sar','fusion']:
                    name=pair[modality+'_run'];rr=ROOT/'runs'/name;t=pd.read_csv(rr/'train_manifest.csv');assert not set(t.event_group)&{event,target};pred_manifest=pd.read_csv(rr/'validation_manifest.csv');assert target in set(pred_manifest.event_group)
                    provenance.append(dict(outer_fold=fold,outer_event=event,role=role,target_event=target,modality=modality,expert_run=name,head_run=run,training_groups=';'.join(sorted(set(t.event_group))),train_samples=len(t),config_sha256=sha(rr/'config.json'),checkpoint_sha256=sha(rr/'last.pt'),predicted_source=str((rr/'validation_probabilities.npy').relative_to(ROOT)),head_training_source=str((d/'training_records.csv').relative_to(ROOT)),head_prediction_source=str((d/'prediction_records.csv').relative_to(ROOT)),exclusion_verified=True))
                    for e in EVENTS:matrix.append(dict(outer_fold=fold,stage=role,run_id=name,target_event=target,event_group=e,trained_on_group=e in set(t.event_group),predicted_group=e==target,outer_excluded=e==event))
        assert set(pred.cache_index)==set(dev[dev.event_group.eq(event)].cache_index);pred['outer_fold']=fold;oof_ids.append(pred)
    table(3,'crossfit_provenance.csv',pd.DataFrame(provenance));table(3,'training_prediction_boolean_matrix.csv',pd.DataFrame(matrix));oof=pd.concat(oof_ids);assert len(oof)==8786 and not oof.cache_index.duplicated().any();table(3,'oof_product_index.csv.gz',oof)
    risk=read(ROOT/'runs/final_s17_risk/config.json');risk_train=pd.read_csv(ROOT/'runs/final_s17_risk/training_records.csv');assert set(risk_train.cache_index)==set(oof.cache_index);assert risk['source_provenance']['training_groups']==EVENTS
    save(sd(3)/'omission_supervision.json',dict(config=risk,actual_training_records_sha256=sha(ROOT/'runs/final_s17_risk/training_records.csv'),target='BCE with y restricted to common-valid, system-predicted-negative pixels; reported omission score sigmoid(logit)*1[p<0.5]',core_labels_used_in_training=False))
    integ=[('Training jobs',38,len(verified)),('Checkpoints',38,len(verified)),('Frozen files',142,len(frozen)),('Prediction arrays',len(proof['prediction_files']),len(proof['prediction_files'])),('Core tiles',8233,int((keep&frame.role.eq('locked_test')).sum())),('Map rows',32,len(pd.read_csv(ROOT/'runs/core_s17_evaluation/map_metrics.csv'))),('Review rows',1824,len(pd.read_csv(ROOT/'runs/core_s17_evaluation/review_budget_results.csv'))),('Simultaneous rows',24,len(pd.read_csv(ROOT/'runs/core_s17_evaluation/review_simultaneous_intervals.csv')))];table(3,'integrity_counts.csv',pd.DataFrame(integ,columns=['evidence','expected','observed']))
    save(sd(3)/'completed_verification_record.json',proof)
    # S04: full six-output, four-event OOF comparisons from each original evaluation.
    model4=['sar','fusion','probability_average','sigmoid_value'];model6=model4+['analytic_value','ridge_value_001'];oofmetrics=[]
    for fold,event in enumerate(EVENTS):
        q=pd.read_csv(ROOT/f'runs/nested_development_fold{fold}/map_metrics.csv');assert set(q.model)==set(model6) and set(q.event_group)=={event};assert len(q)==6
        q['source_csv']=f'runs/nested_development_fold{fold}/map_metrics.csv';q['samples']=int((dev.event_group==event).sum());q['exploratory']=q.model.isin(model6[4:]);q['evaluation_role']='outer event held out';oofmetrics.append(q)
    om=pd.concat(oofmetrics,ignore_index=True);om['rank_IoU']=om.groupby('event_group').flood_iou.rank(method='min',ascending=False).astype(int);effects=[]
    for event in EVENTS:
        q=om[om.event_group.eq(event)].set_index('model');fusion=q.loc['fusion']
        for model in model6:
            for metric,sign in [('flood_iou',1),('brier',-1),('fn_km2',-1),('fp_km2',-1)]:effects.append(dict(event_group=event,model=model,metric=metric,value=float(q.loc[model,metric]),fusion_baseline=float(fusion[metric]),difference=float(q.loc[model,metric]-fusion[metric]),beneficial_effect=float(sign*(q.loc[model,metric]-fusion[metric])),exploratory=model in model6[4:]))
    effect=pd.DataFrame(effects)
    for metric in effect.metric.unique():
        mask=effect.metric.eq(metric);scale=effect.loc[mask,'beneficial_effect'].abs().max();effect.loc[mask,'colour_scale']=scale;effect.loc[mask,'colour_value']=effect.loc[mask,'beneficial_effect']/scale
    ranks=om[om.model.isin(model4)].copy();ranks['rank_IoU']=ranks.groupby('event_group').flood_iou.rank(method='min',ascending=False).astype(int)
    table(4,'all_oof_metrics.csv',om);table(4,'effects_vs_fusion.csv',effect);table(4,'core_model_ranks.csv',ranks);table(4,'event_composition.csv',groups[groups.event_group.isin(EVENTS)]);table(4,'crossfit_sources.csv',pd.DataFrame(provenance))
    # S05: all sixteen observed outputs per core region, independent networks vs combinations.
    maps=pd.read_csv(ROOT/'runs/core_s17_evaluation/map_metrics.csv');assert len(maps)==32 and sha(ROOT/'runs/core_s17_evaluation/map_metrics.csv')==proof['csv_sha256']['map_metrics.csv']
    models=['sar','optical','rgb_fusion','fusion','ordinary_gate','value_gate','probability_average','lower_entropy_expert','inverse_entropy_blend','stacker','terramind_tiny','terramind_tiny_random','fusion_ensemble','sar_temperature','fusion_temperature','value_gate_temperature'];assert set(maps.model)==set(models)
    families=['Single modality']*2+['Multimodal experts']*2+['Learned routing']*2+['Combination strategies']*4+['Foundation controls']*2+['Ensemble']+['Calibrated outputs']*3
    sources=dict(sar='final_s17_sar',optical='final_s17_optical',rgb_fusion='final_s17_rgb_fusion',fusion='final_s17_fusion',ordinary_gate='final_s17_ordinary_gate',value_gate='final_s17_value',stacker='final_s17_stacker',terramind_tiny='final_terramind_s17',terramind_tiny_random='final_terramind_random_s17')
    metadata=[]
    for model,family in zip(models,families):
        base_model=model.replace('_temperature','');run=sources.get(base_model);cfg=read(ROOT/'runs'/run/'config.json') if run else {};parameters=cfg.get('parameters');kind='trained network' if model in ['sar','optical','rgb_fusion','fusion','terramind_tiny','terramind_tiny_random'] else 'trained head with two experts' if model in ['ordinary_gate','value_gate','stacker'] else 'calibrated probabilities' if model.endswith('temperature') else 'three-member probability ensemble' if model=='fusion_ensemble' else 'deterministic probability combination'
        if model in ['probability_average','lower_entropy_expert','inverse_entropy_blend']:run='final_s17_sar;final_s17_fusion'
        if model=='fusion_ensemble':run='final_s17_fusion;final_s29_fusion;final_s43_fusion'
        metadata.append(dict(model=model,family=family,output_type=kind,training_run=run,parameters=parameters,parameters_definition='head alone; two experts also required' if 'head' in kind else 'single network' if kind=='trained network' else 'underlying model; calibration does not define a new network' if model.endswith('temperature') else 'not represented as an independent network',calibrated=model.endswith('temperature'),ensemble_seeds='17;29;43' if model=='fusion_ensemble' else '',release_sha256=release_sha))
    meta=pd.DataFrame(metadata);maps=maps.merge(meta,on='model',validate='many_to_one');table(5,'all_core_map_metrics.csv',maps);table(5,'model_provenance.csv',meta)
    benchmark=[];pairs=[]
    for region,g in maps.groupby('region'):
        q=g.set_index('model');fusion=q.loc['fusion']
        for model in models:
            for metric,sign in [('flood_iou',1),('fn_km2',-1),('fp_km2',-1),('brier',-1),('confident_omission_01',-1)]:benchmark.append(dict(region=region,model=model,metric=metric,value=float(q.loc[model,metric]),difference_from_fusion=float(q.loc[model,metric]-fusion[metric]),beneficial_effect=float(sign*(q.loc[model,metric]-fusion[metric])),release_sha256=release_sha))
        for a,b,label in [('terramind_tiny','terramind_tiny_random','Pretrained minus random'),('fusion_ensemble','fusion','Ensemble minus fusion')]:
            for metric in ['flood_iou','brier','fn_km2','fp_km2']:pairs.append(dict(region=region,comparison=label,first_model=a,second_model=b,metric=metric,first_value=q.loc[a,metric],second_value=q.loc[b,metric],difference=q.loc[a,metric]-q.loc[b,metric]))
        for model in ['sar','fusion','value_gate']:
            for metric in ['flood_iou','fn_km2','fp_km2']:assert abs(q.loc[model,metric]-q.loc[model+'_temperature',metric])<1e-12
    b=pd.DataFrame(benchmark)
    for metric in b.metric.unique():
        mask=b.metric.eq(metric);scale=b.loc[mask,'beneficial_effect'].abs().max();b.loc[mask,'colour_scale']=scale;b.loc[mask,'colour_value']=b.loc[mask,'beneficial_effect']/scale
    table(5,'benchmark_cells.csv',b);table(5,'paired_controls.csv',pd.DataFrame(pairs));table(5,'capacity_comparison.csv',maps[maps.output_type.eq('trained network')]);assert set(maps[maps.output_type.eq('trained network')].parameters)=={1964529.,5873617.}
    # Bind each figure to actual compact inputs and preserve all main figures.
    common=['configs/core_release_s17.json','reports/completed_batch_verification.json','data/processed/sen2gf3/manifest.csv','data/acquisition_dates.csv','reports/spatial_overlap_pairs.csv','reports/training_artifact_audit.json','reports/label_semantics_audit.json','reports/retained_support_audit.json','experiments/fit_review_head.py','experiments/evaluate_nested_development.py','runs/core_s17_evaluation/map_metrics.csv']+[f'runs/nested_development_fold{j}/map_metrics.csv' for j in range(4)]
    bindings=[dict(path=name,bytes=(ROOT/name).stat().st_size,sha256=sha(ROOT/name)) for name in common]
    for n in range(1,6):
        save(sd(n)/'input_bindings.json',bindings);save(sd(n)/'main_figure_preservation.json',preserved);save(sd(n)/'scientific_audit.json',dict(passed=True,release_sha256=release_sha,checked_at_utc=datetime.now(timezone.utc).isoformat(),all_142_frozen_hashes_verified=True,all_38_training_artifacts_rechecked=True,actual_source_derivation_script='derive_s01_s05.py',source_files={str(q.name):sha(q) for q in sd(n).iterdir() if q.is_file() and q.name!='scientific_audit.json'}))
        shutil.copyfile(Path(__file__),OUT/f'S{n:02d}'/'scripts'/'derive_s01_s05.py')
    save(OUT/'batch_source_audit.json',dict(passed=True,elapsed_seconds=time.time()-start,release_sha256=release_sha,main_figures_preserved=preserved,stages=stages.to_dict('records'),site_records=9,retained_tiles=17019,checkpoint_rechecks=38,frozen_hashes=142,overlap_component=chosen,example_scene=sid))
    print(json.dumps(read(OUT/'batch_source_audit.json'),indent=2),flush=True)
if __name__=='__main__':main()
