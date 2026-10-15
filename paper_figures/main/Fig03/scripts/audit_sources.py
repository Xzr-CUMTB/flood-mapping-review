"""Supplementary provenance binding and independent aggregate numerical audit."""
import os,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime,timezone
ROOT=Path(os.path.abspath(__file__)).parents[4];SD=ROOT/'figures/main/Fig03/source_data'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
proof=json.loads((SD/'panel_a_crossfit_provenance.json').read_text());bindings=json.loads((SD/'input_bindings.json').read_text());seen={b['path'] for b in bindings};models=[]
for fold in proof:
    rr=ROOT/'runs'/fold['head_run'];spec=json.loads((rr/'input_spec.json').read_text())
    for pair in spec['pairs']+spec['predict_pairs']:
        for field in ['sar_run','fusion_run']:
            run=ROOT/'runs'/pair[field];cfg=json.loads((run/'config.json').read_text());train=pd.read_csv(run/'train_manifest.csv');valid=pd.read_csv(run/'validation_patch_metrics.csv');assert not set(train.event_group)&set(pair['groups']);assert fold['event'] not in set(train.event_group)
            for n in ['config.json','last.pt','train_manifest.csv','validation_patch_metrics.csv','validation_probabilities.npy']:
                path=run/n;rel=str(path.relative_to(ROOT))
                if rel not in seen:bindings.append(dict(path=rel,bytes=path.stat().st_size,sha256=sha(path)));seen.add(rel)
            models.append(dict(fold=fold['fold'],outer_event=fold['event'],run=pair[field],role='inner supervision' if pair in spec['pairs'] else 'outer evaluation',modality=cfg['modality'],parameters=cfg.get('parameters',1964529),epochs=cfg['epochs'],seed=cfg['seed'],training_events=cfg['train_groups'],target_events=','.join(pair['groups'])))
    predicted=pd.read_csv(rr/'prediction_records.csv')
    for rn in predicted.sar_run.unique():
        table=pd.read_csv(ROOT/'runs'/rn/'validation_patch_metrics.csv').set_index('prediction_index');selected=table.loc[predicted[predicted.sar_run==rn].prediction_index]
        assert selected.cache_index.tolist()==predicted[predicted.sar_run==rn].cache_index.tolist()
verify=json.loads((ROOT/'reports/completed_batch_verification.json').read_text());corefile=ROOT/'runs/core_s17_evaluation/map_metrics.csv'
assert verify['passed'];print('Verification keys',list(verify),flush=True)
for p in [corefile,ROOT/'reports/completed_batch_verification.json',ROOT/'configs/core_release_s17.json',ROOT/'models/review_heads.py',ROOT/'experiments/evaluate_nested_development.py']:
    bindings.append(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)))
# Validate metrics against algebra and joint histograms independently of rendering.
d=pd.read_csv(SD/'panel_de_event_metrics.csv');summ=pd.read_csv(SD/'panel_bc_event_summary.csv');z=np.load(SD/'joint_statistics.npz')
for fold,event in enumerate(summ.event):
    rows=d[d.event_group==event].set_index('model');v=rows.loc['value_gate'];f=rows.loc['fusion'];assert np.isclose((v.true_flood_km2-v.fn_km2)/(v.true_flood_km2+v.fp_km2),v.flood_iou)
    assert np.isclose(f.brier-v.brier,summ.loc[fold,'fusion_minus_gate_brier']);assert np.isclose(z[f'b_{fold}'].sum(),z[f'c_{fold}'].sum());assert np.isclose(z[f'b_{fold}'].sum()/1e6,v.valid_area_km2)
    cc=z[f'c_{fold}'];bs=z[f'benefit_sum_{fold}'];assert np.isclose(bs.sum()/cc.sum(),f.brier-v.brier)
pd.DataFrame(models).drop_duplicates().to_csv(SD/'panel_a_model_training.csv',index=False)
(SD/'input_bindings.json').write_text(json.dumps(bindings,indent=2),encoding='utf-8')
audit=json.loads((SD/'scientific_audit.json').read_text());audit['checks'].update(expert_training_manifests_exclude_outer_events=True,expert_saved_array_index_matches=True,independent_aggregate_arithmetic=True);audit['checked_at_utc']=datetime.now(timezone.utc).isoformat();audit['source_files_bound']=len(bindings);audit['passed']=all(audit['checks'].values());(SD/'scientific_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8');print('Supplementary source audit passed',len(bindings),flush=True)
