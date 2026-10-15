"""End-to-end evaluator I/O on explicitly synthetic files, never study test predictions."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,sys,os
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import experiments.infer_frozen_core as inference
import experiments.evaluate_frozen_core as evaluation


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2))


def main():
    fixture=ROOT/'runs/software_checks/frozen_evaluator_io_fixture'
    fixture.mkdir(parents=True,exist_ok=True)
    # Re-exercise the full evaluator after code changes, retaining the labelled fixture files.
    previous=fixture/'runs/core_s17_evaluation/complete.json'
    if previous.exists():previous.unlink()
    (fixture/'FIXTURE_NOT_RESEARCH_DATA.txt').write_text('All data in this directory are synthetic software fixtures. Never report as study results.\n')
    protocol=json.loads((ROOT/'configs/evaluation_protocol_v1.json').read_text())
    write(fixture/'configs/evaluation_protocol_v1.json',protocol)
    release_path=fixture/'configs/core_release_s17.json'
    write(release_path,{'status':'frozen_for_core_evaluation','sha256':{
        'configs/evaluation_protocol_v1.json':inference.sha(fixture/'configs/evaluation_protocol_v1.json')}})
    release_hash=inference.sha(release_path)
    source=fixture/'runs/core_s17_predictions';source.mkdir(parents=True,exist_ok=True)
    fsource=fixture/'runs/core_s17_foundation_predictions';fsource.mkdir(parents=True,exist_ok=True)
    frame=pd.DataFrame({'sample_id':range(1,7),'cache_index':range(6),'event_group':['2023_holdout']*6,
        'provisional_site':['Beijing_region']*3+['Harbin']*3,
        'lon':[115.7,116.,116.3,126.,126.3,126.6],'lat':[39.1,39.4,39.7,45.2,45.5,45.8]})
    for directory in [source,fsource]:frame.to_csv(directory/'prediction_manifest.csv',index=False)
    write(source/'complete.json',{'status':'complete','release_sha256':release_hash,'models':inference.NAMES})
    write(fsource/'complete.json',{'status':'complete','release_sha256':release_hash})
    rng=np.random.default_rng(8319);truth=(rng.random((6,256,256))<.2).astype('uint8')
    for directory,names in [(source,inference.NAMES),(fsource,['terramind_tiny','terramind_tiny_random'])]:
        for name in names:
            probability=np.clip(.15+.5*truth+rng.normal(0,.25,truth.shape),0,1).astype('float32')
            np.save(directory/(name+'.npy'),probability)
    np.save(source/'router_unit_features.npy',rng.random((6,16,5)))
    data=fixture/'data/processed/sen2gf3';data.mkdir(parents=True,exist_ok=True)
    np.save(data/'labels_uint8.npy',truth);np.save(data/'support_uint8.npy',np.full(truth.shape,7,dtype='uint8'))
    np.save(data/'row_area_m2.npy',np.full((6,256),100.,dtype='float32'))
    calibration={'temperatures':{name:{'temperature':1.1} for name in ['sar','fusion','value_gate']},
        'operating_points':{scenario:{name:{'threshold':.5 if name=='fusion' else .4,'target_rate':.1,'achieved_rate':.1}
                            for name in ['sar','fusion','value_gate']} for scenario in ['matched_fpr','matched_recall']}}
    write(fixture/'runs/oof_calibration_s17/complete.json',calibration)
    # The changed roots are confined to this isolated Python check process.
    inference.ROOT=fixture;evaluation.ROOT=fixture;evaluation.main()
    out=fixture/'runs/core_s17_evaluation'
    maps=pd.read_csv(out/'map_metrics.csv');points=pd.read_csv(out/'matched_operating_point_metrics.csv')
    router=pd.read_csv(out/'router_value_unit_diagnostics.csv')
    assert len(maps)==32 and len(points)==12 and len(router)==96
    assert np.isfinite(maps[['flood_iou','brier','nll']].to_numpy()).all()
    assert np.isfinite(points[['false_positive_rate','recall']].to_numpy()).all()
    report={'passed':True,'checked_at_utc':datetime.now(timezone.utc).isoformat(),
        'scope':'Complete evaluator file I/O using synthetic 256x256 software fixtures; no study test predictions',
        'map_rows':len(maps),'operating_point_rows':len(points),'router_unit_rows':len(router),
        'source_sha256':{relative:inference.sha(ROOT/relative) for relative in
            ['experiments/evaluate_frozen_core.py','experiments/evaluation_metrics.py','experiments/operating_points.py']}}
    write(ROOT/'reports/frozen_evaluator_io_checks.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
