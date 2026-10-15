"""Recompute fixed evaluation selections and intervals from frozen sufficient statistics."""
import os
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
from pathlib import Path
import argparse, json, sys
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.evaluation_metrics import metric_result
from publication.evaluation_core import ALL_NAMES, UNIT_FIELDS, evaluate_review, uncertainty_rows


def compare_csv(actual,expected,keys):
    a=pd.read_csv(actual).sort_values(keys).reset_index(drop=True)
    b=pd.read_csv(expected).sort_values(keys).reset_index(drop=True)
    if a.shape!=b.shape or list(a.columns)!=list(b.columns):raise ValueError('CSV schema mismatch: '+actual.name)
    errors={}
    for col in a:
        if pd.api.types.is_numeric_dtype(a[col]) and pd.api.types.is_numeric_dtype(b[col]):
            left=a[col].to_numpy(float);right=b[col].to_numpy(float)
            if not np.allclose(left,right,rtol=1e-10,atol=1e-8,equal_nan=True):raise ValueError('Numeric mismatch: '+actual.name+'/'+col)
            finite=np.isfinite(left)&np.isfinite(right)
            errors[col]=float(np.max(np.abs(left[finite]-right[finite]))) if finite.any() else 0.
        elif not a[col].fillna('').equals(b[col].fillna('')):raise ValueError('Categorical mismatch: '+actual.name+'/'+col)
    return dict(file=actual.name,rows=len(a),maximum_absolute_numeric_error=max(errors.values(),default=0),columns=errors)


def reproduce(data,out):
    data=Path(data);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    ev=data/'evaluation';protocol=json.loads((data/'provenance/configs/evaluation_protocol_v1.json').read_text())
    schema=json.loads((ev/'array_schema.json').read_text())
    assert schema['models']==ALL_NAMES and schema['unit_fields']==UNIT_FIELDS
    frame=pd.read_csv(ev/'prediction_manifest.csv')
    assert len(frame)==8233 and frame.sample_id.is_unique and frame.cache_index.is_unique
    assert frame.prediction_index.tolist()==list(range(len(frame)))
    masks={protocol[k]['name']:frame.provisional_site.isin(protocol[k]['provisional_sites']).to_numpy() for k in ['primary_region','secondary_region']}
    assert (np.stack(list(masks.values())).sum(axis=0)==1).all()
    metrics=np.load(ev/'patch_metric_sums.npy',mmap_mode='r');units=np.load(ev/'review_units.npy',mmap_mode='r')
    assert metrics.shape==(16,8233,10) and units.shape==(16,8233,16,14)
    expected_maps=pd.read_csv(ev/'map_metrics.csv').set_index(['region','model']);map_errors=[]
    for region,mask in masks.items():
        for i,name in enumerate(ALL_NAMES):
            recomputed=metric_result(metrics[i,mask].sum(axis=0))
            original=expected_maps.loc[(region,name)]
            for key,value in recomputed.items():
                if not np.isclose(value,original[key],rtol=1e-11,atol=1e-8):raise ValueError('Map mismatch '+region+'/'+name+'/'+key)
            # Re-evaluate ECE from actual saved reliability-bin weights and means.
            rel=pd.read_csv(ev/(region+'_'+name+'_reliability.csv'))
            ece=float(np.dot(rel.area_m2,np.abs(rel.mean_probability-rel.flood_fraction))/rel.area_m2.sum())
            if not np.isclose(ece,original.area_weighted_ece,atol=1e-12):raise ValueError('Reliability ECE mismatch')
            fn=units[i,mask,:,1].sum()/1e6;fp=units[i,mask,:,2].sum()/1e6
            assert np.isclose(fn,recomputed['fn_km2'],atol=1e-8) and np.isclose(fp,recomputed['fp_km2'],atol=1e-8)
            map_errors.append(dict(region=region,model=name,fields_verified=len(recomputed)+1))
    print('32 map rows recomputed; all patch/unit FN/FP totals agree.',flush=True)
    pd.DataFrame(uncertainty_rows(metrics,frame,masks,protocol)).to_csv(out/'map_paired_spatial_intervals.csv',index=False)
    print('Map intervals rebuilt from original patch sums, block assignments and RNG seed.',flush=True)
    evaluate_review(units,frame,masks,protocol,out)
    comparisons=[]
    for name,keys in [('map_paired_spatial_intervals.csv',['region','first','second','metric','block_size_m']),
                      ('review_budget_results.csv',['region','model','policy','budget_fraction']),
                      ('review_paired_spatial_intervals.csv',['region','first_model','first_policy','second_model','second_policy','endpoint','block_size_m']),
                      ('review_simultaneous_intervals.csv',['region','first_model','first_policy','second_model','second_policy','endpoint'])]:
        comparisons.append(compare_csv(out/name,ev/name,keys))
    with np.load(ev/'primary_review_selections_packed.npz',allow_pickle=False) as original, np.load(out/'primary_review_selections_packed.npz',allow_pickle=False) as new:
        assert set(original.files)==set(new.files)
        for key in original.files:assert np.array_equal(original[key],new[key]),key
        selection_count=len(original.files)
    budget=pd.read_csv(out/'review_budget_results.csv')
    assert (budget.actual_area_fraction<=budget.budget_fraction+1e-12).all()
    assert (budget.residual_fn_km2<=budget.initial_fn_km2+1e-9).all()
    assert (budget.residual_fp_km2<=budget.initial_fp_km2+1e-9).all()
    report=dict(passed=True,scope='Actual frozen patch and review-unit sufficient statistics; no retraining or study prediction replacement',
        map_rows=len(map_errors),csv_comparisons=comparisons,primary_selection_sets=selection_count,
        bootstrap='Fresh deterministic resampling from original contributions using original algorithm and seed, compared with all archived intervals; not recovered from interval endpoints',
        limitation='4096-bin average precision is preserved in original CSV, not independently rebuilt from the sufficient statistics. Pixel inference requires public original imagery plus released model states.',
        budget_and_fn_fp_conservation=True,core_units=int((units[0,:,:,0]>0).sum()))
    (out/'evaluation_reproduction.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['passed','map_rows','primary_selection_sets','core_units']}),flush=True)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();reproduce(args.data,args.output)
