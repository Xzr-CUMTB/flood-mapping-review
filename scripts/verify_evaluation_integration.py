"""Exercise result-table and interval generation with labelled synthetic software fixtures."""
from pathlib import Path
from datetime import datetime,timezone
import json,sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.evaluation_metrics import patch_metrics,review_units
from experiments.evaluate_frozen_core import ALL_NAMES,UNIT_FIELDS,evaluate_review,uncertainty_rows


def main():
    out=ROOT/'runs/software_checks/evaluation_integration_fixture';out.mkdir(parents=True,exist_ok=True)
    (out/'FIXTURE_NOT_RESEARCH_DATA.txt').write_text('Synthetic software fixture for arithmetic and table integration. Never use as empirical evidence.\n')
    protocol=json.loads((ROOT/'configs/evaluation_protocol_v1.json').read_text())
    frame=pd.DataFrame({'lon':[115.7,116.,116.3,126.,126.3,126.6],
                        'lat':[39.1,39.4,39.7,45.2,45.5,45.8]})
    masks={'Beijing_Baoding':np.array([True]*3+[False]*3),'Harbin':np.array([False]*3+[True]*3)}
    rng=np.random.default_rng(20261012)
    metrics=np.zeros((len(ALL_NAMES),6,10));units=np.empty((len(ALL_NAMES),6,16,len(UNIT_FIELDS)))
    units.fill(np.nan)
    for index in range(6):
        truth=rng.random((16,16))<.2;weights=np.full((16,16),10.)
        sar,fusion=rng.random((16,16)),rng.random((16,16))
        for m,name in enumerate(ALL_NAMES):
            p=np.clip(.15+.5*truth+rng.normal(0,.25,(16,16)),0,1)
            risk=rng.random((16,16)) if name in ['value_gate','value_gate_temperature','fusion','fusion_temperature'] else None
            u=review_units(p,truth,weights,sar,fusion,risk=risk,stacker=fusion,size=4,
                           ensemble_variance=rng.random((16,16))*.1,ensemble_mi=rng.random((16,16))*.1)
            metrics[m,index]=patch_metrics(p,truth,weights)
            for j,key in enumerate(UNIT_FIELDS):units[m,index,:,j]=u.get(key,np.full(16,np.nan))
    evaluate_review(units,frame,masks,protocol,out)
    map_intervals=uncertainty_rows(metrics,frame,masks,protocol)
    assert len(map_intervals)>0
    result=pd.read_csv(out/'review_budget_results.csv')
    assert (result.actual_area_fraction<=result.budget_fraction+1e-12).all()
    assert (result.residual_fn_km2<=result.initial_fn_km2+1e-12).all()
    assert (result.residual_fp_km2<=result.initial_fp_km2+1e-12).all()
    for _,group in result.groupby(['region','model','budget_fraction']):
        oracle=group[group.policy.eq('oracle_fn_density')].iloc[0]
        # Equal-cost units make FN-density sorting exactly optimal in this software fixture.
        assert oracle.residual_fn_km2<=group.residual_fn_km2.min()+1e-12
    intervals=pd.read_csv(out/'review_simultaneous_intervals.csv')
    assert len(intervals)==24 and intervals.family_estimates.eq(12).all()
    assert np.isfinite(intervals[['simultaneous_lower95','simultaneous_upper95']].to_numpy()).all()
    assert (intervals.simultaneous_lower95<=intervals.difference_first_minus_second+1e-12).all()
    assert (intervals.simultaneous_upper95>=intervals.difference_first_minus_second-1e-12).all()
    report={'passed':True,'checked_at_utc':datetime.now(timezone.utc).isoformat(),
        'scope':'Synthetic software fixture; no study data or core predictions used',
        'models_exercised':len(ALL_NAMES),'budget_rows':len(result),'simultaneous_interval_rows':len(intervals),
        'checks':['region/model/policy table integration','budget and residual-error monotonicity',
                  'equal-cost oracle ordering','paired spatial intervals','12-estimate simultaneous family']}
    (ROOT/'reports/evaluation_integration_checks.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
