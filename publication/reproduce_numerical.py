"""CPU reproduction of published effects from actual frozen result tables."""
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd


def reproduce(data, out):
    data=Path(data);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    maps=pd.read_csv(data/'evaluation/map_metrics.csv')
    review=pd.read_csv(data/'evaluation/review_budget_results.csv')
    intervals=pd.read_csv(data/'evaluation/review_simultaneous_intervals.csv')
    claims=[]
    expected={'Beijing_Baoding':[16.39,5.12,22.43,17.4], 'Harbin':[26.28,5.13,44.60,49.1]}
    def row(region,model,policy):
        q=review[(review.region==region)&(review.model==model)&(review.policy==policy)&np.isclose(review.budget_fraction,.1)]
        if len(q)!=1:raise ValueError('Nonunique review endpoint')
        return q.iloc[0]
    def interval(region,first,policy,second,second_policy):
        q=intervals[(intervals.region==region)&(intervals.first_model==first)&(intervals.first_policy==policy)&
            (intervals.second_model==second)&(intervals.second_policy==second_policy)&(intervals.endpoint=='residual_fn_km2')]
        if len(q)!=1:raise ValueError('Nonunique simultaneous interval')
        return q.iloc[0]
    for region,target in expected.items():
        m=maps[maps.region==region].set_index('model')
        if len(m)!=16 or not m.index.is_unique:raise ValueError('Expected 16 unique frozen maps')
        fusion=row(region,'fusion','mean_entropy');value=row(region,'value_gate','mean_entropy')
        learned=row(region,'value_gate','learned_omission_density');proxy=row(region,'value_gate','fn_probability_density')
        gain=100*(m.loc['fusion','flood_iou']-m.loc['sar','flood_iou'])
        inversion=value.residual_fn_km2-fusion.residual_fn_km2
        reduction=learned.residual_fn_km2-proxy.residual_fn_km2
        relative=100*reduction/learned.residual_fn_km2
        ci_inversion=interval(region,'value_gate','mean_entropy','fusion','mean_entropy')
        ci_reduction=interval(region,'value_gate','learned_omission_density','value_gate','fn_probability_density')
        for i,(name,number,unit,denominator,decimals,ci) in enumerate([
            ('Fusion minus SAR IoU',gain,'percentage points','IoU is TP/(TP+FP+FN)',2,None),
            ('Value minus fusion residual FN at 10% H review',inversion,'km2','Same regional valid-area budget',2,ci_inversion),
            ('Learned minus proxy residual FN on fixed value map',reduction,'km2','Same fixed map and regional valid-area budget',2,ci_reduction),
            ('Relative FN reduction replacing learned ranking',relative,'percent','Learned-ranking residual FN: '+repr(float(learned.residual_fn_km2))+' km2',1,None)]):
            rounded=float(f'{number:.{decimals}f}')
            claims.append(dict(region=region,claim=name,value=float(number),unit=unit,denominator=denominator,
                rounded=rounded,manuscript_value=target[i],decimal_places=decimals,
                rounding='Round unrounded final effect to nearest displayed decimal; no rounding of inputs',
                simultaneous_lower95=None if ci is None else float(ci.simultaneous_lower95),
                simultaneous_upper95=None if ci is None else float(ci.simultaneous_upper95),
                ci_scope='No ratio CI claimed' if i==3 else '12-estimate regional simultaneous family for review differences; map interval separately in map_paired_spatial_intervals.csv',
                passed=rounded==target[i]))
    pd.DataFrame(claims).to_csv(out/'paper_key_results_recomputed.csv',index=False)
    report=dict(passed=all(c['passed'] for c in claims),scope='Real frozen results; no synthetic data or retraining',
                claims=claims,review_rows=len(review),map_rows=len(maps),simultaneous_rows=len(intervals))
    (out/'numerical_reproduction.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    lines=['# Paper result reproduction','', 'Values are recalculated from unrounded frozen CSV records.','',
           '| Region | Effect | Recomputed | Unit | Rounded manuscript value | Status |', '|---|---|---:|---|---:|---|']
    for c in claims:lines.append(f"| {c['region']} | {c['claim']} | {c['value']:.12g} | {c['unit']} | {c['rounded']} | {'PASS' if c['passed'] else 'FAIL'} |")
    (out/'PAPER_RESULTS_RECOMPUTED.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    if not report['passed']:raise ValueError('Manuscript/frozen-data mismatch; see numerical_reproduction.json')
    print(json.dumps(dict(passed=True,claims=len(claims),map_rows=len(maps),review_rows=len(review))),flush=True)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();reproduce(args.data,args.output)
