"""Keep panel tables minimal and record the exact focal rank comparison."""
import os,json
from pathlib import Path
import pandas as pd
SD=Path(os.path.abspath(__file__)).parents[1]/'source_data'
a=pd.read_csv(SD/'panel_a_budget_response.csv')
cols=['region','model','policy','budget_fraction','actual_area_fraction','selected_units','initial_fn_km2','corrected_fn_km2','residual_fn_km2','initial_fp_km2','corrected_fp_km2','residual_fp_km2','used_area_km2','total_valid_area_km2','interval','release_sha256']
a[cols].to_csv(SD/'panel_a_budget_response.csv',index=False)
r=pd.read_csv(SD/'panel_b_rankings.csv');rows=[]
for region,g in r.groupby('region'):
    q=g.set_index('model');f=q.loc['fusion'];v=q.loc['value_gate'];rows.append(dict(region=region,first='value_gate',second='fusion',delta_iou=v.flood_iou-f.flood_iou,delta_residual_fn_km2=v.residual_fn_km2-f.residual_fn_km2,inversion=bool(v.flood_iou>f.flood_iou and v.residual_fn_km2>f.residual_fn_km2),same_policy='mean_entropy',budget_fraction=.1,rank_set='SAR;Fusion;Ordinary;Value',release_sha256=v.release_sha256))
pd.DataFrame(rows).to_csv(SD/'panel_b_inversion_comparisons.csv',index=False)
assert all(r['inversion'] for r in rows)
print('Panel-specific source tables finalized')
