"""Check rounded SI numerical tables against the actual full-precision sources."""
from pathlib import Path
import argparse,json,re
import numpy as np
import pandas as pd

def verify(data,out):
 data=Path(data);out=Path(out);out.mkdir(parents=True,exist_ok=True);checks=[]
 def equal(table,row,field,printed,actual,decimals):
  a=float(str(printed).replace('−','-').replace(',',''));expected=float(f'{float(actual):.{decimals}f}')
  checks.append(dict(table=table,row=int(row),field=field,manuscript=a,rounded_source=expected,full_precision=float(actual),passed=bool(np.isclose(a,expected,atol=10**(-decimals-3),rtol=0))))
 maps=pd.read_csv(data/'evaluation/map_metrics.csv');review=pd.read_csv(data/'evaluation/review_budget_results.csv');op=pd.read_csv(data/'evaluation/matched_operating_point_metrics.csv');ci=pd.read_csv(data/'evaluation/review_simultaneous_intervals.csv')
 names=maps[maps.region.eq('Beijing_Baoding')].model.tolist()
 for n,region in [('S3a','Beijing_Baoding'),('S3b','Harbin')]:
  table=pd.read_csv(data/'manuscript_tables'/('Table_'+n+'.csv'));source=maps[maps.region==region].set_index('model')
  assert len(table)==len(names)==16
  for i,row in table.iterrows():
   for col,key,d in [('IoU','flood_iou',4),('FN (km²)','fn_km2',2),('FP (km²)','fp_km2',2),('Brier','brier',5),(table.columns[-1],'confident_omission_01',4)]:equal(n,i,col,row[col],source.loc[names[i],key],d)
 oof=pd.read_csv(data/'figure_data/S04/source_data/all_oof_metrics.csv')
 table=pd.read_csv(data/'manuscript_tables/Table_S4.csv');groups=['2020_flood_season','2021_Henan','2024_April','2024_June'];models=['sar','fusion','probability_average','sigmoid_value','analytic_value','ridge_value_001']
 assert len(table)==24
 for i,row in table.iterrows():
  s=oof[(oof.event_group==groups[i//6])&(oof.model==models[i%6])]
  assert len(s)==1;s=s.iloc[0]
  for col,key,d in [('IoU','flood_iou',4),('Brier','brier',4),('FN (km²)','fn_km2',2),('FP (km²)','fp_km2',2)]:equal('S4',i,col,row[col],s[key],d)
 for n,region in [('S5a','Beijing_Baoding'),('S5b','Harbin')]:
  table=pd.read_csv(data/'manuscript_tables'/('Table_'+n+'.csv'))
  for i,row in table.iterrows():
   scenario,model=row['Target/map'].split('/');s=op[(op.region==region)&(op.scenario=={'FPR':'matched_fpr','Recall':'matched_recall'}[scenario])&(op.model=={'SAR':'sar','Fusion':'fusion','Value':'value_gate'}[model])]
   assert len(s)==1;s=s.iloc[0]
   for col,key,d in [('Threshold','threshold',5),('FPR','false_positive_rate',5),('Recall','recall',4),('FN (km²)','fn_km2',2),('FP (km²)','fp_km2',2)]:equal(n,i,col,row[col],s[key],d)
 contrast=pd.read_csv(data/'figure_data/S10/source_data/contrast_key.csv').set_index('contrast')
 for n,region in [('S6a','Beijing_Baoding'),('S6b','Harbin')]:
  table=pd.read_csv(data/'manuscript_tables'/('Table_'+n+'.csv'))
  for i,row in table.iterrows():
   c=contrast.loc[row.Contrast];s=ci[(ci.region==region)&(ci.first_model==c.first_model)&(ci.first_policy==c.first_policy)&(ci.second_model==c.second_model)&(ci.second_policy==c.second_policy)&(ci.endpoint=={'FN':'residual_fn_km2','FP':'residual_fp_km2'}[row.Endpoint])]
   assert len(s)==1;s=s.iloc[0];equal(n,i,'Difference',row['Difference (km²)'],s.difference_first_minus_second,2)
   lo,hi=re.findall(r'-?\d+(?:\.\d+)?',row['Simultaneous 95% interval (km²)'].replace('−','-'))
   equal(n,i,'Interval lower',lo,s.simultaneous_lower95,2);equal(n,i,'Interval upper',hi,s.simultaneous_upper95,2)
 table=pd.read_csv(data/'manuscript_tables/Table_S7.csv')
 for i,row in table.iterrows():
  model,policy=row['Map/ranking'].split('/');s=review[(review.region=={'Beijing–Baoding':'Beijing_Baoding','Harbin':'Harbin'}[row.Region])&(review.model=={'Fusion':'fusion','Ordinary':'ordinary_gate','Value':'value_gate'}[model])&(review.policy=={'H':'mean_entropy','L':'learned_omission_density','P':'fn_probability_density'}[policy])&np.isclose(review.budget_fraction,.1)]
  assert len(s)==1;equal('S7',i,'Residual FN',row['Residual FN (km²)'],s.iloc[0].residual_fn_km2,2)
 report=dict(passed=all(c['passed'] for c in checks),numeric_cells_checked=len(checks),tables=['S3a','S3b','S4','S5a','S5b','S6a','S6b','S7'],
  descriptive_tables='S1 dates, S2 settings and S8 definitions are extracted verbatim with their original metadata sources, rather than treated as inferential result cells.',failures=[c for c in checks if not c['passed']],checks=checks)
 (out/'manuscript_table_verification.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
 print(json.dumps({k:report[k] for k in ['passed','numeric_cells_checked','tables','failures']}),flush=True)
 if not report['passed']:raise ValueError('Manuscript table mismatch; scientific results were not modified')
 return report

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();verify(a.data,a.output)
