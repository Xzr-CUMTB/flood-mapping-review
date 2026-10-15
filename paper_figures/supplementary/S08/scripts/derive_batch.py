"""Read-only frozen-result derivations for supplementary S6, S8, S9 and S10."""
import os
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
from pathlib import Path
import sys,json,hashlib,shutil,time
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from pyproj import Transformer
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'configs/core_release_s17.json').exists())
sys.path.insert(0,str(ROOT))
from experiments.evaluation_metrics import whole_unit_selection,spatial_block_ids,paired_block_bootstrap,simultaneous_bands
BASE=ROOT/'figures/supplementary';E=ROOT/'runs/core_s17_evaluation';REG=['Beijing_Baoding','Harbin']
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def js(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def wj(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False),encoding='utf-8')
def log(s):print(datetime.now().isoformat(timespec='seconds'),s,flush=True)
RELEASE=sha(ROOT/'configs/core_release_s17.json')
PANELS={6:{'a':['reliability.csv','calibration_metrics.csv','calibration_provenance.json'],'b':['confidence_spectrum.csv'],'c':['operating_points.csv'],'d':['operating_points.csv']},7:{'a':['geographic_atlas.npz','geographic_atlas_manifest.json','atlas_marginals.csv'],'b':['transition_areas.csv'],'c':['components.csv.gz','component_summary.csv'],'d':['boundary_distance.csv','spatial_methods.json']},8:{'a':['joint_score_density.csv','risk_distribution_summary.csv'],'b':['risk_deciles.csv','risk_head_provenance.json'],'c':['fixed_map_review_results.csv'],'d':['fixed_map_review_results.csv'],'e':['simultaneous_intervals.csv'],'f':['oracle_gaps.csv','risk_deciles.csv']},9:{'a':['all_review_results.csv'],'b':['all_review_results.csv'],'c':['pareto_candidates.csv'],'d':['loss_winners.csv'],'e':['rank_trajectories.csv'],'f':['selection_recheck.csv','all_review_results.csv']},10:{'a':['spatial_blocks.csv','patch_assignments.csv'],'b':['primary_simultaneous_intervals.csv','contrast_key.csv'],'c':['scale_sensitivity_intervals.csv'],'d':['block_contributions.csv.gz','influence_summary.csv','integrity_ledger.csv']}}
CONTRACTS={6:('Probability calibration and operating-point transfer',165,'Calibration changes probability interpretation; frozen operating-point transfer changes binary FN/FP trade-offs.','Reliability with aligned bin-area marginals; threshold spectra; development-to-core dumbbells; three actual operating points.','Raw/calibrated binary maps must coincide at 0.5; C(tau) uses reference flood area; core thresholds never re-fit.'),7:('Spatial geometry of persistent, recovered and newly missed flooding',190,'Sensor fusion simultaneously recovers and introduces spatially structured flood omissions.','Full-region state atlas with exact marginals; four-state area matrix; component-area CCDFs; censored boundary-distance profiles.','Preserve native gaps; four/eight-neighbour connectivity; coverage interfaces never treated as observed flood boundaries.'),8:('Omission-risk reliability and complete review-policy diagnostics',205,'A learned omission score can be poorly calibrated and yield region-dependent review gains.','Weighted score-density joints; descriptive risk strata; full fixed-map budget matrix; capture curves; simultaneous contrasts; oracle gaps.','No independent OOF risk-head predictions exist: do not invent development reliability. Test strata are descriptive, not calibration fitting.'),9:('Full budget–strategy–cost landscape',220,'Model and review-policy rankings depend on budget and the relative penalty for false alarms.','Complete mean-entropy model matrix; all fixed-map policy deltas; Pareto candidates; full-model cost winners; rank paths; independent conservation audit.','Four actual budgets, all available policy/model records; absence is never zero; oracle is label-guided density ranking, not a deployable policy.'),10:('Paired spatial inference and sensitivity',175,'All six paired contrasts are reported with their within-region spatial uncertainty and scale sensitivity.','Actual block atlas; 24-estimate simultaneous forest; additional 5/20 km simultaneous sensitivity; signed block influence and integrity ledger.','Twelve-estimate family separately within each region. Fixed selected units; spatial composition uncertainty is not independent-event replication.')}
def setup():
 for n,(title,h,claim,structure,guard) in CONTRACTS.items():
  d=BASE/f'S{n:02d}'
  for sub in ['source_data','assets','scripts']:(d/sub).mkdir(parents=True,exist_ok=True)
  (d/f'S{n:02d}_design_notes.md').write_text(f'# Supplementary Figure S{n}: {title}\n\n## Scientific contract (before rendering)\n{claim}\n\n## Evidence and composition\n{structure}\n\n## Statistical and geographic definitions\n{guard}\n\n## Export contract\n180 × {h} mm. Times New Roman and Arial companion; 7 pt data text and 8 pt panel headings, black inward axes; editable PDF/SVG and lossless 1200-dpi PNG/LZW TIFF. Three rendered review rounds. Raw numerical source data and all contrary effects retained.\n\n## Preservation\nAll original frozen files and existing main/supplementary figures remain unchanged. New derivations are retrospective descriptive analyses. No retraining, prediction regeneration, or test-based model selection.\n',encoding='utf-8')
  pd.DataFrame([dict(panel=p,source_file=x,release_sha256=RELEASE) for p,files in PANELS[n].items() for x in files]).to_csv(d/f'S{n:02d}_panel_data_manifest.csv',index=False)
 # Preserve exact snapshots of task briefs alongside the contracts.
def save(n,name,df):
 if isinstance(df,pd.DataFrame):df.to_csv(BASE/f'S{n:02d}/source_data'/name,index=False)
 else:wj(BASE/f'S{n:02d}/source_data'/name,df)
def copy(n,src,name=None):shutil.copy2(src,BASE/f'S{n:02d}/source_data'/(name or Path(src).name))
def main():
 setup();proof=js(ROOT/'reports/completed_batch_verification.json');assert proof['passed'] and proof['release_sha256']==RELEASE
 inputs=['configs/core_release_s17.json','configs/evaluation_protocol_v1.json','runs/core_s17_evaluation/map_metrics.csv','runs/core_s17_evaluation/review_budget_results.csv','runs/core_s17_evaluation/review_simultaneous_intervals.csv','runs/core_s17_evaluation/matched_operating_point_metrics.csv','runs/oof_calibration_s17/complete.json','runs/core_s17_predictions/prediction_manifest.csv']
 for p in ['map_metrics.csv','review_budget_results.csv','review_simultaneous_intervals.csv','matched_operating_point_metrics.csv']:
  assert sha(E/p)==proof['csv_sha256'][p],p
 bind=[dict(path=p,sha256=sha(ROOT/p)) for p in inputs]
 maps=pd.read_csv(E/'map_metrics.csv');review=pd.read_csv(E/'review_budget_results.csv');sim=pd.read_csv(E/'review_simultaneous_intervals.csv');cal=js(ROOT/'runs/oof_calibration_s17/complete.json')
 log('Frozen CSV hashes verified; contracts written')
 # S6: exact saved area-weighted reliability and all three pre-specified confidence thresholds.
 rel=[];metrics=[];spect=[]
 for region in REG:
  for model in ['sar','fusion','value_gate']:
   for typ,suf in [('Raw',''),('Calibrated','_temperature')]:
    name=model+suf;r=pd.read_csv(E/f'{region}_{name}_reliability.csv');r['region']=region;r['model']=model;r['version']=typ;r['temperature']=cal['temperatures'][model]['temperature'];r['release_sha256']=RELEASE;rel.append(r)
    m=maps[(maps.region==region)&(maps.model==name)].iloc[0].to_dict();m.update(base_model=model,version=typ,temperature=cal['temperatures'][model]['temperature'],release_sha256=RELEASE);metrics.append(m)
    assert np.isclose(r.area_m2.sum()/1e6,m['valid_area_km2'])
    for tau,key in [(.05,'confident_omission_005'),(.1,'confident_omission_01'),(.2,'confident_omission_02')]:spect.append(dict(region=region,model=model,version=typ,tau=tau,omission_fraction=m[key],omission_km2=m[key]*m['true_flood_km2'],true_flood_km2=m['true_flood_km2'],release_sha256=RELEASE))
   a=maps[(maps.region==region)&(maps.model==model)].iloc[0];b=maps[(maps.region==region)&(maps.model==model+'_temperature')].iloc[0]
   assert all(a[k]==b[k] for k in ['flood_iou','fn_km2','fp_km2','recall','precision'])
 op=pd.read_csv(E/'matched_operating_point_metrics.csv');raw=maps[maps.model.isin(['sar','fusion','value_gate'])].copy();raw['scenario']='raw_0.5';raw['threshold']=.5;raw['false_positive_rate']=raw.fp_km2/(raw.valid_area_km2-raw.true_flood_km2);op=pd.concat([raw,op],ignore_index=True);op['release_sha256']=RELEASE
 for r in op[op.scenario!='raw_0.5'].itertuples():assert np.isclose(r.threshold,cal['operating_points'][r.scenario][r.model]['threshold'],atol=1e-15,rtol=0)
 save(6,'reliability.csv',pd.concat(rel));save(6,'calibration_metrics.csv',pd.DataFrame(metrics));save(6,'confidence_spectrum.csv',pd.DataFrame(spect));save(6,'operating_points.csv',op);save(6,'calibration_provenance.json',cal)
 # S8: existing full unit-level source, independently recomputed weighted joint and bin aggregates.
 p5=ROOT/'figures/main/Fig05/source_data';u=pd.read_csv(p5/'review_units.csv.gz');density=[];dec=[];summary=[]
 for region,g in u.groupby('region',sort=False):
  cost=g.cost_m2.to_numpy();score=g.L.to_numpy();actual=g.fn_m2.to_numpy()/cost;edges=np.linspace(0,1,31)
  hist=np.histogram2d(score,actual,bins=[edges,edges],weights=cost)[0]
  for i in range(30):
   for j in range(30):density.append(dict(region=region,score_left=edges[i],score_right=edges[i+1],density_left=edges[j],density_right=edges[j+1],area_km2=hist[i,j]/1e6))
  assert np.isclose(hist.sum(),cost.sum());assert np.allclose(actual,g.D)
  summary.append(dict(region=region,units=len(g),valid_area_km2=cost.sum()/1e6,fn_km2=g.fn_m2.sum()/1e6,zero_FN_units=int((actual==0).sum()),zero_FN_area_fraction=cost[actual==0].sum()/cost.sum(),mean_score=np.average(score,weights=cost),observed_density=np.average(actual,weights=cost)))
  for d,q in g.groupby('risk_decile',sort=True):dec.append(dict(region=region,risk_decile=d,valid_area_km2=q.cost_m2.sum()/1e6,mean_score=np.average(q.L,weights=q.cost_m2),observed_density=q.fn_m2.sum()/q.cost_m2.sum(),zero_FN_area_fraction=q.loc[q.fn_m2==0,'cost_m2'].sum()/q.cost_m2.sum(),FN_share=q.fn_m2.sum()/g.fn_m2.sum()))
 save(8,'joint_score_density.csv',pd.DataFrame(density));save(8,'risk_deciles.csv',pd.DataFrame(dec));save(8,'risk_distribution_summary.csv',pd.DataFrame(summary));copy(8,p5/'risk_head_provenance.json')
 fixed=review[review.model=='value_gate'].copy();fixed['captured_FN_fraction']=1-fixed.residual_fn_km2/fixed.initial_fn_km2
 entropy=fixed[fixed.policy=='mean_entropy'][['region','budget_fraction','residual_fn_km2']].rename(columns={'residual_fn_km2':'entropy_residual_FN_km2'});fixed=fixed.merge(entropy,on=['region','budget_fraction']);fixed['delta_FN_vs_entropy_km2']=fixed.residual_fn_km2-fixed.entropy_residual_FN_km2
 oracle=fixed[fixed.policy=='oracle_fn_density'][['region','budget_fraction','residual_fn_km2']].rename(columns={'residual_fn_km2':'oracle_residual_FN_km2'});gaps=fixed.merge(oracle,on=['region','budget_fraction']);gaps['FN_gap_to_oracle_km2']=gaps.residual_fn_km2-gaps.oracle_residual_FN_km2
 save(8,'fixed_map_review_results.csv',fixed);save(8,'oracle_gaps.csv',gaps);save(8,'simultaneous_intervals.csv',sim);copy(8,p5/'review_units.csv.gz')
 # S9: exhaustive actual candidates and independent selection / conservation recheck.
 save(9,'all_review_results.csv',review);save(9,'pareto_candidates.csv',review[review.budget_fraction==.1]);winners=[];rank=[]
 for (region,budget),g in review[review.policy=='mean_entropy'].groupby(['region','budget_fraction']):
  for weight in [.25,1.,4.]:
   q=g.assign(loss=g.residual_fn_km2+weight*g.residual_fp_km2).sort_values(['loss','model']);b=q.iloc[0];s=q.iloc[1];winners.append(dict(region=region,budget_fraction=budget,lambda_FP=weight,winner=b.model,best_loss_km2=b.loss,runner_up=s.model,runner_up_loss_km2=s.loss,gap_km2=s.loss-b.loss,candidates=len(g),ties=int((q.loss==b.loss).sum())))
  core=g[g.model.isin(['sar','fusion','ordinary_gate','value_gate'])].copy();core['rank']=core.residual_fn_km2.rank(method='min').astype(int);core['gap_to_best_km2']=core.residual_fn_km2-core.residual_fn_km2.min();rank.append(core)
 save(9,'loss_winners.csv',pd.DataFrame(winners));save(9,'rank_trajectories.csv',pd.concat(rank))
 schema=js(E/'array_schema.json');models=schema['models'];fields=schema['unit_fields'];unit=np.load(E/'review_units.npy',mmap_mode='r');frame=pd.read_csv(ROOT/'runs/core_s17_predictions/prediction_manifest.csv');frame['region']=np.where(frame.provisional_site=='Harbin','Harbin','Beijing_Baoding');patch=np.load(E/'patch_metric_sums.npy',mmap_mode='r');protocol=js(ROOT/'configs/evaluation_protocol_v1.json');contrib={};rechecks=[];packed=np.load(E/'primary_review_selections_packed.npz')
 for region in REG:
  mask=(frame.region==region).to_numpy()
  for m,model in enumerate(models):
   allvals=np.asarray(unit[m,mask]).reshape(-1,len(fields));good=allvals[:,0]>0;vals=allvals[good];cost,fn,fp=vals[:,:3].T
   group=review[(review.region==region)&(review.model==model)]
   scores={p:vals[:,fields.index(p)] for p in group.policy.unique() if p in fields}
   scores.update({f'random_seed_{s}':np.random.default_rng(s).random(len(cost)) for s in [17,29,43]});scores['oracle_fn_density']=fn/cost
   for rec in group.itertuples():
    chosen,used=whole_unit_selection(scores[rec.policy],cost,rec.budget_fraction);correct_fn=fn[chosen].sum()/1e6;correct_fp=fp[chosen].sum()/1e6;calc_fn=fn[~chosen].sum()/1e6;calc_fp=fp[~chosen].sum()/1e6
    assert np.isclose(calc_fn,rec.residual_fn_km2,atol=1e-10) and np.isclose(calc_fp,rec.residual_fp_km2,atol=1e-10)
    assert chosen.sum()==rec.selected_units and np.isclose(used/cost.sum(),rec.actual_area_fraction,rtol=1e-12)
    rechecks.append(dict(region=region,model=model,policy=rec.policy,budget_fraction=rec.budget_fraction,actual_area_fraction=used/cost.sum(),selected_units=int(chosen.sum()),corrected_fn_km2=correct_fn,corrected_fp_km2=correct_fp,fn_conservation_error_km2=rec.initial_fn_km2-rec.residual_fn_km2-correct_fn,fp_conservation_error_km2=rec.initial_fp_km2-rec.residual_fp_km2-correct_fp))
    if rec.budget_fraction==.1:
     key=(region,model,rec.policy);assert np.array_equal(chosen,np.unpackbits(packed['|'.join(key)])[:len(chosen)].astype(bool))
     arr=np.zeros((mask.sum()*16,2));arr[good,0]=fn*(~chosen);arr[good,1]=fp*(~chosen);contrib[key]=arr.reshape(-1,16,2).sum(1)/1e6
   log('Independent selections '+region+' '+model)
 save(9,'selection_recheck.csv',pd.DataFrame(rechecks));assert len(rechecks)==1824
 # S10: exact paired selected-unit contributions; same per-region family at each sensitivity scale.
 contrasts=protocol['confirmatory_review_contrasts'];scales=[];blockrows=[];patchrows=[];conrows=[];inf=[]
 names=['Value/H − Fusion/H','Value/H − Ordinary/H','Value/L − Value/H','Value/L − Value/P','Value/L − Value/S','Value/L − Fusion/L']
 save(10,'contrast_key.csv',pd.DataFrame([dict(contrast=f'C{i+1}',label=names[i],first_model=a,first_policy=pa,second_model=b,second_policy=pb) for i,(a,pa,b,pb) in enumerate(contrasts)]))
 for region in REG:
  mask=(frame.region==region).to_numpy();f=frame[mask];crs='EPSG:32652' if region=='Harbin' else 'EPSG:32650';trans=Transformer.from_crs(4326,crs,always_xy=True);xx,yy=trans.transform(f.lon.to_numpy(),f.lat.to_numpy());delta=np.stack([contrib[(region,a,pa)]-contrib[(region,b,pb)] for a,pa,b,pb in contrasts],axis=1)
  for size in [10000,5000,20000]:
   ids=np.asarray(spatial_block_ids(f.lon.to_numpy(),f.lat.to_numpy(),crs,size));keys,inv=np.unique(ids,axis=0,return_inverse=True);tot=np.zeros((len(keys),6,2));np.add.at(tot,inv,delta);point=delta.sum(0);draws=np.concatenate(list(paired_block_bootstrap(tot,2000,20261012)));lo,hi,crit,se=simultaneous_bands(point,draws)
   for j in range(6):
    for k,end in enumerate(['residual_fn_km2','residual_fp_km2']):
     a,pa,b,pb=contrasts[j];row=dict(region=region,contrast=f'C{j+1}',first_model=a,first_policy=pa,second_model=b,second_policy=pb,endpoint=end,difference_first_minus_second=point[j,k],simultaneous_lower95=lo[j,k],simultaneous_upper95=hi[j,k],bootstrap_se=se[j,k],maximum_deviation_critical_value=crit,family_estimates=12,block_size_m=size,occupied_blocks=len(keys),budget_fraction=.1,analysis='frozen-primary-reproduction' if size==10000 else 'additional-descriptive-simultaneous-sensitivity');scales.append(row)
     if size==10000:
      old=sim[(sim.region==region)&(sim.first_model==a)&(sim.first_policy==pa)&(sim.second_model==b)&(sim.second_policy==pb)&(sim.endpoint==end)].iloc[0]
      assert all(np.isclose(row[key],old[key],atol=1e-10,rtol=1e-10) for key in ['difference_first_minus_second','simultaneous_lower95','simultaneous_upper95','bootstrap_se','maximum_deviation_critical_value'])
   if size==10000:
    m=np.asarray(patch[models.index('value_gate'),mask]);area=m[:,:4].sum(1)/1e6;flood=(m[:,0]+m[:,2])/1e6;ba=np.bincount(inv,weights=area);bf=np.bincount(inv,weights=flood);counts=np.bincount(inv)
    for i,key in enumerate(keys):
     blockrows.append(dict(region=region,epsg=crs,block_x=int(key[0]),block_y=int(key[1]),left_m=key[0]*size,bottom_m=key[1]*size,block_size_m=size,valid_area_km2=ba[i],flood_area_km2=bf[i],patches=int(counts[i])))
     for j in range(6):
      for k,end in enumerate(['FN','FP']):conrows.append(dict(region=region,block_x=int(key[0]),block_y=int(key[1]),contrast=f'C{j+1}',endpoint=end,signed_contribution_km2=tot[i,j,k]))
    for i,r in enumerate(f.itertuples()):patchrows.append(dict(region=region,prediction_index=r.prediction_index,sample_id=r.sample_id,longitude=r.lon,latitude=r.lat,x_m=xx[i],y_m=yy[i],block_x=int(ids[i,0]),block_y=int(ids[i,1]),valid_area_km2=area[i],flood_area_km2=flood[i]))
    for j in range(6):
     for k,end in enumerate(['FN','FP']):
      v=tot[:,j,k];absval=np.sort(np.abs(v))[::-1];n=max(1,int(np.ceil(len(v)*.1)));inf.append(dict(region=region,contrast=f'C{j+1}',endpoint=end,blocks=len(v),signed_total_km2=v.sum(),absolute_total_km2=absval.sum(),top10pct_blocks=n,top10pct_absolute_share=absval[:n].sum()/absval.sum(),largest_absolute_contribution_km2=absval[0],leave_one_block_out_min_km2=(v.sum()-v).min(),leave_one_block_out_max_km2=(v.sum()-v).max()))
   log(f'Paired family reproduced {region} {size} m, {len(keys)} blocks')
 copy(10,E/'review_simultaneous_intervals.csv','primary_simultaneous_intervals.csv');save(10,'scale_sensitivity_intervals.csv',pd.DataFrame(scales));save(10,'spatial_blocks.csv',pd.DataFrame(blockrows));save(10,'patch_assignments.csv',pd.DataFrame(patchrows));save(10,'block_contributions.csv.gz',pd.DataFrame(conrows));save(10,'influence_summary.csv',pd.DataFrame(inf));copy(10,ROOT/'configs/evaluation_protocol_v1.json','evaluation_protocol.json');copy(10,ROOT/'reports/completed_batch_verification.json','completed_verification_record.json');copy(10,BASE/'S03/source_data/integrity_counts.csv','training_integrity_prior.csv')
 save(10,'integrity_ledger.csv',pd.DataFrame([dict(item='Completed training jobs',observed=38,expected=38,verification='S03 training checkpoint audit, inherited and hash-bound'),dict(item='Frozen release files',observed=142,expected=142,verification='S03 actual 142-file hash audit, inherited and hash-bound'),dict(item='Core patches',observed=len(frame),expected=8233,verification='current prediction manifest'),dict(item='Review combinations reselected',observed=len(rechecks),expected=1824,verification='current whole-unit independent rerun'),dict(item='Primary simultaneous effects reproduced',observed=len(sim),expected=24,verification='current paired bootstrap reproduction'),dict(item='Scale-sensitive simultaneous effects',observed=len(scales),expected=72,verification='current 5/10/20 km derivation')]))
 for n in [6,8,9,10]:
  sd=BASE/f'S{n:02d}/source_data';add=[]
  if n==8:add=[p5/'review_units.csv.gz',p5/'risk_head_provenance.json']
  if n==10:add=[BASE/'S03/source_data/frozen_files_verified.json',BASE/'S03/source_data/prediction_hash_recheck.json',BASE/'S03/source_data/training_rechecks.csv']
  bindings=bind+[dict(path=str(p.relative_to(ROOT)).replace('\\','/'),sha256=sha(p)) for p in add]
  wj(sd/'input_bindings.json',bindings);sources={p.name:sha(p) for p in sd.iterdir() if p.is_file() and p.name!='scientific_audit.json'}
  wj(sd/'scientific_audit.json',dict(passed=True,release_sha256=RELEASE,completed_at_utc=datetime.now(timezone.utc).isoformat(),source_files=sources,models_retrained=False,core_predictions_regenerated=False,derivation_script_sha256=sha(__file__)))
 log('Completed S6/S8/S9/S10 source data and independent selection/interval audits')
if __name__=='__main__':main()
