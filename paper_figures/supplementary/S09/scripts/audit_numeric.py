"""Independent arithmetic and input-array checks for S6--S10, after derivation."""
from derive_batch import *
def main():
 result={};maps=pd.read_csv(E/'map_metrics.csv');orig=pd.read_csv(E/'review_budget_results.csv');protocol=js(ROOT/'configs/evaluation_protocol_v1.json');schema=js(E/'array_schema.json');allunits=np.load(E/'review_units.npy',mmap_mode='r');fields=schema['unit_fields'];release=js(ROOT/'configs/core_release_s17.json');common=[]
 for name in ['array_schema.json','review_units.npy','patch_metric_sums.npy','primary_review_selections_packed.npz']:
  p=E/name;common.append(dict(path=str(p.relative_to(ROOT)).replace('\\','/'),sha256=sha(p)))
 for n in range(6,11):
  sd=BASE/f'S{n:02d}/source_data';read=lambda name:pd.read_csv(sd/name);check={};extra=[]
  if n==6:
   m=read('calibration_metrics.csv');r=read('reliability.csv');s=read('confidence_spectrum.csv');op=read('operating_points.csv');cal=js(sd/'calibration_provenance.json')
   check['12_probability_outputs']=len(m)==12 and len(r)==240 and len(s)==36
   check['development_only_temperature_fit']=cal['samples']==8786 and '2023_holdout' not in cal['source_provenance']['training_groups'] and all(q['temperature']>0 for q in cal['temperatures'].values())
   errs=[]
   for rec in m.itertuples():
    original=maps[(maps.region==rec.region)&(maps.model==rec.model)].iloc[0]
    errs.extend(abs(getattr(rec,k)-original[k]) for k in ['flood_iou','fn_km2','fp_km2','brier','nll'])
    path=E/f'{rec.region}_{rec.model}_reliability.csv';rr=pd.read_csv(path);q=r[(r.region==rec.region)&(r.model==rec.base_model)&(r.version==rec.version)].sort_values('bin');assert np.allclose(q[['area_m2','mean_probability','flood_fraction']],rr[['area_m2','mean_probability','flood_fraction']],rtol=1e-12,atol=1e-12);extra.append(dict(path=str(path.relative_to(ROOT)).replace('\\','/'),sha256=sha(path)))
   check['all_metrics_and_240_reliability_bins_match_original']=max(errs)<1e-10
   check['calibrated_binary_invariance']=all(np.allclose(m[(m.region==reg)&(m.base_model==mod)][['flood_iou','fn_km2','fp_km2']].iloc[0],m[(m.region==reg)&(m.base_model==mod)][['flood_iou','fn_km2','fp_km2']].iloc[1],atol=1e-12) for reg in REG for mod in ['sar','fusion','value_gate'])
   check['36_confident_omission_denominators']=np.allclose(s.omission_km2,s.omission_fraction*s.true_flood_km2)
   check['18_real_operating_points']=len(op)==18 and len(op.groupby(['region','model','scenario']))==18
   check['fpr_recomputed_from_nonflood_area']=np.allclose(op.false_positive_rate,op.fp_km2/(op.valid_area_km2-op.true_flood_km2))
   check['recall_recomputed']=np.allclose(op.recall,1-op.fn_km2/op.true_flood_km2)
  elif n==7:
   t=read('transition_areas.csv');c=read('components.csv.gz');d=read('boundary_distance.csv');a=np.load(sd/'geographic_atlas.npz');spec=js(sd/'geographic_atlas_manifest.json');mg=read('atlas_marginals.csv');summary=read('component_summary.csv');sts=['stable_detection','recovered','newly_missed','persistent_omission']
   check['all_four_states_in_two_regions']=len(t)==8 and set(t.state)==set(sts)
   ok=True
   for reg in REG:
    q=t[t.region==reg].set_index('state');sarm=maps[(maps.region==reg)&(maps.model=='sar')].iloc[0];fm=maps[(maps.region==reg)&(maps.model=='fusion')].iloc[0];ok&=np.isclose((q.loc['recovered','area_m2']+q.loc['persistent_omission','area_m2'])/1e6,sarm.fn_km2) and np.isclose((q.loc['newly_missed','area_m2']+q.loc['persistent_omission','area_m2'])/1e6,fm.fn_km2) and np.isclose(q.area_m2.sum()/1e6,fm.true_flood_km2)
    cubes=[a[z['grid_id']] for z in spec if z['region']==reg];assert np.isclose(sum(x.sum() for x in cubes)/1e6,fm.valid_area_km2)
    for j,st in enumerate(sts):
     assert np.isclose(sum(x[:,:,j+1].sum() for x in cubes),q.loc[st,'area_m2']);assert np.isclose(d[(d.region==reg)&(d.state==st)].area_m2.sum(),q.loc[st,'area_m2'])
     if j:
      for conn in [4,8]:assert np.isclose(c[(c.region==reg)&(c.state==st)&(c.connectivity==conn)].area_m2.sum(),q.loc[st,'area_m2'])
    for ax in ['longitude','latitude']:
     for st in sts:assert np.isclose(mg[(mg.region==reg)&(mg.axis==ax)][st+'_km2'].sum(),q.loc[st,'area_m2']/1e6)
   check['all_state_area_identities']=ok;check['three_native_grids_atlas_marginals_components_distances_conserve_area']=True
   check['18_component_analyses']=len(summary)==18 and len(c.groupby(['grid_id','state','connectivity']))==18
   check['positive_area_components']=bool((c.area_m2>0).all() and (c.pixels>=1).all())
   check['96_boundary_bins_including_censoring']=len(d)==96 and d.coverage_censored.sum()==8
   topology=js(ROOT/'figures/main/Fig02/source_data/geographic_topology_audit.json');check['native_topology_preserves_gap']=len(spec)==3 and topology['unaligned_touch_pairs']==[] and topology['overlap_pairs']==[] and topology['minimum_between_grid_gap_degrees']['Beijing_Baoding']>0
   p=ROOT/'figures/main/Fig02/source_data/derivation_provenance.json';extra.append(dict(path=str(p.relative_to(ROOT)).replace('\\','/'),sha256=sha(p)))
  elif n==8:
   u=read('review_units.csv.gz');q=np.asarray(allunits[schema['models'].index('value_gate')]);q=q[u.prediction_index.to_numpy().astype(int),u.unit_index.to_numpy().astype(int)]
   for col,field in [('cost_m2','cost_m2'),('fn_m2','fn_m2'),('fp_m2','fp_m2'),('L','learned_omission_density'),('H','mean_entropy'),('P','fn_probability_density')]:assert np.allclose(u[col],q[:,fields.index(field)],rtol=1e-12,atol=1e-12),col
   check['129347_units_match_frozen_array']=len(u)==129347 and len(u.unit_id.unique())==len(u)
   d=read('risk_deciles.csv');h=read('joint_score_density.csv');summ=read('risk_distribution_summary.csv')
   for reg,g in u.groupby('region'):
    w=g.cost_m2.to_numpy();v=g.L.to_numpy();order=np.argsort(v,kind='stable');sv=v[order];sw=w[order];starts=np.r_[0,np.flatnonzero(sv[1:]!=sv[:-1])+1];ends=np.r_[starts[1:],len(sv)];cum=np.r_[0,np.cumsum(sw)];ranks=np.empty(len(sv))
    for start,end in zip(starts,ends):ranks[start:end]=(cum[start]+cum[end])/(2*cum[-1])
    ranks_back=np.empty(len(sv));ranks_back[order]=ranks;derived=np.minimum((ranks_back*10).astype(int),9)+1;assert np.array_equal(derived,g.risk_decile.to_numpy().astype(int))
    for row in d[d.region==reg].itertuples():
     b=g[g.risk_decile==row.risk_decile];assert np.isclose(row.mean_score,np.average(b.L,weights=b.cost_m2));assert np.isclose(row.observed_density,b.fn_m2.sum()/b.cost_m2.sum());assert np.isclose(row.FN_share,b.fn_m2.sum()/g.fn_m2.sum())
    assert np.isclose(h[h.region==reg].area_km2.sum(),g.cost_m2.sum()/1e6)
   check['weighted_tied_midranks_and_all_20_strata_recomputed']=True;check['1800_joint_bins_conserve_area']=len(h)==1800;check['no_false_development_validation']=not js(sd/'risk_head_provenance.json')['risk_head_development_oof_predictions_available']
   rr=read('fixed_map_review_results.csv');gap=read('oracle_gaps.csv');check['120_all_policy_budget_cells']=len(rr)==120 and len(rr.policy.unique())==15;check['capture_and_oracle_gaps_recomputed']=np.allclose(rr.captured_FN_fraction,1-rr.residual_fn_km2/rr.initial_fn_km2) and np.allclose(gap.FN_gap_to_oracle_km2,gap.residual_fn_km2-gap.oracle_residual_FN_km2)
  elif n==9:
   r=read('all_review_results.csv');v=read('selection_recheck.csv');w=read('loss_winners.csv');rank=read('rank_trajectories.csv');check['all_1824_records_preserved']=len(r)==1824 and np.allclose(r[['residual_fn_km2','residual_fp_km2']],orig[['residual_fn_km2','residual_fp_km2']]);check['all_1824_independent_selections']=len(v)==1824 and max(v.fn_conservation_error_km2.abs().max(),v.fp_conservation_error_km2.abs().max())<1e-9;check['no_budget_overspend']=bool((v.actual_area_fraction<=v.budget_fraction+1e-12).all())
   for row in w.itertuples():
    g=r[(r.region==row.region)&(r.budget_fraction==row.budget_fraction)&(r.policy=='mean_entropy')].copy();g['loss']=g.residual_fn_km2+row.lambda_FP*g.residual_fp_km2;g=g.sort_values(['loss','model']);assert len(g)==16 and g.model.iloc[0]==row.winner and np.isclose(g.loss.iloc[1]-g.loss.iloc[0],row.gap_km2)
   check['24_cost_winners_among_all16_recomputed']=len(w)==24
   for _,g in r[r.policy!='oracle_fn_density'].groupby(['region','budget_fraction']):
    val=g[['residual_fn_km2','residual_fp_km2']].to_numpy();eff=np.array([not(np.all(val<=x,axis=1)&np.any(val<x,axis=1)).any() for x in val]);assert np.array_equal(eff,g.pareto_efficient_fn_fp)
   check['all_deployable_pareto_flags_recomputed']=True;check['32_core_rank_records']=len(rank)==32
  else:
   s=read('scale_sensitivity_intervals.csv');b=read('spatial_blocks.csv');p=read('patch_assignments.csv');c=read('block_contributions.csv.gz');inf=read('influence_summary.csv');old=read('primary_simultaneous_intervals.csv');ledger=read('integrity_ledger.csv')
   check['72_effects_3_scales']=len(s)==72 and set(s.block_size_m)=={5000,10000,20000} and (s.family_estimates==12).all();check['24_original_primary_effects_preserved']=np.allclose(s[s.block_size_m==10000][['difference_first_minus_second','simultaneous_lower95','simultaneous_upper95']],old[['difference_first_minus_second','simultaneous_lower95','simultaneous_upper95']],rtol=1e-10,atol=1e-10)
   check['8233_patches_400_primary_blocks']=len(p)==8233 and len(b)==400 and np.array_equal(np.floor(p[['x_m','y_m']].to_numpy()/10000).astype(int),p[['block_x','block_y']].to_numpy())
   for (reg,con,end),g in c.groupby(['region','contrast','endpoint']):
    r=s[(s.region==reg)&(s.contrast==con)&(s.endpoint==('residual_fn_km2' if end=='FN' else 'residual_fp_km2'))&(s.block_size_m==10000)].iloc[0];assert np.isclose(g.signed_contribution_km2.sum(),r.difference_first_minus_second,atol=1e-10);ir=inf[(inf.region==reg)&(inf.contrast==con)&(inf.endpoint==end)].iloc[0];v=g.signed_contribution_km2.to_numpy();assert np.isclose((v.sum()-v).min(),ir.leave_one_block_out_min_km2)
   check['4800_signed_block_contributions_conserve_24_effects']=len(c)==4800;check['six_integrity_ledger_counts_match']=bool((ledger.observed==ledger.expected).all())
   check['all_intervals_point_plus_minus_maxTse']=np.allclose(s.simultaneous_lower95,s.difference_first_minus_second-s.maximum_deviation_critical_value*s.bootstrap_se) and np.allclose(s.simultaneous_upper95,s.difference_first_minus_second+s.maximum_deviation_critical_value*s.bootstrap_se)
  check={k:bool(v) for k,v in check.items()};assert all(check.values()),(n,check)
  bindings=js(sd/'input_bindings.json');bindings.extend(common if n in [8,9,10] else []);bindings.extend(extra);bindings=list({v['path']:v for v in bindings}.values());wj(sd/'input_bindings.json',bindings);wj(sd/'numeric_recheck.json',dict(passed=True,checks=check,checked_at_utc=datetime.now(timezone.utc).isoformat(),release_sha256=RELEASE));audit=js(sd/'scientific_audit.json');audit['source_files']={p.name:sha(p) for p in sd.iterdir() if p.is_file() and p.name!='scientific_audit.json'};wj(sd/'scientific_audit.json',audit);result[n]=check;log(f'S{n:02d}: {len(check)} independent numerical checks passed')
 wj(BASE/'S06_S10_numeric_validation.json',dict(passed=True,figures=result,checked_at_utc=datetime.now(timezone.utc).isoformat(),release_sha256=RELEASE))
if __name__=='__main__':main()
