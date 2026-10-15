"""Retrospective budget-cost summaries from immutable completed-evaluation CSVs."""
import os,json,hashlib,time,shutil,sys,platform
os.environ.setdefault('MKL_THREADING_LAYER','SEQUENTIAL')
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import matplotlib,PIL
ROOT=Path(os.path.abspath(__file__)).parents[4];OUT=ROOT/'figures/main/Fig06';SD=OUT/'source_data';AS=OUT/'assets'
for p in [SD,AS]:p.mkdir(parents=True,exist_ok=True)
MODELS=['sar','fusion','ordinary_gate','value_gate'];REGIONS=['Beijing_Baoding','Harbin'];BUDGETS=[.01,.05,.1,.2];WEIGHTS=[.25,1.,4.]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(name,data):(SD/name).write_text(json.dumps(data,indent=2),encoding='utf-8')
def main():
    start=time.time();checks={};release=read(ROOT/'configs/core_release_s17.json');release_sha=sha(ROOT/'configs/core_release_s17.json');proof=read(ROOT/'reports/completed_batch_verification.json');assert proof['passed'] and proof['training_jobs']==38 and proof['release_sha256']==release_sha
    protocol=read(ROOT/'configs/evaluation_protocol_v1.json');assert protocol['budget_fractions']==BUDGETS and protocol['tradeoff_checks']['lambda_FP']==WEIGHTS and protocol['review_unit_pixels']==[64,64] and protocol['classification_threshold']==.5
    disk=shutil.disk_usage(ROOT);total=sum((Path(p)/n).stat().st_size for p,ds,fs in os.walk(ROOT) for n in fs if not (Path(p)/n).is_symlink());save('resource_check.json',dict(project_bytes_before=total,disk_free_bytes=disk.free,estimated_output_and_temporary_bytes=150_000_000,legacy_storage_cap_removed_by_user=True));assert disk.free>500_000_000
    targets=['configs/core_release_s17.json','configs/evaluation_protocol_v1.json','reports/completed_batch_verification.json','reports/13_core_computational_results.md','reports/14_implemented_computational_methods.md','experiments/evaluate_frozen_core.py','experiments/evaluation_metrics.py','runs/core_s17_predictions/prediction_manifest.csv']+['runs/core_s17_evaluation/'+n for n in ['complete.json','review_budget_results.csv','review_paired_spatial_intervals.csv','review_simultaneous_intervals.csv','map_metrics.csv','review_selection_encoding.json']]
    bindings=[]
    for name in targets:
        p=ROOT/name;digest=sha(p)
        if name in release['sha256']:assert release['sha256'][name]==digest,name
        if p.name in proof['csv_sha256']:assert proof['csv_sha256'][p.name]==digest,name
        bindings.append(dict(path=name,bytes=p.stat().st_size,sha256=digest))
    checks['release_and_evaluation_hashes_match']=True;checks['frozen_protocol_values_match']=True
    preserved=[]
    for name in ['Fig01','Fig02','Fig03','Fig04','Fig05']:
        prior=ROOT/'figures/main'/name;v=read(prior/(name+'_validation.json'));assert v['passed']
        for r in v['sha256_manifest']:assert sha(prior/r['path'])==r['sha256'],(name,r['path'])
        checks[name+'_preserved']=True;preserved.append(dict(figure=name,validation_sha256=sha(prior/(name+'_validation.json')),bound_files=len(v['sha256_manifest'])))
    save('prior_figure_preservation.json',preserved);save('input_bindings.json',bindings)
    ev=ROOT/'runs/core_s17_evaluation';raw=pd.read_csv(ev/'review_budget_results.csv');q=raw[raw.model.isin(MODELS)&raw.policy.eq('mean_entropy')].copy();assert len(q)==32 and not q.duplicated(['region','model','budget_fraction']).any();assert set(map(tuple,q[['region','model','budget_fraction']].to_numpy()))==set((r,m,b) for r in REGIONS for m in MODELS for b in BUDGETS)
    maps=pd.read_csv(ev/'map_metrics.csv');maps=maps[maps.model.isin(MODELS)].copy();assert len(maps)==8
    frame=pd.read_csv(ROOT/'runs/core_s17_predictions/prediction_manifest.csv');assert len(frame)==8233
    allrows=[];maxerror=0
    for row in q.itertuples(index=False):
        base=maps[(maps.region==row.region)&maps.model.eq(row.model)].iloc[0];assert abs(base.fn_km2-row.initial_fn_km2)<1e-9 and abs(base.fp_km2-row.initial_fp_km2)<1e-9
        assert row.actual_area_fraction<=row.budget_fraction+1e-12 and 0<=row.residual_fn_km2<=row.initial_fn_km2 and 0<=row.residual_fp_km2<=row.initial_fp_km2
        rr=q[(q.region==row.region)&q.model.eq(row.model)&q.budget_fraction.eq(row.budget_fraction)].iloc[0]
        for weight in WEIGHTS:
            loss=row.residual_fn_km2+weight*row.residual_fp_km2;err=abs(loss-rr['area_loss_lambda_'+str(weight)]);assert err<1e-9;maxerror=max(maxerror,err)
            allrows.append(dict(region=row.region,model=row.model,policy=row.policy,budget_fraction=row.budget_fraction,lambda_fp=weight,residual_fn_km2=row.residual_fn_km2,residual_fp_km2=row.residual_fp_km2,weighted_loss_km2eq=loss,actual_area_fraction=row.actual_area_fraction,reviewed_valid_area_km2=row.actual_area_fraction*base.valid_area_km2,total_valid_area_km2=base.valid_area_km2,selected_units=row.selected_units,release_sha256=release_sha))
    table=pd.DataFrame(allrows);table['rank']=table.groupby(['region','budget_fraction','lambda_fp']).weighted_loss_km2eq.rank(method='min').astype(int);atlas=[]
    for (region,budget,weight),g in table.groupby(['region','budget_fraction','lambda_fp'],sort=True):
        s=g.sort_values(['weighted_loss_km2eq','model'],kind='stable');best=float(s.weighted_loss_km2eq.iloc[0]);second=float(s.weighted_loss_km2eq.iloc[1]);ids=s.loc[s.weighted_loss_km2eq.eq(best),'model'].tolist();tied=len(ids)>1;assert (g['rank']==1).sum()==len(ids)
        atlas.append(dict(region=region,budget_fraction=budget,lambda_fp=weight,best_model=';'.join(ids),second_model=s.model.iloc[1],best_loss_km2eq=best,second_loss_km2eq=second,margin_km2eq=second-best,tied_best=tied,number_of_tied_best=len(ids),candidate_count=4,release_sha256=release_sha))
        table.loc[g.index,'minimum_loss_km2eq']=best;table.loc[g.index,'regret_km2eq']=g.weighted_loss_km2eq-best;table.loc[g.index,'tied_best']=tied;table.loc[g.index,'is_best']=g.weighted_loss_km2eq.eq(best)
    atlas=pd.DataFrame(atlas);assert len(table)==96 and len(atlas)==24 and (table.regret_km2eq>=0).all() and (atlas.margin_km2eq>=0).all()
    q['release_sha256']=release_sha;q.to_csv(SD/'verified_core_review_results.csv',index=False);maps['release_sha256']=release_sha;maps.to_csv(SD/'baseline_map_metrics.csv',index=False);table.to_csv(SD/'all_budget_cost_candidates.csv',index=False);atlas.to_csv(SD/'panel_a_decision_atlas.csv',index=False)
    checks.update(all_32_fixed_policy_results_present=True,all_96_weighted_losses_reproduced=True,all_24_candidate_minima_and_margins_valid=True,initial_errors_match_map_metrics=True,actual_area_budget_and_correction_valid=True,regret_nonnegative=True)
    param=[];curves=[];markers=[]
    for region in REGIONS:
        base=q[q.region.eq(region)&q.budget_fraction.eq(.1)].set_index('model');v=base.loc['value_gate'];f=base.loc['fusion'];dn=float(v.residual_fn_km2-f.residual_fn_km2);dp=float(v.residual_fp_km2-f.residual_fp_km2);root=-dn/dp if dp!=0 else None;positive=root if root is not None and root>0 else None
        param.append(dict(region=region,budget_fraction=.1,policy='mean_entropy',first_model='value_gate',second_model='fusion',delta_FN_km2=dn,delta_FP_km2=dp,algebraic_root_lambda=root,positive_break_even_lambda=positive,crossover_in_display_range=positive is not None and positive<=4,first_lower_FN_and_FP=dn<0 and dp<0,second_lower_FN_and_FP=dn>0 and dp>0,interpretation='Analytical point-estimate reweighting of fixed residual errors',release_sha256=release_sha))
        for weight in sorted(set(np.linspace(0,4,401).tolist()+WEIGHTS+([positive] if positive is not None and positive<=4 else []))):
            delta=dn+weight*dp;direct=(v.residual_fn_km2+weight*v.residual_fp_km2)-(f.residual_fn_km2+weight*f.residual_fp_km2);assert abs(delta-direct)<1e-10;curves.append(dict(region=region,lambda_fp=weight,delta_weighted_loss_km2eq=delta,release_sha256=release_sha))
        for weight in WEIGHTS:markers.append(dict(region=region,lambda_fp=weight,delta_weighted_loss_km2eq=dn+weight*dp,release_sha256=release_sha))
        if positive is not None:assert abs(dn+positive*dp)<1e-12
    pd.DataFrame(param).to_csv(SD/'panel_b_analytical_parameters.csv',index=False);pd.DataFrame(curves).to_csv(SD/'panel_b_analytical_lines.csv',index=False);pd.DataFrame(markers).to_csv(SD/'panel_b_preset_weight_markers.csv',index=False)
    assert param[0]['region']=='Beijing_Baoding' and param[0]['second_lower_FN_and_FP'];assert param[1]['crossover_in_display_range'];checks['pairwise_crossovers_reproduced']=True
    bands=pd.read_csv(ev/'review_simultaneous_intervals.csv');pair=bands[bands.first_model.eq('value_gate')&bands.second_model.eq('fusion')&bands.first_policy.eq('mean_entropy')&bands.second_policy.eq('mean_entropy')].copy();assert len(pair)==4;pair['release_sha256']=release_sha;pair.to_csv(SD/'pairwise_marginal_intervals_provenance.csv',index=False)
    save('uncertainty_scope.json',dict(display='point estimates and exact algebraic reweighting; no weighted-loss or root confidence intervals',joint_bootstrap_draws_saved_in_evaluation=False,marginal_FNs_and_FPs='Saved simultaneous intervals belong to the existing 12-estimate family per region; not transformed into a ratio interval',scope='Retrospective within-region core-test evaluation; not prospective model selection',core_files=[p.name for p in ev.iterdir() if p.is_file()],release_sha256=release_sha))
    c=table[table.budget_fraction.eq(.1)].copy();assert len(c)==24;c.to_csv(SD/'panel_c_primary_regret.csv',index=False)
    ledger=[]
    for r in atlas[atlas.budget_fraction.eq(.1)].itertuples(index=False):
        assert not r.tied_best
        best=c[c.region.eq(r.region)&c.lambda_fp.eq(r.lambda_fp)&c.model.eq(r.best_model)].iloc[0];runner=c[c.region.eq(r.region)&c.lambda_fp.eq(r.lambda_fp)&c.model.eq(r.second_model)].iloc[0]
        ledger.append(dict(scenario=f'{r.region}_B10_L{r.lambda_fp:g}',region=r.region,budget_fraction=.1,lambda_fp=r.lambda_fp,candidate_models=';'.join(MODELS),policy='mean_entropy',best_model=r.best_model,second_model=r.second_model,best_loss_km2eq=r.best_loss_km2eq,second_loss_km2eq=r.second_loss_km2eq,margin_km2eq=r.margin_km2eq,residual_FN_km2=best.residual_fn_km2,residual_FP_km2=best.residual_fp_km2,reviewed_valid_area_km2=best.reviewed_valid_area_km2,actual_area_fraction=best.actual_area_fraction,selected_units=int(best.selected_units),runner_up_FN_km2=runner.residual_fn_km2,runner_up_FP_km2=runner.residual_fp_km2,displayed=r.lambda_fp in [.25,4.],release_sha256=release_sha))
    ledger=pd.DataFrame(ledger);ledger.to_csv(SD/'panel_d_all_primary_scenarios.csv',index=False);ledger[ledger.displayed].to_csv(SD/'panel_d_displayed_scenarios.csv',index=False)
    save('analysis_rules.json',dict(status='Exploratory decision visualization of completed frozen evaluation',candidate_models=MODELS,policy='mean_entropy',budgets=BUDGETS,preset_FP_weights=WEIGHTS,primary_budget=.1,loss='Residual FN + lambda_FP * residual FP',loss_units='km2-equivalent',lambda_definition='Dimensionless FP weight relative to unit FN weight; evaluation preference',best='Minimum point-estimate loss over the four included candidates',margin='Second order statistic minus minimum (zero for a tied minimum)',rank='Minimum rank for exact equal full-precision losses, no rounding or jitter',regret='Candidate loss minus four-candidate minimum',continuous_lambda='Analytical line at fixed 10% review selections only; no budget interpolation',scenario_selection='Both regions, 10% budget and the two preset weight endpoints 0.25 and 4; all six primary scenarios also saved',statistical_claim='Retrospective observed comparison; model identities and crossover are point estimates',release_sha256=release_sha))
    save('scientific_audit.json',dict(passed=all(checks.values()),checks=checks,release_sha256=release_sha,max_weighted_loss_reproduction_error=maxerror,core_patches=len(frame),base_review_rows=len(q),weighted_rows=len(table),atlas_cells=len(atlas),exact_tied_best_cells=int(atlas.tied_best.sum()),primary_regret_cells=len(c),displayed_ledger_rows=4,best_candidate_counts=atlas.best_model.value_counts().to_dict(),finished_at_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.time()-start))
    (AS/'environment.json').write_text(json.dumps(dict(python=sys.version,platform=platform.platform(),numpy=np.__version__,pandas=pd.__version__,matplotlib=matplotlib.__version__,Pillow=PIL.__version__,new_dependencies_installed=False),indent=2),encoding='utf-8')
    print(json.dumps(dict(checks=checks,parameters=param,winners=atlas.best_model.value_counts().to_dict()),indent=2))
if __name__=='__main__':main()
