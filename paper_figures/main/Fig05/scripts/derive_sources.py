"""Read-only fixed-map ranking diagnostics; exploratory figure derivations only."""
import os,sys,json,hashlib,time,shutil
os.environ.setdefault('MKL_THREADING_LAYER','SEQUENTIAL')
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd
ROOT=Path(os.path.abspath(__file__)).parents[4];OUT=ROOT/'figures/main/Fig05';SD=OUT/'source_data';AS=OUT/'assets'
for p in [SD,AS]:p.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(ROOT))
from experiments.evaluation_metrics import entropy,tile_sum,whole_unit_selection
REGIONS=['Beijing_Baoding','Harbin'];POLICIES=['mean_entropy','learned_omission_density','fn_probability_density','random_seed_17','random_seed_29','random_seed_43','oracle_fn_density'];CODE={'mean_entropy':'H','learned_omission_density':'L','fn_probability_density':'P'}
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(n,x):(SD/n).write_text(json.dumps(x,indent=2),encoding='utf-8')
def log(s):print(datetime.now().isoformat(timespec='seconds'),s,flush=True)
def midrank(v,w):
    """Area-weighted empirical CDF midpoint; exact ties stay together."""
    order=np.argsort(v,kind='stable');x=v[order];ww=w[order];starts=np.r_[0,np.flatnonzero(x[1:]!=x[:-1])+1];weight=np.add.reduceat(ww,starts);mid=(weight.cumsum()-.5*weight)/weight.sum();rank=np.empty(len(v));rank[order]=np.repeat(mid,np.diff(np.r_[starts,len(v)]));return rank
def wcorr(x,y,w):
    mx=np.average(x,weights=w);my=np.average(y,weights=w);return float(np.sum(w*(x-mx)*(y-my))/np.sqrt(np.sum(w*(x-mx)**2)*np.sum(w*(y-my)**2)))
def wquant(x,w,q):
    if not len(x):return np.full(len(q),np.nan)
    order=np.argsort(x,kind='stable');xx=x[order];ww=w[order];cum=ww.cumsum()/ww.sum();return xx[np.minimum(np.searchsorted(cum,q),len(xx)-1)]
def bins_table(region,name,H,xe,ye):
    i,j=np.indices(H.shape);return pd.DataFrame(dict(region=region,kind=name,x_lower=xe[i.ravel()],x_upper=xe[i.ravel()+1],y_lower=ye[j.ravel()],y_upper=ye[j.ravel()+1],area_km2=H.ravel()))

def main():
    start=time.time();checks={};bindings={};release=read(ROOT/'configs/core_release_s17.json');proof=read(ROOT/'reports/completed_batch_verification.json');release_sha=sha(ROOT/'configs/core_release_s17.json');assert proof['passed'] and proof['release_sha256']==release_sha
    disk=shutil.disk_usage(ROOT);total=sum((Path(p)/n).stat().st_size for p,ds,fs in os.walk(ROOT) for n in fs if not (Path(p)/n).is_symlink());save('resource_check.json',dict(project_bytes_before=total,disk_free_bytes=disk.free,estimated_figure_and_temporary_bytes=600_000_000,legacy_storage_cap_removed_by_user=True));assert disk.free>1_000_000_000
    for rel,digest in release['sha256'].items():
        assert sha(ROOT/rel)==digest,rel;bindings[rel]=dict(path=rel,bytes=(ROOT/rel).stat().st_size,sha256=digest)
    checks['frozen_release_preserved']=True;log('142-file release checked')
    for fig in ['Fig01','Fig02','Fig03','Fig04']:
        prior=ROOT/'figures/main'/fig;v=read(prior/(fig+'_validation.json'));assert v['passed']
        for item in v['sha256_manifest']:assert sha(prior/item['path'])==item['sha256'],(fig,item['path'])
        checks[fig+'_preserved']=True
    ev=ROOT/'runs/core_s17_evaluation';targets=['configs/core_release_s17.json','reports/completed_batch_verification.json','runs/core_s17_predictions/complete.json','runs/core_s17_predictions/prediction_manifest.csv','runs/core_s17_predictions/value_gate.npy','runs/core_s17_predictions/risk.npy','runs/core_s17_evaluation/review_units.npy','runs/core_s17_evaluation/primary_review_selections_packed.npz','runs/core_s17_evaluation/array_schema.json','runs/core_s17_evaluation/review_selection_encoding.json','runs/final_s17_risk/config.json','runs/final_s17_risk/input_spec.json','runs/final_s17_risk/training_records.csv','runs/final_s17_risk/complete.json','runs/final_s17_risk/last.pt','figures/main/Fig04/source_data/interval_reproduction.json','figures/main/Fig04/source_data/geographic_manifest.csv']
    targets += ['runs/core_s17_evaluation/'+n for n in ['review_budget_results.csv','review_paired_spatial_intervals.csv','review_simultaneous_intervals.csv','map_metrics.csv','router_value_unit_diagnostics.csv']]
    for rel in targets:
        p=ROOT/rel;digest=sha(p);bindings[rel]=dict(path=rel,bytes=p.stat().st_size,sha256=digest)
        if p.name in proof['csv_sha256']:assert digest==proof['csv_sha256'][p.name]
        if rel in ['runs/core_s17_predictions/value_gate.npy','runs/core_s17_predictions/risk.npy']:assert digest==next(r['sha256'] for r in proof['prediction_files'] if r['path'].replace('\\','/')==rel)
    save('input_bindings.json',list(bindings.values()));checks['prediction_and_evaluation_hashes_match']=True
    # Risk training targets come from full-system held-out maps; the risk head has no saved OOF risk evaluation.
    rc=read(ROOT/'runs/final_s17_risk/config.json');tr=pd.read_csv(ROOT/'runs/final_s17_risk/training_records.csv');sources=[];assert rc['kind']=='risk' and rc['seed']==17 and rc['epochs']==10 and not tr.event_group.eq('2023_holdout').any()
    for run,g in tr.groupby('system_run'):
        cfg=read(ROOT/'runs'/run/'config.json');rec=pd.read_csv(ROOT/'runs'/run/'prediction_records.csv');q=rec.iloc[g.system_prediction_index.to_numpy()];assert q.cache_index.tolist()==g.cache_index.tolist();targets=set(g.event_group);assert not targets.intersection(cfg['source_provenance']['training_groups'])
        sources.append(dict(system_run=run,target_groups=sorted(targets),system_training_groups=cfg['source_provenance']['training_groups'],risk_training_patches=len(g)))
    riskproof=dict(kind='conditional sigmoid score',training_objective='unweighted pixel BCE on common-valid predicted-negative locations from full-system event-held-out maps',risk_head_parameters=rc['parameters'],risk_head_epochs=10,seed=17,training_patches=len(tr),system_oof_sources=sources,core_model_and_risk_weights_fixed_before_test=True,risk_head_development_oof_predictions_available=False,panel_c_scope='frozen 2023 core test score-outcome correspondence; no fitted-development surrogate or test recalibration',saved_risk_array='unmasked sigmoid score; predicted-negative mask applied when forming unit score',release_sha256=release_sha)
    save('risk_head_provenance.json',riskproof);checks['risk_training_full_system_oof_provenance']=True
    schema=read(ev/'array_schema.json');uall=np.load(ev/'review_units.npy',mmap_mode='r')[schema['models'].index('value_gate')];fields=schema['unit_fields'];frame=pd.read_csv(ROOT/'runs/core_s17_predictions/prediction_manifest.csv');geo=pd.read_csv(ROOT/'figures/main/Fig04/source_data/geographic_manifest.csv');assert geo.cache_index.tolist()==frame.cache_index.tolist();frame=geo
    p=np.load(ROOT/'runs/core_s17_predictions/value_gate.npy',mmap_mode='r');risk=np.load(ROOT/'runs/core_s17_predictions/risk.npy',mmap_mode='r');lab=np.load(ROOT/'data/processed/sen2gf3/labels_uint8.npy',mmap_mode='r');sup=np.load(ROOT/'data/processed/sen2gf3/support_uint8.npy',mmap_mode='r');area=np.load(ROOT/'data/processed/sen2gf3/row_area_m2.npy',mmap_mode='r')
    maxima=np.zeros(6)
    for j,row in enumerate(frame.itertuples(index=False)):
        ci=int(row.cache_index);valid=np.asarray(sup[ci])==7;y=np.asarray(lab[ci])==1;w=np.asarray(area[ci],dtype='float64')[:,None]*valid;prob=np.asarray(p[j],dtype='float64');r=np.asarray(risk[j],dtype='float64');assert np.isfinite(r).all() and np.all((r>=0)&(r<=1));neg=prob<.5;cost=tile_sum(w)
        numer=[tile_sum(w*entropy(prob)),tile_sum(w*r*neg),tile_sum(w*prob*neg)];scores=[np.divide(v,cost,out=np.zeros(16),where=cost>0) for v in numer];actual=np.column_stack([cost,tile_sum(w*y*neg),tile_sum(w*(~y)*(~neg)),*scores]);expected=np.asarray(uall[j])[:,[0,1,2,3,7,6]];maxima=np.maximum(maxima,abs(actual-expected).max(axis=0))
        if j%1500==0:log(f'Risk/entropy/proxy pixel audit {j}/8233')
    assert np.all(maxima[:3]<1e-6) and np.all(maxima[3:]<1e-12),maxima;checks['all_core_pixels_and_three_scores_verified']=True
    ix=np.repeat(frame.prediction_index.to_numpy(),16);j=np.tile(np.arange(16),len(frame));g=frame.iloc[ix];u=np.asarray(uall).reshape(-1,len(fields));unit=pd.DataFrame(dict(unit_id=[f'{a}:{b}' for a,b in zip(ix,j)],region=g.region.to_numpy(),sample_id=g.sample_id.to_numpy(),prediction_index=ix,cache_index=g.cache_index.to_numpy(),unit_index=j,cost_m2=u[:,0],fn_m2=u[:,1],fp_m2=u[:,2],H=u[:,3],L=u[:,7],P=u[:,6],left=g.left.to_numpy()+(j%4)*64*g.xres.to_numpy(),right=g.left.to_numpy()+(j%4+1)*64*g.xres.to_numpy(),top=g.top.to_numpy()+(j//4)*64*g.yres.to_numpy(),bottom=g.top.to_numpy()+(j//4+1)*64*g.yres.to_numpy(),split='frozen_2023_core',release_sha256=release_sha));unit=unit[unit.cost_m2>0].copy();unit['D']=unit.fn_m2/unit.cost_m2
    original=pd.read_csv(ev/'review_budget_results.csv');original=original[original.model=='value_gate'].copy();original['release_sha256']=release_sha;original.to_csv(SD/'all_fixed_map_review_results.csv',index=False);packed=np.load(ev/'primary_review_selections_packed.npz');selections={};brows=[];primary={};upset=[];stats=[];arrays={};densitytables=[];meanrows=[];quantrows=[]
    for region in REGIONS:
        q=unit[unit.region==region];cost=q.cost_m2.to_numpy();fn=q.fn_m2.to_numpy();fp=q.fp_m2.to_numpy();w=cost/1e6;D=q.D.to_numpy();ranks={name:midrank(q[name].to_numpy(),cost) for name in ['H','L','P','D']}
        for name,values in ranks.items():unit.loc[q.index,'rank_'+name]=values
        for policy in POLICIES:
            if policy in CODE:score=q[CODE[policy]].to_numpy()
            elif policy=='oracle_fn_density':score=D
            else:score=np.random.default_rng(int(policy.rsplit('_',1)[1])).random(len(q))
            for b in [.01,.05,.1,.2]:
                selected,used=whole_unit_selection(score,cost,b);row=original[(original.region==region)&(original.policy==policy)&original.budget_fraction.eq(b)].iloc[0];rf=fn[~selected].sum()/1e6;rp=fp[~selected].sum()/1e6;captured=fn[selected].sum()/1e6
                assert np.isclose(rf,row.residual_fn_km2,rtol=0,atol=1e-9) and np.isclose(rp,row.residual_fp_km2,rtol=0,atol=1e-9) and np.isclose(used/cost.sum(),row.actual_area_fraction,rtol=0,atol=1e-13) and selected.sum()==row.selected_units
                assert abs(fn.sum()/1e6-captured-rf)<1e-9 and abs(fp.sum()/1e6-fp[selected].sum()/1e6-rp)<1e-9
                selections[f'{region}|{policy}|{b}']=np.packbits(selected);brows.append(dict(region=region,base_model='value_gate',policy=policy,budget_fraction=b,actual_area_fraction=used/cost.sum(),selected_units=int(selected.sum()),initial_fn_km2=fn.sum()/1e6,captured_fn_km2=captured,capture_fraction=captured/(fn.sum()/1e6),residual_fn_km2=rf,residual_fp_km2=rp,release_sha256=release_sha))
                if b==.1:
                    assert np.array_equal(np.unpackbits(packed[f'{region}|value_gate|{policy}'])[:len(selected)].astype(bool),selected)
                    if policy in CODE:primary[(region,CODE[policy])]=selected;unit.loc[q.index,'selected_'+CODE[policy]]=selected
        membership=primary[(region,'H')].astype(int)+2*primary[(region,'L')]+4*primary[(region,'P')];unit.loc[q.index,'membership_HLP']=membership
        for cell in range(8):
            mask=membership==cell;upset.append(dict(region=region,intersection_id=cell,H=bool(cell&1),L=bool(cell&2),P=bool(cell&4),units=int(mask.sum()),valid_area_km2=w[mask].sum(),fn_area_km2=fn[mask].sum()/1e6,fn_density=fn[mask].sum()/cost[mask].sum() if mask.any() else 0,release_sha256=release_sha))
        assert np.isclose(sum(r['valid_area_km2'] for r in upset if r['region']==region),w.sum())
        for code,bit in [('H',1),('L',2),('P',4)]:
            rows=[v for v in upset if v['region']==region and v['intersection_id']&bit];sel=primary[(region,code)];assert np.isclose(sum(r['valid_area_km2'] for r in rows),w[sel].sum()) and np.isclose(sum(r['fn_area_km2'] for r in rows),fn[sel].sum()/1e6)
        positive=D>0;xe=np.linspace(0,np.log(2),61);ye=np.geomspace(1e-4,1,61);H=np.histogram2d(q.H[positive],D[positive],bins=[xe,ye],weights=w[positive])[0];arrays[region+'_joint_H']=H;arrays['entropy_edges']=xe;arrays['positive_D_edges']=ye;assert np.isclose(H.sum(),w[positive].sum());densitytables.append(bins_table(region,'entropy_positive_FN',H,xe,ye))
        re=np.linspace(0,1,21);cop=np.histogram2d(ranks['H'],ranks['D'],bins=[re,re],weights=w)[0];arrays[region+'_copula']=cop;arrays['rank_edges']=re;assert np.isclose(cop.sum(),w.sum());densitytables.append(bins_table(region,'weighted_midrank_H_D',cop,re,re))
        se=np.linspace(0,1,41);de=np.r_[0,np.geomspace(.001,1,41)];hh=np.histogram2d(q.L,D,bins=[se,de],weights=w)[0];arrays[region+'_joint_L']=hh;arrays['score_edges']=se;arrays['all_D_edges']=de;assert np.isclose(hh.sum(),w.sum());densitytables.append(bins_table(region,'learned_score_FN',hh,se,de))
        for i in range(len(se)-1):
            mask=(q.L.to_numpy()>=se[i])&(q.L.to_numpy()<(se[i+1] if i<len(se)-2 else 1.00001))
            meanrows.append(dict(region=region,bin=i,lower=se[i],upper=se[i+1],valid_area_km2=w[mask].sum(),mean_score=np.average(q.L.to_numpy()[mask],weights=w[mask]) if mask.any() else np.nan,mean_D=fn[mask].sum()/cost[mask].sum() if mask.any() else np.nan,zero_FN_area_fraction=w[mask&~positive].sum()/w[mask].sum() if mask.any() else np.nan))
        decile=np.minimum((ranks['L']*10).astype(int),9);unit.loc[q.index,'risk_decile']=decile+1
        for d in range(10):
            mask=decile==d;pos=mask&positive;quart=wquant(D[pos],cost[pos],[.25,.5,.75]);quantrows.append(dict(region=region,risk_decile=d+1,valid_area_km2=w[mask].sum(),positive_FN_area_km2=w[pos].sum(),zero_FN_area_fraction=w[mask&~positive].sum()/w[mask].sum(),actual_FN_share=fn[mask].sum()/fn.sum(),mean_score=np.average(q.L.to_numpy()[mask],weights=w[mask]),observed_mean_density=fn[mask].sum()/cost[mask].sum(),positive_D_q25=quart[0],positive_D_median=quart[1],positive_D_q75=quart[2]))
        topH=ranks['H']>=.9;topD=ranks['D']>=.9
        stats.append(dict(region=region,units=len(q),valid_area_km2=w.sum(),initial_FN_km2=fn.sum()/1e6,zero_FN_units=int((~positive).sum()),zero_FN_area_fraction=w[~positive].sum()/w.sum(),entropy_FN_weighted_midrank_correlation=wcorr(ranks['H'],ranks['D'],cost),risk_FN_weighted_midrank_correlation=wcorr(ranks['L'],ranks['D'],cost),proxy_FN_weighted_midrank_correlation=wcorr(ranks['P'],ranks['D'],cost),entropy_top_decile_D_top_decile_intersection_km2=w[topH&topD].sum(),entropy_top_decile_D_top_decile_area_jaccard=w[topH&topD].sum()/w[topH|topD].sum(),FN_fraction_in_lower_entropy_half=fn[ranks['H']<=.5].sum()/fn.sum(),learned_mean_score=np.average(q.L,weights=cost),observed_mean_FN_density=fn.sum()/cost.sum(),release_sha256=release_sha))
        log('Regional distributions and exact selections: '+json.dumps(stats[-1]))
    unit.to_csv(SD/'review_units.csv.gz',index=False);save('review_selection_encoding.json',dict(order='Rows of review_units.csv.gz filtered to the key region, stable manifest and row-major unit order',encoding='numpy.packbits, big bit order',primary_fraction=.1));np.savez_compressed(SD/'review_selections_packed.npz',**selections);np.savez_compressed(SD/'density_statistics.npz',**arrays)
    pd.DataFrame(stats).to_csv(SD/'panel_a_distribution_summary.csv',index=False);pd.concat(densitytables).to_csv(SD/'panel_ac_density_bins.csv.gz',index=False);pd.DataFrame(meanrows).to_csv(SD/'panel_c_score_bin_means.csv',index=False);pd.DataFrame(quantrows).to_csv(SD/'panel_c_conditional_quantiles.csv',index=False);pd.DataFrame(upset).to_csv(SD/'panel_d_disjoint_intersections.csv',index=False)
    b=pd.DataFrame(brows);zero=b[b.budget_fraction==.01].copy();zero['budget_fraction']=0.;zero['actual_area_fraction']=0.;zero['selected_units']=0;zero['captured_fn_km2']=0.;zero['capture_fraction']=0.;zero['residual_fn_km2']=zero.initial_fn_km2
    for region in REGIONS:zero.loc[zero.region==region,'residual_fp_km2']=unit.loc[unit.region==region,'fp_m2'].sum()/1e6
    pd.concat([zero,b]).sort_values(['region','policy','budget_fraction']).to_csv(SD/'panel_b_budget_capture.csv',index=False)
    bands=pd.read_csv(ev/'review_simultaneous_intervals.csv');f=bands[(bands.first_model=='value_gate')&(bands.second_model=='value_gate')&(bands.first_policy=='learned_omission_density')&bands.second_policy.isin(['mean_entropy','fn_probability_density'])].copy();assert len(f)==8
    for r in f.itertuples(index=False):
        one=b[(b.region==r.region)&b.policy.eq(r.first_policy)&b.budget_fraction.eq(.1)].iloc[0];two=b[(b.region==r.region)&b.policy.eq(r.second_policy)&b.budget_fraction.eq(.1)].iloc[0];assert np.isclose(one[r.endpoint]-two[r.endpoint],r.difference_first_minus_second,rtol=0,atol=1e-9)
    f['release_sha256']=release_sha;f.to_csv(SD/'panel_f_simultaneous_intervals.csv',index=False);save('interval_provenance.json',dict(source='frozen review_simultaneous_intervals.csv',primary_budget=.1,block_size_m=10000,bootstrap_replicates=2000,seed=20261012,family_estimates_per_region=12,all_family_reproduction_record_sha256=sha(ROOT/'figures/main/Fig04/source_data/interval_reproduction.json'),all_24_family_estimates_already_reproduced=True,scope='Within-region spatial composition conditional on fixed fitted models and selected review units'))
    checks.update(all_56_policy_region_budget_records_reproduced=True,all_14_primary_masks_identical=True,initial_FN_fixed_across_rankings=True,FN_FP_conservation=True,disjoint_intersection_accounting=True,all_histograms_conserve_area=True,all_eight_intervals_match_frozen_results=True)
    make_geographic_cases(frame,unit,p,lab,sup,area,release_sha)
    checks['geographic_cases_native_and_selection_exact']=True
    save('analysis_rules.json',dict(analysis_status='Exploratory descriptive derivations from frozen outputs',rank='Weighted empirical CDF midpoint: (area strictly below score + half area tied at score) / total area; exact ties receive identical ranks; no random jitter',rank_association='Area-weighted Pearson correlation of weighted midranks',rank_matrix='20x20 fixed percentile bins on [0,1]; weights km2',entropy_joint='Positive FN units only; entropy linear [0,ln2], observed density logarithmic [1e-4,1]; zero-FN valid-area mass separately recorded',risk_joint='All valid units; fixed score bins [0,1]; first observed-density bin [0,.001], then logarithmic bins to 1',conditional='Risk groups by floor(10*weighted midrank), exact ties unsplit; positive-FN weighted empirical quartiles and separate zero-FN valid-area fractions',selection='Frozen stable descending ranking; accept whole positive-cost units that fit remaining budget, skip otherwise',upset='Exact H/L/P membership bit mask; H=1,L=2,P=4; zero=neither. Each unit contributes once to its disjoint cell.',risk_probability_claim=False,release_sha256=release_sha))
    save('scientific_audit.json',dict(passed=all(checks.values()),checks=checks,release_sha256=release_sha,finished_at_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.time()-start,algorithm_version='fig05-derivation-v1',core_patches=len(frame),positive_cost_units=len(unit),core_valid_pixels=int(frame.common_valid_pixels.sum()),pixel_unit_max_abs_errors=maxima.tolist(),input_files_bound=len(bindings)));log('Fig05 scientific derivation PASSED')

def make_geographic_cases(frame,unit,p,lab,sup,area,release_sha):
    """Both directions of local recovery differences, deterministic contiguous native windows."""
    patch=frame.copy();agg=[]
    for ix,g in unit.groupby('prediction_index',sort=False):
        h=g.selected_H.to_numpy(dtype=bool);l=g.selected_L.to_numpy(dtype=bool);fn=g.fn_m2.to_numpy()/1e6;cost=g.cost_m2.to_numpy()/1e6
        agg.append(dict(prediction_index=ix,H_only_area=cost[h&~l].sum(),L_only_area=cost[l&~h].sum(),H_only_FN=fn[h&~l].sum(),L_only_FN=fn[l&~h].sum(),both_FN=fn[h&l].sum(),neither_FN=fn[~h&~l].sum(),xor_units=int((h^l).sum())))
    patch=patch.merge(pd.DataFrame(agg),on='prediction_index');patch.to_csv(SD/'panel_e_patch_statistics.csv',index=False);candidates=[]
    for gid,g in patch.groupby('grid_id'):
        lookup={(int(r.grid_row//256),int(r.grid_col//256)):int(r.prediction_index) for r in g.itertuples()}
        for rr,cc in sorted(lookup):
            ids=[lookup.get((rr,cc+x)) for x in range(4)]
            if any(i is None for i in ids):continue
            q=patch.set_index('prediction_index').loc[ids];cov=q.common_valid_pixels.sum()/(4*256*256);h=q.H_only_FN.sum();l=q.L_only_FN.sum();candidates.append(dict(region=q.region.iloc[0],grid_id=gid,row_start=rr,col_start=cc,coverage=cov,H_only_area_km2=q.H_only_area.sum(),L_only_area_km2=q.L_only_area.sum(),H_only_FN_km2=h,L_only_FN_km2=l,H_minus_L_recovery_km2=h-l,eligible=bool(cov>=.85 and q.xor_units.sum()>=2),prediction_indices=';'.join(map(str,ids))))
    candidates=pd.DataFrame(candidates);candidates.to_csv(SD/'panel_e_all_candidate_windows.csv',index=False);raw=np.load(ROOT/'data/processed/sen2gf3/x_uint8.npy',mmap_mode='r');cases=[]
    for region in REGIONS:
        eligible=candidates[(candidates.region==region)&candidates.eligible];assert len(eligible)
        for direction,ascending in [('H_favouring',False),('L_favouring',True)]:
            r=eligible.sort_values(['H_minus_L_recovery_km2','grid_id','row_start','col_start'],ascending=[ascending,True,True,True],kind='stable').iloc[0];assert (r.H_minus_L_recovery_km2>0 if not ascending else r.H_minus_L_recovery_km2<0),(region,direction)
            ids=[int(v) for v in r.prediction_indices.split(';')];g=frame.iloc[ids];arr=dict(rgb=np.zeros((256,1024,3),np.uint8),label=np.full((256,1024),255,np.uint8),valid=np.zeros((256,1024),bool),probability=np.zeros((256,1024),np.float32),area_m2=np.zeros((256,1024)),selected_H=np.zeros((4,16),bool),selected_L=np.zeros((4,16),bool),selected_P=np.zeros((4,16),bool))
            for n,row in enumerate(g.itertuples(index=False)):
                ci=int(row.cache_index);sl=(slice(None),slice(256*n,256*(n+1)));arr['rgb'][sl]=np.asarray(raw[ci,2:5]).transpose(1,2,0);arr['label'][sl]=lab[ci];arr['valid'][sl]=np.asarray(sup[ci])==7;arr['area_m2'][sl]=np.asarray(area[ci],dtype='float64')[:,None];arr['probability'][sl]=p[row.prediction_index];q=unit[unit.prediction_index==row.prediction_index]
                for code in ['H','L','P']:
                    sel=np.zeros(16,bool);sel[q.unit_index.to_numpy()]=q['selected_'+code].to_numpy(bool);arr['selected_'+code][:,n*4:(n+1)*4]=sel.reshape(4,4)
            extent=[float(g.left.min()),float(g.right.max()),float(g.bottom.min()),float(g.top.max())];arr['extent']=np.array(extent);case_id=region+'_'+direction;np.savez_compressed(SD/f'panel_e_{case_id}.npz',**arr)
            q=unit[unit.prediction_index.isin(ids)];q.to_csv(SD/f'panel_e_{case_id}_units.csv',index=False);h=arr['selected_H'].repeat(64,0).repeat(64,1);l=arr['selected_L'].repeat(64,0).repeat(64,1);fn=arr['valid']&(arr['label']==1)&(arr['probability']<.5);w=arr['area_m2'];cat={}
            for name,mask in [('H_only',h&~l),('L_only',l&~h),('both',h&l),('neither',~h&~l)]:cat[name]=float(w[fn&mask].sum()/1e6)
            assert np.isclose(cat['H_only'],r.H_only_FN_km2,atol=1e-10) and np.isclose(cat['L_only'],r.L_only_FN_km2,atol=1e-10) and np.isclose(sum(cat.values()),q.fn_m2.sum()/1e6)
            cases.append(dict(case_id=case_id,region=region,direction=direction,prediction_indices=ids,cache_indices=g.cache_index.tolist(),sample_ids=g.sample_id.tolist(),extent_lonlat=extent,crs='EPSG:4326',grid_id=r.grid_id,candidate_score=r.to_dict(),FN_by_selection_km2=cat,initial_FN_km2=sum(cat.values()),H_corrected_FN_km2=cat['H_only']+cat['both'],L_corrected_FN_km2=cat['L_only']+cat['both'],H_residual_FN_km2=cat['L_only']+cat['neither'],L_residual_FN_km2=cat['H_only']+cat['neither'],release_sha256=release_sha));log('Geographic '+case_id+': '+json.dumps(cat))
    save('panel_e_case_selection.json',dict(rule='All fully occupied 1x4 contiguous native-grid patch windows with >=85% common-valid pixel coverage and >=2 H/L-disagreement units. Per region, show the maximum and minimum H-only-minus-L-only captured FN; grid id, row, column break ties. Both directions must actually exist. Windows illustrate outcome-selected local contrasts.',fixed_selection='Actual whole-region 10% masks retained in crops; no local reranking',map_layers='Low-saturation RGB; reference flood contours; categorical H-only/L-only/both/neither review units; actual initial FN raster. Captured/residual FN are resolved by category membership, not by score colour.',cases=cases))
if __name__=='__main__':main()
