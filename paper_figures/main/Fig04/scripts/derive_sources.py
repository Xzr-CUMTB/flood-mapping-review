"""Read-only Fig04 derivation from frozen maps and whole-unit review evidence."""
import os,sys,json,hashlib,time
os.environ.setdefault('MKL_THREADING_LAYER','SEQUENTIAL')
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd
ROOT=Path(os.path.abspath(__file__)).parents[4];OUT=ROOT/'figures/main/Fig04';SD=OUT/'source_data'
SD.mkdir(parents=True,exist_ok=True);(OUT/'assets').mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT))
from experiments.evaluation_metrics import entropy,tile_sum,whole_unit_selection,spatial_block_ids,paired_block_bootstrap,simultaneous_bands
REGIONS=['Beijing_Baoding','Harbin'];MODELS=['sar','fusion','ordinary_gate','value_gate']
EVAL=ROOT/'runs/core_s17_evaluation'

def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(n,x):(SD/n).write_text(json.dumps(x,indent=2),encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def log(s):print(datetime.now().isoformat(timespec='seconds'),s,flush=True)

def native_grids(frame):
    specs=[];parts=[]
    for region,g in frame.groupby('region',sort=False):
        dx=float(g.xres.median());dy=float(-g.yres.median());left=g.left.min();top=g.top.max()
        px=np.mod((g.left-left)/dx,256);py=np.mod((top-g.top)/dy,256)
        px=np.where((px<.0001)|(px>255.9999),0,px);py=np.where((py<.0001)|(py>255.9999),0,py)
        h=g.assign(phase_x=px.round(4),phase_y=py.round(4))
        for j,(_,q) in enumerate(h.groupby(['phase_x','phase_y'],sort=True)):
            q=q.copy();gid=region+'_'+str(j);x0=float(q.left.min());y0=float(q.top.max());cx=(q.left-x0)/dx;ry=(y0-q.top)/dy
            assert max(abs(cx-cx.round()).max(),abs(ry-ry.round()).max())<.0001
            q['grid_id']=gid;q['grid_col']=cx.round().astype(int);q['grid_row']=ry.round().astype(int)
            specs.append(dict(grid_id=gid,region=region,left=x0,top=y0,dx=dx,dy=dy));parts.append(q)
    return specs,pd.concat(parts).sort_values('prediction_index')

def main():
    start=time.time();started=datetime.now(timezone.utc).isoformat();checks={};bindings={}
    proof=read(ROOT/'reports/completed_batch_verification.json');release=read(ROOT/'configs/core_release_s17.json');release_sha=sha(ROOT/'configs/core_release_s17.json')
    assert proof['passed'] and proof['training_jobs']==38 and proof['release_sha256']==release_sha
    for rel,digest in release['sha256'].items():
        assert sha(ROOT/rel)==digest,rel
        bindings[rel]=dict(path=rel,bytes=(ROOT/rel).stat().st_size,sha256=digest)
    checks['all_frozen_release_hashes_match']=True;log('Frozen release verified')
    for fig in ['Fig01','Fig02','Fig03']:
        prior=ROOT/'figures/main'/fig;v=read(prior/(fig+'_validation.json'))
        for item in v['sha256_manifest']:assert sha(prior/item['path'])==item['sha256'],(fig,item['path'])
        checks[fig+'_unchanged']=True
    targets=['configs/core_release_s17.json','configs/evaluation_protocol_v1.json','reports/completed_batch_verification.json','runs/core_s17_predictions/prediction_manifest.csv','runs/core_s17_evaluation/array_schema.json','runs/core_s17_evaluation/complete.json','runs/core_s17_evaluation/review_units.npy','runs/core_s17_evaluation/primary_review_selections_packed.npz','runs/core_s17_evaluation/review_selection_encoding.json']
    targets += [f'runs/core_s17_evaluation/{n}' for n in ['map_metrics.csv','review_budget_results.csv','review_paired_spatial_intervals.csv','review_simultaneous_intervals.csv','router_value_unit_diagnostics.csv']]
    targets += [f'runs/core_s17_predictions/{m}.npy' for m in MODELS]
    for rel in targets:
        p=ROOT/rel;digest=sha(p);bindings[rel]=dict(path=rel,bytes=p.stat().st_size,sha256=digest)
        if p.name in proof['csv_sha256']:assert digest==proof['csv_sha256'][p.name],rel
        if p.suffix=='.npy' and '/core_s17_predictions/' in rel:assert digest==next(x['sha256'] for x in proof['prediction_files'] if x['path'].replace('\\','/')==rel),rel
    save('input_bindings.json',list(bindings.values()));checks['completed_csv_and_prediction_hashes_match']=True
    frame=pd.read_csv(ROOT/'runs/core_s17_predictions/prediction_manifest.csv');frame['region']=np.where(frame.provisional_site.eq('Harbin'),'Harbin','Beijing_Baoding')
    assert len(frame)==8233 and frame.prediction_index.tolist()==list(range(8233))
    specs,frame=native_grids(frame);frame.to_csv(SD/'geographic_manifest.csv',index=False);save('native_grid_specs.json',specs)
    protocol=read(ROOT/'configs/evaluation_protocol_v1.json');schema=read(EVAL/'array_schema.json');models=schema['models'];fields=schema['unit_fields']
    units=np.load(EVAL/'review_units.npy',mmap_mode='r');packed=np.load(EVAL/'primary_review_selections_packed.npz');budget=pd.read_csv(EVAL/'review_budget_results.csv');maps=pd.read_csv(EVAL/'map_metrics.csv');bands=pd.read_csv(EVAL/'review_simultaneous_intervals.csv')
    assert units.shape==(16,8233,16,14)
    budget=budget[budget.model.isin(MODELS)&budget.policy.eq('mean_entropy')].copy();assert len(budget)==32
    pred={m:np.load(ROOT/f'runs/core_s17_predictions/{m}.npy',mmap_mode='r') for m in MODELS}
    lab=np.load(ROOT/'data/processed/sen2gf3/labels_uint8.npy',mmap_mode='r');sup=np.load(ROOT/'data/processed/sen2gf3/support_uint8.npy',mmap_mode='r');area=np.load(ROOT/'data/processed/sen2gf3/row_area_m2.npy',mmap_mode='r')
    # All 8,233 patches, all four fixed models: independent pixel-to-unit checks.
    maxima=np.zeros(4);truth_areas=np.zeros(len(frame))
    for j,row in enumerate(frame.itertuples(index=False)):
        ci=int(row.cache_index);valid=np.asarray(sup[ci])==7;y=np.asarray(lab[ci])==1;w=np.broadcast_to(np.asarray(area[ci],dtype='float64')[:,None],(256,256))*valid
        cost=tile_sum(w);truth_areas[j]=(w*y).sum()/1e6
        for m in MODELS:
            p=np.asarray(pred[m][j],dtype='float64');assert np.isfinite(p).all() and np.all((p>=0)&(p<=1));h=tile_sum(w*entropy(p));score=np.divide(h,cost,out=np.zeros(16),where=cost>0)
            test=np.column_stack([cost,tile_sum(w*y*(p<.5)),tile_sum(w*(~y)*(p>=.5)),score]);saved=np.asarray(units[models.index(m),j,:,:4]);maxima=np.maximum(maxima,np.max(abs(test-saved),axis=0))
        if j%1000==0:log(f'Pixel-to-unit verification {j}/8233')
    assert np.all(maxima[:3]<1e-6) and maxima[3]<1e-12,maxima
    checks['all_four_models_pixel_unit_areas_and_entropy_verified']=True
    selections={};verified=[];primary={};contributions={}
    for region in REGIONS:
        mask=frame.region.eq(region).to_numpy()
        for m in MODELS:
            allvalues=np.asarray(units[models.index(m),mask]).reshape(-1,len(fields));good=allvalues[:,0]>0;u=allvalues[good];cost,fn,fp=u[:,:3].T
            mm=maps[(maps.region==region)&(maps.model==m)].iloc[0]
            for v,k in [(cost.sum()/1e6,'valid_area_km2'),(fn.sum()/1e6,'fn_km2'),(fp.sum()/1e6,'fp_km2')]:assert np.isclose(v,mm[k],rtol=1e-12,atol=1e-9)
            assert np.isclose((truth_areas[mask].sum()-fn.sum()/1e6)/(truth_areas[mask].sum()+fp.sum()/1e6),mm.flood_iou,atol=1e-12)
            for b in [.01,.05,.1,.2]:
                chosen,used=whole_unit_selection(u[:,fields.index('mean_entropy')],cost,b);row=budget[(budget.region==region)&(budget.model==m)&budget.budget_fraction.eq(b)].iloc[0]
                qf=fn[chosen].sum()/1e6;qfp=fp[chosen].sum()/1e6;rf=fn[~chosen].sum()/1e6;rp=fp[~chosen].sum()/1e6
                assert chosen.sum()==row.selected_units and np.isclose(used/cost.sum(),row.actual_area_fraction,rtol=0,atol=1e-13)
                assert used<=cost.sum()*b+1e-7 and np.isclose(rf,row.residual_fn_km2,atol=1e-10) and np.isclose(rp,row.residual_fp_km2,atol=1e-10)
                full=np.zeros(len(good),dtype=bool);full[good]=chosen;selections[f'{region}|{m}|{b}']=full.reshape(-1,16)
                rec=row.to_dict();rec.update(corrected_fn_km2=qf,corrected_fp_km2=qfp,used_area_km2=used/1e6,total_valid_area_km2=cost.sum()/1e6,fn_conservation_error_km2=fn.sum()/1e6-qf-rf,fp_conservation_error_km2=fp.sum()/1e6-qfp-rp,release_sha256=release_sha,interval='none; paired differences at 10% are reported in panel f');verified.append(rec)
                if b==.1:
                    saved=np.unpackbits(packed[f'{region}|{m}|mean_entropy'])[:len(chosen)].astype(bool);assert np.array_equal(saved,chosen)
                    primary[(region,m)]=full.reshape(-1,16);contributions[(region,m)]=np.stack([allvalues[:,1],allvalues[:,2]],axis=1)*(~full[:,None])
    table=pd.DataFrame(verified);assert abs(table[['fn_conservation_error_km2','fp_conservation_error_km2']].to_numpy()).max()<1e-9
    table.to_csv(SD/'verified_review_budget_results.csv',index=False);np.savez_compressed(SD/'review_selection_masks.npz',**selections)
    checks.update(all_32_budget_selections_reproduced=True,primary_packed_masks_identical=True,fn_fp_conservation=True,variable_cost_budget_respected=True,map_metrics_independently_reproduced=True)
    zero=table[table.budget_fraction==.01].copy();zero['budget_fraction']=0.;zero['actual_area_fraction']=0.;zero['selected_units']=0;zero['used_area_km2']=0.;zero['corrected_fn_km2']=0.;zero['corrected_fp_km2']=0.;zero['residual_fn_km2']=zero.initial_fn_km2;zero['residual_fp_km2']=zero.initial_fp_km2
    # Initial map values form the observed no-review baseline, with no interpolation.
    a=pd.concat([zero,table],ignore_index=True).sort_values(['region','model','budget_fraction']);a.to_csv(SD/'panel_a_budget_response.csv',index=False)
    deltas=[];ranks=[];decomp=[];flows=[]
    for region in REGIONS:
        aa=a[a.region==region];f=aa[aa.model=='fusion'].set_index('budget_fraction');v=aa[aa.model=='value_gate'].set_index('budget_fraction')
        for b in [0.,.01,.05,.1,.2]:deltas.append(dict(region=region,budget_fraction=b,delta_residual_fn_km2=v.loc[b,'residual_fn_km2']-f.loc[b,'residual_fn_km2'],release_sha256=release_sha))
        t=table[(table.region==region)&table.budget_fraction.eq(.1)].merge(maps[maps.region==region][['model','flood_iou']],on='model');t['iou_rank']=t.flood_iou.rank(ascending=False,method='min').astype(int);t['residual_fn_rank']=t.residual_fn_km2.rank(method='min').astype(int);t['rank_comparison_set']='sar;fusion;ordinary_gate;value_gate';ranks.append(t)
        df=v.loc[.1,'initial_fn_km2']-f.loc[.1,'initial_fn_km2'];dq=v.loc[.1,'corrected_fn_km2']-f.loc[.1,'corrected_fn_km2'];dr=v.loc[.1,'residual_fn_km2']-f.loc[.1,'residual_fn_km2']
        decomp.append(dict(region=region,delta_initial_fn_km2=df,delta_corrected_fn_km2=dq,negative_delta_corrected_fn_km2=-dq,delta_residual_fn_km2=dr,conservation_error_km2=dr-(df-dq),source_records=f'{region}|fusion/value_gate|mean_entropy|0.1',release_sha256=release_sha))
        for m in ['fusion','value_gate']:
            r=t[t.model==m].iloc[0]
            for state,col in [('corrected','corrected_fn_km2'),('remaining','residual_fn_km2')]:flows.append(dict(region=region,model=m,flow_id=f'{region}|{m}|{state}',source='initial FN',target=state+' FN',area_km2=r[col],initial_fn_km2=r.initial_fn_km2,release_sha256=release_sha))
    pd.DataFrame(deltas).to_csv(SD/'panel_a_paired_differences.csv',index=False);pd.concat(ranks).to_csv(SD/'panel_b_rankings.csv',index=False);pd.DataFrame(decomp).to_csv(SD/'panel_c_decomposition.csv',index=False);pd.DataFrame(flows).to_csv(SD/'panel_d_conservation_flows.csv',index=False)
    assert all(r['delta_residual_fn_km2']<0 for r in deltas if r['budget_fraction']==.01)
    assert all(r['delta_residual_fn_km2']>0 for r in deltas if r['budget_fraction'] in [.05,.1,.2]);checks['both_regions_observed_budget_ordering_verified']=True
    # Reproduce the entire prespecified twelve-estimate simultaneous family in each region.
    interval_checks=[]
    for region in REGIONS:
        mask=frame.region.eq(region).to_numpy();g=frame.loc[mask];cache={}
        for contrast in protocol['confirmatory_review_contrasts']:
            for m,policy in [contrast[:2],contrast[2:]]:
                if (m,policy) in cache:continue
                u=np.asarray(units[models.index(m),mask]).reshape(-1,len(fields));good=u[:,0]>0;selected=np.unpackbits(packed[f'{region}|{m}|{policy}'])[:good.sum()].astype(bool);sel=np.zeros(len(good),bool);sel[good]=selected
                cache[(m,policy)]=(u[:,1:3]*(~sel[:,None])).reshape(-1,16,2).sum(axis=1)
        differences=np.stack([(cache[(m,p)]-cache[(n,q)])/1e6 for m,p,n,q in protocol['confirmatory_review_contrasts']],axis=1)
        crs=next(protocol[k]['spatial_block_crs'] for k in ['primary_region','secondary_region'] if protocol[k]['name']==region);blocks=spatial_block_ids(g.lon.to_numpy(),g.lat.to_numpy(),crs,10000);keys,inverse=np.unique(np.asarray(blocks),axis=0,return_inverse=True);totals=np.zeros((len(keys),6,2));np.add.at(totals,inverse,differences)
        draws=np.concatenate(list(paired_block_bootstrap(totals,protocol['spatial_uncertainty']['bootstrap_replicates'],protocol['spatial_uncertainty']['seed'])));point=differences.sum(axis=0);lo,hi,crit,se=simultaneous_bands(point,draws)
        for j,(m,p,n,q) in enumerate(protocol['confirmatory_review_contrasts']):
            for k,endpoint in enumerate(['residual_fn_km2','residual_fp_km2']):
                saved=bands[(bands.region==region)&(bands.first_model==m)&(bands.first_policy==p)&(bands.second_model==n)&(bands.second_policy==q)&(bands.endpoint==endpoint)].iloc[0]
                actual=[point[j,k],lo[j,k],hi[j,k],crit,se[j,k]];expected=saved[['difference_first_minus_second','simultaneous_lower95','simultaneous_upper95','maximum_deviation_critical_value','bootstrap_se']].astype(float).to_numpy();assert np.allclose(actual,expected,rtol=1e-10,atol=1e-9)
                interval_checks.append(dict(region=region,contrast=j,endpoint=endpoint,max_abs_error=float(np.max(abs(np.asarray(actual)-expected))),occupied_blocks=len(keys)))
    save('interval_reproduction.json',dict(algorithm='paired spatial blocks; max standardized centered bootstrap deviation',replicates=2000,seed=20261012,family_estimates=12,checks=interval_checks));checks['all_24_simultaneous_estimates_reproduced']=True
    bands['release_sha256']=release_sha;bands.to_csv(SD/'panel_f_all_simultaneous_intervals.csv',index=False)
    bands[(bands.first_model=='value_gate')&(bands.second_model=='fusion')&(bands.first_policy=='mean_entropy')&(bands.second_policy=='mean_entropy')].to_csv(SD/'panel_f_paired_intervals.csv',index=False)
    primarytable=table[table.budget_fraction==.1].copy();primarytable.to_csv(SD/'panel_f_endpoints_and_costs.csv',index=False)
    log('Budget, rank, conservation and simultaneous intervals verified')
    # Lightweight geography-indexed summaries: all units of the two mapped models.
    unitframes=[];patch=frame.copy();patch['reference_flood_km2']=truth_areas
    for m in ['fusion','value_gate']:
        selected=np.zeros((len(frame),16),bool)
        for region in REGIONS:selected[frame.region==region]=primary[(region,m)]
        u=np.asarray(units[models.index(m)])
        for name,arr in [('fn',u[:,:,1]),('corrected_fn',u[:,:,1]*selected),('residual_fn',u[:,:,1]*(~selected))]:patch[m+'_'+name+'_km2']=arr.sum(axis=1)/1e6
        patch[m+'_selected_units']=selected.sum(axis=1)
        ix=np.repeat(frame.prediction_index.to_numpy(),16);unit=np.tile(np.arange(16),len(frame));g=frame.iloc[ix]
        uf=pd.DataFrame(dict(region=g.region.to_numpy(),sample_id=g.sample_id.to_numpy(),cache_index=g.cache_index.to_numpy(),prediction_index=ix,unit_index=unit,unit_id=[f'{i}:{j}' for i,j in zip(ix,unit)],model=m,cost_m2=u[:,:,0].ravel(),mean_entropy=u[:,:,3].ravel(),selected=selected.ravel(),initial_fn_km2=u[:,:,1].ravel()/1e6,corrected_fn_km2=(u[:,:,1]*selected).ravel()/1e6,residual_fn_km2=(u[:,:,1]*(~selected)).ravel()/1e6,initial_fp_km2=u[:,:,2].ravel()/1e6,left=g.left.to_numpy()+(unit%4)*64*g.xres.to_numpy(),right=g.left.to_numpy()+(unit%4+1)*64*g.xres.to_numpy(),top=g.top.to_numpy()+(unit//4)*64*g.yres.to_numpy(),bottom=g.top.to_numpy()+(unit//4+1)*64*g.yres.to_numpy(),release_sha256=release_sha));unitframes.append(uf)
    uf=pd.concat(unitframes,ignore_index=True);uf.to_csv(SD/'panel_e_geographic_review_units.csv.gz',index=False)
    for region in REGIONS:
        mask=frame.region==region;patch.loc[mask,'selection_xor_units']=(primary[(region,'fusion')]^primary[(region,'value_gate')]).sum(axis=1)
    patch.to_csv(SD/'panel_e_patch_summaries.csv',index=False)
    candidates=[]
    for gid,g in patch.groupby('grid_id'):
        index={(int(r.grid_row//256),int(r.grid_col//256)):int(r.prediction_index) for r in g.itertuples()}
        for r,c in sorted(index):
            ids=[index.get((r+dy,c+dx)) for dy in range(2) for dx in range(5)]
            if any(i is None for i in ids):continue
            q=patch.iloc[ids];coverage=float(q.common_valid_pixels.sum()/(10*256*256));gain=float(q.fusion_corrected_fn_km2.sum()-q.value_gate_corrected_fn_km2.sum());delta=float(q.value_gate_residual_fn_km2.sum()-q.fusion_residual_fn_km2.sum());xor=int(q.selection_xor_units.sum());eligible=coverage>=.85 and gain>0 and delta>0 and xor>=4
            candidates.append(dict(region=q.region.iloc[0],grid_id=gid,row_start=r,col_start=c,coverage_fraction=coverage,fusion_recovery_advantage_km2=gain,value_minus_fusion_residual_fn_km2=delta,selection_xor_units=xor,eligible=eligible,prediction_indices=';'.join(map(str,ids))))
    candidates=pd.DataFrame(candidates);candidates.to_csv(SD/'panel_e_all_candidate_windows.csv',index=False)
    x=np.load(ROOT/'data/processed/sen2gf3/x_uint8.npy',mmap_mode='r');caseinfo=[]
    for region in REGIONS:
        eligible=candidates[(candidates.region==region)&candidates.eligible];assert len(eligible)
        chosen=eligible.sort_values(['fusion_recovery_advantage_km2','value_minus_fusion_residual_fn_km2','grid_id','row_start','col_start'],ascending=[False,False,True,True,True],kind='stable').iloc[0];ids=[int(v) for v in chosen.prediction_indices.split(';')];gg=frame.iloc[ids];H,W=512,1280
        arrays=dict(rgb=np.zeros((H,W,3),np.uint8),label=np.full((H,W),255,np.uint8),valid=np.zeros((H,W),bool),area_m2=np.zeros((H,W)),fusion=np.zeros((H,W),np.float32),value_gate=np.zeros((H,W),np.float32),selected_fusion=np.zeros((8,20),bool),selected_value_gate=np.zeros((8,20),bool))
        for n,row in enumerate(gg.itertuples(index=False)):
            dy,dx=divmod(n,5);sl=(slice(dy*256,(dy+1)*256),slice(dx*256,(dx+1)*256));ci=int(row.cache_index);arrays['rgb'][sl]=np.asarray(x[ci,2:5]).transpose(1,2,0);arrays['label'][sl]=lab[ci];arrays['valid'][sl]=np.asarray(sup[ci])==7;arrays['area_m2'][sl]=np.asarray(area[ci],dtype='float64')[:,None]
            for m in ['fusion','value_gate']:
                arrays[m][sl]=pred[m][row.prediction_index];sel=uf[(uf.model==m)&(uf.prediction_index==row.prediction_index)].sort_values('unit_index').selected.to_numpy().reshape(4,4);arrays['selected_'+m][dy*4:(dy+1)*4,dx*4:(dx+1)*4]=sel
        extent=[float(gg.left.min()),float(gg.right.max()),float(gg.bottom.min()),float(gg.top.max())];arrays['extent']=np.array(extent);np.savez_compressed(SD/f'panel_e_{region}_window.npz',**arrays)
        caseunits=uf[(uf.region==region)&uf.prediction_index.isin(ids)];caseunits.to_csv(SD/f'panel_e_{region}_units.csv',index=False)
        summary={}
        for m in ['fusion','value_gate']:
            sel=arrays['selected_'+m].repeat(64,axis=0).repeat(64,axis=1);fn=(arrays[m]<.5)&(arrays['label']==1)&arrays['valid'];w=arrays['area_m2'];vv=[float(w[fn].sum()/1e6),float(w[fn&sel].sum()/1e6),float(w[fn&~sel].sum()/1e6)];uu=caseunits[caseunits.model==m];assert np.allclose(vv,[uu.initial_fn_km2.sum(),uu.corrected_fn_km2.sum(),uu.residual_fn_km2.sum()],atol=1e-10);summary[m]=dict(initial_fn_km2=vv[0],corrected_fn_km2=vv[1],residual_fn_km2=vv[2],selected_units=int(arrays['selected_'+m].sum()))
        caseinfo.append(dict(region=region,grid_id=chosen.grid_id,rows=2,columns=5,extent_lonlat=extent,crs='EPSG:4326',prediction_indices=ids,sample_ids=gg.sample_id.tolist(),eligible_windows=len(eligible),candidate_score=chosen.to_dict(),statistics=summary,release_sha256=release_sha));log(f'Geographic case {region}: '+json.dumps(summary))
    save('panel_e_case_selection.json',dict(rule='All fully occupied contiguous 2-row x 5-column native-grid windows. Require common-valid pixel coverage >= 85%, Fusion-minus-Value corrected FN > 0, Value-minus-Fusion remaining FN > 0, and >= 4 units with different selection. Rank by decreasing Fusion recovery advantage, decreasing residual difference, then ascending grid id, row and column. Select first eligible window per region.',interpretation='Outcome-selected illustrative examples of fixed regional 10% selections; all candidate scores retained; review is not reallocated within the crop.',cases=caseinfo,display='Raw RGB channels globally transformed identically within each regional pair; actual probability/label masks determine overlays; nearest-neighbour raster rendering.'))
    checks.update(geographic_cases_contiguous=True,case_pixel_error_masks_match_unit_sums=True,case_selections_are_regional_primary_selections=True)
    save('scientific_audit.json',dict(passed=all(checks.values()),checks=checks,release_sha256=release_sha,started_at_utc=started,finished_at_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.time()-start,algorithm_version='fig04-source-v1',pixel_unit_max_abs_errors=maxima.tolist(),core_patches=len(frame),models=MODELS,valid_pixels=int(frame.common_valid_pixels.sum()),source_files_bound=len(bindings),frozen_files_checked=len(release['sha256']),maximum_conservation_error_km2=float(abs(table[['fn_conservation_error_km2','fp_conservation_error_km2']].to_numpy()).max())))
    log('Fig04 scientific derivation PASSED')
if __name__=='__main__':main()
