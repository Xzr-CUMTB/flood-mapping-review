"""Read-only full-population outer-event diagnostic derivation; no training."""
from pathlib import Path
import os,json,hashlib,time
os.environ.setdefault('MKL_THREADING_LAYER','SEQUENTIAL')
import numpy as np
import pandas as pd
ROOT=Path(os.path.abspath(__file__)).parents[4];OUT=ROOT/'figures/main/Fig03';SD=OUT/'source_data';SD.mkdir(exist_ok=True)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(n,x):(SD/n).write_text(json.dumps(x,indent=2),encoding='utf-8')
def main():
    release=read(ROOT/'configs/core_release_s17.json');release_sha=sha(ROOT/'configs/core_release_s17.json')
    checks={};bindings=[]
    for rel,digest in release['sha256'].items():
        assert sha(ROOT/rel)==digest,rel
    checks['all_frozen_release_files_preserved']=True
    for fig in ['Fig01','Fig02']:
        prior=ROOT/'figures/main'/fig;v=read(prior/(fig+'_validation.json'))
        for item in v['sha256_manifest']:assert sha(prior/item['path'])==item['sha256'],item['path']
        checks[fig+'_unchanged']=True
    y=np.load(ROOT/'data/processed/sen2gf3/labels_uint8.npy',mmap_mode='r');sup=np.load(ROOT/'data/processed/sen2gf3/support_uint8.npy',mmap_mode='r');ar=np.load(ROOT/'data/processed/sen2gf3/row_area_m2.npy',mmap_mode='r')
    u=np.linspace(-np.log(101),np.log(101),101);be=np.sign(u)*.01*np.expm1(abs(u));be[0]=-1;be[-1]=1;ce=np.linspace(0,1,41)
    metrics=[];summaries=[];proofs=[];records_all=[];cal=[];bins=[];routebins=[];delta=[];arrays={};maxgate=0.;maxprob=0.
    for fold in range(4):
        run=ROOT/'runs'/f'nested_s17_outer{fold}_value';cfg=read(run/'config.json');spec=read(run/'input_spec.json');rec=pd.read_csv(run/'prediction_records.csv');train=pd.read_csv(run/'training_records.csv');event=rec.event_group.unique().tolist();assert len(event)==1;event=event[0]
        assert read(run/'complete.json')['status']=='complete' and cfg['epochs']==10 and cfg['seed']==17 and cfg['temperature']==.05
        assert event not in cfg['source_provenance']['training_groups'] and event not in set(train.event_group)
        for pair in spec['pairs']:
            for k in ['sar_run','fusion_run']:
                ec=read(ROOT/'runs'/pair[k]/'config.json');assert event not in ec['train_groups'].split(',');assert not set(pair['groups']) & set(ec['train_groups'].split(','));assert ec['checkpoint_selection_mode']=='fixed_last' and ec['epochs']==20
        for k in ['sar_run','fusion_run']:
            for rn in rec[k].unique():
                ec=read(ROOT/'runs'/rn/'config.json');assert event not in ec['train_groups'].split(',');assert ec['checkpoint_selection_mode']=='fixed_last'
        proofs.append(dict(fold=fold,event=event,head_run=run.name,training_groups=cfg['source_provenance']['training_groups'],inner_sources=cfg['source_provenance']['sources'],kind=cfg['kind'],parameters=cfg['parameters'],epochs=cfg['epochs'],seed=cfg['seed'],temperature=cfg['temperature']))
        for n in ['config.json','input_spec.json','training_records.csv','prediction_records.csv','complete.json','prediction_complete.json','last.pt','validation_values.npy','validation_gates.npy','validation_probabilities.npy']:
            p=run/n;bindings.append(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)))
        pa={rn:np.load(ROOT/'runs'/rn/'validation_probabilities.npy',mmap_mode='r') for rn in set(rec.sar_run)|set(rec.fusion_run)}
        v=np.load(run/'validation_values.npy',mmap_mode='r');g=np.load(run/'validation_gates.npy',mmap_mode='r');p=np.load(run/'validation_probabilities.npy',mmap_mode='r');assert len(v)==len(rec)
        B=np.zeros((100,100));C=np.zeros((40,40));CW=np.zeros_like(C);BS=np.zeros((100,4));mom=np.zeros(10);sums={m:np.zeros(6) for m in ['sar','fusion','value_gate']};pixels=0
        for j,row in enumerate(rec.itertuples(index=False)):
            idx=int(row.cache_index);m=np.asarray(sup[idx])==7;yy=np.asarray(y[idx])[m].astype(np.float64);w=np.broadcast_to(np.asarray(ar[idx],dtype=np.float64)[:,None],(256,256))[m];ss=np.asarray(pa[row.sar_run][row.prediction_index])[m].astype(np.float64);ff=np.asarray(pa[row.fusion_run][row.prediction_index])[m].astype(np.float64);vv=np.asarray(v[j])[m].astype(np.float64);gg=np.asarray(g[j])[m].astype(np.float64);pp=np.asarray(p[j])[m].astype(np.float64)
            assert all(np.isfinite(a).all() for a in [ss,ff,vv,gg,pp])
            maxgate=max(maxgate,float(np.max(abs(gg-1/(1+np.exp(-vv/.05))))));maxprob=max(maxprob,float(np.max(abs(pp-(ss+gg*(ff-ss))))))
            actual=(ss-yy)**2-(ff-yy)**2;benefit=(ff-yy)**2-(pp-yy)**2;dis=abs(ff-ss)
            bi=np.minimum(np.searchsorted(be,vv,side='right')-1,99);bj=np.minimum(np.searchsorted(be,actual,side='right')-1,99);ci=np.minimum((dis*40).astype(int),39);cj=np.minimum((gg*40).astype(int),39)
            B+=np.bincount(bi*100+bj,weights=w,minlength=10000).reshape(100,100);C+=np.bincount(ci*40+cj,weights=w,minlength=1600).reshape(40,40);CW+=np.bincount(ci*40+cj,weights=w*benefit,minlength=1600).reshape(40,40)
            for k,z in enumerate([np.ones_like(vv),vv,actual,actual**2]):BS[:,k]+=np.bincount(bi,weights=w*z,minlength=100)
            for k,z in enumerate([np.ones_like(vv),vv,actual,vv*vv,actual*actual,vv*actual,(np.sign(vv)==np.sign(actual)),gg,benefit,(vv-actual)**2]):mom[k]+=np.sum(w*z)
            for model,pr in [('sar',ss),('fusion',ff),('value_gate',pp)]:
                pos=pr>=.5;truth=yy==1;sums[model]+=np.array([w[pos&truth].sum(),w[pos&~truth].sum(),w[~pos&truth].sum(),w.sum(),np.sum(w*(pr-yy)**2),w[truth].sum()])
            pixels+=len(w)
            if j%1000==0:print(f'fold {fold} {j}/{len(rec)}',flush=True)
        assert np.isclose(B.sum(),mom[0]) and np.isclose(C.sum(),mom[0])
        arrays[f'b_{fold}']=B;arrays[f'c_{fold}']=C;arrays[f'benefit_sum_{fold}']=CW;arrays[f'cal_{fold}']=BS
        mx,my=mom[1:3]/mom[0];corr=(mom[5]/mom[0]-mx*my)/np.sqrt((mom[3]/mom[0]-mx*mx)*(mom[4]/mom[0]-my*my))
        summaries.append(dict(event=event,tiles=len(rec),valid_pixels=pixels,valid_area_km2=mom[0]/1e6,predicted_mean=mx,observed_mean=my,area_weighted_pearson=corr,sign_agreement=mom[6]/mom[0],mean_gate=mom[7]/mom[0],fusion_minus_gate_brier=mom[8]/mom[0],value_mse=mom[9]/mom[0]))
        raw=pd.read_csv(ROOT/'runs'/f'nested_development_fold{fold}'/'map_metrics.csv')
        for model,z in sums.items():
            tp,fp,fn,area,brier,truth=z;metric=dict(fold=fold,event_group=event,model=model,flood_iou=tp/(tp+fp+fn),brier=brier/area,fn_km2=fn/1e6,fp_km2=fp/1e6,valid_area_km2=area/1e6,true_flood_km2=truth/1e6);orig=raw[raw.model==('sigmoid_value' if model=='value_gate' else model)].iloc[0]
            for k in ['flood_iou','brier','fn_km2','fp_km2','valid_area_km2']:assert np.isclose(metric[k],orig[k],rtol=2e-6,atol=2e-7),(event,model,k,metric[k],orig[k])
            metrics.append(metric)
        records_all.append(rec.assign(head_run=run.name,head_prediction_index=np.arange(len(rec))))
        for i in range(100):cal.append(dict(event=event,lower=be[i],upper=be[i+1],area_m2=BS[i,0],predicted_sum=BS[i,1],observed_sum=BS[i,2],observed_square_sum=BS[i,3]))
        for i in range(100):
            for k in range(100):bins.append(dict(event=event,x_lower=be[i],x_upper=be[i+1],y_lower=be[k],y_upper=be[k+1],area_m2=B[i,k]))
        for i in range(40):
            for k in range(40):routebins.append(dict(event=event,disagreement_lower=ce[i],disagreement_upper=ce[i+1],gate_lower=ce[k],gate_upper=ce[k+1],area_m2=C[i,k],fusion_minus_gate_brier_area_sum=CW[i,k]))
        print(json.dumps(summaries[-1]),flush=True)
    assert maxgate<1e-6 and maxprob<2e-7
    df=pd.DataFrame(metrics);df.to_csv(SD/'panel_de_event_metrics.csv',index=False)
    for event,group in df.groupby('event_group',sort=False):
        f=group[group.model=='fusion'].iloc[0];v0=group[group.model=='value_gate'].iloc[0];delta.append(dict(event_group=event,delta_iou=v0.flood_iou-f.flood_iou,delta_brier=v0.brier-f.brier,delta_fn_km2=v0.fn_km2-f.fn_km2,delta_fp_km2=v0.fp_km2-f.fp_km2,valid_area_km2=f.valid_area_km2,true_flood_km2=f.true_flood_km2))
    pd.DataFrame(delta).to_csv(SD/'panel_de_paired_effects.csv',index=False);pd.DataFrame(summaries).to_csv(SD/'panel_bc_event_summary.csv',index=False);pd.DataFrame(cal).to_csv(SD/'panel_b_calibration.csv',index=False);pd.DataFrame(bins).to_csv(SD/'panel_b_joint_bins.csv.gz',index=False);pd.DataFrame(routebins).to_csv(SD/'panel_c_joint_bins.csv.gz',index=False);pd.concat(records_all).to_csv(SD/'oof_prediction_index.csv.gz',index=False)
    np.savez_compressed(SD/'joint_statistics.npz',b_edges=be,c_edges=ce,**arrays)
    core=pd.read_csv(ROOT/'runs/core_s17_evaluation/map_metrics.csv');core=core[core.model.isin(['fusion','ordinary_gate','value_gate'])].copy();core['release_sha256']=release_sha;core.to_csv(SD/'panel_f_frozen_metrics.csv',index=False)
    save('panel_a_crossfit_provenance.json',proofs);save('input_bindings.json',bindings)
    checks.update(outer_event_exclusion=True,head_prediction_indices_match=True,saved_gate_formula=True,saved_probability_formula=True,full_domain_metrics_match_original_csv=True,all_pixels_binned=True)
    save('scientific_audit.json',dict(passed=all(checks.values()),checks=checks,release_sha256=release_sha,release_files=len(release['sha256']),max_gate_absolute_error=maxgate,max_mixture_absolute_error=maxprob,events=summaries,model_name_mapping='nested sigmoid_value = saved nested value-kind head; same route function as final value_gate; distinct event-specific trained parameters',density_domain='support==7; all valid pixels included; area weighted; no visualization sampling',no_retraining=True))
    print('Fig03 derivation PASSED',flush=True)
if __name__=='__main__':main()
