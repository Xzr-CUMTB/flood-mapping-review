"""Cross-event temperature-calibration diagnostic on the completed development experts."""
from pathlib import Path
from datetime import datetime,timezone
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit,logit

ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data/processed/sen2gf3'
RUNS={'sar':'dev_sar_seed17','fusion':'dev_fusion_seed17'}


def main():
    out=ROOT/'runs/calibration_development';out.mkdir(parents=True,exist_ok=True)
    tables={name:pd.read_csv(ROOT/'runs'/run/'validation_patch_metrics.csv') for name,run in RUNS.items()}
    frame=tables['sar']
    if frame.cache_index.tolist()!=tables['fusion'].cache_index.tolist():raise ValueError('Paired sample mismatch')
    groups=sorted(frame.event_group.unique())
    if len(groups)!=2 or '2023_holdout' in groups:raise ValueError('Expected two development validation groups')
    probs={name:np.load(ROOT/'runs'/run/'validation_probabilities.npy',mmap_mode='r') for name,run in RUNS.items()}
    ymap=np.load(DATA/'labels_uint8.npy',mmap_mode='r');support=np.load(DATA/'support_uint8.npy',mmap_mode='r');areas=np.load(DATA/'row_area_m2.npy',mmap_mode='r')
    sampled={g:{k:[] for k in ['sar','fusion','y','weight']} for g in groups}
    for j,row in frame.iterrows():
        idx=int(row.cache_index);valid=np.flatnonzero(np.asarray(support[idx]).ravel()==7)
        chosen=np.random.default_rng(17000+idx).choice(valid,size=min(1024,len(valid)),replace=False)
        sampled[row.event_group]['y'].append((np.asarray(ymap[idx]).ravel()[chosen]==1).astype('float64'))
        sampled[row.event_group]['weight'].append(np.asarray(areas[idx],dtype='float64')[chosen//256]*len(valid)/len(chosen))
        for name,p in probs.items():
            sampled[row.event_group][name].append(logit(np.clip(np.asarray(p[j]).ravel()[chosen].astype('float64'),1e-7,1-1e-7)))
    sampled={g:{k:np.concatenate(v) for k,v in values.items()} for g,values in sampled.items()}
    temperatures={};fit_records=[]
    for held in groups:
        training=[g for g in groups if g!=held][0];s=sampled[training];weight=s['weight']/s['weight'].sum()
        for name in RUNS:
            def objective(log_temperature):
                z=s[name]/np.exp(log_temperature)
                return float(np.sum(weight*(np.logaddexp(0,z)-s['y']*z)))
            fit=minimize_scalar(objective,bounds=(-3,3),method='bounded',options={'xatol':1e-6})
            if not fit.success:raise ValueError('Temperature optimization failed')
            t=float(np.exp(fit.x));temperatures[(held,name)]=t
            fit_records.append({'model':name,'calibration_event':training,'evaluation_event':held,'temperature':t,
                                'calibration_sampled_pixels':len(s['y']),'calibration_area_weighted_nll':float(fit.fun)})
    sums={}
    for j,row in frame.iterrows():
        idx=int(row.cache_index);valid=np.asarray(support[idx])==7;y=np.asarray(ymap[idx])==1
        w=np.broadcast_to(np.asarray(areas[idx],dtype='float64')[:,None],(256,256))*valid
        for name,source in probs.items():
            raw=np.clip(np.asarray(source[j],dtype='float64'),1e-7,1-1e-7)
            calibrated=expit(logit(raw)/temperatures[(row.event_group,name)])
            if not np.array_equal(raw>=.5,calibrated>=.5):raise ValueError('Positive temperature changed the decision threshold')
            for mode,p in [('raw',raw),('cross_event_temperature',calibrated)]:
                pred=p>=.5;safe=np.clip(p,1e-12,1-1e-12)
                values=np.array([w[y&pred].sum(),w[~y&pred].sum(),w[y&~pred].sum(),w.sum(),
                                 (w*(p-y)**2).sum(),(w*(-(y*np.log(safe)+(~y)*np.log1p(-safe)))).sum(),
                                 w[y&(p<=.1)].sum(),w[y].sum()])
                for group in [row.event_group,'ALL']:
                    key=(name,mode,group);sums[key]=sums.get(key,np.zeros(8))+values
    records=[]
    for (name,mode,group),(tp,fp,fn,area,bs,nll,conf,truth) in sums.items():
        records.append({'model':name,'mode':mode,'event_group':group,'flood_iou':tp/(tp+fp+fn),'fn_rate':fn/truth,
                        'brier':bs/area,'nll':nll/area,'confident_omission_fraction':conf/truth,
                        'confident_omission_km2':conf/1e6,'true_flood_km2':truth/1e6})
    pd.DataFrame(records).to_csv(out/'metrics.csv',index=False);pd.DataFrame(fit_records).to_csv(out/'temperature_fits.csv',index=False)
    report={'status':'complete','scope':'development diagnostic; checkpoints selected on both validation events, not independent final-test evidence',
            'calibration':'positive scalar temperature; weighted NLL fit on the other development event',
            'sampling':'up to 1024 valid pixels per patch, fixed seed 17000+cache_index, inverse-sampling geodesic-area weights',
            'calibration_not_reused_as_OOF_supervision':True,'core_test_evaluated':False,
            'completed_at_utc':datetime.now(timezone.utc).isoformat(),'fits':fit_records}
    (out/'complete.json').write_text(json.dumps(report,indent=2));print(pd.DataFrame(records).to_string(index=False))


if __name__=='__main__':main()
