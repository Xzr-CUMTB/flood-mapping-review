"""Paired H1/H2 development screening; no plots, no final-test access, no trained risk head."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data/processed/sen2gf3'

def whole_unit_selection(score,cost,budget):
    order=np.argsort(-score,kind='stable');chosen=np.zeros(len(score),dtype=bool);used=0.
    for i in order:
        if used+cost[i]<=budget+1e-9:chosen[i]=True;used+=cost[i]
    return chosen,used

def run(sar_run,fusion_run):
    sa=ROOT/'runs'/sar_run;fu=ROOT/'runs'/fusion_run
    for p in [sa,fu]:
        if not (p/'complete.json').exists():raise ValueError('Completed model runs required')
    a=pd.read_csv(sa/'validation_patch_metrics.csv');b=pd.read_csv(fu/'validation_patch_metrics.csv')
    assert a.cache_index.tolist()==b.cache_index.tolist()
    assert '2023_holdout' not in set(a.event_group)
    ps=np.load(sa/'validation_probabilities.npy',mmap_mode='r');pf=np.load(fu/'validation_probabilities.npy',mmap_mode='r')
    ymap=np.load(DATA/'labels_uint8.npy',mmap_mode='r');support=np.load(DATA/'support_uint8.npy',mmap_mode='r');areas=np.load(DATA/'row_area_m2.npy',mmap_mode='r')
    units=[];quadrant=0.;quadrant_flood=0.;total=0.;flood=0.;by_threshold={str(t):{'sar':0.,'fusion':0.} for t in [.05,.1,.2]};aggregate={}
    entropy=lambda p:-(p*np.log(p)+(1-p)*np.log1p(-p))
    for j,row in a.iterrows():
        idx=int(row.cache_index);y=np.asarray(ymap[idx])==1;valid=np.asarray(support[idx])==7
        w=np.broadcast_to(np.asarray(areas[idx],dtype='float64')[:,None],(256,256))*valid
        s=np.clip(np.asarray(ps[j],dtype='float64'),1e-7,1-1e-7);f=np.clip(np.asarray(pf[j],dtype='float64'),1e-7,1-1e-7)
        hs=entropy(s);hf=entropy(f);dh=hf-hs;dl=(f-y)**2-(s-y)**2
        mean=(s+f)/2;models=[('sar',s,hs),('fusion',f,hf),('probability_average',mean,entropy(mean))]
        for name,p,h in models:
            pred=p>=.5
            values=np.array([w[y&pred].sum(),w[~y&pred].sum(),w[y&~pred].sum(),w.sum(),
                             (w*(p-y)**2).sum(),(w*(-(y*np.log(p)+(~y)*np.log1p(-p)))).sum(),w[y&(p<=.1)].sum(),w[y].sum()])
            for group in [row.event_group,'ALL']:
                key=(name,group);aggregate[key]=aggregate.get(key,np.zeros(8))+values
        mask=(dh<0)&(dl>0);quadrant+=float(w[mask].sum());quadrant_flood+=float(w[mask&y].sum());total+=float(w.sum());flood+=float(w[y].sum())
        for threshold in [.05,.1,.2]:
            by_threshold[str(threshold)]['sar']+=float(w[y&(s<=threshold)].sum())
            by_threshold[str(threshold)]['fusion']+=float(w[y&(f<=threshold)].sum())
        for rr in range(0,256,64):
            for cc in range(0,256,64):
                tile=np.s_[rr:rr+64,cc:cc+64];weight=w[tile];cost=float(weight.sum())
                if cost<=0:continue
                for name,p,h in models:
                    q=p[tile];truth=y[tile];negative=q<.5
                    fn=float(weight[truth&negative].sum());fp=float(weight[(~truth)&(~negative)].sum())
                    units.append({'model':name,'cache_index':idx,'row':rr,'col':cc,'event_group':row.event_group,
                                  'cost_m2':cost,'fn_m2':fn,'fp_m2':fp,'mean_entropy':float((weight*h[tile]).sum()/cost),
                                  'total_entropy':float((weight*h[tile]).sum()),'mean_disagreement':float((weight*np.abs(s-f)[tile]).sum()/cost),
                                  'fn_probability_density':float((weight*q*negative).sum()/cost)})
    out=ROOT/'runs/development_pair';out.mkdir(parents=True,exist_ok=True)
    df=pd.DataFrame(units);df.to_csv(out/'review_units.csv',index=False)
    metrics=[]
    for (model,group),(tp,fp,fn,area,bs,nll,conf,truth) in aggregate.items():
        ratio=lambda a,b:float(a/b) if b>0 else None
        metrics.append({'model':model,'event_group':group,'flood_iou':ratio(tp,tp+fp+fn),'fn_rate':ratio(fn,truth),
                        'brier':ratio(bs,area),'nll':ratio(nll,area),'confident_omission_fraction':ratio(conf,truth),
                        'fn_km2':fn/1e6,'fp_km2':fp/1e6,'valid_area_km2':area/1e6,'true_flood_km2':truth/1e6})
    pd.DataFrame(metrics).to_csv(out/'baseline_and_average_metrics.csv',index=False)
    rows=[]
    for model,g in df.groupby('model',sort=False):
        cost=g.cost_m2.to_numpy();fn=g.fn_m2.to_numpy();fp=g.fp_m2.to_numpy()
        scores={k:g[k].to_numpy() for k in ['mean_entropy','total_entropy','mean_disagreement','fn_probability_density']}
        scores.update({f'random_seed_{seed}':np.random.default_rng(seed).random(len(g)) for seed in [17,29,43]})
        scores['oracle_fn_density']=fn/cost
        for budget in [.01,.05,.1,.2]:
            limit=budget*cost.sum()
            for method,score in scores.items():
                selected,used=whole_unit_selection(score,cost,limit)
                rows.append({'model':model,'budget_fraction':budget,'method':method,'actual_area_fraction':used/cost.sum(),
                             'initial_fn_km2':fn.sum()/1e6,'residual_fn_km2':fn[~selected].sum()/1e6,
                             'corrected_fn_km2':fn[selected].sum()/1e6,'residual_fp_km2':fp[~selected].sum()/1e6,
                             'correction_assumption':'perfect correction of all errors inside selected whole units; not human performance'})
    pd.DataFrame(rows).to_csv(out/'idealized_review_screen.csv',index=False)
    for values in by_threshold.values():
        values.update(sar_km2=values['sar']/1e6,fusion_km2=values['fusion']/1e6,delta_fraction=(values['fusion']-values['sar'])/flood if flood else None)
    report={'scope':'development screening; checkpoints selected on this validation group; not final test or OOF supervision',
            'sar_run':sar_run,'fusion_run':fusion_run,'paired_patches':len(a),'valid_area_km2':total/1e6,
            'true_flood_km2':flood/1e6,'lower_entropy_higher_brier_all_area_fraction':quadrant/total,
            'lower_entropy_higher_brier_flood_area_fraction':quadrant_flood/flood if flood else None,
            'confident_omissions':by_threshold,'review_unit_pixels':[64,64],
            'review_selection':'whole-unit descending score, skip units exceeding remaining valid-area budget; not an optimal knapsack claim',
            'learned_risk_head_evaluated':False,'real_human_review_completed':False}
    (out/'paired_screening.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--sar-run',default='dev_sar_seed17');p.add_argument('--fusion-run',default='dev_fusion_seed17');a=p.parse_args();run(a.sar_run,a.fusion_run)
