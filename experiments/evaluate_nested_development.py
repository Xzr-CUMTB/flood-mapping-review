"""Review completed nested development folds and fixed analytic routing diagnostics."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, json, sys, time
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.train_baseline import atomic_json
from experiments.evaluation_metrics import patch_metrics, metric_result, review_units, whole_unit_selection


def analytic_gate(value, sar, fusion, ridge=0.):
    d2=(fusion-sar)**2
    adjustment=np.divide(value,2*(d2+ridge),out=np.zeros_like(value,dtype=np.float64),where=(d2+ridge)>1e-12)
    g=np.clip(.5+adjustment,0.,1.)
    return sar+g*(fusion-sar),g


def evaluate_fold(fold):
    directory=ROOT/'runs'/f'nested_s17_outer{fold}_value'
    if not (directory/'complete.json').exists():
        return None
    config=json.loads((directory/'config.json').read_text())
    records=pd.read_csv(directory/'prediction_records.csv')
    groups=set(records.event_group)
    if '2023_holdout' in groups or groups.intersection(config['source_provenance']['training_groups']):
        raise ValueError('Expected event-held-out development predictions')
    if not groups.issubset(config['source_provenance']['outer_excluded_groups']):
        raise ValueError('Whole pipeline did not exclude the target event')
    out=ROOT/'runs'/f'nested_development_fold{fold}'
    if (out/'complete.json').exists():
        return json.loads((out/'complete.json').read_text())
    out.mkdir(parents=True,exist_ok=True)
    data=ROOT/'data/processed/sen2gf3'
    labels,support,areas=[np.load(data/name,mmap_mode='r') for name in ['labels_uint8.npy','support_uint8.npy','row_area_m2.npy']]
    sources={name:np.load(ROOT/'runs'/name/'validation_probabilities.npy',mmap_mode='r')
             for name in set(records.sar_run)|set(records.fusion_run)}
    pg=np.load(directory/'validation_probabilities.npy',mmap_mode='r')
    values=np.load(directory/'validation_values.npy',mmap_mode='r')
    gates=np.load(directory/'validation_gates.npy',mmap_mode='r')
    names=['sar','fusion','probability_average','sigmoid_value','analytic_value','ridge_value_001']
    sums={name:np.zeros(10) for name in names}
    units={name:[] for name in names}
    mechanism=np.zeros(8);start=time.monotonic()
    for j,row in enumerate(records.itertuples()):
        idx=int(row.cache_index);k=int(row.prediction_index)
        y=np.asarray(labels[idx])==1;valid=np.asarray(support[idx])==7
        w=np.broadcast_to(np.asarray(areas[idx],dtype=np.float64)[:,None],(256,256))*valid
        s=np.asarray(sources[row.sar_run][k],dtype=np.float64);f=np.asarray(sources[row.fusion_run][k],dtype=np.float64)
        v=np.asarray(values[j],dtype=np.float64);g=np.asarray(gates[j],dtype=np.float64)
        analytic,ga=analytic_gate(v,s,f);ridge,gr=analytic_gate(v,s,f,.01)
        predictions={'sar':s,'fusion':f,'probability_average':(s+f)/2,'sigmoid_value':np.asarray(pg[j],dtype=np.float64),
                     'analytic_value':analytic,'ridge_value_001':ridge}
        target=(s-y)**2-(f-y)**2
        mechanism+=np.array([w.sum(),(w*(v-target)**2).sum(),(w*v).sum(),(w*target).sum(),
                             w[g<.1].sum(),w[g>.9].sum(),w[((predictions['sigmoid_value']>=.5)!=(f>=.5))].sum(),
                             (w*ga).sum()])
        for name,p in predictions.items():
            sums[name]+=patch_metrics(p,y,w)
            u=review_units(p,y,w,s,f)
            units[name].append(np.stack([u[key] for key in ['cost_m2','fn_m2','fp_m2','mean_entropy','fn_probability_density']],axis=1))
        if (j+1)%500==0:print(json.dumps({'fold':fold,'patches':j+1,'total':len(records)}),flush=True)
    metrics=[{'fold':fold,'event_group':','.join(sorted(groups)),'model':name,**metric_result(value)} for name,value in sums.items()]
    pd.DataFrame(metrics).to_csv(out/'map_metrics.csv',index=False)
    reviews=[]
    for name,parts in units.items():
        matrix=np.concatenate(parts);matrix=matrix[matrix[:,0]>0]
        cost,fn,fp,ent,proxy=matrix.T
        for fraction in [.01,.05,.1,.2]:
            for policy,score in [('mean_entropy',ent),('fn_probability_density',proxy),('oracle_fn_density',fn/cost)]:
                chosen,used=whole_unit_selection(score,cost,fraction)
                reviews.append({'fold':fold,'event_group':','.join(sorted(groups)),'model':name,'policy':policy,
                    'budget_fraction':fraction,'actual_area_fraction':used/cost.sum(),
                    'residual_fn_km2':fn[~chosen].sum()/1e6,'residual_fp_km2':fp[~chosen].sum()/1e6})
    pd.DataFrame(reviews).to_csv(out/'review_metrics.csv',index=False)
    area,mse,vsum,targetsum,gzero,gone,changed,analytic_sum=mechanism
    result={'status':'complete','fold':fold,'samples':len(records),'event_groups':sorted(groups),
        'scope':'Nested development OOF; analytic/ridge rules are development diagnostics after observing fold 0 sigmoid results',
        'mechanism':{'value_target_area_weighted_mse':mse/area,'predicted_mean_value':vsum/area,'actual_mean_value':targetsum/area,
                     'gate_below_01_fraction':gzero/area,'gate_above_09_fraction':gone/area,
                     'decision_changed_vs_fusion_fraction':changed/area,'mean_analytic_gate':analytic_sum/area},
        'core_test_evaluated':False,'completed_at_utc':datetime.now(timezone.utc).isoformat(),'seconds':time.monotonic()-start}
    atomic_json(out/'complete.json',result);print(json.dumps(result),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--fold',type=int,choices=[0,1,2,3]);args=parser.parse_args()
    for fold in ([args.fold] if args.fold is not None else range(4)):evaluate_fold(fold)
