"""Read-only Fig. 1 derivation from frozen arrays; no model fitting or threshold selection."""
import os
os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'
from pathlib import Path
import sys, json, hashlib, time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from scipy import ndimage
from PIL import Image
import rasterio
from rasterio.transform import from_bounds
from pyproj import Geod

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parents[1]
SD = OUT/'source_data'; AS = OUT/'assets'
SD.mkdir(exist_ok=True, parents=True); AS.mkdir(exist_ok=True, parents=True)
sys.path.insert(0, str(ROOT))
from experiments.evaluation_metrics import entropy, spatial_block_ids, metric_result

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(16*1024*1024),b''): h.update(b)
    return h.hexdigest()

def writej(path,x):
    path.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')

def main():
    started=datetime.now(timezone.utc).isoformat()
    rule=json.loads((SD/'case_selection_rule.json').read_text('utf-8'))
    proof=json.loads((ROOT/'reports/completed_batch_verification.json').read_text('utf-8'))
    assert proof['passed']
    release=ROOT/'configs/core_release_s17.json'
    assert sha(release)==proof['release_sha256']
    ev=ROOT/'runs/core_s17_evaluation'
    sources={str(release.relative_to(ROOT)):sha(release)}
    for fn in ['map_metrics.csv','map_paired_spatial_intervals.csv','review_budget_results.csv','review_simultaneous_intervals.csv']:
        h=sha(ev/fn); assert h==proof['csv_sha256'][fn]
        sources[str((ev/fn).relative_to(ROOT))]=h
    frame=pd.read_csv(ROOT/'runs/core_s17_predictions/prediction_manifest.csv')
    frame['region']=np.where(frame.provisional_site.eq('Harbin'),'Harbin','Beijing_Baoding')
    assert len(frame)==8233 and frame.prediction_index.tolist()==list(range(8233))
    frame.to_csv(SD/'core_geographic_manifest.csv',index=False)
    cache=ROOT/'data/processed/sen2gf3'
    x=np.load(cache/'x_uint8.npy',mmap_mode='r'); y=np.load(cache/'labels_uint8.npy',mmap_mode='r')
    support=np.load(cache/'support_uint8.npy',mmap_mode='r'); area=np.load(cache/'row_area_m2.npy',mmap_mode='r')
    pred={m:np.load(ROOT/f'runs/core_s17_predictions/{m}.npy',mmap_mode='r') for m in ['sar','fusion']}
    for m in pred:
        p=ROOT/f'runs/core_s17_predictions/{m}.npy'
        h=sha(p); expected=next(z['sha256'] for z in proof['prediction_files'] if z['path'].replace('\\','/').endswith('/'+m+'.npy'))
        assert h==expected
        sources[str(p.relative_to(ROOT))]=h
    print('Verified release, CSVs and both full prediction hashes.',flush=True)
    maps=pd.read_csv(ev/'map_metrics.csv')
    maps[maps.model.isin(['sar','fusion','value_gate','sar_temperature','fusion_temperature'])].to_csv(SD/'panel_d_h_map_metrics.csv',index=False)
    ci=pd.read_csv(ev/'map_paired_spatial_intervals.csv')
    ci[(ci.block_size_m==10000)&(ci['first'].isin(['fusion','fusion_temperature']))&(ci.second.isin(['sar','sar_temperature']))&ci.metric.isin(['flood_iou','confident_omission_01'])].to_csv(SD/'panel_g_intervals.csv',index=False)
    budget=pd.read_csv(ev/'review_budget_results.csv')
    budget[(budget.model.isin(['fusion','value_gate']))&(budget.policy=='mean_entropy')&(budget.budget_fraction==.1)].to_csv(SD/'panel_h_review.csv',index=False)
    sci=pd.read_csv(ev/'review_simultaneous_intervals.csv')
    sci[(sci.first_model=='value_gate')&(sci.second_model=='fusion')&(sci.first_policy=='mean_entropy')&(sci.second_policy=='mean_entropy')].to_csv(SD/'panel_h_simultaneous_intervals.csv',index=False)
    xedges=np.linspace(-np.log(2),np.log(2),161); yedges=np.linspace(-1,1,161)
    summaries=[]; candidates=[]; cases=[]; ecdf_list=[]
    pooled=np.zeros((160,160)); pooled_flood=np.zeros_like(pooled)
    for region, group in frame.groupby('region',sort=False):
        hist=np.zeros_like(pooled); hist_f=np.zeros_like(hist)
        totals=np.zeros(6) # all area, flood area, all Q, flood Q, SAR conf, fusion conf
        ps=[];pf=[];weights=[]
        lon0,lon1=group.left.min(),group.right.max();lat0,lat1=group.bottom.min(),group.top.max()
        coslat=np.cos(np.deg2rad((lat0+lat1)/2))
        nw=1400;nh=max(350,int(nw*(lat1-lat0)/((lon1-lon0)*coslat)))
        if nh>1800: nw=int(nw*1800/nh);nh=1800
        mosaic=np.zeros((nh*nw,3),np.float64); count=np.zeros(nh*nw); flood=np.zeros(nh*nw)
        rowcol=np.indices((32,32)); rr=rowcol[0]*8+3.5; cc=rowcol[1]*8+3.5
        rgbhist=np.zeros((3,256),np.int64)
        for n,rec in enumerate(group.itertuples()):
            idx=rec.prediction_index; k=rec.cache_index
            valid=np.asarray(support[k])==7; truth=np.asarray(y[k])==1
            w=np.broadcast_to(np.asarray(area[k],np.float64)[:,None],(256,256))
            s=np.asarray(pred['sar'][idx],np.float64);f=np.asarray(pred['fusion'][idx],np.float64)
            sv,fv,yy,ww=s[valid],f[valid],truth[valid],w[valid]
            dh=entropy(fv)-entropy(sv); db=(fv-yy)**2-(sv-yy)**2
            hist+=np.histogram2d(dh,db,bins=[xedges,yedges],weights=ww)[0]
            hist_f+=np.histogram2d(dh[yy],db[yy],bins=[xedges,yedges],weights=ww[yy])[0]
            q=(dh<0)&(db>0)
            totals+=np.array([ww.sum(),ww[yy].sum(),ww[q].sum(),ww[q&yy].sum(),ww[yy&(sv<=.1)].sum(),ww[yy&(fv<=.1)].sum()])
            ps.append(sv[yy].astype('float32'));pf.append(fv[yy].astype('float32'));weights.append(ww[yy].astype('float32'))
            vf=valid.mean(); fa=ww[yy].sum()/ww.sum()
            om=truth&valid&(f<=.1)
            if vf>=.95 and .1<=fa<=.9 and om.any():
                lab,nlab=ndimage.label(om,np.ones((3,3),int))
                ca=np.bincount(lab.ravel(),weights=np.where(om,w,0).ravel(),minlength=nlab+1)[1:]
                candidates.append({'region':region,'sample_id':int(rec.sample_id),'prediction_index':idx,'cache_index':k,
                    'common_fraction':vf,'flood_fraction':fa,'largest_connected_omission_m2':float(ca.max()),
                    'lon':rec.lon,'lat':rec.lat,'sar_confident_fraction':float(ww[yy&(sv<=.1)].sum()/ww[yy].sum()),
                    'fusion_confident_fraction':float(ww[yy&(fv<=.1)].sum()/ww[yy].sum())})
            # Geographic display only: each 8x8 cell contributes its valid-pixel mean at its true centre.
            rgb=np.asarray(x[k,2:5]).transpose(1,2,0)
            for ch in range(3): rgbhist[ch]+=np.bincount(rgb[:,:,ch][valid],minlength=256)
            vc=valid.reshape(32,8,32,8).sum((1,3)); good=vc>0
            rs=(rgb.astype(float)*valid[:,:,None]).reshape(32,8,32,8,3).sum((1,3))
            fc=(truth&valid).reshape(32,8,32,8).sum((1,3))
            xx=rec.left+(cc+.5)*rec.xres; yygeo=rec.top+(rr+.5)*rec.yres
            mx=np.clip(((xx-lon0)/(lon1-lon0)*nw).astype(int),0,nw-1)
            my=np.clip(((lat1-yygeo)/(lat1-lat0)*nh).astype(int),0,nh-1)
            flat=(my*nw+mx)[good]
            np.add.at(count,flat,vc[good]);np.add.at(flood,flat,fc[good]);np.add.at(mosaic,flat,rs[good])
            if n%800==0:print(region,n,'/',len(group),flush=True)
        pooled+=hist;pooled_flood+=hist_f
        for model,j in [('sar',4),('fusion',5)]:
            ref=maps[(maps.region==region)&(maps.model==model)].iloc[0]
            assert np.isclose(totals[j]/totals[1],ref.confident_omission_01,atol=1e-10)
            assert np.isclose(totals[0]/1e6,ref.valid_area_km2,rtol=1e-10)
        np.savez_compressed(SD/f'panel_e_{region}.npz',area_m2=hist,flood_area_m2=hist_f,xedges=xedges,yedges=yedges)
        summaries.append(dict(region=region,valid_area_m2=totals[0],reference_flood_area_m2=totals[1],
            lower_entropy_higher_brier_all_m2=totals[2],lower_entropy_higher_brier_flood_m2=totals[3],
            quadrant_share_all_valid=totals[2]/totals[0],quadrant_share_reference_flood=totals[3]/totals[1],
            sar_confident_omission=totals[4]/totals[1],fusion_confident_omission=totals[5]/totals[1]))
        for model,parts in [('sar',ps),('fusion',pf)]:
            p=np.concatenate(parts);w=np.concatenate(weights).astype(np.float64)
            order=np.argsort(p,kind='stable');p=p[order];w=w[order]
            # Equal probabilities pooled exactly. Keep exact compact step table as reproducible source.
            starts=np.r_[0,np.flatnonzero(np.diff(p)!=0)+1]; pu=p[starts];wu=np.add.reduceat(w,starts)
            cdf=np.cumsum(wu)/wu.sum()
            np.savez_compressed(SD/f'panel_f_exact_{region}_{model}.npz',probability=pu,area_m2=wu,cdf=cdf)
            # Drawing subset chosen by CDF increments <= 0.0001; exact threshold explicitly inserted.
            ix=np.unique(np.r_[0,np.searchsorted(cdf,np.arange(0,1,.0001)).clip(0,len(cdf)-1),len(cdf)-1])
            rows=pd.DataFrame({'region':region,'model':model,'probability':pu[ix].astype(float),'ecdf':cdf[ix]})
            at=float(wu[pu<=.1].sum()/wu.sum())
            rows=pd.concat([rows,pd.DataFrame([dict(region=region,model=model,probability=.1,ecdf=at)])],ignore_index=True).sort_values('probability')
            ecdf_list.append(rows)
            print('Exact ECDF',region,model,len(p),'pixels',len(pu),'unique probabilities',flush=True)
        avg=np.divide(mosaic,count[:,None],out=np.zeros_like(mosaic),where=count[:,None]>0).reshape(nh,nw,3)
        lohi=[]
        for ch in range(3):
            cum=rgbhist[ch].cumsum();lo,hi=np.searchsorted(cum,[cum[-1]*.02,cum[-1]*.98]);lohi.append([int(lo),int(hi)])
            avg[:,:,ch]=np.clip((avg[:,:,ch]-lo)/max(hi-lo,1),0,1)
        observed=count.reshape(nh,nw)>0
        rgba=np.dstack([np.round(avg*255).astype('uint8'),observed.astype('uint8')*255])
        frac=np.divide(flood,count,out=np.zeros_like(count),where=count>0).reshape(nh,nw)
        Image.fromarray(rgba).save(AS/f'{region}_rgb_context.png')
        np.savez_compressed(SD/f'panel_b_{region}.npz',rgb=rgba,flood_fraction=frac.astype('float32'),observed=observed,
            extent=np.array([lon0,lon1,lat0,lat1]),stretch=np.array(lohi))
        with rasterio.open(AS/f'{region}_context.tif','w',driver='GTiff',height=nh,width=nw,count=5,dtype='float32',
            crs='EPSG:4326',transform=from_bounds(lon0,lat0,lon1,lat1,nw,nh),compress='deflate') as dst:
            dst.write(np.moveaxis(avg.astype('float32'),-1,0),indexes=[1,2,3]);dst.write(observed.astype('float32'),4);dst.write(frac.astype('float32'),5)
    pd.concat(ecdf_list).to_csv(SD/'panel_f_ecdf_display.csv',index=False)
    cand=pd.DataFrame(candidates);cand.to_csv(SD/'panel_c_all_eligible_cases.csv',index=False)
    for region,group in cand.groupby('region'):
        geo=frame[frame.region==region];clon=np.average(geo.lon,weights=geo.common_area_km2);clat=np.average(geo.lat,weights=geo.common_area_km2)
        target=float(group.largest_connected_omission_m2.quantile(.75))
        group=group.copy();group['target_distance']=abs(group.largest_connected_omission_m2-target)
        group['geographic_tie_distance']=((group.lon-clon)*np.cos(np.deg2rad(clat)))**2+(group.lat-clat)**2
        selected=group.sort_values(['target_distance','geographic_tie_distance','sample_id']).iloc[0].to_dict()
        rec=frame[frame.sample_id==selected['sample_id']].iloc[0]; k=int(rec.cache_index);i=int(rec.prediction_index)
        label_path=ROOT/f'data/extracted/sen2gf3/Sen2GF3Floods/label/label_{int(rec.sample_id)}.tif'
        with rasterio.open(label_path) as ds:
            assert ds.crs.to_epsg()==4326 and np.array_equal(ds.read(1),y[k])
            assert np.allclose(ds.bounds,[rec.left,rec.bottom,rec.right,rec.top])
        selected.update(target_75th_percentile_m2=target,eligible_cases=len(group),bounds=[rec.left,rec.right,rec.bottom,rec.top],
            label_path=str(label_path.relative_to(ROOT)),label_sha256=sha(label_path),crs='EPSG:4326')
        np.savez_compressed(SD/f'panel_c_{region}.npz',x=x[k],label=y[k],support=support[k],row_area_m2=area[k],
            sar=pred['sar'][i],fusion=pred['fusion'][i],value_gate=np.load(ROOT/'runs/core_s17_predictions/value_gate.npy',mmap_mode='r')[i],bounds=selected['bounds'])
        cases.append(selected)
    writej(SD/'panel_c_selected_cases.json',cases)
    pool=np.sum([[s[k] for k in ['valid_area_m2','reference_flood_area_m2','lower_entropy_higher_brier_all_m2','lower_entropy_higher_brier_flood_m2']] for s in summaries],axis=0)
    summaries.append(dict(region='Pooled_area_weighted',valid_area_m2=pool[0],reference_flood_area_m2=pool[1],
        lower_entropy_higher_brier_all_m2=pool[2],lower_entropy_higher_brier_flood_m2=pool[3],quadrant_share_all_valid=pool[2]/pool[0],quadrant_share_reference_flood=pool[3]/pool[1]))
    pd.DataFrame(summaries).to_csv(SD/'panel_e_quadrant_shares.csv',index=False)
    np.savez_compressed(SD/'panel_e_pooled.npz',area_m2=pooled,flood_area_m2=pooled_flood,xedges=xedges,yedges=yedges)
    # Preserve regional spatial heterogeneity in a source table; only fixed-protocol intervals plotted.
    sums=np.load(ev/'patch_metric_sums.npy',mmap_mode='r');schema=json.loads((ev/'array_schema.json').read_text())
    blockrows=[]
    for region,group in frame.groupby('region'):
        ids=spatial_block_ids(group.lon,group.lat,'EPSG:32652' if region=='Harbin' else 'EPSG:32650',10000)
        group=group.copy();group['block']=[str(z) for z in ids]
        for block,bg in group.groupby('block'):
            vals={m:metric_result(sums[schema['models'].index(m),bg.prediction_index].sum(axis=0)) for m in ['sar','fusion']}
            blockrows.append(dict(region=region,block_id=block,n_patches=len(bg),valid_area_km2=vals['sar']['valid_area_km2'],
                reference_flood_km2=vals['sar']['true_flood_km2'],delta_iou=vals['fusion']['flood_iou']-vals['sar']['flood_iou'] if vals['sar']['flood_iou'] is not None and vals['fusion']['flood_iou'] is not None else None,
                delta_confident_omission=vals['fusion']['confident_omission_01']-vals['sar']['confident_omission_01'] if vals['sar']['confident_omission_01'] is not None else None))
    pd.DataFrame(blockrows).to_csv(SD/'panel_g_spatial_blocks.csv',index=False)
    spec={'started_at_utc':started,'completed_at_utc':datetime.now(timezone.utc).isoformat(),'release_sha256':proof['release_sha256'],
      'source_hashes':sources,'script_sha256':sha(Path(__file__)),'case_rule_sha256':sha(SD/'case_selection_rule.json'),
      'n_patches':len(frame),'regions':frame.groupby('region').size().to_dict(),'common_support':'support == 7',
      'area_weight':'float32 saved geodesic row areas promoted to float64, broadcast to columns',
      'joint_entropy':'natural logarithms; same clipping [1e-7,1-1e-7] as frozen evaluator',
      'joint_brier':'unclipped frozen float32 predictions promoted to float64; label 0/1',
      'joint_quadrant':'delta_entropy < 0 AND delta_brier > 0, exact tests before histogramming',
      'joint_histogram':'160 x 160 equal bins over full theoretical range; area sum per bin; no pixel subsampling',
      'ecdf':'exact area-weighted empirical CDF on common-valid reference-flooded pixels; unique probability sums saved; plotted subset has <=0.0001 vertical loss between retained steps except genuine probability mass jumps, threshold inserted exactly',
      'mosaic':'8x8 cell valid-pixel means placed by geographic centre into a WGS84 overview grid; no outside-coverage imagery; display only, statistics are full-resolution',
      'frozen_csv_crosschecks':'both regions, both models confident omissions and valid area match at 1e-10 tolerance',
      'foundation_status':json.loads((ROOT/'runs/core_s17_foundation_predictions/complete.json').read_text('utf-8'))}
    writej(SD/'derivation_provenance.json',spec)
    print(json.dumps({'quadrants':summaries,'cases':cases},indent=2),flush=True)

if __name__=='__main__':main()
