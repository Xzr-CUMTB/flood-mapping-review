"""Independent arithmetic, geometry and frozen-source checks for Fig. 2."""
import os
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
from pathlib import Path
import json,hashlib,gc
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from pyproj import Geod,Transformer
from derive_spatial import SegmentDistance
ROOT=Path(__file__).resolve().parents[4];OUT=Path(__file__).resolve().parents[1];SD=OUT/'source_data'
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    checks={};details={};pro=json.loads((SD/'derivation_provenance.json').read_text());release=json.loads((ROOT/'configs/core_release_s17.json').read_text())
    def ck(name,test):checks[name]=bool(test);assert test,name
    ck('release_identity',sha(ROOT/'configs/core_release_s17.json')==pro['release_sha256'])
    ck('derived_script_hash',sha(OUT/'scripts/derive_spatial.py')==pro['script_sha256'])
    ck('analysis_rules_preserved',sha(OUT/'Fig02_design_notes.md')==pro['design_rule_sha256'])
    frozen_bad=[]
    for i,(path,expected) in enumerate(release['sha256'].items()):
        if sha(ROOT/path)!=expected:frozen_bad.append(path)
        if i%30==0:print('Frozen source rehash',i,flush=True)
    ck('all_142_frozen_files_preserved',not frozen_bad);details['frozen_files_rehashed']=len(release['sha256'])
    # Fig. 1 final exports and reproducible sources must remain byte-identical.
    fig1=ROOT/'figures/main/Fig01';v1=json.loads((fig1/'Fig01_validation.json').read_text());bad1=[r['path'] for r in v1['sha256_manifest'] if sha(fig1/r['path'])!=r['sha256']]
    ck('figure_1_unchanged',not bad1);details['figure1_files_checked']=len(v1['sha256_manifest'])
    metrics=pd.read_csv(ROOT/'runs/core_s17_evaluation/map_metrics.csv');trans=pd.read_csv(SD/'panel_c_transitions.csv');boundary=pd.read_csv(SD/'panel_e_boundary_distance.csv');conditions=pd.read_csv(SD/'panel_f_conditions.csv');components=pd.read_csv(SD/'panel_b_components.csv.gz');frame=pd.read_csv(SD/'geographic_manifest.csv')
    n=[];quant=[]
    for region in ['Beijing_Baoding','Harbin']:
        t=trans[trans.region==region].set_index('state');e=boundary[boundary.region==region];h=conditions[conditions.region==region];m=metrics[metrics.region==region].set_index('model')
        for group,q in [('boundary',e),('conditions',h)]:
            for col,expected in [('flood_m2',m.loc['fusion','true_flood_km2']),('sar_fn_m2',m.loc['sar','fn_km2']),('fusion_fn_m2',m.loc['fusion','fn_km2']),('sar_conf_m2',m.loc['sar','true_flood_km2']*m.loc['sar','confident_omission_01']),('fusion_conf_m2',m.loc['fusion','true_flood_km2']*m.loc['fusion','confident_omission_01'])]:ck(region+'_'+group+'_'+col,np.isclose(q[col].sum()/1e6,expected,rtol=1e-10))
        ck(region+'_transition_flood_partition',np.isclose(t.area_m2.sum()/1e6,m.loc['fusion','true_flood_km2'],rtol=1e-10))
        ck(region+'_transition_conservation',np.isclose((t.loc['newly_missed','area_m2']-t.loc['recovered','area_m2'])/1e6,m.loc['fusion','fn_km2']-m.loc['sar','fn_km2'],rtol=1e-10))
        a=components[(components.region==region)&(components.connectivity==8)];b=components[(components.region==region)&(components.connectivity==4)]
        ck(region+'_connectivity_mass',np.isclose(a.area_m2.sum(),b.area_m2.sum(),rtol=1e-12));ck(region+'_connectivity_order',len(a)<=len(b) and a.area_m2.max()>=b.area_m2.max())
        q=cond_total=h.flood_m2.sum();ck(region+'_condition_effect_weighting',np.isclose(np.sum(h.delta_fn_pp*h.flood_m2/q),100*(m.loc['fusion','fn_km2']-m.loc['sar','fn_km2'])/m.loc['fusion','true_flood_km2'],rtol=1e-10))
        quant.append(dict(region=region,confident_area_km2=float(a.area_m2.sum()/1e6),n_components_8=len(a),n_components_4=len(b),largest_ha=float(a.area_m2.max()/1e4),area_share_ge_1ha=float(a.loc[a.area_m2>=1e4,'area_m2'].sum()/a.area_m2.sum()),censored_component_area_share=float(a.loc[a.touches_unknown,'area_m2'].sum()/a.area_m2.sum()),confident_share_within_50m=float(e.loc[e.bin<3,'fusion_conf_m2'].sum()/e.fusion_conf_m2.sum()),reference_share_distance_unresolved=float(e.loc[e.bin==10,'flood_m2'].sum()/e.flood_m2.sum()),newly_missed_km2=float(t.loc['newly_missed','area_m2']/1e6),recovered_km2=float(t.loc['recovered','area_m2']/1e6)))
    pd.DataFrame(quant).to_csv(SD/'verified_spatial_findings.csv',index=False)
    for spec in json.loads((SD/'native_grid_specs.json').read_text()):
        labs=np.load(OUT/f"native_grids/{spec['grid_id']}_component8.npy",mmap_mode='r');size=int(components[(components.grid_id==spec['grid_id'])&(components.connectivity==8)].component_id.max());counts=np.zeros(size+1,np.int32)
        for rec in frame[frame.grid_id==spec['grid_id']].itertuples():counts[np.unique(labs[rec.grid_row:rec.grid_row+256,rec.grid_col:rec.grid_col+256])]+=1
        n.append(dict(grid_id=spec['grid_id'],components_crossing_real_tile_boundaries=int((counts[1:]>1).sum()),maximum_tiles_in_component=int(counts[1:].max())))
        del labs
    pd.DataFrame(n).to_csv(SD/'component_cross_tile_audit.csv',index=False);ck('components_join_across_true_tile_edges',sum(r['components_crossing_real_tile_boundaries'] for r in n)>0)
    z=np.load(SD/'panel_a_geographic_window.npz');valid=z['label']!=255;truth=z['label']==1;w=np.broadcast_to(z['row_area_m2'][:,None].astype(float),truth.shape);marg=pd.read_csv(SD/'panel_a_marginals.csv')
    for name,mask in [('reference',truth),('sar_fn',truth&(z['sar']<.5)),('fusion_fn',truth&(z['fusion']<.5)),('fusion_conf',truth&(z['fusion']<=.1)),('valid',valid)]:
        for axis in ['longitude','latitude']:ck('map_'+axis+'_'+name,np.isclose(marg[marg.axis==axis][name+'_km2'].sum(),w[mask].sum()/1e6,rtol=1e-10))
    tr=pd.read_csv(SD/'panel_d_transect.csv');r=int(z['profile_row']);ck('transect_valid_alignment',np.array_equal(tr.valid,valid[r]));ck('transect_sample_ids',np.array_equal(tr.sample_id,z['sample_id'][r]));ck('transect_probabilities',np.allclose(tr.p_sar,z['sar'][r],equal_nan=True) and np.allclose(tr.p_fusion,z['fusion'][r],equal_nan=True))
    geod=Geod(ellps='WGS84');dd=np.r_[0,np.cumsum(geod.inv(tr.longitude.to_numpy()[:-1],tr.latitude.to_numpy()[:-1],tr.longitude.to_numpy()[1:],tr.latitude.to_numpy()[1:])[2])]/1000;ck('transect_geodesic_distance',np.allclose(dd,tr.distance_km,atol=1e-8));ck('transect_one_real_path',tr.latitude.nunique()==1 and np.all(np.diff(tr.longitude)>0))
    # Check geographic crop directly against cached inputs and frozen probabilities.
    selection=json.loads((SD/'panel_a_selection.json').read_text());r0,r1,c0,c1=selection['grid_window'];x=np.load(ROOT/'data/processed/sen2gf3/x_uint8.npy',mmap_mode='r');y=np.load(ROOT/'data/processed/sen2gf3/labels_uint8.npy',mmap_mode='r');support=np.load(ROOT/'data/processed/sen2gf3/support_uint8.npy',mmap_mode='r');p={name:np.load(ROOT/f'runs/core_s17_predictions/{name}.npy',mmap_mode='r') for name in ['sar','fusion']};crop_count=0
    for rec in frame[frame.grid_id==selection['grid_id']].itertuples():
        top=max(r0,rec.grid_row);bottom=min(r1,rec.grid_row+256);left=max(c0,rec.grid_col);right=min(c1,rec.grid_col+256)
        if top>=bottom or left>=right:continue
        src=np.s_[top-rec.grid_row:bottom-rec.grid_row,left-rec.grid_col:right-rec.grid_col];dst=np.s_[top-r0:bottom-r0,left-c0:right-c0];v=support[(rec.cache_index,)+src]==7
        ck('crop_input_'+str(rec.sample_id),np.array_equal(z['x'][(slice(None),)+dst],x[(rec.cache_index,slice(None))+src]))
        ck('crop_reference_'+str(rec.sample_id),np.array_equal(z['label'][dst],np.where(v,y[(rec.cache_index,)+src],255)))
        for name in p:ck('crop_'+name+'_'+str(rec.sample_id),np.allclose(z[name][dst],np.where(v,p[name][(rec.prediction_index,)+src],np.nan),equal_nan=True))
        crop_count+=1
    details['window_source_tiles_rechecked']=crop_count
    # Independently brute-force every actual boundary segment for fixed audit points.
    errors=[]
    for region in ['Beijing_Baoding','Harbin']:
        transformer=Transformer.from_crs(4326,32652 if region=='Harbin' else 32650,always_xy=True);records=frame[frame.region==region].iloc[np.linspace(0,len(frame[frame.region==region])-1,12).astype(int)];points=[]
        for rec in records.itertuples():
            rr,cc=np.nonzero((support[rec.cache_index]==7)&(y[rec.cache_index]==1))
            if len(rr):j=len(rr)//2;points.append(transformer.transform(rec.left+(cc[j]+.5)*rec.xres,rec.top+(rr[j]+.5)*rec.yres))
        points=np.array(points);segs=np.load(SD/f'boundary_segments_{region}.npz')
        for name in ['reference','unknown']:
            segments=segs[name];fast=SegmentDistance(segments).query(points);brute=np.full(len(points),np.inf)
            for st in range(0,len(segments),100000):
                a=segments[st:st+100000,:2];v=segments[st:st+100000,2:]-a;l2=(v*v).sum(1)
                for j,point in enumerate(points):
                    delta=point-a;t=np.clip((delta*v).sum(1)/l2,0,1);brute[j]=min(brute[j],float(np.sqrt(((delta-t[:,None]*v)**2).sum(1)).min()))
            err=float(np.max(abs(brute-fast)));errors.append(dict(region=region,boundary=name,n_points=len(points),maximum_error_m=err));ck(region+'_'+name+'_distance_bruteforce',err<1e-8);del segments;gc.collect()
    details['distance_bruteforce']=errors;details['cross_tile_components']=n
    result=dict(passed=all(checks.values()),checked_at_utc=datetime.now(timezone.utc).isoformat(),release_sha256=pro['release_sha256'],checks=checks,details=details,findings=quant)
    (SD/'scientific_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(dict(passed=result['passed'],checks=len(checks),findings=quant),indent=2),flush=True)
if __name__=='__main__':main()
