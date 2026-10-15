"""Exploratory spatial diagnostics of read-only frozen predictions, with native-grid topology."""
import os
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
from pathlib import Path
import json,hashlib,time,gc
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial import cKDTree
from pyproj import Transformer,Geod

ROOT=Path(__file__).resolve().parents[4];OUT=Path(__file__).resolve().parents[1]
SD=OUT/'source_data';AS=OUT/'assets';WORK=OUT/'native_grids'
for p in [SD,AS,WORK]:p.mkdir(parents=True,exist_ok=True)
REGIONS=['Beijing_Baoding','Harbin'];GEOD=Geod(ellps='WGS84')
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def writej(p,x):p.write_text(json.dumps(x,indent=2),encoding='utf-8')
def log(s):print(datetime.now().isoformat(timespec='seconds'),s,flush=True)
def grids_for(frame):
    specs=[];parts=[]
    for region,g in frame.groupby('region',sort=False):
        dx=float(g.xres.median());dy=float(-g.yres.median());left=g.left.min();top=g.top.max()
        px=np.mod((g.left-left)/dx,256);py=np.mod((top-g.top)/dy,256)
        px=np.where((px<.0001)|(px>255.9999),0,px);py=np.where((py<.0001)|(py>255.9999),0,py)
        h=g.copy();h['phase_x']=px.round(4);h['phase_y']=py.round(4)
        for j,(_,q) in enumerate(h.groupby(['phase_x','phase_y'],sort=True)):
            q=q.copy();gid=region+'_'+str(j);x0=float(q.left.min());y0=float(q.top.max())
            cx=(q.left-x0)/dx;ry=(y0-q.top)/dy
            assert max(abs(cx-cx.round()).max(),abs(ry-ry.round()).max())<.0001
            q['grid_id']=gid;q['grid_col']=cx.round().astype(int);q['grid_row']=ry.round().astype(int)
            spec=dict(grid_id=gid,region=region,left=x0,top=y0,dx=dx,dy=dy,height=int(q.grid_row.max()+256),width=int(q.grid_col.max()+256),n_patches=len(q))
            specs.append(spec);parts.append(q)
    return specs,pd.concat(parts).sort_values('prediction_index')
def segment_arrays(lab,spec,transformer):
    ref=[];unknown=[];H,W=lab.shape
    def add(r,c,vertical,target):
        if not len(r):return
        if vertical:
            x=spec['left']+(c+1)*spec['dx'];y=spec['top']-r*spec['dy'];x2=x;y2=y-spec['dy']
        else:
            x=spec['left']+c*spec['dx'];y=spec['top']-(r+1)*spec['dy'];x2=x+spec['dx'];y2=y
        a,b=transformer.transform(x,y);u,v=transformer.transform(x2,y2);target.append(np.column_stack([a,b,u,v]))
    for start in range(0,H,256):
        end=min(start+256,H);a=np.asarray(lab[start:end,:-1]);b=np.asarray(lab[start:end,1:])
        r,c=np.nonzero((a<2)&(b<2)&(a!=b));add(r+start,c,True,ref)
        r,c=np.nonzero((a==255)!=(b==255));add(r+start,c,True,unknown)
        end2=min(end,H-1);a=np.asarray(lab[start:end2]);b=np.asarray(lab[start+1:end2+1])
        r,c=np.nonzero((a<2)&(b<2)&(a!=b));add(r+start,c,False,ref)
        r,c=np.nonzero((a==255)!=(b==255));add(r+start,c,False,unknown)
    # Outer raster edges are unknown coverage interfaces, not reference flood boundaries.
    for col,edge in [(-1,0),(W-1,W-1)]:
        r=np.flatnonzero(lab[:,edge]!=255);add(r,np.full(len(r),col),True,unknown)
    for row,edge in [(-1,0),(H-1,H-1)]:
        c=np.flatnonzero(lab[edge]!=255);add(np.full(len(c),row),c,False,unknown)
    return np.concatenate(ref),np.concatenate(unknown)
class SegmentDistance:
    def __init__(self,segments):
        self.a=segments[:,:2];self.v=segments[:,2:]-self.a;self.l2=(self.v**2).sum(1)
        self.half=float(np.sqrt(self.l2).max()/2);self.tree=cKDTree(self.a+self.v/2)
    def query(self,points):
        k=8;answer=np.empty(len(points));pending=np.arange(len(points))
        while len(pending):
            kk=min(k,len(self.a));dist,ix=self.tree.query(points[pending],k=kk,workers=2)
            if kk==1:dist=dist[:,None];ix=ix[:,None]
            delta=points[pending,None,:]-self.a[ix];t=np.clip((delta*self.v[ix]).sum(2)/self.l2[ix],0,1)
            best=np.sqrt(((delta-t[:,:,None]*self.v[ix])**2).sum(2)).min(1)
            answer[pending]=best
            # This bound guarantees that an excluded segment cannot be nearer.
            retry=(dist[:,-1]<=best+self.half+1e-8) if kk<len(self.a) else np.zeros(len(pending),bool)
            pending=pending[retry];k*=2
        return answer
def main():
    start=datetime.now(timezone.utc).isoformat();proof=json.loads((ROOT/'reports/completed_batch_verification.json').read_text());release=json.loads((ROOT/'configs/core_release_s17.json').read_text())
    assert proof['passed']; hashes={}
    targets=['configs/core_release_s17.json','runs/core_s17_predictions/prediction_manifest.csv','runs/core_s17_evaluation/map_metrics.csv']
    targets += [f'data/processed/sen2gf3/{n}.npy' for n in ['x_uint8','labels_uint8','support_uint8','row_area_m2']]
    targets += [f'runs/core_s17_predictions/{n}.npy' for n in ['sar','fusion','value_gate']]
    for p in targets:
        hh=sha(ROOT/p);hashes[p]=hh
        expected=release['sha256'].get(p)
        if p.endswith('core_release_s17.json'):expected=proof['release_sha256']
        if p.endswith('map_metrics.csv'):expected=proof['csv_sha256']['map_metrics.csv']
        if p.startswith('runs/core_s17_predictions') and p.endswith('.npy'):expected=next(r['sha256'] for r in proof['prediction_files'] if r['path'].replace('\\','/')==p)
        if expected:assert hh==expected,p
        log('Hash verified '+p)
    frame=pd.read_csv(ROOT/'runs/core_s17_predictions/prediction_manifest.csv');frame['region']=np.where(frame.provisional_site.eq('Harbin'),'Harbin','Beijing_Baoding')
    assert len(frame)==8233 and frame.prediction_index.tolist()==list(range(8233))
    specs,frame=grids_for(frame);frame.to_csv(SD/'geographic_manifest.csv',index=False);writej(SD/'native_grid_specs.json',specs)
    # Independent geometric nonoverlap/adjacency audit over nearby rectangle centres.
    overlap=[];unaligned_touch=[];mindist={}
    for region,g in frame.groupby('region'):
        coords=g[['lon','lat']].to_numpy();neighbors=cKDTree(coords).query_pairs(.05);rr=list(g.itertuples());min_gap=np.inf
        for i,j in neighbors:
            a,b=rr[i],rr[j];ox=min(a.right,b.right)-max(a.left,b.left);oy=min(a.top,b.top)-max(a.bottom,b.bottom)
            if ox>1e-9 and oy>1e-9:overlap.append([a.sample_id,b.sample_id,ox,oy])
            if a.grid_id!=b.grid_id:
                gap=np.hypot(max(-ox,0),max(-oy,0));min_gap=min(min_gap,gap)
                if gap<1e-9:unaligned_touch.append([a.sample_id,b.sample_id])
        mindist[region]=None if not np.isfinite(min_gap) else float(min_gap)
    assert not overlap and not unaligned_touch,(overlap[:3],unaligned_touch[:3])
    writej(SD/'geographic_topology_audit.json',dict(overlap_pairs=overlap,unaligned_touch_pairs=unaligned_touch,minimum_between_grid_gap_degrees=mindist,grid_specs=specs))
    cache=ROOT/'data/processed/sen2gf3';x=np.load(cache/'x_uint8.npy',mmap_mode='r');y=np.load(cache/'labels_uint8.npy',mmap_mode='r');support=np.load(cache/'support_uint8.npy',mmap_mode='r');areas=np.load(cache/'row_area_m2.npy',mmap_mode='r')
    preds={m:np.load(ROOT/f'runs/core_s17_predictions/{m}.npy',mmap_mode='r') for m in ['sar','fusion']}
    transitions=[];conditions=[];allcomp=[];grid_stats=[];segments={r:[[],[]] for r in REGIONS}
    for spec in specs:
        gid=spec['grid_id'];region=spec['region'];H,W=spec['height'],spec['width'];group=frame[frame.grid_id==gid]
        lab=np.lib.format.open_memmap(WORK/(gid+'_label.npy'),mode='w+',dtype='uint8',shape=(H,W));lab[:]=255
        code=np.lib.format.open_memmap(WORK/(gid+'_code.npy'),mode='w+',dtype='uint8',shape=(H,W));code[:]=0
        row_area=np.zeros(H,np.float32)
        log('Assembling native grid '+gid+f' {H}x{W}')
        for rec in group.itertuples():
            i=rec.prediction_index;k=rec.cache_index;r=rec.grid_row;c=rec.grid_col;valid=support[k]==7;truth=y[k]==1;sw=preds['sar'][i];fw=preds['fusion'][i];weight=np.broadcast_to(areas[k].astype(float)[:,None],(256,256));flood=valid&truth
            assert np.all(lab[r:r+256,c:c+256]==255)
            lab[r:r+256,c:c+256]=np.where(valid,y[k],255)
            co=np.where(valid,1,0).astype('uint8');state=(sw<.5).astype('uint8')+2*(fw<.5).astype('uint8')
            co[flood]=2+state[flood];co[flood&(fw<=.1)]|=8;co[flood&(sw<=.1)]|=16
            code[r:r+256,c:c+256]=co
            old=row_area[r:r+256];assert np.all((old==0)|np.isclose(old,areas[k],rtol=1e-6));row_area[r:r+256]=areas[k]
            for j,name in enumerate(['stable_detection','recovered','newly_missed','persistent_omission']):
                mask=flood&(state==j);transitions.append(dict(region=region,sample_id=rec.sample_id,state=name,area_m2=weight[mask].sum(),confident_fusion_m2=weight[mask&(fw<=.1)].sum()))
            for hh in range(4):
                mask=flood&(x[k,0]//64==hh);ww=weight[mask];ss=sw[mask];ff=fw[mask]
                conditions.append(dict(region=region,sample_id=rec.sample_id,hh_bin=hh,n_pixels=int(mask.sum()),flood_m2=ww.sum(),sar_fn_m2=ww[ss<.5].sum(),fusion_fn_m2=ww[ff<.5].sum(),sar_conf_m2=ww[ss<=.1].sum(),fusion_conf_m2=ww[ff<=.1].sum()))
        lab.flush();code.flush();np.save(WORK/(gid+'_row_area.npy'),row_area)
        transformer=Transformer.from_crs(4326,32652 if region=='Harbin' else 32650,always_xy=True)
        seg=segment_arrays(lab,spec,transformer)
        for j in range(2):segments[region][j].append(seg[j])
        log(gid+f' reference/coverage edges: {len(seg[0])}/{len(seg[1])}')
        mask=(code&8)>0
        for conn in [8,4]:
            labels,n=ndimage.label(mask,structure=np.ones((3,3),int) if conn==8 else ndimage.generate_binary_structure(2,1))
            sums=np.zeros((n+1,5),float)
            for st in range(0,H,256):
                en=min(H,st+256);rr,cc=np.nonzero(mask[st:en]);rr+=st;ix=labels[rr,cc];ww=row_area[rr].astype(float)
                for col,value in enumerate([ww,np.ones(len(rr)),rr*ww,cc*ww]):sums[:,col]+=np.bincount(ix,weights=value,minlength=n+1)
                edge=(rr==0)|(rr==H-1)|(cc==0)|(cc==W-1)
                for dr in [-1,0,1]:
                    for dc in [-1,0,1]:edge|=lab[np.clip(rr+dr,0,H-1),np.clip(cc+dc,0,W-1)]==255
                sums[:,4]+=np.bincount(ix,weights=edge.astype(float),minlength=n+1)
            ids=np.arange(1,n+1);s=sums[1:];df=pd.DataFrame(dict(region=region,grid_id=gid,connectivity=conn,component_id=ids,area_m2=s[:,0],n_pixels=s[:,1].astype(int),centroid_row=s[:,2]/s[:,0],centroid_col=s[:,3]/s[:,0],touches_unknown=s[:,4]>0))
            allcomp.append(df);grid_stats.append(dict(grid_id=gid,connectivity=conn,n_components=n,area_m2=float(s[:,0].sum()),largest_m2=float(s[:,0].max()),edge_censored_area_m2=float(s[s[:,4]>0,0].sum())))
            if conn==8:
                np.save(WORK/(gid+'_component8.npy'),labels)
            log(gid+f' connectivity={conn}, components={n}, omitted area={s[:,0].sum()/1e6:.4f} km2')
            del labels,sums;gc.collect()
        del mask,code,lab;gc.collect()
    trans=pd.DataFrame(transitions);trans.to_csv(SD/'panel_c_patch_transitions.csv.gz',index=False)
    tc=trans.groupby(['region','state'],sort=False)[['area_m2','confident_fusion_m2']].sum().reset_index();tc['fraction_flood']=tc.area_m2/tc.groupby('region').area_m2.transform('sum');tc.to_csv(SD/'panel_c_transitions.csv',index=False)
    cond=pd.DataFrame(conditions);cond.to_csv(SD/'panel_f_patch_conditions.csv.gz',index=False)
    cf=cond.groupby(['region','hh_bin'],sort=False)[['n_pixels','flood_m2','sar_fn_m2','fusion_fn_m2','sar_conf_m2','fusion_conf_m2']].sum().reset_index()
    cf['delta_fn_pp']=100*(cf.fusion_fn_m2-cf.sar_fn_m2)/cf.flood_m2;cf['delta_conf_pp']=100*(cf.fusion_conf_m2-cf.sar_conf_m2)/cf.flood_m2;cf.to_csv(SD/'panel_f_conditions.csv',index=False)
    comp=pd.concat(allcomp,ignore_index=True);comp.to_csv(SD/'panel_b_components.csv.gz',index=False);pd.DataFrame(grid_stats).to_csv(SD/'panel_b_grid_summary.csv',index=False)
    maps=pd.read_csv(ROOT/'runs/core_s17_evaluation/map_metrics.csv');checks={}
    for region in REGIONS:
        tt=tc[tc.region==region].set_index('state');sar=maps[(maps.region==region)&(maps.model=='sar')].iloc[0];fus=maps[(maps.region==region)&(maps.model=='fusion')].iloc[0]
        checks[region]=dict(sar_fn_km2=float((tt.loc['recovered'].area_m2+tt.loc['persistent_omission'].area_m2)/1e6),fusion_fn_km2=float((tt.loc['newly_missed'].area_m2+tt.loc['persistent_omission'].area_m2)/1e6),delta_fn_km2=float((tt.loc['newly_missed'].area_m2-tt.loc['recovered'].area_m2)/1e6),reference_flood_km2=float(tt.area_m2.sum()/1e6))
        for key,expect in [('sar_fn_km2',sar.fn_km2),('fusion_fn_km2',fus.fn_km2),('reference_flood_km2',fus.true_flood_km2)]:assert np.isclose(checks[region][key],expect,rtol=1e-10)
        for conn in [4,8]:assert np.isclose(comp[(comp.region==region)&(comp.connectivity==conn)].area_m2.sum()/1e6,fus.true_flood_km2*fus.confident_omission_01,rtol=1e-10)
    writej(SD/'area_conservation_checks.json',checks)
    # Full-region reference-boundary distances; missing coverage is handled explicitly.
    bins=np.array([0,10,20,50,100,200,500,1000,2000,5000,np.inf]);distance_rows=[]
    for region in REGIONS:
        ref=np.concatenate(segments[region][0]);unknown=np.concatenate(segments[region][1]);segments[region]=None
        np.savez_compressed(SD/f'boundary_segments_{region}.npz',reference=ref,unknown=unknown,epsg=32652 if region=='Harbin' else 32650)
        rd=SegmentDistance(ref);ud=SegmentDistance(unknown);del ref,unknown;gc.collect();transformer=Transformer.from_crs(4326,32652 if region=='Harbin' else 32650,always_xy=True)
        sums=np.zeros((11,6));counts=np.zeros(11,np.int64);max_dist=0.
        for n,rec in enumerate(frame[frame.region==region].itertuples()):
            i=rec.prediction_index;k=rec.cache_index;rr,cc=np.nonzero((support[k]==7)&(y[k]==1))
            if not len(rr):continue
            lon=rec.left+(cc+.5)*rec.xres;lat=rec.top+(rr+.5)*rec.yres;xx,yy=transformer.transform(lon,lat);pts=np.column_stack([xx,yy]);dr=rd.query(pts);du=ud.query(pts);resolved=dr<du-1e-7
            max_dist=max(max_dist,float(dr[resolved].max(initial=0)));idx=np.searchsorted(bins,dr,side='right')-1;idx[~resolved]=10;w=areas[k,rr].astype(float);ss=preds['sar'][i,rr,cc];ff=preds['fusion'][i,rr,cc]
            counts+=np.bincount(idx,minlength=11)
            for col,mask in enumerate([np.ones(len(rr),bool),ss<.5,ff<.5,ss<=.1,ff<=.1,(ff>.1)&(ff<.5)]):sums[:,col]+=np.bincount(idx,weights=w*mask,minlength=11)
            if n%500==0:log(region+f' boundary distances {n} patches')
        for j in range(11):
            row=dict(region=region,bin=j,left_m=float(bins[j]) if j<10 else None,right_m=float(bins[j+1]) if j<10 else None,n_pixels=int(counts[j]),max_resolved_distance_m=max_dist)
            row.update(dict(zip(['flood_m2','sar_fn_m2','fusion_fn_m2','sar_conf_m2','fusion_conf_m2','fusion_other_fn_m2'],sums[j])));distance_rows.append(row)
        assert np.isclose(sums[:,0].sum()/1e6,checks[region]['reference_flood_km2'],rtol=1e-10)
        del rd,ud;gc.collect()
    pd.DataFrame(distance_rows).to_csv(SD/'panel_e_boundary_distance.csv',index=False)
    # Geographic example and continuous transect selected by the saved rule.
    bb=comp[(comp.region=='Beijing_Baoding')&(comp.connectivity==8)].sort_values(['area_m2','grid_id','component_id']);cdf=bb.area_m2.cumsum()/bb.area_m2.sum();target=float(bb.loc[cdf>=.75].iloc[0].area_m2)
    bb=bb.copy();bb['target_distance']=abs(bb.area_m2-target);selected=bb.sort_values(['target_distance','grid_id','component_id']).iloc[0];spec=next(s for s in specs if s['grid_id']==selected.grid_id)
    H,W=spec['height'],spec['width'];r0=int(np.clip(int(selected.centroid_row)-640,0,H-1280));c0=int(np.clip(int(selected.centroid_col)-640,0,W-1280));r1=r0+1280;c1=c0+1280
    labels=np.load(WORK/(selected.grid_id+'_component8.npy'),mmap_mode='r');lab=np.load(WORK/(selected.grid_id+'_label.npy'),mmap_mode='r');code=np.load(WORK/(selected.grid_id+'_code.npy'),mmap_mode='r');wa=np.load(WORK/(selected.grid_id+'_row_area.npy'),mmap_mode='r')
    cube=np.zeros((6,1280,1280),np.uint8);ss=np.full((1280,1280),np.nan,np.float32);ff=ss.copy();sample=np.full((1280,1280),-1,np.int32)
    for rec in frame[frame.grid_id==selected.grid_id].itertuples():
        t=max(r0,rec.grid_row);b=min(r1,rec.grid_row+256);l=max(c0,rec.grid_col);r=min(c1,rec.grid_col+256)
        if t>=b or l>=r:continue
        src=np.s_[t-rec.grid_row:b-rec.grid_row,l-rec.grid_col:r-rec.grid_col];dst=np.s_[t-r0:b-r0,l-c0:r-c0];cube[(slice(None),)+dst]=x[(rec.cache_index,slice(None))+src];ss[dst]=preds['sar'][(rec.prediction_index,)+src];ff[dst]=preds['fusion'][(rec.prediction_index,)+src];sample[dst]=rec.sample_id
    crop_lab=np.array(lab[r0:r1,c0:c1]);crop_code=np.array(code[r0:r1,c0:c1]);valid=crop_lab!=255;ss[~valid]=np.nan;ff[~valid]=np.nan
    extent=[spec['left']+c0*spec['dx'],spec['left']+c1*spec['dx'],spec['top']-r1*spec['dy'],spec['top']-r0*spec['dy']]
    selectedmask=np.array(labels[r0:r1,c0:c1])==selected.component_id;profile_row=int(np.argmax(selectedmask.sum(1)))
    np.savez_compressed(SD/'panel_a_geographic_window.npz',x=cube,label=crop_lab,code=crop_code,sar=ss,fusion=ff,sample_id=sample,row_area_m2=wa[r0:r1],extent=extent,selected_component=selectedmask,profile_row=profile_row)
    lon=extent[0]+(np.arange(1280)+.5)*spec['dx'];lat=np.full(1280,extent[3]-(profile_row+.5)*spec['dy']);distance=np.r_[0,np.cumsum(GEOD.inv(lon[:-1],lat[:-1],lon[1:],lat[1:])[2])]/1000
    tr=pd.DataFrame(dict(distance_km=distance,longitude=lon,latitude=lat,sample_id=sample[profile_row],valid=valid[profile_row],reference=crop_lab[profile_row],p_sar=ss[profile_row],p_fusion=ff[profile_row],delta_probability=ff[profile_row]-ss[profile_row]))
    for j,name in enumerate(['HH','HV','R','G','B','NIR']):tr[name]=np.where(valid[profile_row],cube[j,profile_row].astype(float),np.nan)
    tr['sar_fn']=valid[profile_row]&(crop_lab[profile_row]==1)&(ss[profile_row]<.5);tr['fusion_fn']=valid[profile_row]&(crop_lab[profile_row]==1)&(ff[profile_row]<.5);tr['fusion_confident']=valid[profile_row]&(crop_lab[profile_row]==1)&(ff[profile_row]<=.1);tr.to_csv(SD/'panel_d_transect.csv',index=False)
    # Exact same-area sums underpin the geographic marginals; fixed 32-pixel bins.
    marg=[];weight=np.broadcast_to(wa[r0:r1,None].astype(float),(1280,1280));flood=crop_lab==1
    for axis in ['longitude','latitude']:
        for j in range(0,1280,32):
            cut=np.s_[:,j:j+32] if axis=='longitude' else np.s_[j:j+32,:]
            lo=extent[0]+j*spec['dx'] if axis=='longitude' else extent[3]-(j+32)*spec['dy'];hi=extent[0]+(j+32)*spec['dx'] if axis=='longitude' else extent[3]-j*spec['dy']
            row=dict(axis=axis,bin=j//32,lower=lo,upper=hi)
            for name,mask in [('reference',flood),('sar_fn',flood&(ss<.5)),('fusion_fn',flood&(ff<.5)),('fusion_conf',flood&(ff<=.1)),('valid',valid)]:row[name+'_km2']=float((weight[cut]*mask[cut]).sum()/1e6)
            marg.append(row)
    pd.DataFrame(marg).to_csv(SD/'panel_a_marginals.csv',index=False)
    selection=selected.to_dict();selection.update(target_area_m2=target,extent=extent,grid_window=[r0,r1,c0,c1],transect_local_row=profile_row,transect_lat=float(lat[0]),window_reference_km2=float(weight[flood].sum()/1e6),window_confident_km2=float(weight[flood&(ff<=.1)].sum()/1e6))
    writej(SD/'panel_a_selection.json',selection)
    writej(SD/'derivation_provenance.json',dict(started_at_utc=start,completed_at_utc=datetime.now(timezone.utc).isoformat(),release_sha256=proof['release_sha256'],source_hashes=hashes,script_sha256=sha(Path(__file__)),design_rule_sha256=sha(OUT/'Fig02_design_notes.md'),native_grids=specs,area_checks=checks,analysis='Exploratory descriptive spatial diagnostics; fixed raw probability thresholds; no refitting',storage_budget='User removed former cap on 2026-10-14',distance_method='Exact nearest projected observed cell-interface segment; midpoint-tree candidate guarantee; resolved iff observed boundary nearer than unknown support edge'))
    log('Completed all spatial derivations.');print(json.dumps(checks,indent=2),flush=True)
if __name__=='__main__':main()
