"""New state-specific native-grid topology and boundary distances; original evidence read-only."""
import os
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
from pathlib import Path
import json,sys,gc,hashlib
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial import cKDTree
from pyproj import Transformer
from derive_batch import ROOT,BASE,RELEASE,sha,js,wj,log
OUT=BASE/'S07';SD=OUT/'source_data';OLD=ROOT/'figures/main/Fig02';STATES=['stable_detection','recovered','newly_missed','persistent_omission']
class SegmentDistance:
 def __init__(self,segments):
  self.a=segments[:,:2];self.v=segments[:,2:]-self.a;self.l2=(self.v**2).sum(1);self.half=float(np.sqrt(self.l2).max()/2);self.tree=cKDTree(self.a+self.v/2)
 def query(self,points):
  k=8;answer=np.empty(len(points));pending=np.arange(len(points))
  while len(pending):
   kk=min(k,len(self.a));dist,ix=self.tree.query(points[pending],k=kk,workers=2)
   if kk==1:dist=dist[:,None];ix=ix[:,None]
   delta=points[pending,None,:]-self.a[ix];t=np.clip((delta*self.v[ix]).sum(2)/self.l2[ix],0,1);best=np.sqrt(((delta-t[:,:,None]*self.v[ix])**2).sum(2)).min(1);answer[pending]=best;retry=(dist[:,-1]<=best+self.half+1e-8) if kk<len(self.a) else np.zeros(len(pending),bool);pending=pending[retry];k*=2
  return answer
def main():
 specs=js(OLD/'source_data/native_grid_specs.json');frame=pd.read_csv(OLD/'source_data/geographic_manifest.csv');components=[];summaries=[];atlas={};marg=[];bindings=[]
 for spec in specs:
  gid=spec['grid_id'];region=spec['region'];H,W=spec['height'],spec['width'];paths=[OLD/f'native_grids/{gid}_{v}.npy' for v in ['code','label','row_area']]
  for p in paths:bindings.append(dict(path=str(p.relative_to(ROOT)).replace('\\','/'),sha256=sha(p)))
  code=np.load(paths[0],mmap_mode='r');lab=np.load(paths[1],mmap_mode='r');weight=np.load(paths[2]);bins=np.zeros((H//32,W//32,5),float)
  for start in range(0,H,256):
   end=min(H,start+256);co=np.asarray(code[start:end])&7;w=weight[start:end,None].astype(float)
   for k in range(5):
    val=(co==k+1) if k<5 else None
    bins[start//32:end//32,:,k]=((co==k+1)*w).reshape((end-start)//32,32,W//32,32).sum((1,3))
  atlas[gid]=bins;atlas[gid+'_extent']=np.array([spec['left'],spec['left']+W*spec['dx'],spec['top']-H*spec['dy'],spec['top']]);
  for axis in ['longitude','latitude']:
   sums=bins.sum(axis=0 if axis=='longitude' else 1)
   for i,row in enumerate(sums):
    low=spec['left']+i*32*spec['dx'] if axis=='longitude' else spec['top']-(i+1)*32*spec['dy'];high=low+32*(spec['dx'] if axis=='longitude' else spec['dy']);marg.append(dict(region=region,grid_id=gid,axis=axis,lower=low,upper=high,nonflood_km2=row[0]/1e6,**{s+'_km2':row[j+1]/1e6 for j,s in enumerate(STATES)}))
  for stateidx in [1,2,3]:
   mask=(code&7)==stateidx+2
   for conn in [8,4]:
    labels,n=ndimage.label(mask,structure=np.ones((3,3),int) if conn==8 else ndimage.generate_binary_structure(2,1));sums=np.zeros(n+1);counts=np.zeros(n+1,dtype=np.int64);touch=np.zeros(n+1,dtype=bool)
    for start in range(0,H,256):
     end=min(H,start+256);rr,cc=np.nonzero(mask[start:end]);rr+=start;ix=labels[rr,cc];sums+=np.bincount(ix,weights=weight[rr].astype(float),minlength=n+1);counts+=np.bincount(ix,minlength=n+1);edge=(rr==0)|(rr==H-1)|(cc==0)|(cc==W-1)
     for dr,dc in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]:edge|=lab[np.clip(rr+dr,0,H-1),np.clip(cc+dc,0,W-1)]==255
     touch[np.unique(ix[edge])]=True
    d=pd.DataFrame(dict(region=region,grid_id=gid,state=STATES[stateidx],connectivity=conn,component_id=np.arange(1,n+1),area_m2=sums[1:],pixels=counts[1:],touches_unknown=touch[1:]));components.append(d);summaries.append(dict(region=region,grid_id=gid,state=STATES[stateidx],connectivity=conn,components=n,total_area_km2=sums[1:].sum()/1e6,largest_area_km2=sums[1:].max()/1e6,unknown_touching_components=int(touch[1:].sum()),unknown_touching_area_km2=sums[1:][touch[1:]].sum()/1e6));log(f'{gid} {STATES[stateidx]} {conn}-neighbour: {n} components');del labels;gc.collect()
   del mask;gc.collect()
  del code,lab;gc.collect()
 np.savez_compressed(SD/'geographic_atlas.npz',**atlas);wj(SD/'geographic_atlas_manifest.json',specs);pd.DataFrame(marg).to_csv(SD/'atlas_marginals.csv',index=False);pd.concat(components).to_csv(SD/'components.csv.gz',index=False);pd.DataFrame(summaries).to_csv(SD/'component_summary.csv',index=False)
 tc=pd.read_csv(OLD/'source_data/panel_c_transitions.csv');tc['release_sha256']=RELEASE;tc.to_csv(SD/'transition_areas.csv',index=False)
 for r in ['Beijing_Baoding','Harbin']:
  for s in STATES[1:]:
   for conn in [4,8]:assert np.isclose(sum(v['total_area_km2'] for v in summaries if v['region']==r and v['state']==s and v['connectivity']==conn),tc[(tc.region==r)&(tc.state==s)].area_m2.iloc[0]/1e6,atol=1e-9)
 # Exact projected reference-boundary distance with explicit nearest-unknown censoring.
 edges=np.array([0,5,10,20,50,100,200,500,1000,2000,5000,np.inf]);rows=[]
 for region in ['Beijing_Baoding','Harbin']:
  p=OLD/f'source_data/boundary_segments_{region}.npz';bindings.append(dict(path=str(p.relative_to(ROOT)).replace('\\','/'),sha256=sha(p)));segments=np.load(p);rd=SegmentDistance(segments['reference']);ud=SegmentDistance(segments['unknown']);trans=Transformer.from_crs(4326,32652 if region=='Harbin' else 32650,always_xy=True);sums=np.zeros((4,12));nums=np.zeros((4,12),np.int64)
  for spec in [q for q in specs if q['region']==region]:
   gid=spec['grid_id'];code=np.load(OLD/f'native_grids/{gid}_code.npy',mmap_mode='r');weight=np.load(OLD/f'native_grids/{gid}_row_area.npy');group=frame[frame.grid_id==gid]
   for i,rec in enumerate(group.itertuples()):
    tile=(np.asarray(code[rec.grid_row:rec.grid_row+256,rec.grid_col:rec.grid_col+256])&7);rr,cc=np.nonzero(tile>=2)
    if not len(rr):continue
    cls=tile[rr,cc]-2;gr=rr+rec.grid_row;gcx=cc+rec.grid_col;lon=spec['left']+(gcx+.5)*spec['dx'];lat=spec['top']-(gr+.5)*spec['dy'];x,y=trans.transform(lon,lat);points=np.column_stack([x,y]);dr=rd.query(points);du=ud.query(points);resolved=dr<du-1e-7;ix=np.searchsorted(edges,dr,side='right')-1;ix[~resolved]=11;w=weight[gr].astype(float)
    for j in range(4):sums[j]+=np.bincount(ix[cls==j],weights=w[cls==j],minlength=12);nums[j]+=np.bincount(ix[cls==j],minlength=12)
    if i%600==0:log(f'{gid}: exact boundary distance {i}/{len(group)} patches')
  for j,state in enumerate(STATES):
   expected=tc[(tc.region==region)&(tc.state==state)].area_m2.iloc[0];assert np.isclose(sums[j].sum(),expected,rtol=1e-10)
   for k in range(12):rows.append(dict(region=region,state=state,bin=k,left_m=float(edges[k]) if k<11 else np.nan,right_m=float(edges[k+1]) if k<11 else np.nan,coverage_censored=k==11,area_m2=sums[j,k],pixels=int(nums[j,k]),total_state_area_m2=expected,release_sha256=RELEASE))
  del rd,ud,segments;gc.collect()
 pd.DataFrame(rows).to_csv(SD/'boundary_distance.csv',index=False)
 wj(SD/'spatial_methods.json',dict(release_sha256=RELEASE,classification='p >= 0.5',states=dict(stable_detection='SAR positive, Fusion positive',recovered='SAR negative, Fusion positive',newly_missed='SAR positive, Fusion negative',persistent_omission='SAR negative, Fusion negative'),area='sum of WGS84 geodetic row pixel areas, common support only',connectivity='native grids, primary 8-neighbour and sensitivity 4-neighbour; actual missing coverage and disjoint grid phases preserved',component_unknown='component includes a pixel adjacent to unknown support within eight neighbours, including raster perimeter; such components retained and flagged, not discarded',boundary='exact nearest projected observed flood/nonflood cell-interface segment; midpoint-tree bound guarantees nearest segment; resolved iff reference distance < nearest unknown edge minus 1e-7 m',crs={'Beijing_Baoding':'EPSG:32650','Harbin':'EPSG:32652'},map_display='32x32 pixel area sums. Colour is flood-area-weighted mixing of the four state colours, with opacity sqrt(flood / valid area); no-data is white, observed nonflood is light grey. This display is not used for area or topology estimates.',new_analysis=True))
 for name in ['native_grid_specs.json','geographic_manifest.csv','panel_c_transitions.csv']:
  p=OLD/'source_data'/name;bindings.append(dict(path=str(p.relative_to(ROOT)).replace('\\','/'),sha256=sha(p)))
 wj(SD/'input_bindings.json',bindings);wj(SD/'scientific_audit.json',dict(passed=True,release_sha256=RELEASE,completed_at_utc=datetime.now(timezone.utc).isoformat(),source_files={p.name:sha(p) for p in SD.iterdir() if p.is_file() and p.name!='scientific_audit.json'},derivation_script_sha256=sha(__file__),original_files_modified=False));log('S7 topology and censored distances complete')
if __name__=='__main__':main()
