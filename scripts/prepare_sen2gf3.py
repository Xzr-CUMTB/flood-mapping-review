# -*- coding: utf-8 -*-
"""Audit complete paired GeoTIFFs and create compact, aligned memory-mapped arrays.

Original files are immutable. No model prediction or test-set model selection occurs here.
"""
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import argparse,csv,hashlib,json,os,sys,time
import numpy as np
import rasterio
from rasterio.warp import reproject,Resampling
from pyproj import Geod

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'data/extracted/sen2gf3/Sen2GF3Floods'
OUT=ROOT/'data/processed/sen2gf3'
GEOD=Geod(ellps='WGS84')

def geographical_group(lon,lat):
    # Provisional spatial association to the paper, never an inference from label content.
    # Unmatched regions are quarantined from dated experiments.
    if 125<lon<129 and 44<lat<48:return 'Harbin','2023_holdout',2023
    if 114<lon<118.5 and 38<lat<41.5:return 'Beijing_region' if lat>=39.4 else 'Baoding_region','2023_holdout',2023
    if 112<lon<116 and 33.5<lat<37:return 'Hebi_region' if lat>=35.4 else 'Zhengzhou_region','2021_Henan',2021
    if 116.5<lon<119 and 30.5<lat<33.5:return 'Hefei_region','2020_flood_season',2020
    if 115<lon<117.5 and 27.5<lat<30.5:return 'Nanchang_region','2020_flood_season',2020
    if 110<lon<112.5 and 22.5<lat<24.5:return 'Wuzhou_region','2024_April',2024
    if 109<lon<112 and 24.5<lat<26.5:return 'Guilin_region','2024_June',2024
    return 'unmatched_geography','unassigned',-1

@lru_cache(maxsize=4096)
def row_areas(top,dy,dx):
    values=[]
    for row in range(256):
        north=top+row*dy;south=north+dy
        area,_=GEOD.polygon_area_perimeter([0,dx,dx,0],[north,north,south,south])
        values.append(abs(area))
    return np.asarray(values,dtype=np.float32)

def aligned_read(path,target_transform,target_crs):
    with rasterio.open(path) as ds:
        if ds.crs!=target_crs or ds.width!=256 or ds.height!=256:raise ValueError('Unexpected CRS/shape '+str(path))
        raw=ds.read()  # Explicit raw bands: band 4 is NIR per publication, despite alpha tag.
        if raw.dtype!=np.uint8:raise ValueError('Unexpected input type')
        support=np.any(raw!=0,axis=0).astype('uint8')
        if ds.nodata is not None:support &= np.any(raw!=ds.nodata,axis=0)
        points=[(0,0),(256,0),(0,256),(256,256)]
        offsets=[np.hypot(*(np.array((~target_transform)*(ds.transform*p))-p)) for p in points]
        metadata={'offset_px':float(max(offsets)),'nodata':ds.nodata,'bands':ds.count,
                  'alpha_tag':any(c.name=='alpha' for c in ds.colorinterp)}
        same=np.allclose(tuple(ds.transform),tuple(target_transform),rtol=0,atol=1e-12)
        if same:return raw,support,metadata
        result=np.zeros(raw.shape,dtype=np.float32);coverage=np.zeros((256,256),dtype=np.uint8)
        for c in range(raw.shape[0]):
            reproject(raw[c].astype(np.float32),result[c],src_transform=ds.transform,src_crs=ds.crs,
                      dst_transform=target_transform,dst_crs=target_crs,resampling=Resampling.bilinear,
                      dst_nodata=0,num_threads=1)
        reproject(support,coverage,src_transform=ds.transform,src_crs=ds.crs,dst_transform=target_transform,
                  dst_crs=target_crs,resampling=Resampling.nearest,dst_nodata=0,num_threads=1)
        return np.clip(np.rint(result),0,255).astype('uint8'),coverage,metadata

def read_sample(sample_id):
    labelpath=SOURCE/'label'/f'label_{sample_id}.tif'
    paths=[SOURCE/'gaofen3'/f'after_gf3_hh_{sample_id}.TIF',SOURCE/'gaofen3'/f'after_gf3_hv_{sample_id}.TIF',SOURCE/'sentinel2'/f'before_s2_{sample_id}.TIF']
    with rasterio.open(labelpath) as ds:
        if (ds.count,ds.height,ds.width)!=(1,256,256) or str(ds.crs)!='EPSG:4326':raise ValueError('Unexpected label grid')
        y=ds.read(1)
        if y.dtype!=np.uint8 or not np.isin(y,[0,1,255]).all():raise ValueError('Unexpected labels')
        valid=np.isin(y,[0,1])
        if ds.nodata is not None and ds.nodata not in [0,1]:valid &= y!=ds.nodata
        transform,crs=ds.transform,ds.crs
        if transform.b or transform.d or transform.a<=0 or transform.e>=0:raise ValueError('Rotated/invalid grid')
        bounds=list(ds.bounds);label_nodata=ds.nodata
    parts=[aligned_read(p,transform,crs) for p in paths]
    if [p[0].shape[0] for p in parts]!=[1,1,4]:raise ValueError('Expected HH,HV,RGB-NIR')
    x=np.concatenate([p[0] for p in parts])
    sar=(parts[0][1]>0)&(parts[1][1]>0);opt=parts[2][1]>0
    support=valid.astype('uint8')+2*sar.astype('uint8')+4*opt.astype('uint8')
    common=support==7
    lon=(bounds[0]+bounds[2])/2;lat=(bounds[1]+bounds[3])/2
    site,group,year=geographical_group(lon,lat)
    areas=row_areas(round(transform.f,11),round(transform.e,14),round(transform.a,14)).copy()
    weights=np.broadcast_to(areas[:,None],(256,256))
    rec={'sample_id':sample_id,'cache_index':sample_id-1,'provisional_site':site,'event_group':group,'year':year,
         'left':bounds[0],'bottom':bounds[1],'right':bounds[2],'top':bounds[3],'lon':lon,'lat':lat,
         'xres':transform.a,'yres':transform.e,'label_nodata':label_nodata,
         'label_valid_pixels':int(valid.sum()),'common_valid_pixels':int(common.sum()),
         'sar_support_pixels':int(sar.sum()),'optical_support_pixels':int(opt.sum()),
         'label_positive_pixels':int(((y==1)&valid).sum()),'common_positive_pixels':int(((y==1)&common).sum()),
         'excluded_positive_pixels':int(((y==1)&valid&~common).sum()),
         'common_area_km2':float(weights[common].sum()/1e6),'common_flood_area_km2':float(weights[common&(y==1)].sum()/1e6),
         'hh_grid_offset_px':parts[0][2]['offset_px'],'hv_grid_offset_px':parts[1][2]['offset_px'],
         'optical_grid_offset_px':parts[2][2]['offset_px'],'optical_alpha_tag':parts[2][2]['alpha_tag'],
         'input_sha256':hashlib.sha256(x.tobytes()).hexdigest(),'label_sha256':hashlib.sha256(y.tobytes()).hexdigest(),
         'bounds_key':','.join(f'{v:.7f}' for v in bounds)}
    return rec,x,y,support,areas

def prepare(workers):
    OUT.mkdir(parents=True,exist_ok=True)
    complete=OUT/'preparation_complete.json'
    if complete.exists():print('Already prepared',flush=True);return
    marker=SOURCE.parent/'extraction_complete.json'
    if not marker.exists():raise ValueError('Verified extraction required')
    identifiers={int(p.stem.rsplit('_',1)[-1]) for p in (SOURCE/'label').glob('*.tif')}
    if identifiers!=set(range(1,21484)):raise ValueError('Sample IDs differ from expected 1..21483')
    for folder,prefix in [('gaofen3','after_gf3_hh_'),('gaofen3','after_gf3_hv_'),('sentinel2','before_s2_')]:
        found={int(p.stem.rsplit('_',1)[-1]) for p in (SOURCE/folder).glob(prefix+'*.TIF')}
        if found!=identifiers:raise ValueError('Unpaired inputs '+prefix)
    used=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())
    if used+12_000_000_000>80_000_000_000:raise ValueError('Cache would violate storage reserve')
    lock=OUT/'prepare.lock'
    with lock.open('x') as handle:handle.write(str(os.getpid()))
    try:
        n=len(identifiers)
        xmap=np.lib.format.open_memmap(OUT/'x_uint8.npy',mode='w+',dtype=np.uint8,shape=(n,6,256,256))
        ymap=np.lib.format.open_memmap(OUT/'labels_uint8.npy',mode='w+',dtype=np.uint8,shape=(n,256,256))
        mmap=np.lib.format.open_memmap(OUT/'support_uint8.npy',mode='w+',dtype=np.uint8,shape=(n,256,256))
        amap=np.lib.format.open_memmap(OUT/'row_area_m2.npy',mode='w+',dtype=np.float32,shape=(n,256))
        start=time.monotonic();rows=[]
        # executor.map is bounded here by small result arrays and an ordered consumer;
        # all submitted tasks perform I/O, and futures can otherwise accumulate arrays.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for begin in range(1,n+1,workers*4):
                for rec,x,y,support,areas in pool.map(read_sample,range(begin,min(begin+workers*4,n+1))):
                    idx=rec['cache_index'];xmap[idx]=x;ymap[idx]=y;mmap[idx]=support;amap[idx]=areas;rows.append(rec)
                if begin==1 or len(rows)%512==0 or len(rows)==n:
                    for m in [xmap,ymap,mmap,amap]:m.flush()
                    progress={'samples':len(rows),'total':n,'elapsed_seconds':round(time.monotonic()-start,1),'pid':os.getpid()}
                    (OUT/'progress.json').write_text(json.dumps(progress));print(json.dumps(progress),flush=True)
        for m in [xmap,ymap,mmap,amap]:m.flush()
        import pandas as pd
        frame=pd.DataFrame(rows)
        frame['duplicate_bounds']=frame.duplicated('bounds_key',keep='first')
        frame['duplicate_input']=frame.duplicated('input_sha256',keep='first')
        frame['eligible']=((frame.common_valid_pixels>=6554)&~frame.duplicate_bounds&~frame.duplicate_input&frame.event_group.ne('unassigned'))
        frame['role']=np.where(frame.event_group.eq('2023_holdout'),'locked_test',np.where(frame.event_group.eq('unassigned'),'metadata_quarantine','development'))
        frame.to_csv(OUT/'manifest.csv',index=False)
        summary=frame.groupby(['provisional_site','event_group'],dropna=False).agg(samples=('sample_id','size'),eligible=('eligible','sum'),min_id=('sample_id','min'),max_id=('sample_id','max'),left=('left','min'),right=('right','max'),bottom=('bottom','min'),top=('top','max'),common_area_km2=('common_area_km2','sum'),common_flood_area_km2=('common_flood_area_km2','sum'),excluded_positive_pixels=('excluded_positive_pixels','sum')).reset_index()
        summary.to_csv(ROOT/'reports/sen2gf3_geographic_audit.csv',index=False)
        qa={'sample_count':n,'paired_files':n*4,'label_values':[0,1,255],
            'optical_alpha_tag_count':int(frame.optical_alpha_tag.sum()),'duplicate_bounds_count':int(frame.duplicate_bounds.sum()),
            'duplicate_input_count':int(frame.duplicate_input.sum()),'metadata_quarantine_count':int(frame.event_group.eq('unassigned').sum()),
            'geometric_offset_px_quantiles':{c:{str(k):float(v) for k,v in frame[c].quantile([0,.5,.95,1]).items()} for c in ['hh_grid_offset_px','hv_grid_offset_px','optical_grid_offset_px']},
            'common_support_rule':'label in {0,1}; non-all-zero coverage in both SAR bands and optical stack, warped nearest; identical primary evaluation domain',
            'band_order':['GF3_HH','GF3_HV','S2_R','S2_G','S2_B','S2_band4_NIR_per_paper'],
            'feature_alignment':'array-based bilinear reprojection to original label grid; uint8 rounded; support nearest',
            'geographical_assignment':'provisional spatial match to paper; unmatched region quarantined; no dates learned from outcomes',
            'raw_file_count':n*4,'cache_bytes':sum(p.stat().st_size for p in OUT.glob('*.npy')),
            'elapsed_seconds':round(time.monotonic()-start,1),'completed_at_utc':datetime.now(timezone.utc).isoformat()}
        (ROOT/'reports/sen2gf3_data_audit.json').write_text(json.dumps(qa,indent=2))
        complete.write_text(json.dumps(qa,indent=2))
        print(json.dumps(qa,indent=2),flush=True);print(summary.to_string(index=False),flush=True)
    finally:lock.unlink(missing_ok=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=4);prepare(p.parse_args().workers)
