"""Audit spatial leakage and freeze a deterministic non-overlapping patch subset."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'data/processed/sen2gf3'

def audit():
    path=OUT/'manifest.csv';frame=pd.read_csv(path).sort_values('sample_id').reset_index(drop=True)
    tree=cKDTree(frame[['lon','lat']].to_numpy())
    radius=float(np.hypot((frame.right-frame.left).max(),(frame.top-frame.bottom).max())+1e-7)
    pairs=tree.query_pairs(radius,output_type='ndarray')
    boxes=frame[['left','bottom','right','top']].to_numpy();rows=[];conflicts=[set() for _ in range(len(frame))]
    cross=[]
    for a,b in pairs:
        width=min(boxes[a,2],boxes[b,2])-max(boxes[a,0],boxes[b,0]);height=min(boxes[a,3],boxes[b,3])-max(boxes[a,1],boxes[b,1])
        overlap=max(0,width)*max(0,height)
        if overlap<=1e-12:continue  # <~0.01 m²; floating-point seam tolerance, not a percentage exclusion.
        smaller=min((boxes[a,2]-boxes[a,0])*(boxes[a,3]-boxes[a,1]),(boxes[b,2]-boxes[b,0])*(boxes[b,3]-boxes[b,1]))
        row={'sample_a':int(frame.loc[a,'sample_id']),'sample_b':int(frame.loc[b,'sample_id']),
             'group_a':frame.loc[a,'event_group'],'group_b':frame.loc[b,'event_group'],'overlap_fraction_of_smaller':float(overlap/smaller)}
        rows.append(row);conflicts[a].add(b);conflicts[b].add(a)
        if row['group_a']!=row['group_b'] and 'unassigned' not in [row['group_a'],row['group_b']]:cross.append(row)
    pd.DataFrame(rows,columns=['sample_a','sample_b','group_a','group_b','overlap_fraction_of_smaller']).to_csv(ROOT/'reports/spatial_overlap_pairs.csv',index=False)
    if cross:raise ValueError('Overlapping geography crosses proposed event groups: '+repr(cross[:5]))
    accepted=set();nonoverlap=np.zeros(len(frame),dtype=bool)
    for i in frame.index:
        if not bool(frame.loc[i,'eligible']):continue
        if not any(j in accepted for j in conflicts[i]):accepted.add(i);nonoverlap[i]=True
    frame['spatial_nonoverlap']=nonoverlap
    # Check exact input content across accepted event groups, including differing footprints.
    hashes=frame[frame.eligible].groupby('input_sha256').event_group.nunique()
    if (hashes>1).any():raise ValueError('Cross-event duplicate image content')
    frame.to_csv(path,index=False)
    counts=frame.groupby(['event_group','role']).agg(total=('sample_id','size'),eligible=('eligible','sum'),nonoverlap=('spatial_nonoverlap','sum')).reset_index()
    counts.to_csv(ROOT/'reports/experiment_split_counts.csv',index=False)
    report={'passed':True,'sample_count':len(frame),'overlap_pair_count':len(rows),'cross_event_overlap_count':len(cross),
            'nonoverlap_selection':'greedy ascending sample ID, independent of labels and model results; any footprint overlap >1e-12 square degrees conflicts',
            'retained_nonoverlap_count':int(nonoverlap.sum()),'test_group_locked':'2023_holdout','metadata_quarantine_group':'unassigned',
            'manifest_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'completed_at_utc':datetime.now(timezone.utc).isoformat()}
    (OUT/'split_audit_passed.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2));print(counts.to_string(index=False))

if __name__=='__main__':audit()
