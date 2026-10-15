"""Verify file-level hashes, readable numeric sources and figure-panel coverage."""
from pathlib import Path
import argparse, csv, hashlib, json, re
import numpy as np
import pandas as pd


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def verify(data,out):
    data=Path(data).resolve();out=Path(out);out.mkdir(parents=True,exist_ok=True)
    failures=[];checked=[]
    with (data/'FILE_MANIFEST.csv').open(encoding='utf-8',newline='') as f:manifest=list(csv.DictReader(f))
    for entry in manifest:
        p=data/entry['path']
        if not p.resolve().is_relative_to(data):raise ValueError('Unsafe manifest path')
        if not p.is_file() or p.stat().st_size!=int(entry['bytes']) or sha(p)!=entry['sha256']:
            failures.append(dict(file=entry['path'],error='Missing file, size or SHA-256 mismatch'));continue
        try:
            if p.suffix=='.json':json.loads(p.read_text(encoding='utf-8'))
            elif p.name.endswith(('.csv','.csv.gz')):
                for chunk in pd.read_csv(p,chunksize=100000,low_memory=False):pass
            elif p.suffix=='.npy':
                a=np.load(p,mmap_mode='r',allow_pickle=False);assert a.dtype.kind!='O'
            elif p.suffix=='.npz':
                with np.load(p,allow_pickle=False) as z:
                    for k in z.files:assert z[k].dtype.kind!='O'
            if p.suffix in ['.json','.md','.py','.txt','.cff']:
                text=p.read_text(encoding='utf-8-sig')
                if re.search(r'(?i)(?<![A-Za-z])[A-Z]:[\\/]',text):raise ValueError('Absolute machine path')
                if re.search(r'gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|-----BEGIN [A-Z ]*PRIVATE KEY-----',text):raise ValueError('Credential pattern')
            checked.append(entry['path'])
        except Exception as exc:failures.append(dict(file=entry['path'],error=str(exc)))
    index=pd.read_csv(data/'PANEL_DATA_INDEX.csv').fillna('')
    names=set(index.figure)
    assert names=={f'Fig{i:02}' for i in range(1,7)}|{f'S{i:02}' for i in range(1,11)}
    if index.duplicated(['figure','panel']).any():failures.append(dict(file='PANEL_DATA_INDEX.csv',error='Duplicate panel mapping'))
    for row in index.itertuples():
        sources=[s.strip() for s in row.source_files.split(';') if s.strip()]
        if not sources:failures.append(dict(file=row.figure+':'+row.panel,error='No panel data'))
        for name in sources:
            if not (data/'figure_data'/row.figure/'source_data'/name).is_file():failures.append(dict(file=row.figure+':'+row.panel,error='Missing '+name))
    expected={entry['path'] for entry in manifest}|{'FILE_MANIFEST.csv','CHECKSUMS_SHA256.txt'}
    actual={p.relative_to(data).as_posix() for p in data.rglob('*') if p.is_file()}
    if expected!=actual:failures.append(dict(file='FILE_MANIFEST.csv',error='Unexpected or missing payload files',extra=sorted(actual-expected),missing=sorted(expected-actual)))
    for line in (data/'CHECKSUMS_SHA256.txt').read_text(encoding='utf-8').splitlines():
        digest,name=line.split('  ',1);path=data/name
        if not path.resolve().is_relative_to(data) or sha(path)!=digest:failures.append(dict(file=name,error='Checksum-text mismatch'))
    for entry in json.loads((data/'COPY_PROVENANCE.json').read_text(encoding='utf-8')):
        if sha(data/entry['path'])!=entry['sha256']:failures.append(dict(file=entry['path'],error='Copy-provenance public hash mismatch'))
    repo=Path(__file__).resolve().parents[1]
    for entry in json.loads((data/'METADATA_SANITIZATION.json').read_text(encoding='utf-8')):
        base=repo if entry['file'].startswith('paper_figures/') else data
        path=base/entry['file']
        if not path.resolve().is_relative_to(base.resolve()) or sha(path)!=entry['public_sha256']:failures.append(dict(file=entry['file'],error='Metadata-change public hash mismatch'))
    result=dict(passed=not failures,files_verified=len(checked),manifest_files=len(manifest),figures=len(names),panels=len(index),failures=failures,
                scope='Actual research payload; synthetic fixtures excluded. SHA256 does not imply repository publication.')
    (out/'payload_verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result),flush=True)
    if failures:raise ValueError('Payload verification failed; inspect report')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();verify(a.data,a.output)
