"""Validate archive paths and storage bounds, then extract an authorized dataset."""
from pathlib import Path,PurePosixPath
from datetime import datetime,timezone
import argparse,hashlib,json,shutil,subprocess,os
import rarfile
ROOT=Path(__file__).resolve().parents[1]
DATA={
 'sen2gf3':('Sen2GF3Floods.rar',4931104688,'eaca85ef80813b56bda8f4a57e88ea88',24_000_000_000),
 'hisea':('HISEA1_flooding_dataset.rar',119729117,'539062492b15772e8a27b682100bf0a3',1_000_000_000),
 'china_flood_events':('China_Flood_Data_2012~2024.rar',234076467,'50b07aa441784f69f87e29dc44fa2b5b',4_000_000_000)}

def extract(dataset):
    name,size,md5,cap=DATA[dataset]
    archive=ROOT/'data/raw'/dataset/name
    out=(ROOT/'data/extracted'/dataset).resolve()
    assert ROOT in out.parents
    out.mkdir(parents=True,exist_ok=True)
    marker=out/'extraction_complete.json'
    if marker.exists():
        old=json.loads(marker.read_text());assert old['archive_md5']==md5
        print('Already extracted',dataset,flush=True);return
    if not archive.exists() or archive.stat().st_size!=size:raise ValueError('Complete verified archive required')
    lock=out/'extraction.lock'
    with lock.open('x') as handle:handle.write(str(os.getpid()))
    try:
        h=hashlib.md5()
        with archive.open('rb') as stream:
            for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
        if h.hexdigest()!=md5:raise ValueError('Whole-file MD5 mismatch')
        with rarfile.RarFile(archive,errors='strict') as rf:
            entries=rf.infolist()
            seen=set();total=0;files=[]
            for info in entries:
                n=info.filename;p=PurePosixPath(n)
                if p.is_absolute() or '..' in p.parts or ':' in n or '\\' in n or n.startswith('-'):
                    raise ValueError('Unsafe member '+n)
                if getattr(info,'file_redir',None) or info.is_symlink():raise ValueError('Archive links not allowed')
                dest=(out/n).resolve()
                if out not in dest.parents and dest!=out:raise ValueError('Archive path escapes output')
                if info.isdir():continue
                key=str(dest).casefold()
                if key in seen:raise ValueError('Duplicate destination')
                seen.add(key);total+=info.file_size;files.append((n,info.file_size))
            if total>cap:raise ValueError('Expanded size exceeds dataset bound')
            used=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())
            if used+total>80_000_000_000 or shutil.disk_usage(ROOT).free<total+20_000_000_000:raise ValueError('Space reserve would be exceeded')
            inventory={'archive':str(archive),'archive_md5':md5,'file_count':len(files),'expanded_bytes':total,'paths_validated':True}
            (ROOT/f'reports/{dataset}_archive_summary.json').write_text(json.dumps(inventory,indent=2))
            print(json.dumps(inventory),flush=True)
        with (out/'extraction_stdout.log').open('w') as stdout,(out/'extraction_stderr.log').open('w') as stderr:
            r=subprocess.run(['tar','-xf',str(archive),'-C',str(out)],stdout=stdout,stderr=stderr)
        if r.returncode:raise RuntimeError('tar extraction failed; see stderr log')
        bad=[n for n,s in files if not (out/n).is_file() or (out/n).stat().st_size!=s]
        if bad:raise ValueError('Extracted file size mismatch: '+repr(bad[:10]))
        inventory['completed_at_utc']=datetime.now(timezone.utc).isoformat()
        marker.write_text(json.dumps(inventory,indent=2))
        print('Extraction complete',dataset,flush=True)
    finally:lock.unlink(missing_ok=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('dataset',choices=DATA);extract(parser.parse_args().dataset)
