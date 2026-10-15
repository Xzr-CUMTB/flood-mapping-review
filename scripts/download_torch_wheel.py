"""Bounded parallel range transfer of the exact official, hash-pinned PyTorch wheel."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import argparse,hashlib,json,os,re,time
import requests

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'vendor/wheels';PARTS=OUT/'torch_parts'
CHUNK=16*1024*1024


def main(seed_existing=None):
    metadata=json.loads((ROOT/'configs/torch_foundation_wheel.json').read_text())
    size=metadata['expected_bytes'];expected=metadata['sha256']
    OUT.mkdir(parents=True,exist_ok=True);PARTS.mkdir(parents=True,exist_ok=True)
    assert ROOT in PARTS.resolve().parents
    final=OUT/'torch-2.7.1+cu128-cp311-cp311-win_amd64.whl'
    lock=PARTS/'download.lock'
    with lock.open('x') as stream:stream.write(str(os.getpid()))
    start_time=time.monotonic()
    count=(size+CHUNK-1)//CHUNK

    def chunk_path(index):return PARTS/f'chunk_{index:04d}.bin'
    def expected_chunk(index):return min(CHUNK,size-index*CHUNK)
    def transfer(index):
        path=chunk_path(index);length=expected_chunk(index);errors=0
        if path.exists() and path.stat().st_size>length:raise ValueError('Oversized local range')
        while (path.stat().st_size if path.exists() else 0)<length:
            offset=path.stat().st_size if path.exists() else 0
            begin=index*CHUNK+offset;end=index*CHUNK+length-1
            host='download.pytorch.org' if index%2==0 else 'download-r2.pytorch.org'
            url=metadata['url'].replace('download-r2.pytorch.org',host)
            try:
                with requests.get(url,headers={'Range':f'bytes={begin}-{end}','Accept-Encoding':'identity'},stream=True,timeout=(20,60)) as response:
                    response.raise_for_status()
                    match=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',response.headers.get('Content-Range',''))
                    if response.status_code!=206 or not match:raise ValueError('Range response required')
                    lo,hi,total=map(int,match.groups())
                    if lo!=begin or hi>end or total!=size:raise ValueError('Incorrect byte range')
                    received=0
                    with path.open('ab') as stream:
                        for data in response.iter_content(256*1024):
                            if received+len(data)>hi-lo+1:raise ValueError('Range response overflow')
                            stream.write(data);received+=len(data)
                    if received!=hi-lo+1:raise requests.RequestException('Truncated range response')
                errors=0
            except requests.RequestException as exc:
                errors+=1
                if errors>=8:raise
                time.sleep(min(2**errors,30))
        return index

    try:
        if not final.exists():
            if seed_existing:
                seed=Path(seed_existing)
                if seed.exists():
                    with seed.open('rb') as stream:
                        for index in range(count):
                            data=stream.read(expected_chunk(index))
                            if not data:break
                            path=chunk_path(index)
                            if not path.exists():path.write_bytes(data)
            with ThreadPoolExecutor(max_workers=8) as pool:
                for future in as_completed([pool.submit(transfer,i) for i in range(count)]):
                    future.result()
                    received=sum(p.stat().st_size for p in PARTS.glob('chunk_*.bin'))
                    progress={'bytes':received,'expected_bytes':size,'elapsed_seconds':time.monotonic()-start_time,'pid':os.getpid()}
                    temp=OUT/'torch_download_progress.tmp';temp.write_text(json.dumps(progress));temp.replace(OUT/'torch_download_progress.json')
                    print(json.dumps(progress),flush=True)
            combined=final.with_suffix('.whl.part')
            digest=hashlib.sha256()
            with combined.open('wb') as dest:
                for index in range(count):
                    path=chunk_path(index)
                    if path.stat().st_size!=expected_chunk(index):raise ValueError('Incomplete chunk')
                    with path.open('rb') as source:
                        for data in iter(lambda:source.read(8*1024*1024),b''):dest.write(data);digest.update(data)
            if combined.stat().st_size!=size or digest.hexdigest()!=expected:raise ValueError('Whole-wheel SHA256 mismatch')
            combined.replace(final)
        else:
            digest=hashlib.sha256()
            with final.open('rb') as stream:
                for data in iter(lambda:stream.read(8*1024*1024),b''):digest.update(data)
            if final.stat().st_size!=size or digest.hexdigest()!=expected:raise ValueError('Existing wheel hash mismatch')
        proof={**metadata,'status':'verified','actual_sha256':digest.hexdigest(),'verified_at_utc':datetime.now(timezone.utc).isoformat()}
        (OUT/'torch_wheel_verification.json').write_text(json.dumps(proof,indent=2))
        # Delete only this downloader's numbered scratch files, after the assembled file is verified.
        for index in range(count):
            path=chunk_path(index)
            assert path.resolve().parent==PARTS.resolve() and ROOT in path.resolve().parents
            path.unlink(missing_ok=True)
        print(json.dumps(proof),flush=True)
    finally:lock.unlink(missing_ok=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seed-existing');args=parser.parse_args();main(args.seed_existing)
