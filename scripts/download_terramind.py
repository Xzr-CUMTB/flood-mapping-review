"""Download the pinned, public Apache-2.0 TerraMind tiny checkpoint and verify SHA256."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,time
import requests

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'models/pretrained/terramind_tiny'
REVISION='2b5ac0a3ed7dd7e922ccfd595b56607f342df343'
SIZE=211873402
SHA='e56ea9ebcd4451078b9ca4893d5cd8a89bbee376ae16829c3e7fbbbc76de0eba'
URL=f'https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-tiny/resolve/{REVISION}/TerraMind_v1_tiny.pt'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    final=OUT/'TerraMind_v1_tiny.pt';partial=OUT/'TerraMind_v1_tiny.pt.part';lock=OUT/'download.lock'
    with lock.open('x') as stream:stream.write(str(os.getpid()))
    try:
        if not final.exists():
            for attempt in range(8):
                offset=partial.stat().st_size if partial.exists() else 0
                if offset==SIZE:break
                if offset>SIZE:raise ValueError('Oversized partial file')
                try:
                    with requests.get(URL,headers={'Range':f'bytes={offset}-','Accept-Encoding':'identity'},stream=True,timeout=(25,60)) as response:
                        response.raise_for_status()
                        if response.status_code==206:
                            if not response.headers.get('Content-Range','').startswith(f'bytes {offset}-'):raise ValueError('Incorrect resume range')
                        elif response.status_code!=200 or offset:raise ValueError('Server did not honor resume')
                        with partial.open('ab') as stream:
                            for chunk in response.iter_content(1024*1024):
                                if offset+len(chunk)>SIZE:raise ValueError('Response exceeds expected checkpoint size')
                                stream.write(chunk);offset+=len(chunk)
                                (OUT/'progress.json').write_text(json.dumps({'bytes':offset,'expected_bytes':SIZE,'pid':os.getpid()}))
                    if offset!=SIZE:raise requests.RequestException('Truncated checkpoint transfer')
                    break
                except requests.RequestException as exc:
                    print(type(exc).__name__,str(exc)[:200],flush=True)
                    if attempt==7:raise
                    time.sleep(min(2**attempt,30))
        target=final if final.exists() else partial
        digest=hashlib.sha256()
        with target.open('rb') as stream:
            for chunk in iter(lambda:stream.read(8*1024*1024),b''):digest.update(chunk)
        if target.stat().st_size!=SIZE or digest.hexdigest()!=SHA:raise ValueError('Checkpoint size/SHA256 mismatch')
        if target==partial:partial.replace(final)
        result={'status':'verified','bytes':SIZE,'sha256':digest.hexdigest(),'revision':REVISION,'url':URL,
                'license':'Apache-2.0','verified_at_utc':datetime.now(timezone.utc).isoformat()}
        (OUT/'verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    finally:lock.unlink(missing_ok=True)


if __name__=='__main__':main()
