"""Resume the public Sen2GF3 V2 archive and verify its repository MD5."""
from pathlib import Path
import hashlib
import json
import shutil
import time
import requests
import os
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/raw/sen2gf3'
OUT.mkdir(parents=True, exist_ok=True)
URL = 'https://china.scidb.cn/download?fileId=e1a0e5b0745088286d10ce78e85183c3'
SIZE = 4931104688
MD5 = 'eaca85ef80813b56bda8f4a57e88ea88'
DEST = OUT / 'Sen2GF3Floods.rar'
PART = DEST.with_suffix('.rar.part')

def verify(path):
    md5, sha = hashlib.md5(), hashlib.sha256()
    with path.open('rb') as src:
        for chunk in iter(lambda: src.read(8 * 1024 * 1024), b''):
            md5.update(chunk)
            sha.update(chunk)
    if path.stat().st_size != SIZE or md5.hexdigest() != MD5:
        raise ValueError('Archive size or checksum does not match ScienceDB')
    return sha.hexdigest()

if __name__ == '__main__':
    if DEST.exists():
        sha = verify(DEST)
    else:
        initial = PART.stat().st_size if PART.exists() else 0
        if shutil.disk_usage(ROOT).free < (SIZE - initial) + 25_000_000_000:
            raise SystemExit('Need download space plus 25 GB working reserve')
        used = sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())
        if used + SIZE - initial > 100_000_000_000:
            raise SystemExit('Project would exceed 100 GB')
        for attempt in range(10):
            offset = PART.stat().st_size if PART.exists() else 0
            if offset == SIZE:
                break
            if offset > SIZE:
                raise ValueError('Partial file exceeds published bounds; preserved for inspection')
            try:
                with requests.get(URL, headers={'Range': f'bytes={offset}-', 'Accept-Encoding': 'identity'},
                                  stream=True, timeout=(15, 60)) as r:
                    r.raise_for_status()
                    if r.status_code == 206:
                        if not r.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                            raise ValueError('Wrong response offset')
                    elif r.status_code == 200 and offset == 0:
                        pass
                    else:
                        raise ValueError('Resume offset not honored; existing partial file preserved')
                    n = offset
                    started = time.monotonic()
                    with PART.open('ab' if offset else 'wb') as stream:
                        for chunk in r.iter_content(4 * 1024 * 1024):
                            n += len(chunk)
                            if n > SIZE:
                                raise ValueError('Response exceeds published size')
                            stream.write(chunk)
                            stream.flush()
                            (OUT / 'progress.json').write_text(json.dumps({'status': 'downloading', 'bytes': n, 'expected_bytes': SIZE, 'pid': os.getpid(), 'utc': datetime.now(timezone.utc).isoformat()}))
                            if n // 250_000_000 != (n - len(chunk)) // 250_000_000:
                                speed = (n - offset) / max(time.monotonic() - started, 1) / 1e6
                                print(f'{n / 1e9:.2f}/{SIZE / 1e9:.2f} GB, {speed:.1f} MB/s', flush=True)
                if n != SIZE:
                    raise requests.RequestException('Incomplete response; preserved for resume')
                break
            except requests.RequestException as exc:
                print(f'Attempt {attempt + 1}: {type(exc).__name__}', flush=True)
                if attempt == 9:
                    raise
                time.sleep(min(2 ** attempt, 30))
        sha = verify(PART)
        PART.replace(DEST)
    result = {'status': 'verified', 'source': URL, 'bytes': SIZE, 'md5': MD5, 'sha256': sha, 'verified_at_utc': datetime.now(timezone.utc).isoformat()}
    (OUT / 'verification.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
