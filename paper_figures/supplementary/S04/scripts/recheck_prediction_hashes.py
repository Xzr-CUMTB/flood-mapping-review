"""Read-only whole-file SHA256 recheck of the completed core prediction products."""
import os,json,hashlib,time
from pathlib import Path
from datetime import datetime,timezone
ROOT=next(p for p in Path(os.path.abspath(__file__)).parents if (p/'configs/core_release_s17.json').is_file())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(16*1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    t=time.time();source=ROOT/'reports/completed_batch_verification.json';proof=json.loads(source.read_text(encoding='utf-8'));rows=[]
    for r in proof['prediction_files']:
        p=ROOT/r['path'];observed=sha(p);rows.append(dict(path=r['path'],bytes=p.stat().st_size,sha256=observed,expected_sha256=r['sha256'],passed=observed==r['sha256'] and p.stat().st_size==r['bytes']));print(p.name,rows[-1]['passed'],flush=True)
    out=dict(passed=all(r['passed'] for r in rows) and len(rows)==13,checked_at_utc=datetime.now(timezone.utc).isoformat(),release_sha256=proof['release_sha256'],prior_record_sha256=sha(source),elapsed_seconds=time.time()-t,files=rows)
    (ROOT/'figures/supplementary/S03/source_data/prediction_hash_recheck.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    assert out['passed']
if __name__=='__main__':main()
