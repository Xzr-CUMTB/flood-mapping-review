"""Wait for the active download, then verify, extract and audit. No plots or training."""
from pathlib import Path
from datetime import datetime,timezone
import json,os,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/stage1_preparation';OUT.mkdir(parents=True,exist_ok=True)
LOCK=OUT/'run.lock'
with LOCK.open('x') as f:f.write(str(os.getpid()))
def status(stage,**extra):
    value={'stage':stage,'pid':os.getpid(),'utc':datetime.now(timezone.utc).isoformat(),**extra}
    (OUT/'status.json').write_text(json.dumps(value,indent=2));print(json.dumps(value),flush=True)
try:
    archive=ROOT/'data/raw/sen2gf3/Sen2GF3Floods.rar'
    marker=archive.parent/'verification.json'
    start=time.monotonic()
    while not (archive.exists() and marker.exists()):
        if time.monotonic()-start>10800:raise TimeoutError('Primary download has not completed within 3 hours')
        part=archive.with_suffix('.rar.part')
        status('waiting_for_verified_archive',received_bytes=part.stat().st_size if part.exists() else 0)
        time.sleep(30)
    status('extracting')
    subprocess.run([sys.executable,'-s',str(ROOT/'scripts/extract_verified_dataset.py'),'sen2gf3'],cwd=ROOT,check=True)
    status('auditing_and_building_cache')
    subprocess.run([sys.executable,'-s',str(ROOT/'scripts/prepare_sen2gf3.py'),'--workers','4'],cwd=ROOT,check=True)
    status('verifying_prepared_cache')
    subprocess.run([sys.executable,'-s',str(ROOT/'scripts/verify_prepared_cache.py')],cwd=ROOT,check=True)
    status('auditing_spatial_overlap')
    subprocess.run([sys.executable,'-s',str(ROOT/'scripts/audit_spatial_splits.py')],cwd=ROOT,check=True)
    status('data_ready_for_split_review')
except BaseException as exc:
    status('failed',error=repr(exc));raise
finally:LOCK.unlink(missing_ok=True)
