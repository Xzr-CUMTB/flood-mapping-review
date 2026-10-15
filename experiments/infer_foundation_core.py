"""Predict the frozen core with both prescribed foundation controls, using no labels."""
from pathlib import Path
from datetime import datetime, timezone
import json, os, shutil, sys, time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.foundation_environment import configure
configure()
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from experiments.train_baseline import FloodDataset, atomic_json
from experiments.infer_frozen_core import check_release, sha
from models.terramind_flood import TerraMindFlood
from scripts.project_inventory import count_bytes


@torch.no_grad()
def main():
    release_path=ROOT/'configs/core_release_s17.json';check_release(release_path)
    source=ROOT/'runs/core_s17_predictions'
    proof=json.loads((source/'complete.json').read_text())
    if proof['status']!='complete' or proof['release_sha256']!=sha(release_path):
        raise ValueError('Primary predictions from this frozen release required')
    frame=pd.read_csv(source/'prediction_manifest.csv')
    if not frame.event_group.eq('2023_holdout').all() or not frame.role.eq('locked_test').all():
        raise ValueError('Expected the frozen core test')
    out=ROOT/'runs/core_s17_foundation_predictions';out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():
        previous=json.loads((out/'complete.json').read_text())
        if previous['release_sha256']!=sha(release_path):raise ValueError('Completed predictions from another release')
        print('Foundation core predictions already completed',flush=True);return
    expected_bytes=len(frame)*256*256*4*2
    replaceable=sum(path.stat().st_size for path in [out/'terramind_tiny.npy',out/'terramind_tiny_random.npy'] if path.exists())
    extra_bytes=max(expected_bytes-replaceable,0)
    if count_bytes(ROOT)+extra_bytes>95_000_000_000 or shutil.disk_usage(ROOT).free<extra_bytes+5_000_000_000:
        raise ValueError('Foundation inference would exceed storage reserve')
    run_release=out/'run_release.json'
    if run_release.exists() and json.loads(run_release.read_text())['release_sha256']!=sha(release_path):
        raise ValueError('Incomplete foundation outputs belong to another release')
    atomic_json(run_release,{'release_sha256':sha(release_path),'restart_rule':'Recompute incomplete maps from the same frozen release'})
    lock=ROOT/'runs/gpu_training.lock'
    with lock.open('x') as stream:stream.write(json.dumps({'pid':os.getpid(),'run_id':'core_s17_foundation_predictions'}))
    try:
        torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        loader=DataLoader(FloodDataset(frame.cache_index,modality='fusion'),batch_size=8,num_workers=0,pin_memory=True)
        start=time.monotonic()
        models={'terramind_tiny':'final_terramind_s17','terramind_tiny_random':'final_terramind_random_s17'}
        for name,run in models.items():
            checkpoint=torch.load(ROOT/'runs'/run/'last.pt',map_location='cpu',weights_only=False)
            model=TerraMindFlood(pretrained=False)
            model.load_state_dict(checkpoint['model'],strict=True);model=model.cuda().eval()
            predictions=np.lib.format.open_memmap(out/(name+'.npy'),mode='w+',dtype='float32',shape=(len(frame),256,256))
            cursor=0
            for step,batch in enumerate(loader,1):
                # Targets are never passed to a model or used for checkpoint selection.
                with torch.autocast('cuda',dtype=torch.float16):p=model(batch['x'].cuda(non_blocking=True)).float().sigmoid()
                array=p.cpu().numpy()
                if not np.isfinite(array).all() or np.any((array<0)|(array>1)):raise ValueError('Invalid foundation probabilities')
                predictions[cursor:cursor+len(p)]=array;cursor+=len(p)
                if step%100==0:
                    progress={'status':'predicting','pid':os.getpid(),'model':name,'patches':cursor,'total':len(frame),'seconds':time.monotonic()-start}
                    atomic_json(out/'status.json',progress);print(json.dumps(progress),flush=True)
            predictions.flush();del predictions,checkpoint,model;torch.cuda.empty_cache()
        frame.to_csv(out/'prediction_manifest.csv',index=False)
        result={'status':'complete','patches':len(frame),'models':list(models),'release_sha256':sha(release_path),
                'completed_at_utc':datetime.now(timezone.utc).isoformat(),'elapsed_seconds':time.monotonic()-start,
                'checkpoint_selection_on_test':False,'peak_gpu_bytes':torch.cuda.max_memory_allocated()}
        atomic_json(out/'complete.json',result);atomic_json(out/'status.json',result);print(json.dumps(result),flush=True)
    except BaseException as exc:
        atomic_json(out/'status.json',{'status':'failed','error':repr(exc),'pid':os.getpid()});raise
    finally:lock.unlink(missing_ok=True)


if __name__=='__main__':main()
