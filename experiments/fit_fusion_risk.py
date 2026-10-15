"""Matched omission-head control on the fusion expert, without learned value routing."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os, random, sys, time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.crossfit_records import assemble
from experiments.review_dataset import ReviewDataset, augment_batch
from experiments.fit_review_head import on_device, forward_features, loss_for
from experiments.train_baseline import atomic_json
from models.review_heads import SpatialHead


def main():
    spec_path=ROOT/'configs/crossfit_seed17/final_value.json'
    spec=json.loads(spec_path.read_text());records,proof=assemble(spec['pairs'])
    config={'run_id':'final_s17_fusion_risk','kind':'fusion_risk','seed':17,'epochs':10,'batch_size':16,'lr':.001,
        'source_provenance':proof,'samples':len(records),'spec_sha256':hashlib.sha256(spec_path.read_bytes()).hexdigest(),
        'checkpoint_selection':'fixed final epoch',
        'supervision':'Flood label on predicted-negative fusion expert OOF pixels; no value router',
        'features':'Same 16-channel risk head: raw6, source probabilities, disagreement, entropies, coverage, system probability and gate=1'}
    out=ROOT/'runs'/config['run_id'];out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():
        previous=json.loads((out/'config.json').read_text())
        if previous['spec_sha256']!=config['spec_sha256']:raise ValueError('Different completed OOF specification')
        print('Fusion omission control already completed',flush=True);return
    lock=ROOT/'runs/gpu_training.lock'
    with lock.open('x') as stream:stream.write(json.dumps({'pid':os.getpid(),'run_id':config['run_id']}))
    try:
        random.seed(17);np.random.seed(17);torch.manual_seed(17);torch.cuda.manual_seed_all(17)
        torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        model=SpatialHead(16).cuda();rng=random.Random(17)
        loader=DataLoader(ReviewDataset(records),batch_size=16,shuffle=True,num_workers=0,pin_memory=True,
                          generator=torch.Generator().manual_seed(17))
        config['parameters']=sum(p.numel() for p in model.parameters())
        config['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__),ROOT/'experiments/crossfit_records.py',ROOT/'experiments/review_dataset.py',
             ROOT/'experiments/fit_review_head.py',ROOT/'models/review_heads.py',ROOT/'experiments/train_baseline.py']}
        atomic_json(out/'config.json',config);records.to_csv(out/'training_records.csv',index=False)
        for relative,expected in config['source_sha256'].items():
            content=(ROOT/relative).read_bytes()
            if hashlib.sha256(content).hexdigest()!=expected:raise ValueError('Source changed during initialization')
            target=out/'source_snapshot'/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(content)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4);scaler=torch.amp.GradScaler('cuda')
        start=time.monotonic();history=[]
        for epoch in range(1,11):
            model.train();total=0.;count=0;epoch_start=time.monotonic()
            for step,batch in enumerate(loader,1):
                batch=augment_batch(on_device(batch),rng)
                batch['system']=batch['fusion'];batch['gate']=torch.ones_like(batch['fusion'])
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.float16):logits=model(forward_features(batch,'risk'))
                loss,n=loss_for(batch,logits,'risk',.05)
                if not torch.isfinite(loss):raise ValueError('Non-finite fusion-risk loss')
                if n:
                    scaler.scale(loss).backward();scaler.unscale_(optimizer);nn.utils.clip_grad_norm_(model.parameters(),1.)
                    scaler.step(optimizer);scaler.update()
                total+=float(loss)*n;count+=n
                if step%100==0:
                    atomic_json(out/'status.json',{'status':'training','pid':os.getpid(),'epoch':epoch,'step':step,'steps_per_epoch':len(loader)})
            row={'epoch':epoch,'loss':total/max(count,1),'supervised_pixels':count,'seconds':time.monotonic()-epoch_start}
            history.append(row);pd.DataFrame(history).to_csv(out/'history.csv',index=False)
            torch.save({'model':model.state_dict(),'epoch':epoch,'config':config},out/'last.pt')
            atomic_json(out/'status.json',{'status':'training','pid':os.getpid(),**row});print(json.dumps(row),flush=True)
        result={'status':'complete','run_id':config['run_id'],'epochs':10,'samples':len(records),
            'elapsed_seconds':time.monotonic()-start,'peak_gpu_bytes':torch.cuda.max_memory_allocated(),
            'core_test_evaluated':False,'completed_at_utc':datetime.now(timezone.utc).isoformat()}
        atomic_json(out/'complete.json',result);atomic_json(out/'status.json',result);print(json.dumps(result),flush=True)
    except BaseException as exc:
        atomic_json(out/'status.json',{'status':'failed','pid':os.getpid(),'error':repr(exc)});raise
    finally:lock.unlink(missing_ok=True)


if __name__=='__main__':main()
