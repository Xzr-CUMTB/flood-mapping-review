# -*- coding: utf-8 -*-
"""Auditable single-GPU baseline training. Test events remain inaccessible by default."""
from pathlib import Path
import os
os.environ.setdefault('PYTHONNOUSERSITE','1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys,argparse,csv,hashlib,json,random,time,platform
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from models.flood_unet import FloodUNet
DATA=ROOT/'data/processed/sen2gf3'

def atomic_json(path,value):
    temp=path.with_suffix('.json.tmp');temp.write_text(json.dumps(value,indent=2),encoding='utf-8');temp.replace(path)

class FloodDataset(Dataset):
    def __init__(self,indices,augment=False,modality='fusion'):
        self.indices=list(map(int,indices));self.augment=augment;self.modality=modality
        self.x=np.load(DATA/'x_uint8.npy',mmap_mode='r');self.y=np.load(DATA/'labels_uint8.npy',mmap_mode='r')
        self.support=np.load(DATA/'support_uint8.npy',mmap_mode='r');self.area=np.load(DATA/'row_area_m2.npy',mmap_mode='r')
    def __len__(self):return len(self.indices)
    def __getitem__(self,i):
        idx=self.indices[i]
        # Avoid reading optical pages for the SAR control; inactive channels are exactly zero.
        x=np.zeros((6,256,256),dtype=np.float32)
        channels={'sar':slice(0,2),'fusion':slice(0,6),'rgb_fusion':slice(0,5),'optical':slice(2,6)}[self.modality]
        x[channels]=np.array(self.x[idx,channels],dtype=np.float32)/255.
        raw=np.array(self.y[idx]);valid=np.array(self.support[idx])==7
        y=np.where(valid,raw,0).astype('float32')
        if self.augment:
            k=random.randrange(4);x=np.rot90(x,k,axes=(-2,-1));y=np.rot90(y,k);valid=np.rot90(valid,k)
            if random.random()<.5:x=x[...,::-1];y=y[:,::-1];valid=valid[:,::-1]
        return {'x':torch.from_numpy(x.copy()),'y':torch.from_numpy(y.copy()),
                'valid':torch.from_numpy(valid.copy()),'area':torch.from_numpy(np.array(self.area[idx])),'index':idx}

class Metrics:
    def __init__(self):self.sums=np.zeros(10,dtype=np.float64)
    def add(self,p,y,valid,area):
        p=p.detach().float();y=y.bool();valid=valid.bool();w=area[:,:,None].double()*valid
        positive=p>=.5
        values=[((positive&y)*w).sum(),((positive&~y)*w).sum(),((~positive&y)*w).sum(),((~positive&~y)*w).sum(),
                (((p-y.float())**2)*w).sum(),w.sum(),((y&(p<=.1))*w).sum(),(y*w).sum(),
                ((-(y.float()*p.clamp_min(1e-7).log()+(~y).float()*(1-p).clamp_min(1e-7).log()))*w).sum(),valid.sum()]
        self.sums+=np.array([float(v) for v in values])
    def result(self):
        tp,fp,fn,tn,bs,area,conf,truth,nll,npixels=self.sums
        ratio=lambda a,b:float(a/b) if b>0 else None
        return {'flood_iou':ratio(tp,tp+fp+fn),'background_iou':ratio(tn,tn+fp+fn),'precision':ratio(tp,tp+fp),
                'recall':ratio(tp,tp+fn),'fn_rate':ratio(fn,tp+fn),'brier':ratio(bs,area),'nll':ratio(nll,area),
                'confident_omission_fraction':ratio(conf,truth),'confident_omission_km2':float(conf/1e6),
                'fn_km2':float(fn/1e6),'fp_km2':float(fp/1e6),'valid_area_km2':float(area/1e6),'true_flood_km2':float(truth/1e6),'valid_pixels':int(npixels)}

@torch.no_grad()
def evaluate(model,loader,device,output=None,frame=None):
    model.eval();metrics=Metrics();rows=[];predmap=None;cursor=0
    if output is not None:predmap=np.lib.format.open_memmap(output/'validation_probabilities.npy',mode='w+',dtype=np.float32,shape=(len(loader.dataset),256,256))
    for batch in loader:
        x=batch['x'].to(device);y=batch['y'].to(device);valid=batch['valid'].to(device);area=batch['area'].to(device)
        with torch.autocast('cuda',dtype=torch.float16):logits=model(x)
        prob=logits.float().sigmoid();metrics.add(prob,y,valid,area)
        if output is not None:
            array=prob.cpu().numpy();predmap[cursor:cursor+len(array)]=array
            for j,idx in enumerate(batch['index'].tolist()):
                m=Metrics();m.add(prob[j:j+1],y[j:j+1],valid[j:j+1],area[j:j+1]);record=m.result()
                pv=prob[j][valid[j]].clamp(1e-7,1-1e-7)
                record.update(cache_index=idx,prediction_index=cursor+j,sample_id=int(frame.loc[idx,'sample_id']),event_group=frame.loc[idx,'event_group'],site=frame.loc[idx,'provisional_site'],
                              mean_entropy=float((-(pv*pv.log()+(1-pv)*(1-pv).log())).mean()))
                rows.append(record)
            cursor+=len(array)
    if predmap is not None:predmap.flush();pd.DataFrame(rows).to_csv(output/'validation_patch_metrics.csv',index=False)
    return metrics.result()

def run(args):
    if not (DATA/'preparation_complete.json').exists():raise ValueError('Full audited cache required')
    if not (DATA/'split_audit_passed.json').exists():raise ValueError('Spatial split audit required')
    frame=pd.read_csv(DATA/'manifest.csv').set_index('cache_index',drop=False)
    usable=frame[frame.eligible&frame.spatial_nonoverlap&frame.role.eq('development')].copy()
    train_groups=args.train_groups.split(',');val_groups=[] if args.no_validation else args.val_groups.split(',')
    if args.no_validation and args.selection!='fixed_last':raise ValueError('No-validation training requires fixed_last')
    if set(train_groups)&set(val_groups):raise ValueError('Group overlap')
    if '2023_holdout' in train_groups+val_groups:raise ValueError('Core test locked; stage1 cannot access it')
    train=usable[usable.event_group.isin(train_groups)];val=usable[usable.event_group.isin(val_groups)]
    if not len(train) or (not len(val) and not args.no_validation):raise ValueError('Empty training/validation group')
    if args.max_train:train=train.sample(min(args.max_train,len(train)),random_state=args.seed).sort_index()
    if args.max_val:val=val.sample(min(args.max_val,len(val)),random_state=args.seed).sort_index()
    assert not set(train.bounds_key)&set(val.bounds_key)
    assert not set(train.input_sha256)&set(val.input_sha256)
    out=ROOT/'runs'/args.run_id;out.mkdir(parents=True,exist_ok=True)
    if (out/'complete.json').exists():
        previous=json.loads((out/'config.json').read_text())
        for key in ['modality','train_groups','val_groups','epochs','width','seed','max_train','max_val','no_validation']:
            if previous.get(key,False if key=='no_validation' else None)!=getattr(args,key):raise ValueError('Run ID already has different '+key)
        print('Run already complete',args.run_id,flush=True);return
    lock=ROOT/'runs/gpu_training.lock'
    with lock.open('x') as handle:handle.write(json.dumps({'pid':os.getpid(),'run_id':args.run_id}))
    try:
        random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
        torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        device=torch.device('cuda');model=FloodUNet(args.modality,args.width).to(device)
        train_loader=DataLoader(FloodDataset(train.index,True,args.modality),batch_size=args.batch_size,shuffle=True,num_workers=0,pin_memory=True,
                                generator=torch.Generator().manual_seed(args.seed))
        val_loader=DataLoader(FloodDataset(val.index,modality=args.modality),batch_size=args.batch_size,num_workers=0,pin_memory=True) if len(val) else None
        optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
        scaler=torch.amp.GradScaler('cuda')
        manifest_hash=hashlib.sha256((DATA/'manifest.csv').read_bytes()).hexdigest()
        config=vars(args).copy();config.update(train_samples=len(train),val_samples=len(val),parameters=sum(p.numel() for p in model.parameters()),
            torch=torch.__version__,python=platform.python_version(),gpu=torch.cuda.get_device_name(0),manifest_sha256=manifest_hash,
            source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'models/flood_unet.py']},
            selection='best validation NLL' if args.selection=='best_val' else 'fixed final epoch; validation not used for selection',
            checkpoint_selection_mode=args.selection,validation_used_for_selection=args.selection=='best_val',
            primary_domain='same common support for SAR and fusion; all-zero padding heuristic audited separately',
            interpretation='development experiment; provisional geography-to-paper association; not final independent test',
            area_interpretation='geodesic pixel area on a deterministic non-overlapping patch subset; full scene coverage is not inferred')
        atomic_json(out/'config.json',config)
        for relative,digest in config['source_sha256'].items():
            content=(ROOT/relative).read_bytes()
            if hashlib.sha256(content).hexdigest()!=digest:raise ValueError('Source changed while starting run')
            snapshot=out/'source_snapshot'/relative;snapshot.parent.mkdir(parents=True,exist_ok=True);snapshot.write_bytes(content)
        train[['sample_id','event_group','bounds_key']].to_csv(out/'train_manifest.csv',index=False)
        val[['sample_id','event_group','bounds_key']].to_csv(out/'validation_manifest.csv',index=False)
        atomic_json(out/'status.json',{'status':'training','pid':os.getpid(),'started_utc':datetime.now(timezone.utc).isoformat()})
        print(json.dumps(config),flush=True)
        best=float('inf');best_epoch=0;start=time.monotonic();history=[]
        for epoch in range(1,args.epochs+1):
            epoch_start=time.monotonic();model.train();loss_sum=0.;pixel_sum=0;steps=0
            for batch in train_loader:
                x=batch['x'].to(device,non_blocking=True);y=batch['y'].to(device,non_blocking=True);valid=batch['valid'].to(device,non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.float16):
                    logits=model(x);pixel_loss=nn.functional.binary_cross_entropy_with_logits(logits,y,reduction='none')
                    loss=(pixel_loss*valid).sum()/valid.sum().clamp_min(1)
                if not torch.isfinite(loss):raise ValueError('Non-finite training loss')
                scaler.scale(loss).backward();scaler.unscale_(optimizer);nn.utils.clip_grad_norm_(model.parameters(),1.)
                scaler.step(optimizer);scaler.update();n=int(valid.sum());loss_sum+=float(loss)*n;pixel_sum+=n;steps+=1
                if steps%100==0:
                    atomic_json(out/'status.json',{'status':'training','epoch':epoch,'step':steps,'steps_per_epoch':len(train_loader),'pid':os.getpid(),'loss':loss_sum/max(pixel_sum,1)})
                    print(json.dumps({'epoch':epoch,'step':steps,'loss':round(loss_sum/max(pixel_sum,1),5)}),flush=True)
            val_result=evaluate(model,val_loader,device) if val_loader is not None and (args.selection=='best_val' or epoch==args.epochs) else {}
            record={'epoch':epoch,'train_bce':loss_sum/max(pixel_sum,1),'seconds':round(time.monotonic()-epoch_start,2),**{'val_'+k:v for k,v in val_result.items()}}
            history.append(record);pd.DataFrame(history).to_csv(out/'history.csv',index=False)
            torch.save({'model':model.state_dict(),'epoch':epoch,'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),'config':config},out/'last.pt')
            if args.selection=='best_val' and val_result['nll']<best:
                best=val_result['nll'];best_epoch=epoch;torch.save({'model':model.state_dict(),'epoch':epoch,'config':config},out/'best.pt')
            print(json.dumps(record),flush=True)
            atomic_json(out/'status.json',{'status':'training','epoch':epoch,'best_epoch':best_epoch,'pid':os.getpid(),'last_validation':val_result})
            if args.selection=='best_val' and epoch-best_epoch>=args.patience:break
        checkpoint=out/('best.pt' if args.selection=='best_val' else 'last.pt')
        fitted=torch.load(checkpoint,map_location=device,weights_only=False);model.load_state_dict(fitted['model'])
        final=evaluate(model,val_loader,device,out,frame) if val_loader is not None else {}
        result={'run_id':args.run_id,'status':'complete','selected_epoch':fitted['epoch'],'metrics':final,
                'elapsed_seconds':round(time.monotonic()-start,2),'peak_allocated_gpu_bytes':torch.cuda.max_memory_allocated(),
                'completed_at_utc':datetime.now(timezone.utc).isoformat(),'core_test_evaluated':False}
        atomic_json(out/'complete.json',result);atomic_json(out/'status.json',result);print(json.dumps(result),flush=True)
    except BaseException as exc:
        atomic_json(out/'status.json',{'status':'failed','error':repr(exc),'pid':os.getpid()});raise
    finally:lock.unlink(missing_ok=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--modality',choices=['sar','fusion','optical','rgb_fusion'],required=True)
    p.add_argument('--train-groups',default='2020_flood_season,2021_Henan');p.add_argument('--val-groups',default='2024_April,2024_June')
    p.add_argument('--epochs',type=int,default=20);p.add_argument('--patience',type=int,default=6);p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--width',type=int,default=16);p.add_argument('--seed',type=int,default=17);p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--max-train',type=int,default=0);p.add_argument('--max-val',type=int,default=0);p.add_argument('--selection',choices=['best_val','fixed_last'],default='best_val')
    p.add_argument('--no-validation',action='store_true',help='Fit all development groups for a fixed epoch count without evaluating any held-out labels')
    run(p.parse_args())
