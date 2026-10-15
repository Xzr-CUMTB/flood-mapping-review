"""Fit spatial value/gating/risk heads from event-held-out predictions; no plots."""
from pathlib import Path
import argparse, hashlib, json, os, random, sys, time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.crossfit_records import assemble
from experiments.review_dataset import ReviewDataset, augment_batch
from experiments.train_baseline import atomic_json, Metrics
from models.review_heads import SpatialHead, evidence_features, omission_features, route


def risk_records(system_runs):
    chunks, sources = [], []
    for run in system_runs:
        path = ROOT/'runs'/run
        config = json.loads((path/'config.json').read_text())
        proof = config['source_provenance']
        if config['kind'] != 'value' or not (path/'prediction_complete.json').exists():
            raise ValueError('Completed whole-system value-head OOF predictions required')
        records = pd.read_csv(path/'prediction_records.csv')
        targets = set(records.event_group)
        if targets.intersection(proof['training_groups']):
            raise ValueError('Gate saw the target event labels')
        if not targets.issubset(proof['outer_excluded_groups']):
            raise ValueError('Target event was not excluded from the complete inner pipeline')
        for source in proof['sources']:
            if targets.intersection(source['expert_training_groups']):
                raise ValueError('Inner expert leaked the target event')
        if '2023_holdout' in targets:
            raise ValueError('Core test cannot provide risk supervision')
        records['system_run'] = run
        records['system_prediction_index'] = np.arange(len(records))
        chunks.append(records)
        sources.append({'system_run': run, 'target_groups': sorted(targets), 'gate_provenance': proof})
    records = pd.concat(chunks, ignore_index=True).sort_values('cache_index').reset_index(drop=True)
    if records.cache_index.duplicated().any():
        raise ValueError('Repeated risk training sample')
    return records, {'training_groups': sorted(records.event_group.unique().tolist()), 'systems': sources,
                     'supervision': 'full-system event-held-out probabilities; target y on predicted-negative pixels'}


def on_device(batch):
    return {key: value.cuda(non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}


def forward_features(batch, kind):
    z = evidence_features(batch['x'], batch['sar'], batch['fusion'], batch['coverage'])
    return omission_features(z, batch['system'], batch['gate']) if kind == 'risk' else z


def loss_for(batch, logits, kind, temperature):
    valid = batch['valid']
    if kind == 'risk':
        valid = valid & (batch['system'] < .5)
        loss = nn.functional.binary_cross_entropy_with_logits(logits.float(), batch['y'], reduction='none')
    elif kind == 'value':
        target = (batch['sar']-batch['y']).square() - (batch['fusion']-batch['y']).square()
        loss = (logits.float().tanh()-target.detach()).square()
    elif kind == 'ordinary_gate':
        probability, _, _ = route(logits, batch['sar'], batch['fusion'], batch['coverage'][:,1], kind, temperature)
        loss = nn.functional.binary_cross_entropy(probability.clamp(1e-7,1-1e-7), batch['y'], reduction='none')
    elif kind == 'stacker':
        loss = nn.functional.binary_cross_entropy_with_logits(logits.float(), batch['y'], reduction='none')
    else:
        raise ValueError(kind)
    count = valid.sum()
    return (loss*valid).sum()/count.clamp_min(1), int(count)


@torch.no_grad()
def predict(model, records, config, out):
    if set(records.event_group).intersection(config['source_provenance']['training_groups']):
        raise ValueError('Head prediction event overlaps its supervised training events')
    loader = DataLoader(ReviewDataset(records), batch_size=config['batch_size'], num_workers=0, pin_memory=True)
    maps = {name: np.lib.format.open_memmap(out/f'validation_{name}.npy',mode='w+',dtype=np.float32,
                                           shape=(len(records),256,256)) for name in ['probabilities','gates','values']}
    model.eval(); cursor=0; metrics=Metrics()
    for batch in loader:
        batch=on_device(batch)
        with torch.autocast('cuda',dtype=torch.float16):
            logits=model(forward_features(batch,config['kind']))
        p,g,v=route(logits,batch['sar'],batch['fusion'],batch['coverage'][:,1],config['kind'],config['temperature'])
        metrics.add(p,batch['y'],batch['valid'],batch['area'])
        count=len(p)
        for name,value in [('probabilities',p),('gates',g),('values',v if v is not None else torch.zeros_like(p))]:
            maps[name][cursor:cursor+count]=value.cpu().numpy()
        cursor+=count
    for value in maps.values():value.flush()
    records.to_csv(out/'prediction_records.csv',index=False)
    result={'status':'complete','metrics':metrics.result(),'samples':len(records),'target_groups':sorted(records.event_group.unique().tolist()),
            'core_test_evaluated':False,'completed_at_utc':datetime.now(timezone.utc).isoformat()}
    atomic_json(out/'prediction_complete.json',result)
    return result


def run(args):
    spec_path=Path(args.spec).resolve()
    spec=json.loads(spec_path.read_text())
    if args.kind=='risk':
        records,proof=risk_records(spec['system_runs'])
    else:
        records,proof=assemble(spec['pairs'],spec.get('excluded_groups',[]))
    out=ROOT/'runs'/args.run_id;out.mkdir(parents=True,exist_ok=True)
    spec_hash=hashlib.sha256(spec_path.read_bytes()).hexdigest()
    if (out/'complete.json').exists():
        previous=json.loads((out/'config.json').read_text())
        if previous['spec_sha256']!=spec_hash or previous['kind']!=args.kind:
            raise ValueError('Run ID already has a different specification')
        print('Already complete',args.run_id,flush=True);return
    lock=ROOT/'runs/gpu_training.lock'
    with lock.open('x') as stream:stream.write(json.dumps({'pid':os.getpid(),'run_id':args.run_id}))
    try:
        random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
        torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        rng=random.Random(args.seed)
        model=SpatialHead(16 if args.kind=='risk' else 13).cuda()
        loader=DataLoader(ReviewDataset(records),batch_size=args.batch_size,shuffle=True,num_workers=0,pin_memory=True,
                          generator=torch.Generator().manual_seed(args.seed))
        config=vars(args).copy();config.update(source_provenance=proof,spec_sha256=spec_hash,samples=len(records),
            parameters=sum(p.numel() for p in model.parameters()),checkpoint_selection='fixed final epoch',
            source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                           [Path(__file__),ROOT/'models/review_heads.py',ROOT/'experiments/review_dataset.py',ROOT/'experiments/crossfit_records.py',ROOT/'experiments/train_baseline.py']},
            input_age_used=False,reason_age_omitted='Scene-to-sample acquisition dates are not independently authenticated',
            probability_calibration_claim=False)
        atomic_json(out/'config.json',config);records.to_csv(out/'training_records.csv',index=False)
        (out/'input_spec.json').write_bytes(spec_path.read_bytes())
        for relative,digest in config['source_sha256'].items():
            content=(ROOT/relative).read_bytes()
            if hashlib.sha256(content).hexdigest()!=digest:raise ValueError('Source changed while starting run')
            snapshot=out/'source_snapshot'/relative;snapshot.parent.mkdir(parents=True,exist_ok=True);snapshot.write_bytes(content)
        optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
        scaler=torch.amp.GradScaler('cuda');history=[];start=time.monotonic()
        for epoch in range(1,args.epochs+1):
            model.train();total=0.;count=0;epoch_start=time.monotonic()
            for step,batch in enumerate(loader,1):
                batch=augment_batch(on_device(batch),rng)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.float16):logits=model(forward_features(batch,args.kind))
                loss,n=loss_for(batch,logits,args.kind,args.temperature)
                if not torch.isfinite(loss):raise ValueError('Non-finite head training loss')
                if n:
                    scaler.scale(loss).backward();scaler.unscale_(optimizer);nn.utils.clip_grad_norm_(model.parameters(),1.)
                    scaler.step(optimizer);scaler.update()
                total+=float(loss)*n;count+=n
                if step%100==0:
                    atomic_json(out/'status.json',{'status':'training','pid':os.getpid(),'epoch':epoch,'step':step,'steps_per_epoch':len(loader)})
            row={'epoch':epoch,'loss':total/max(count,1),'supervised_pixels':count,'seconds':time.monotonic()-epoch_start}
            history.append(row);pd.DataFrame(history).to_csv(out/'history.csv',index=False)
            torch.save({'model':model.state_dict(),'config':config,'epoch':epoch},out/'last.pt')
            print(json.dumps(row),flush=True)
            atomic_json(out/'status.json',{'status':'training','pid':os.getpid(),**row})
        prediction=None
        if spec.get('predict_pairs'):
            if args.kind=='risk':raise ValueError('Risk head prediction uses the separate locked-test evaluator')
            targets,_=assemble(spec['predict_pairs'])
            prediction=predict(model,targets,config,out)
        result={'status':'complete','run_id':args.run_id,'kind':args.kind,'epochs':args.epochs,'samples':len(records),
                'elapsed_seconds':time.monotonic()-start,'peak_gpu_bytes':torch.cuda.max_memory_allocated(),
                'prediction':prediction,'core_test_evaluated':False,'completed_at_utc':datetime.now(timezone.utc).isoformat()}
        atomic_json(out/'complete.json',result);atomic_json(out/'status.json',result);print(json.dumps(result),flush=True)
    except BaseException as exc:
        atomic_json(out/'status.json',{'status':'failed','pid':os.getpid(),'error':repr(exc)});raise
    finally:
        lock.unlink(missing_ok=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',required=True);parser.add_argument('--spec',required=True)
    parser.add_argument('--kind',choices=['value','ordinary_gate','stacker','risk'],required=True)
    parser.add_argument('--epochs',type=int,default=10);parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--seed',type=int,default=17);parser.add_argument('--lr',type=float,default=.001)
    parser.add_argument('--temperature',type=float,default=.05)
    run(parser.parse_args())
