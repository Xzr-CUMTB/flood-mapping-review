"""Fixed single-GPU TerraMind adaptation, using the same audited data and masked BCE."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json, os, random, sys, time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.foundation_environment import configure
configure()
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from experiments.train_baseline import FloodDataset, atomic_json, evaluate
from models.terramind_flood import TerraMindFlood


def run(args):
    data = ROOT/'data/processed/sen2gf3'
    proof = json.loads((ROOT/'reports/terramind_integration_checks.json').read_text())
    model_path = ROOT/'models/terramind_flood.py'
    if not proof['passed'] or proof['model_source_sha256'] != hashlib.sha256(model_path.read_bytes()).hexdigest():
        raise ValueError('Reviewed TerraMind input and weight integration required')
    pretrained_hash=None
    if not args.random_init:
        expected=json.loads((ROOT/'configs/foundation_experiments_s17.json').read_text())['pretrained_weight_sha256']
        pretrained_hash=hashlib.sha256((ROOT/'models/pretrained/terramind_tiny/TerraMind_v1_tiny.pt').read_bytes()).hexdigest()
        if pretrained_hash != expected:raise ValueError('Pretrained weight file changed after verification')
    if not (data/'split_audit_passed.json').exists():
        raise ValueError('Audited spatial splits required')
    frame = pd.read_csv(data/'manifest.csv').set_index('cache_index', drop=False)
    available = frame[frame.eligible & frame.spatial_nonoverlap & frame.role.eq('development')]
    train_groups = args.train_groups.split(',')
    val_groups = [] if args.no_validation else args.val_groups.split(',')
    if '2023_holdout' in train_groups+val_groups or set(train_groups) & set(val_groups):
        raise ValueError('Training/validation group leakage')
    train = available[available.event_group.isin(train_groups)]
    val = available[available.event_group.isin(val_groups)]
    if not len(train) or (not args.no_validation and not len(val)):
        raise ValueError('Empty declared split')
    if set(train.input_sha256) & set(val.input_sha256) or set(train.bounds_key) & set(val.bounds_key):
        raise ValueError('Repeated inputs between splits')
    out = ROOT/'runs'/args.run_id
    out.mkdir(parents=True, exist_ok=True)
    if (out/'complete.json').exists():
        config = json.loads((out/'config.json').read_text())
        for key, value in vars(args).items():
            if config.get(key) != value:
                raise ValueError('A completed run uses different settings: '+key)
        print('Already complete', args.run_id, flush=True)
        return
    lock = ROOT/'runs/gpu_training.lock'
    with lock.open('x') as stream:
        stream.write(json.dumps({'pid': os.getpid(), 'run_id': args.run_id}))
    try:
        random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
        torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        model = TerraMindFlood(pretrained=not args.random_init).cuda()
        config = {**vars(args), 'model': 'TerraMind 1.0 tiny encoder with new GF3/NIR stems and a four-layer feature decoder',
            'train_samples': len(train), 'val_samples': len(val), 'parameters': sum(p.numel() for p in model.parameters()),
            'pretrained_loading': model.pretrained_loading, 'selection': 'fixed last epoch; no validation selection',
            'pretrained_weight_sha256':pretrained_hash,
            'manifest_sha256': hashlib.sha256((data/'manifest.csv').read_bytes()).hexdigest(),
            'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0),
            'normalization': {'RGB': 'Pinned pretrained mean/std on original 0-255 R,G,B',
                              'GF3_HH_HV_and_NIR': 'New stems, uint8/127.5 - 1; not physical backscatter or reflectance'},
            'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in [Path(__file__), model_path, ROOT/'experiments/foundation_environment.py', ROOT/'experiments/train_baseline.py']}}
        atomic_json(out/'config.json', config)
        for relative, expected in config['source_sha256'].items():
            content = (ROOT/relative).read_bytes()
            if hashlib.sha256(content).hexdigest() != expected:
                raise ValueError('Source changed during initialization')
            target = out/'source_snapshot'/relative
            target.parent.mkdir(parents=True, exist_ok=True);target.write_bytes(content)
        train[['sample_id','event_group','bounds_key']].to_csv(out/'train_manifest.csv', index=False)
        val[['sample_id','event_group','bounds_key']].to_csv(out/'validation_manifest.csv', index=False)
        loader = DataLoader(FloodDataset(train.index, augment=True, modality='fusion'), batch_size=args.batch_size,
            shuffle=True, num_workers=0, pin_memory=True, generator=torch.Generator().manual_seed(args.seed))
        val_loader = DataLoader(FloodDataset(val.index, modality='fusion'), batch_size=args.batch_size,
            num_workers=0, pin_memory=True) if len(val) else None
        optimizer = torch.optim.AdamW(model.optimizer_groups(args.backbone_lr, args.adaptation_lr), weight_decay=1e-4)
        scaler = torch.amp.GradScaler('cuda')
        history = [];start = time.monotonic()
        for epoch in range(1, args.epochs+1):
            model.train();total = 0.;count = 0;epoch_start = time.monotonic()
            for step, batch in enumerate(loader, 1):
                x, y, valid = [batch[key].cuda(non_blocking=True) for key in ['x','y','valid']]
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.float16):
                    logits = model(x)
                    loss = (nn.functional.binary_cross_entropy_with_logits(logits, y, reduction='none')*valid).sum()/valid.sum()
                if not torch.isfinite(loss):
                    raise ValueError('Non-finite foundation training loss')
                scaler.scale(loss).backward();scaler.unscale_(optimizer);nn.utils.clip_grad_norm_(model.parameters(), 1.)
                scaler.step(optimizer);scaler.update()
                n = int(valid.sum());total += float(loss)*n;count += n
                if step % 100 == 0:
                    progress = {'status':'training','pid':os.getpid(),'epoch':epoch,'step':step,'steps_per_epoch':len(loader),
                                'loss':total/count,'peak_gpu_bytes':torch.cuda.max_memory_allocated()}
                    atomic_json(out/'status.json', progress);print(json.dumps(progress), flush=True)
            row = {'epoch':epoch,'train_bce':total/count,'seconds':time.monotonic()-epoch_start}
            history.append(row);pd.DataFrame(history).to_csv(out/'history.csv', index=False)
            torch.save({'model':model.state_dict(),'config':config,'epoch':epoch,'optimizer':optimizer.state_dict(),
                        'scaler':scaler.state_dict()},out/'last.pt')
            atomic_json(out/'status.json', {'status':'training','pid':os.getpid(),**row});print(json.dumps(row), flush=True)
        model.eval()
        metrics = evaluate(model, val_loader, torch.device('cuda'), out, frame) if val_loader is not None else {}
        result = {'status':'complete','run_id':args.run_id,'selected_epoch':args.epochs,'metrics':metrics,
            'elapsed_seconds':time.monotonic()-start,'peak_gpu_bytes':torch.cuda.max_memory_allocated(),
            'completed_at_utc':datetime.now(timezone.utc).isoformat(),'core_test_evaluated':False}
        atomic_json(out/'complete.json',result);atomic_json(out/'status.json',result);print(json.dumps(result),flush=True)
    except BaseException as exc:
        atomic_json(out/'status.json',{'status':'failed','pid':os.getpid(),'error':repr(exc)});raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--train-groups',default='2020_flood_season,2021_Henan')
    parser.add_argument('--val-groups',default='2024_April,2024_June')
    parser.add_argument('--no-validation',action='store_true');parser.add_argument('--random-init',action='store_true')
    parser.add_argument('--epochs',type=int,default=20);parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--seed',type=int,default=17);parser.add_argument('--backbone-lr',type=float,default=1e-4)
    parser.add_argument('--adaptation-lr',type=float,default=1e-3)
    run(parser.parse_args())
