"""One locked-test inference pass for the frozen seed-17 pipeline; no selection."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json, os, shutil, sys, time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.train_baseline import FloodDataset, atomic_json
from models.flood_unet import FloodUNet
from models.review_heads import SpatialHead, evidence_features, omission_features, route
from scripts.project_inventory import count_bytes
from experiments.evaluation_metrics import tile_sum

EXPERT_RUNS = {'sar': 'final_s17_sar', 'fusion': 'final_s17_fusion',
              'rgb_fusion': 'final_s17_rgb_fusion', 'optical': 'final_s17_optical',
              'fusion_seed29': 'final_s29_fusion', 'fusion_seed43': 'final_s43_fusion'}
EXPERTS = list(EXPERT_RUNS)
HEADS = ['value', 'ordinary_gate', 'stacker', 'risk', 'fusion_risk']
NAMES = EXPERTS+['value_gate', 'ordinary_gate', 'stacker', 'risk', 'fusion_risk']


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def check_release(path):
    release = json.loads(path.read_text())
    if release.get('status') != 'frozen_for_core_evaluation':
        raise ValueError('A frozen release record is required')
    for relative, expected in release['sha256'].items():
        if sha(ROOT/relative) != expected:
            raise ValueError('Frozen file changed: '+relative)
    for relative, snapshot_path in release.get('source_snapshot',{}).items():
        if sha(ROOT/snapshot_path)!=release['sha256'][relative]:
            raise ValueError('Frozen source snapshot changed: '+relative)
    return release


@torch.no_grad()
def run(args):
    release_path = Path(args.release).resolve()
    release = check_release(release_path)
    if (ROOT/'runs/nested_queue/queue.lock').exists():
        raise ValueError('Finish the training controller before core inference')
    protocol = json.loads((ROOT/'configs/evaluation_protocol_v1.json').read_text())
    data = ROOT/'data/processed/sen2gf3'
    frame = pd.read_csv(data/'manifest.csv')
    frame = frame[frame.eligible & frame.spatial_nonoverlap & frame.event_group.eq(protocol['core_group'])].copy()
    if not len(frame) or not frame.role.eq('locked_test').all():
        raise ValueError('Expected the frozen eligible core-test group')
    expected_bytes = len(frame)*(256*256*4*len(NAMES)+16*5*8)
    out = ROOT/'runs/core_s17_predictions'
    if (out/'complete.json').exists():
        previous = json.loads((out/'complete.json').read_text())
        if previous['release_sha256'] != sha(release_path):
            raise ValueError('Completed core outputs belong to another release')
        print('Core predictions already completed', flush=True)
        return
    occupied = count_bytes(ROOT)
    replaceable=sum(path.stat().st_size for path in [out/(name+'.npy') for name in NAMES]+[out/'router_unit_features.npy'] if path.exists())
    extra_bytes=max(expected_bytes-replaceable,0)
    if occupied+extra_bytes > 95_000_000_000 or shutil.disk_usage(ROOT).free < extra_bytes+5_000_000_000:
        raise ValueError('Core inference would exceed the storage reserve')
    lock = ROOT/'runs/gpu_training.lock'
    out.mkdir(parents=True, exist_ok=True)
    run_release=out/'run_release.json'
    if run_release.exists() and json.loads(run_release.read_text())['release_sha256'] != sha(release_path):
        raise ValueError('Incomplete core outputs belong to another release')
    atomic_json(run_release,{'release_sha256':sha(release_path),'restart_rule':'Recompute incomplete prediction files from the same frozen release'})
    with lock.open('x') as stream:
        stream.write(json.dumps({'pid': os.getpid(), 'run_id': 'core_s17_predictions'}))
    try:
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        models = {}
        for name in EXPERTS+HEADS:
            directory = ROOT/'runs'/EXPERT_RUNS.get(name, 'final_s17_'+name)
            config = json.loads((directory/'config.json').read_text())
            model = FloodUNet(config['modality'], config['width']) if name in EXPERTS else SpatialHead(16 if name in ['risk', 'fusion_risk'] else 13)
            checkpoint = torch.load(directory/'last.pt', map_location='cpu', weights_only=False)
            model.load_state_dict(checkpoint['model'], strict=True)
            models[name] = model.cuda().eval()
        raw = FloodDataset(frame.cache_index, modality='fusion')
        loader = DataLoader(raw, batch_size=args.batch_size, num_workers=0, pin_memory=True)
        maps = {name: np.lib.format.open_memmap(out/(name+'.npy'), mode='w+', dtype=np.float32,
                                              shape=(len(frame), 256, 256)) for name in NAMES}
        router_fields=['mean_predicted_brier_value','mean_gate','gate_below_01_fraction',
                       'gate_above_09_fraction','mean_expert_disagreement']
        router_units=np.lib.format.open_memmap(out/'router_unit_features.npy',mode='w+',dtype=np.float64,
                                               shape=(len(frame),16,len(router_fields)))
        atomic_json(out/'router_unit_schema.json',{'fields':router_fields,'unit_order':'4x4 row-major 64x64',
            'weighting':'Common-valid geodesic area','test_labels_used':False})
        frame['prediction_index'] = np.arange(len(frame))
        frame.to_csv(out/'prediction_manifest.csv', index=False)
        start, cursor = time.monotonic(), 0
        for step, batch in enumerate(loader, 1):
            x = batch['x'].cuda(non_blocking=True)
            idx = batch['index'].numpy()
            bits = np.asarray(raw.support[idx])
            coverage = torch.from_numpy(np.stack([(bits & 2) > 0, (bits & 4) > 0], axis=1).astype('float32')).cuda()
            with torch.autocast('cuda', dtype=torch.float16):
                probabilities = {name: model(x).float().sigmoid() for name, model in models.items() if name in EXPERTS}
                z = evidence_features(x, probabilities['sar'], probabilities['fusion'], coverage)
                value_logits, ordinary_logits = models['value'](z), models['ordinary_gate'](z)
                stacker = models['stacker'](z).float().sigmoid()
            p, g, v = route(value_logits, probabilities['sar'], probabilities['fusion'], coverage[:, 1], 'value', .05)
            ordinary, _, _ = route(ordinary_logits, probabilities['sar'], probabilities['fusion'], coverage[:, 1], 'ordinary_gate', .05)
            with torch.autocast('cuda', dtype=torch.float16):
                risk = models['risk'](omission_features(z, p, g)).float().sigmoid()
                fusion_risk = models['fusion_risk'](omission_features(z, probabilities['fusion'], torch.ones_like(g))).float().sigmoid()
            probabilities.update(value_gate=p, ordinary_gate=ordinary, stacker=stacker, risk=risk, fusion_risk=fusion_risk)
            n = len(x)
            for name, probability in probabilities.items():
                array = probability.cpu().numpy()
                if not np.isfinite(array).all() or np.any((array < 0) | (array > 1)):
                    raise ValueError('Invalid probabilities: '+name)
                maps[name][cursor:cursor+n] = array
            ga,va=g.cpu().numpy(),v.cpu().numpy()
            disagreement=(probabilities['sar']-probabilities['fusion']).abs().cpu().numpy()
            row_area=batch['area'].numpy()
            for k in range(n):
                weight=np.broadcast_to(row_area[k,:,None].astype(np.float64),(256,256))*(bits[k]==7)
                cost=tile_sum(weight)
                for column,field in enumerate([va[k],ga[k],ga[k]<.1,ga[k]>.9,disagreement[k]]):
                    router_units[cursor+k,:,column]=np.divide(tile_sum(weight*field),cost,
                        out=np.full(16,np.nan),where=cost>0)
            cursor += n
            if step % 50 == 0:
                progress = {'status': 'predicting', 'pid': os.getpid(), 'patches': cursor, 'total': len(frame),
                            'seconds': time.monotonic()-start, 'test_metrics_used_for_selection': False}
                atomic_json(out/'status.json', progress)
                print(json.dumps(progress), flush=True)
        for array in maps.values():
            array.flush()
        router_units.flush()
        result = {'status': 'complete', 'patches': cursor, 'models': NAMES, 'release_sha256': sha(release_path),
                  'prediction_dtype': 'float32', 'manifest_sha256': release['sha256']['data/processed/sen2gf3/manifest.csv'],
                  'completed_at_utc': datetime.now(timezone.utc).isoformat(), 'elapsed_seconds': time.monotonic()-start,
                  'peak_gpu_bytes': torch.cuda.max_memory_allocated(), 'checkpoint_selection_on_test': False,
                  'risk_interpretation': 'Conditional omission score trained on OOF predicted-negative pixels; not calibrated'}
        atomic_json(out/'complete.json', result)
        atomic_json(out/'status.json', result)
        print(json.dumps(result), flush=True)
    except BaseException as exc:
        atomic_json(out/'status.json', {'status': 'failed', 'pid': os.getpid(), 'error': repr(exc)})
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--release', default=str(ROOT/'configs/core_release_s17.json'))
    parser.add_argument('--batch-size', type=int, default=16)
    run(parser.parse_args())
