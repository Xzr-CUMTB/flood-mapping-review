"""Fit final positive temperatures using completed development OOF maps only."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, sys, os
# CPU calibration shares a process with Torch utilities. Use MKL's sequential backend
# so Conda MKL does not initialize a second Intel OpenMP runtime beside Torch's copy.
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import logit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.fit_review_head import risk_records
from experiments.train_baseline import atomic_json
from experiments.operating_points import match_positive_rate


def main():
    out = ROOT/'runs/oof_calibration_s17'
    out.mkdir(parents=True, exist_ok=True)
    spec = json.loads((ROOT/'configs/crossfit_seed17/final_risk.json').read_text())
    records, proof = risk_records(spec['system_runs'])
    manifest = ROOT/'data/processed/sen2gf3/manifest.csv'
    frame = pd.read_csv(manifest)
    expected = set(frame.loc[frame.eligible & frame.spatial_nonoverlap & frame.role.eq('development'), 'cache_index'])
    if set(records.cache_index) != expected:
        raise ValueError('OOF calibration requires every eligible development sample exactly once')
    data = ROOT/'data/processed/sen2gf3'
    labels = np.load(data/'labels_uint8.npy', mmap_mode='r')
    support = np.load(data/'support_uint8.npy', mmap_mode='r')
    areas = np.load(data/'row_area_m2.npy', mmap_mode='r')
    sources, sampled = {}, {key: [] for key in ['sar', 'fusion', 'value_gate', 'y', 'weight']}
    for row in records.itertuples():
        idx = int(row.cache_index)
        valid = np.flatnonzero(np.asarray(support[idx]).ravel() == 7)
        chosen = np.random.default_rng(17000+idx).choice(valid, min(1024, len(valid)), replace=False)
        sampled['y'].append((np.asarray(labels[idx]).ravel()[chosen] == 1).astype('float64'))
        sampled['weight'].append(np.asarray(areas[idx], dtype='float64')[chosen//256]*len(valid)/len(chosen))
        for name, run, j in [('sar', row.sar_run, row.prediction_index),
                             ('fusion', row.fusion_run, row.prediction_index),
                             ('value_gate', row.system_run, row.system_prediction_index)]:
            if run not in sources:
                sources[run] = np.load(ROOT/'runs'/run/'validation_probabilities.npy', mmap_mode='r')
            probability = np.asarray(sources[run][int(j)]).ravel()[chosen].astype('float32')
            if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
                raise ValueError('Invalid OOF probabilities')
            sampled[name].append(probability)
    sampled = {key: np.concatenate(value) for key, value in sampled.items()}
    logits={name:logit(np.clip(sampled[name].astype('float64'),1e-7,1-1e-7)) for name in ['sar','fusion','value_gate']}
    weights = sampled['weight']/sampled['weight'].sum()
    fits = {}
    for name in ['sar', 'fusion', 'value_gate']:
        def loss(log_temperature):
            z = logits[name]/np.exp(log_temperature)
            return float(np.dot(weights, np.logaddexp(0, z)-sampled['y']*z))
        fit = minimize_scalar(loss, bounds=(-3, 3), method='bounded', options={'xatol': 1e-6})
        if not fit.success:
            raise ValueError('Temperature optimization failed')
        fits[name] = {'temperature': float(np.exp(fit.x)), 'fit_nll': float(fit.fun),
                      'raw_nll': loss(0), 'log_temperature_bounds': [-3, 3],
                      'near_bound': bool(abs(fit.x) > 2.99)}
    records.to_csv(out/'source_records.csv', index=False)
    y=sampled['y'].astype(bool)
    fusion_positive=sampled['fusion']>=.5
    targets={'matched_fpr':float(np.dot(weights[~y],fusion_positive[~y])/weights[~y].sum()),
             'matched_recall':float(np.dot(weights[y],fusion_positive[y])/weights[y].sum())}
    operating_points={}
    for scenario,target in targets.items():
        selected=~y if scenario=='matched_fpr' else y
        operating_points[scenario]={}
        for name in ['sar','fusion','value_gate']:
            if name=='fusion':
                point={'threshold':.5,'achieved_rate':target,'target_rate':target,'absolute_rate_gap':0.,
                       'rule':'Reference fusion threshold 0.5'}
            else:
                point=match_positive_rate(sampled[name][selected],weights[selected],target)
            operating_points[scenario][name]=point
    result = {'status': 'complete', 'completed_at_utc': datetime.now(timezone.utc).isoformat(),
              'manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest(),
              'source_provenance': proof, 'temperatures': fits, 'samples': len(records),
              'operating_points':operating_points,
              'operating_point_scope':'Thresholds match the fusion 0.5 FPR or recall on sampled, weighted development OOF data; core rates are not forced to match',
              'sampled_pixels': len(sampled['y']), 'sampling_seed': '17000 + cache_index',
              'sampling': 'Up to 1024 common-valid pixels per patch, inverse-sampling geodesic area weights',
              'interpretation': 'Final calibration fit on development event-held-out maps; fit NLL is not an independent evaluation',
              'risk_head_calibrated': False, 'core_test_evaluated': False,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    atomic_json(out/'complete.json', result)
    print(json.dumps({'temperatures': fits, 'samples': len(records), 'sampled_pixels': len(sampled['y'])}), flush=True)


if __name__ == '__main__':
    main()
