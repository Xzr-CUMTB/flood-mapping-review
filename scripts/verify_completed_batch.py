"""Post-run verification without rewriting any file bound into the frozen release."""
import os
os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, sys
import numpy as np
import pandas as pd
import psutil
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_completed_training import planned_jobs, audit
from experiments.infer_frozen_core import check_release, sha
from scripts.project_inventory import count_bytes


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    torch.set_num_threads(2)
    release_path = ROOT/'configs/core_release_s17.json'
    release = check_release(release_path)
    release_hash = sha(release_path)
    print('Frozen source, data, weights and snapshots verified', flush=True)
    original_audit_path = ROOT/'reports/training_artifact_audit.json'
    original_audit_hash = sha(original_audit_path)
    original_audit = read(original_audit_path)
    assert original_audit['passed'] and original_audit['verified_jobs'] == 38
    plan, jobs = planned_jobs()
    manifest = pd.read_csv(ROOT/'data/processed/sen2gf3/manifest.csv')
    dev = manifest[manifest.eligible & manifest.spatial_nonoverlap & manifest.role.eq('development')]
    verified = [audit(job, dev) for job in jobs]
    assert len(verified) == 38 and verified == original_audit['verified']
    assert sha(original_audit_path) == original_audit_hash
    print('All 38 actual checkpoints reverified; original pre-test audit preserved', flush=True)
    calibration = read(ROOT/'runs/oof_calibration_s17/complete.json')
    assert calibration['status'] == 'complete' and calibration['core_test_evaluated'] is False
    assert calibration['samples'] == 8786
    for value in calibration['temperatures'].values():
        assert np.isfinite(value['temperature']) and value['temperature'] > 0
    protocol = read(ROOT/'configs/evaluation_protocol_v1.json')
    core = manifest[manifest.eligible & manifest.spatial_nonoverlap & manifest.event_group.eq(protocol['core_group'])]
    assert len(core) == 8233 and core.role.eq('locked_test').all()
    output_proofs = []
    for folder in ['core_s17_predictions', 'core_s17_foundation_predictions']:
        directory = ROOT/'runs'/folder
        complete = read(directory/'complete.json')
        assert complete['status'] == 'complete' and complete['release_sha256'] == release_hash
        assert complete['patches'] == len(core)
        frame = pd.read_csv(directory/'prediction_manifest.csv')
        assert frame.sample_id.tolist() == core.sample_id.tolist()
        assert frame.cache_index.tolist() == core.cache_index.tolist()
        for name in complete['models']:
            path = directory/(name+'.npy')
            array = np.load(path, mmap_mode='r')
            assert array.shape == (len(core), 256, 256) and array.dtype == np.float32
            minimum, maximum = 1., 0.
            for start in range(0, len(array), 64):
                block = np.asarray(array[start:start+64])
                assert np.isfinite(block).all()
                lo, hi = float(block.min()), float(block.max())
                assert 0 <= lo <= hi <= 1
                minimum, maximum = min(minimum, lo), max(maximum, hi)
            output_proofs.append({'path':str(path.relative_to(ROOT)), 'shape':list(array.shape),
                                 'dtype':str(array.dtype), 'minimum':minimum, 'maximum':maximum,
                                 'bytes':path.stat().st_size, 'sha256':sha(path)})
            del array
        print(folder+' complete: all saved probability values verified', flush=True)
    out = ROOT/'runs/core_s17_evaluation'
    complete = read(out/'complete.json')
    assert complete['status'] == 'complete' and complete['release_sha256'] == release_hash
    assert complete['patches'] == 8233 and sum(complete['regions'].values()) == 8233
    maps = pd.read_csv(out/'map_metrics.csv')
    budgets = pd.read_csv(out/'review_budget_results.csv')
    bands = pd.read_csv(out/'review_simultaneous_intervals.csv')
    points = pd.read_csv(out/'matched_operating_point_metrics.csv')
    summary = read(ROOT/'reports/core_result_summary.json')
    report = (ROOT/'reports/13_core_computational_results.md').read_text(encoding='utf-8')
    assert summary['status'] == 'complete' and summary['core_completion'] == complete
    assert len(maps) == summary['map_rows'] == 32
    assert len(budgets) == summary['budget_rows'] == 1824
    assert len(bands) == summary['simultaneous_rows'] == 24
    assert len(points) == 12
    assert not maps.duplicated(['region','model']).any()
    assert not budgets.duplicated(['region','model','policy','budget_fraction']).any()
    for table in [maps, budgets, bands, points]:
        assert np.isfinite(table.select_dtypes(include='number').to_numpy()).all()
    assert (budgets.actual_area_fraction <= budgets.budget_fraction+1e-9).all()
    for endpoint in ['fn','fp']:
        assert (budgets['residual_'+endpoint+'_km2'] >= 0).all()
        assert (budgets['residual_'+endpoint+'_km2'] <= budgets['initial_'+endpoint+'_km2']+1e-8).all()
    for row in maps.itertuples():
        line = f'| {row.region} | {row.model} | {row.flood_iou:.4f} | {row.fn_km2:.2f} | {row.fp_km2:.2f} | {row.brier:.5f} | {row.confident_omission_01:.4f} |'
        assert line in report
    for row in bands.itertuples():
        subset = budgets[budgets.region.eq(row.region) & budgets.budget_fraction.eq(row.budget_fraction)]
        a = subset[subset.model.eq(row.first_model) & subset.policy.eq(row.first_policy)]
        b = subset[subset.model.eq(row.second_model) & subset.policy.eq(row.second_policy)]
        assert len(a) == len(b) == 1
        difference = float(a.iloc[0][row.endpoint]-b.iloc[0][row.endpoint])
        assert np.isclose(difference, row.difference_first_minus_second, atol=1e-8, rtol=1e-8)
        assert row.simultaneous_lower95 <= difference <= row.simultaneous_upper95
        line = f'| {row.region} | {row.first_model}/{row.first_policy} | {row.second_model}/{row.second_policy} | {row.endpoint} | {row.difference_first_minus_second:.2f} | [{row.simultaneous_lower95:.2f}, {row.simultaneous_upper95:.2f}] |'
        assert line in report
    for row in points.itertuples():
        line = f'| {row.region} | {row.scenario} | {row.model} | {row.threshold:.5f} | {row.false_positive_rate:.5f} | {row.recall:.4f} | {row.fn_km2:.2f} | {row.fp_km2:.2f} |'
        assert line in report
    print('Result CSV arithmetic and every reported table row verified', flush=True)
    queues = {}
    for name in ['nested','foundation','post_training','ensemble','core_completion']:
        directory = ROOT/'runs'/(name+'_queue')
        queues[name] = read(directory/'status.json')
        assert queues[name]['stage'] == 'completed' and not (directory/'queue.lock').exists()
    assert not (ROOT/'runs/gpu_training.lock').exists()
    active = []
    names = {'run_nested_experiments.py','run_foundation_experiments.py','run_post_training_checks.py',
             'run_ensemble_controls.py','run_core_completion.py','train_baseline.py','train_foundation.py',
             'fit_review_head.py','fit_fusion_risk.py','infer_frozen_core.py','infer_foundation_core.py',
             'evaluate_frozen_core.py'}
    for process in psutil.process_iter(['pid','name','cmdline']):
        try:
            if 'python' in (process.info['name'] or '').lower() and any(Path(x).name in names for x in (process.info['cmdline'] or [])):
                if Path(process.cwd()).resolve() == ROOT.resolve():
                    active.append(process.info)
        except psutil.NoSuchProcess:
            continue
    assert not active
    occupied = count_bytes(ROOT)
    assert occupied < 100_000_000_000
    result = {'passed':True,'checked_at_utc':datetime.now(timezone.utc).isoformat(),
              'batch_completed_at_utc':queues['core_completion']['utc'], 'training_jobs':38,
              'release_sha256':release_hash, 'frozen_files_verified':len(release['sha256']),
              'pre_test_audit_preserved':True, 'training_checkpoint_rechecks':len(verified),
              'prediction_files':output_proofs, 'map_rows':len(maps), 'budget_rows':len(budgets),
              'simultaneous_rows':len(bands), 'operating_point_rows':len(points),
              'all_reported_table_rows_match_csv':True,'active_experiment_processes':active,
              'workspace_bytes':occupied,'submission_ready_assessed':False,
              'csv_sha256':{p.name:sha(p) for p in out.glob('*.csv')},
              'report_sha256':sha(ROOT/'reports/13_core_computational_results.md'),
              'verifier_sha256':sha(Path(__file__))}
    destination = ROOT/'reports/completed_batch_verification.json'
    temporary = destination.with_suffix('.tmp')
    temporary.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    temporary.replace(destination)
    print(json.dumps({k:result[k] for k in ['passed','training_jobs','workspace_bytes','batch_completed_at_utc']}),flush=True)


if __name__ == '__main__':
    main()
