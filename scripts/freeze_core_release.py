"""Bind completed training, calibration and evaluation code before core predictions."""
from pathlib import Path
from datetime import datetime, timezone
import json, sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.infer_frozen_core import sha
from experiments.train_baseline import atomic_json


def main():
    release_path = ROOT/'configs/core_release_s17.json'
    if release_path.exists():
        from experiments.infer_frozen_core import check_release
        check_release(release_path)
        print('Existing frozen release verified; no changes made', flush=True)
        return
    review_path = ROOT/'reports/core_release_review_s17.json'
    review = json.loads(review_path.read_text())
    if not review.get('passed') or review.get('core_test_predictions_inspected'):
        raise ValueError('Completed pre-test review required')
    checks_path = ROOT/'reports/evaluation_arithmetic_checks.json'
    checks = json.loads(checks_path.read_text())
    if not checks.get('passed') or checks.get('core_test_predictions_read'):
        raise ValueError('Evaluation arithmetic must pass before test release')
    for relative, expected in checks['source_sha256'].items():
        if sha(ROOT/relative) != expected:
            raise ValueError('Evaluation implementation changed after its checks')
    io_checks_path=ROOT/'reports/frozen_evaluator_io_checks.json'
    io_checks=json.loads(io_checks_path.read_text())
    if not io_checks.get('passed'):raise ValueError('Full evaluator I/O check required')
    for relative, expected in io_checks['source_sha256'].items():
        if sha(ROOT/relative) != expected:raise ValueError('Evaluator changed after its I/O check')
    audit_path = ROOT/'reports/training_artifact_audit.json'
    audit = json.loads(audit_path.read_text())
    if not audit.get('passed') or audit.get('pending') or audit.get('core_test_predictions_inspected'):
        raise ValueError('Completed training artifact audit required')
    if audit['auditor_sha256'] != sha(ROOT/'scripts/audit_completed_training.py'):
        raise ValueError('Training auditor changed after its checks')
    for job in audit['verified']:
        directory = ROOT/'runs'/job['run_id']
        if (sha(directory/'config.json') != job['config_sha256']
                or sha(directory/'complete.json') != job['complete_sha256']):
            raise ValueError('Training artifacts changed after their checks')
    plan_path = ROOT/'configs/crossfit_seed17/plan.json'
    plan = json.loads(plan_path.read_text())
    if sha(ROOT/'data/processed/sen2gf3/manifest.csv') != plan['manifest_sha256']:
        raise ValueError('Frozen development manifest changed')
    paths = [plan_path, review_path, checks_path, audit_path, io_checks_path, ROOT/'scripts/audit_completed_training.py',
             ROOT/'reports/evaluation_integration_checks.json', ROOT/'configs/evaluation_protocol_v1.json',
             ROOT/'data/processed/sen2gf3/manifest.csv', ROOT/'data/processed/sen2gf3/split_audit_passed.json']
    paths.extend(ROOT/'data/processed/sen2gf3'/name for name in
                 ['x_uint8.npy','labels_uint8.npy','support_uint8.npy','row_area_m2.npy','preparation_complete.json'])
    paths.extend(ROOT/'configs'/name for name in ['environment_experiments_freeze.txt','environment_foundation_freeze.txt'])
    for job in plan['jobs']:
        directory = ROOT/'runs'/job['run_id']
        complete = json.loads((directory/'complete.json').read_text())
        config = json.loads((directory/'config.json').read_text())
        if complete.get('status') != 'complete' or complete.get('core_test_evaluated'):
            raise ValueError('All planned development jobs must complete before release')
        if config['seed'] != 17 or config['epochs'] != job['epochs']:
            raise ValueError('Training settings differ from the frozen development plan')
        if job['type'] == 'expert' and complete['selected_epoch'] != config['epochs']:
            raise ValueError('Fixed final epoch not used')
        paths.extend([directory/'config.json', directory/'complete.json'])
        if job['run_id'].startswith('final_'):
            paths.append(directory/'last.pt')
    calibration_path = ROOT/'runs/oof_calibration_s17/complete.json'
    calibration = json.loads(calibration_path.read_text())
    if calibration['status'] != 'complete' or calibration['core_test_evaluated']:
        raise ValueError('Development-only final calibration required')
    if calibration['source_sha256'] != sha(ROOT/'experiments/calibrate_oof.py'):
        raise ValueError('Calibration implementation differs from its completed fit')
    paths.append(calibration_path)
    foundation_plan_path = ROOT/'configs/foundation_experiments_s17.json'
    foundation_plan = json.loads(foundation_plan_path.read_text())
    paths.append(foundation_plan_path)
    for job in foundation_plan['jobs']:
        directory = ROOT/'runs'/job['run_id']
        completed = json.loads((directory/'complete.json').read_text())
        if completed.get('status') != 'complete' or completed.get('core_test_evaluated'):
            raise ValueError('All prescribed foundation controls must complete before release')
        paths.extend([directory/'config.json', directory/'complete.json'])
        if job['run_id'].startswith('final_'):
            paths.append(directory/'last.pt')
    directory = ROOT/'runs/final_s17_fusion_risk'
    matched_risk = json.loads((directory/'complete.json').read_text())
    if matched_risk.get('status') != 'complete' or matched_risk.get('core_test_evaluated'):
        raise ValueError('Matched fusion omission control must complete before release')
    paths.extend([directory/'config.json', directory/'complete.json', directory/'last.pt'])
    ensemble_plan_path = ROOT/'configs/ensemble_controls.json'
    ensemble_plan = json.loads(ensemble_plan_path.read_text())
    paths.append(ensemble_plan_path)
    for seed, run_id in zip(ensemble_plan['seeds'][1:], ensemble_plan['new_members']):
        directory = ROOT/'runs'/run_id
        completed = json.loads((directory/'complete.json').read_text())
        config = json.loads((directory/'config.json').read_text())
        if (completed.get('status') != 'complete' or completed.get('core_test_evaluated')
                or config['seed'] != seed or config['modality'] != 'fusion'
                or config['epochs'] != ensemble_plan['epochs']
                or completed['selected_epoch'] != ensemble_plan['epochs']):
            raise ValueError('All prescribed ensemble controls must complete with fixed final epochs')
        paths.extend([directory/'config.json', directory/'complete.json', directory/'last.pt'])
    paths.extend(ROOT/relative for relative in [
        'models/flood_unet.py', 'models/review_heads.py', 'experiments/infer_frozen_core.py',
        'experiments/evaluation_metrics.py', 'experiments/evaluate_frozen_core.py',
        'experiments/calibrate_oof.py', 'experiments/train_baseline.py', 'models/terramind_flood.py',
        'experiments/foundation_environment.py', 'experiments/infer_foundation_core.py',
        'experiments/train_foundation.py', 'experiments/fit_fusion_risk.py', 'experiments/operating_points.py',
        'scripts/freeze_core_release.py', 'scripts/verify_evaluation_arithmetic.py',
        'scripts/verify_evaluation_integration.py', 'scripts/verify_frozen_evaluator_io.py',
        'experiments/fit_review_head.py','experiments/review_dataset.py','experiments/crossfit_records.py',
        'scripts/prepare_sen2gf3.py','scripts/audit_spatial_splits.py','scripts/extract_verified_dataset.py',
        'scripts/project_inventory.py','scripts/run_nested_experiments.py','scripts/run_foundation_experiments.py',
        'scripts/run_post_training_checks.py','scripts/run_ensemble_controls.py'])
    # Additional model comparisons must be specified, trained and reviewed before release.
    for relative in review.get('additional_frozen_files', []):
        paths.append(ROOT/relative)
    result = {'status': 'frozen_for_core_evaluation', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'training_jobs': len(plan['jobs'])+len(foundation_plan['jobs'])+1+len(ensemble_plan['new_members']),
              'primary_pipeline_seed': 17, 'ensemble_control_seeds': ensemble_plan['seeds'], 'selection_on_test': False,
              'sha256': {path.relative_to(ROOT).as_posix(): sha(path) for path in paths},
              'additional_comparisons': review.get('additional_comparisons', []),
              'interpretation': 'Primary pipeline uses seed 17; the three-seed fusion ensemble is an uncertainty control, not full-pipeline seed replication'}
    snapshot_root=ROOT/'runs/frozen_release_s17/source_snapshot'
    snapshot={}
    for path in paths:
        if path.suffix.lower() not in ['.py','.json','.txt','.csv']:continue
        relative=path.resolve().relative_to(ROOT.resolve())
        target=snapshot_root/relative;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(path.read_bytes())
        if sha(target)!=result['sha256'][relative.as_posix()]:raise ValueError('Source snapshot changed while freezing')
        snapshot[relative.as_posix()]=target.relative_to(ROOT).as_posix()
    result['source_snapshot']=snapshot
    result['snapshot_scope']='Exact code/configuration/manifest copies; large input arrays and weights remain at their SHA256-bound paths'
    atomic_json(release_path, result)
    print(json.dumps({'status': result['status'], 'files_bound': len(result['sha256'])}), flush=True)


if __name__ == '__main__':
    main()
