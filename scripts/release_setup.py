"""Fresh-checkout setup and explicit development-review recording; no training."""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')


def initialize():
    for name in ['reports', 'runs', 'data', 'vendor/wheels']:
        (ROOT/name).mkdir(parents=True, exist_ok=True)
    print('Local runtime directories created; no datasets or models downloaded.')


def freeze_environment(name):
    # Versions rather than pip's local file:// build paths; records the running interpreter.
    packages = sorted({d.metadata['Name']+'=='+d.version for d in metadata.distributions()
                       if d.metadata.get('Name')})
    path = ROOT/'configs'/('environment_'+name+'_freeze.txt')
    if (ROOT/'configs/core_release_s17.json').exists():
        raise ValueError('An active frozen release exists; its environment record must not change.')
    path.write_text('# Observed runtime: Python '+sys.version.split()[0]+'\n'
                    +'\n'.join(packages)+'\n', encoding='utf-8')
    print(str(path.relative_to(ROOT)))


def review(acknowledged):
    if not acknowledged:
        raise ValueError('Inspect development outputs first, then supply --reviewed.')
    if (ROOT/'configs/core_release_s17.json').exists():
        raise ValueError('Do not revise review evidence after core release.')
    import pandas as pd
    manifest = ROOT/'data/processed/sen2gf3/manifest.csv'
    original = read(ROOT/'provenance/original_core_release_s17.json')
    expected = original['sha256']['data/processed/sen2gf3/manifest.csv']
    digest = sha(manifest)
    if digest != expected:
        raise ValueError('Prepared manifest differs from the study protocol; investigate before proceeding.')
    frame = pd.read_csv(manifest)
    retained = frame[frame.eligible & frame.spatial_nonoverlap]
    audit = read(ROOT/'provenance/retained_support_audit.json')
    if len(retained) != audit['retained_eligible_nonoverlap_samples']:
        raise ValueError('Retained sample count differs from historical benchmark audit.')
    queue = read(ROOT/'runs/development_queue/status.json')
    if queue.get('stage') != 'completed':
        raise ValueError('Initial development queue has not completed.')
    folder = ROOT/'runs/development_pair'
    evidence = ['paired_screening.json', 'baseline_and_average_metrics.csv',
                'idealized_review_screen.csv', 'review_units.csv']
    hashes = {}
    for name in evidence:
        path = folder/name
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError('Missing development evidence: '+name)
        hashes[name] = sha(path)
    for name in ['sar', 'fusion', 'rgb_fusion']:
        completed = read(ROOT/'runs'/('dev_'+name+'_seed17')/'complete.json')
        if completed.get('status') != 'complete' or completed.get('core_test_evaluated'):
            raise ValueError('Development fit incomplete or inconsistent: '+name)
    for name in ['label_semantics_audit', 'retained_support_audit']:
        shutil.copyfile(ROOT/'provenance'/(name+'.json'), ROOT/'reports'/(name+'.json'))
    write(folder/'reviewed_for_crossfit.json', {
        'passed': True, 'reviewed_at_utc': datetime.now(timezone.utc).isoformat(),
        'manifest_sha256': digest, 'core_test_predictions_inspected': False,
        'review_type': 'Operator explicitly acknowledged development-output review using --reviewed',
        'development_evidence_sha256': hashes,
        'data_audits': 'Historical published-benchmark evidence, reused after exact manifest verification',
        'acceptance_criterion': 'Data integrity and protocol readiness, not favourable effect size'})
    print('Development review recorded for the verified manifest; no core predictions read.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['initialize', 'freeze-environment', 'record-development-review'])
    parser.add_argument('--environment', choices=['experiments', 'foundation'])
    parser.add_argument('--reviewed', action='store_true')
    args = parser.parse_args()
    if args.action == 'initialize':
        initialize()
    elif args.action == 'freeze-environment':
        if args.environment is None:
            parser.error('--environment is required for freeze-environment')
        freeze_environment(args.environment)
    else:
        review(args.reviewed)


if __name__ == '__main__':
    main()
