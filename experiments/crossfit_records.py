"""Provenance checks for expert predictions used as cross-fitted supervision."""
from pathlib import Path
import hashlib, json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/processed/sen2gf3'


def assemble(pairs, excluded_groups=()):
    """Each pair names sar_run, fusion_run and the predicted groups to retain.

    excluded_groups is the outer held-out set. Every source expert must exclude it,
    including experts that generate supervision for the inner gate.
    """
    chunks, proofs = [], []
    manifest_sha = hashlib.sha256((DATA/'manifest.csv').read_bytes()).hexdigest()
    for pair in pairs:
        configs, tables = [], []
        for modality, field in [('sar', 'sar_run'), ('fusion', 'fusion_run')]:
            directory = ROOT/'runs'/pair[field]
            config = json.loads((directory/'config.json').read_text())
            complete = json.loads((directory/'complete.json').read_text())
            if complete.get('status') != 'complete' or complete.get('core_test_evaluated'):
                raise ValueError('Completed development-only source required')
            if config.get('checkpoint_selection_mode') != 'fixed_last':
                raise ValueError('OOF targets require a fixed final checkpoint, not validation selection')
            if complete['selected_epoch'] != config['epochs']:
                raise ValueError('Source checkpoint did not reach its fixed epoch count')
            if config['modality'] != modality or config['manifest_sha256'] != manifest_sha:
                raise ValueError('Source modality or data manifest mismatch')
            training = set(config['train_groups'].split(','))
            if training.intersection(excluded_groups):
                raise ValueError('Outer-event labels entered an inner expert')
            table = pd.read_csv(directory/'validation_patch_metrics.csv')
            if set(table.event_group).intersection(training):
                raise ValueError('Prediction event overlaps expert training events')
            if '2023_holdout' in set(table.event_group):
                raise ValueError('Core test cannot supply training targets')
            configs.append(config)
            tables.append(table)
        if tables[0].cache_index.tolist() != tables[1].cache_index.tolist():
            raise ValueError('Experts did not predict identical samples in identical order')
        if configs[0]['train_groups'] != configs[1]['train_groups'] or configs[0]['seed'] != configs[1]['seed']:
            raise ValueError('Expert comparison uses different training groups or seeds')
        selected = tables[0][tables[0].event_group.isin(pair['groups'])].copy()
        if set(selected.event_group) != set(pair['groups']):
            raise ValueError('Requested prediction group is missing')
        selected = selected[['cache_index', 'sample_id', 'event_group', 'prediction_index']]
        selected['sar_run'] = pair['sar_run']
        selected['fusion_run'] = pair['fusion_run']
        chunks.append(selected)
        proofs.append({'sar_run': pair['sar_run'], 'fusion_run': pair['fusion_run'],
                       'expert_training_groups': configs[0]['train_groups'].split(','),
                       'target_groups': pair['groups'], 'outer_excluded_groups': list(excluded_groups)})
    if not chunks:
        raise ValueError('No cross-fitted sources supplied')
    records = pd.concat(chunks, ignore_index=True).sort_values('cache_index').reset_index(drop=True)
    if records.cache_index.duplicated().any():
        raise ValueError('Repeated target sample in cross-fitted records')
    if set(records.event_group).intersection(excluded_groups):
        raise ValueError('Outer-event labels entered gate supervision')
    return records, {'manifest_sha256': manifest_sha, 'sources': proofs,
                     'training_groups': sorted(records.event_group.unique().tolist()),
                     'outer_excluded_groups': list(excluded_groups)}
