"""Recompute fixed format-audit samples serially to check the threaded cache."""
from pathlib import Path
from datetime import datetime, timezone
import json
import numpy as np
import pandas as pd
from prepare_sen2gf3 import read_sample

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data/processed/sen2gf3'


def main():
    if not (DATA/'preparation_complete.json').exists():
        raise ValueError('Completed preparation required')
    frame = pd.read_csv(DATA/'manifest.csv')
    if frame.cache_index.tolist() != list(range(21483)):
        raise ValueError('Cache row order mismatch')
    arrays = [np.load(DATA/name, mmap_mode='r') for name in
              ['x_uint8.npy', 'labels_uint8.npy', 'support_uint8.npy', 'row_area_m2.npy']]
    checked = []
    for sample in np.linspace(1,21483,24,dtype=int):
        record, *actual = read_sample(int(sample))
        for array, rebuilt in zip(arrays, actual):
            if not np.array_equal(array[sample-1], rebuilt):
                raise ValueError(f'Serial/threaded cache mismatch at sample {sample}')
        if int(frame.loc[sample-1,'common_valid_pixels']) != record['common_valid_pixels']:
            raise ValueError('Manifest coverage mismatch')
        checked.append(int(sample))
    report = {'passed': True, 'method': 'exact equality of serial recomputation versus threaded cache',
              'fixed_sample_ids': checked, 'arrays_checked': ['inputs', 'labels', 'coverage', 'pixel_area'],
              'core_test_predictions_evaluated': False, 'completed_at_utc': datetime.now(timezone.utc).isoformat()}
    (ROOT/'reports/prepared_cache_verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report))


if __name__ == '__main__':
    main()
