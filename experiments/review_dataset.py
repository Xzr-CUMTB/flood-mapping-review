"""Memory-mapped raw inputs paired with provenance-checked OOF predictions."""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from experiments.train_baseline import FloodDataset

ROOT = Path(__file__).resolve().parents[1]


class ReviewDataset(Dataset):
    def __init__(self, records):
        self.records = records.reset_index(drop=True)
        self.raw = FloodDataset(self.records.cache_index, augment=False, modality='fusion')
        names = set(self.records.sar_run) | set(self.records.fusion_run)
        if 'system_run' in self.records:
            names |= set(self.records.system_run)
        self.probability = {name: np.load(ROOT/'runs'/name/'validation_probabilities.npy', mmap_mode='r') for name in names}
        self.gates = {name: np.load(ROOT/'runs'/name/'validation_gates.npy', mmap_mode='r')
                      for name in set(self.records.system_run)} if 'system_run' in self.records else {}

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records.iloc[index]
        item = self.raw[index]
        idx = int(row.cache_index)
        support = np.asarray(self.raw.support[idx])
        item['coverage'] = torch.from_numpy(np.stack([(support & 2) > 0, (support & 4) > 0]).astype('float32'))
        for field, run in [('sar', row.sar_run), ('fusion', row.fusion_run)]:
            item[field] = torch.from_numpy(np.array(self.probability[run][int(row.prediction_index)], dtype='float32'))
        if self.gates:
            j = int(row.system_prediction_index)
            item['system'] = torch.from_numpy(np.array(self.probability[row.system_run][j], dtype='float32'))
            item['gate'] = torch.from_numpy(np.array(self.gates[row.system_run][j], dtype='float32'))
        return item


def augment_batch(batch, rng):
    k = rng.randrange(4)
    flip = rng.random() < .5
    for key in ['x', 'y', 'valid', 'coverage', 'sar', 'fusion', 'system', 'gate']:
        if key not in batch:
            continue
        batch[key] = torch.rot90(batch[key], k, (-2, -1))
        if flip:
            batch[key] = batch[key].flip(-1)
    return batch
