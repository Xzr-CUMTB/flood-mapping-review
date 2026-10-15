"""Streaming area metrics and whole-unit review; no inference or training."""
import numpy as np

FIELDS = ['tp_m2', 'fp_m2', 'fn_m2', 'tn_m2', 'brier_area', 'nll_area',
          'conf_005_m2', 'conf_01_m2', 'conf_02_m2', 'valid_pixels']


def entropy(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1-1e-7)
    return -(p*np.log(p)+(1-p)*np.log1p(-p))


def patch_metrics(p, y, w, threshold=.5):
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=bool)
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Non-finite or out-of-range probability')
    pred = p >= threshold
    safe = np.clip(p, 1e-7, 1-1e-7)
    return np.array([w[y & pred].sum(), w[~y & pred].sum(), w[y & ~pred].sum(), w[~y & ~pred].sum(),
                     (w*(p-y)**2).sum(), (w*(-(y*np.log(safe)+(~y)*np.log1p(-safe)))).sum(),
                     w[y & (p <= .05)].sum(), w[y & (p <= .1)].sum(), w[y & (p <= .2)].sum(),
                     np.count_nonzero(w)], dtype=np.float64)


def metric_result(sums):
    tp, fp, fn, tn, bs, nll, c005, c01, c02, pixels = sums
    area, truth = tp+fp+fn+tn, tp+fn
    ratio = lambda x, y: float(x/y) if y > 0 else None
    return {'flood_iou': ratio(tp, tp+fp+fn), 'precision': ratio(tp, tp+fp),
            'recall': ratio(tp, truth), 'fn_rate': ratio(fn, truth), 'brier': ratio(bs, area),
            'nll': ratio(nll, area), 'fn_km2': fn/1e6, 'fp_km2': fp/1e6,
            'valid_area_km2': area/1e6, 'true_flood_km2': truth/1e6, 'valid_pixels': int(pixels),
            'confident_omission_005': ratio(c005, truth), 'confident_omission_01': ratio(c01, truth),
            'confident_omission_02': ratio(c02, truth)}


class ProbabilityHistogram:
    def __init__(self, bins=4096, reliability_bins=20):
        self.bins, self.reliability_bins = bins, reliability_bins
        self.positive = np.zeros(bins)
        self.negative = np.zeros(bins)
        self.reliability = np.zeros((reliability_bins, 3))

    def add(self, p, y, w):
        active = w > 0
        p, y, w = np.asarray(p)[active], np.asarray(y)[active], w[active]
        b = np.minimum((p*self.bins).astype(int), self.bins-1)
        self.positive += np.bincount(b, weights=w*y, minlength=self.bins)
        self.negative += np.bincount(b, weights=w*(~y), minlength=self.bins)
        r = np.minimum((p*self.reliability_bins).astype(int), self.reliability_bins-1)
        for j, values in enumerate([w, w*p, w*y]):
            self.reliability[:, j] += np.bincount(r, weights=values, minlength=self.reliability_bins)

    def result(self):
        tp = self.positive[::-1].cumsum()
        fp = self.negative[::-1].cumsum()
        precision = np.divide(tp, tp+fp, out=np.zeros_like(tp), where=(tp+fp) > 0)
        recall = tp/max(tp[-1], 1e-20)
        ap = float(np.dot(np.diff(np.r_[0., recall]), precision)) if tp[-1] > 0 else None
        w, ps, ys = self.reliability.T
        mean_p = np.divide(ps, w, out=np.zeros_like(w), where=w > 0)
        mean_y = np.divide(ys, w, out=np.zeros_like(w), where=w > 0)
        ece = float(np.dot(w, abs(mean_p-mean_y))/max(w.sum(), 1e-20))
        return {'area_weighted_binned_average_precision': ap, 'area_weighted_ece': ece,
                'precision_recall_bins': self.bins, 'reliability_bins': self.reliability_bins,
                'reliability': [{'bin': i, 'area_m2': float(w[i]), 'mean_probability': float(mean_p[i]),
                                 'flood_fraction': float(mean_y[i])} for i in range(len(w))]}


def tile_sum(array, size=64):
    h, w = array.shape
    if h % size or w % size:
        raise ValueError('Review units must tile each patch exactly')
    return array.reshape(h//size, size, w//size, size).sum(axis=(1, 3)).ravel()


def review_units(p, y, w, sar, fusion, risk=None, stacker=None, size=64, ensemble_variance=None, ensemble_mi=None):
    cost = tile_sum(w, size)
    neg = p < .5
    fn, fp = tile_sum(w*y*neg, size), tile_sum(w*(~y)*(~neg), size)
    fields = {'cost_m2': cost, 'fn_m2': fn, 'fp_m2': fp,
              'total_entropy': tile_sum(w*entropy(p), size)}
    means = {'mean_entropy': fields['total_entropy'],
             'negative_entropy_density': tile_sum(w*entropy(p)*neg, size),
             'mean_disagreement': tile_sum(w*np.abs(sar-fusion), size),
             'fn_probability_density': tile_sum(w*p*neg, size)}
    if risk is not None:
        means['learned_omission_density'] = tile_sum(w*risk*neg, size)
    if stacker is not None:
        means['stacker_omission_density'] = tile_sum(w*stacker*neg, size)
    if ensemble_variance is not None:
        means['ensemble_variance'] = tile_sum(w*ensemble_variance, size)
        means['negative_ensemble_variance'] = tile_sum(w*ensemble_variance*neg, size)
    if ensemble_mi is not None:
        means['ensemble_mutual_information'] = tile_sum(w*ensemble_mi, size)
        means['negative_ensemble_mutual_information'] = tile_sum(w*ensemble_mi*neg, size)
    for name, numerator in means.items():
        fields[name] = np.divide(numerator, cost, out=np.zeros_like(cost), where=cost > 0)
    return fields


def whole_unit_selection(score, cost, fraction):
    if not np.isfinite(score).all() or not np.isfinite(cost).all() or np.any(cost <= 0):
        raise ValueError('Selection requires finite scores and positive finite costs')
    if not 0 <= fraction <= 1:
        raise ValueError('Invalid review fraction')
    limit, used = float(cost.sum()*fraction), 0.
    chosen = np.zeros(len(cost), dtype=bool)
    for i in np.argsort(-score, kind='stable'):
        if used+cost[i] <= limit+1e-9:
            chosen[i] = True
            used += cost[i]
    return chosen, used


def spatial_block_ids(longitude, latitude, crs, size_m):
    from pyproj import Transformer
    x, y = Transformer.from_crs('EPSG:4326', crs, always_xy=True).transform(longitude, latitude)
    return list(zip(np.floor(np.asarray(x)/size_m).astype(int), np.floor(np.asarray(y)/size_m).astype(int)))


def paired_block_bootstrap(block_sums, replicates=2000, seed=20261012):
    """Yield resampled sums with identical draws for every compared column."""
    rng = np.random.default_rng(seed)
    n = len(block_sums)
    if n < 2:
        raise ValueError('At least two occupied spatial blocks required')
    for start in range(0, replicates, 100):
        draws = rng.integers(0, n, size=(min(100, replicates-start), n))
        yield block_sums[draws].sum(axis=1)


def simultaneous_bands(point, draws, coverage=.95):
    """Max standardized centered bootstrap deviations over one prespecified family."""
    point, draws = np.asarray(point), np.asarray(draws)
    se = draws.std(axis=0, ddof=1)
    centered = np.divide(np.abs(draws-point), se, out=np.zeros_like(draws), where=se > 1e-15)
    maximum = centered.reshape(len(draws), -1).max(axis=1)
    critical = float(np.quantile(maximum, coverage))
    return point-critical*se, point+critical*se, critical, se
