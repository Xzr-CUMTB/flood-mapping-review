"""Check operational metrics against independent arithmetic before test release."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, sys
import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.evaluation_metrics import (patch_metrics, metric_result, ProbabilityHistogram,
    review_units, whole_unit_selection, paired_block_bootstrap, simultaneous_bands, entropy)
from experiments.evaluate_frozen_core import get_maps, group_sums
from experiments.operating_points import match_positive_rate


def main():
    p = np.array([[.9, .8], [.2, .1]])
    y = np.array([[True, False], [True, False]])
    w = np.array([[1., 2.], [3., 4.]])
    sums = patch_metrics(p, y, w)
    assert np.allclose(sums[:4], [1, 2, 3, 4])
    result = metric_result(sums)
    assert abs(result['flood_iou']-1/6) < 1e-12
    assert abs(result['brier']-brier_score_loss(y.ravel(), p.ravel(), sample_weight=w.ravel())) < 1e-12
    assert abs(result['nll']-log_loss(y.ravel(), p.ravel(), sample_weight=w.ravel())) < 1e-12
    hist = ProbabilityHistogram()
    hist.add(p, y, w)
    expected_ap = average_precision_score(y.ravel(), p.ravel(), sample_weight=w.ravel())
    assert abs(hist.result()['area_weighted_binned_average_precision']-expected_ap) < 1e-12
    # Invalid pixels must not contribute, including to probability histograms.
    masked = w.copy(); masked[0, 0] = 0
    assert patch_metrics(p, y, masked)[0] == 0
    histogram = ProbabilityHistogram(); histogram.add(p, y, masked)
    assert histogram.positive.sum() == 3 and histogram.negative.sum() == 6
    # Skip an expensive high-score unit that cannot fit; preserve stable score ties.
    selected, used = whole_unit_selection(np.array([4., 3., 2., 2.]), np.array([8., 6., 4., 3.]), .5)
    assert selected.tolist() == [True, False, False, False] and used == 8
    selected, used = whole_unit_selection(np.array([4., 3., 2.]), np.array([12., 6., 4.]), .5)
    assert selected.tolist() == [False, True, True] and used == 10
    selected, used = whole_unit_selection(np.array([1., 1., 1.]), np.ones(3), .34)
    assert selected.tolist() == [True, False, False] and used == 1
    # A 64x64 unit contains all 4096 pixels; no point-center approximation for cost.
    unit = review_units(np.full((64, 64), .2), np.ones((64, 64), dtype=bool), np.ones((64, 64)),
                        np.full((64, 64), .1), np.full((64, 64), .2), risk=np.full((64, 64), .7))
    assert unit['cost_m2'][0] == 4096 and unit['fn_m2'][0] == 4096 and unit['fp_m2'][0] == 0
    assert abs(unit['learned_omission_density'][0]-.7) < 1e-12
    # All-positive predictions remove omissions while retaining the false-positive penalty.
    all_positive = metric_result(patch_metrics(np.ones((2, 2)), y, w))
    assert all_positive['fn_km2'] == 0 and all_positive['fp_km2'] == 6/1e6
    # Positive temperatures preserve 0.5 decisions, including values exactly at threshold.
    raw = {name: np.array([[[0., .49], [.5, 1.]]], dtype='float32')
           for name in ['sar', 'fusion', 'rgb_fusion', 'optical', 'value_gate', 'ordinary_gate', 'stacker', 'risk']}
    maps = get_maps(raw, 0, {name: {'temperature': .6} for name in ['sar', 'fusion', 'value_gate']})
    for name in ['sar', 'fusion', 'value_gate']:
        assert np.array_equal(maps[name] >= .5, maps[name+'_temperature'] >= .5)
    # Three constant members give a hand-computable mean, variance and Jensen gap.
    ensemble_raw={name:np.full((1,2,2),.1) for name in ['sar','fusion','value_gate']}
    ensemble_raw.update(fusion_seed29=np.full((1,2,2),.5),fusion_seed43=np.full((1,2,2),.9))
    ensemble_maps=get_maps(ensemble_raw,0,{name:{'temperature':1.} for name in ['sar','fusion','value_gate']})
    assert np.allclose(ensemble_maps['fusion_ensemble'],.5)
    members=np.array([.1,.5,.9])
    assert abs(members.var()-8/75)<1e-12
    expected_mi=(2/3)*(np.log(2)+.1*np.log(.1)+.9*np.log(.9))
    assert abs(float(entropy(.5)-entropy(members).mean())-expected_mi)<1e-12
    assert abs(unit['negative_entropy_density'][0]-(-.2*np.log(.2)-.8*np.log(.8)))<1e-12
    conditional=review_units(np.array([[.2,.8],[.2,.8]]),y,w,p,p,size=2,
        ensemble_variance=np.full((2,2),.4),ensemble_mi=np.full((2,2),.3))
    assert abs(conditional['negative_ensemble_variance'][0]-.16)<1e-12
    assert abs(conditional['negative_ensemble_mutual_information'][0]-.12)<1e-12
    # A tied pair cannot be split to hit the requested weighted positive rate.
    matched=match_positive_rate(np.array([.8,.8,.2]),np.array([1.,2.,7.]),.4)
    assert matched['threshold']==.8 and abs(matched['achieved_rate']-.3)<1e-12
    absent=match_positive_rate(np.array([.8,.8,.2]),np.array([1.,2.,7.]),0.)
    assert absent['threshold']>.8 and absent['achieved_rate']==0.
    at_point=patch_metrics(p,y,w,threshold=.85)
    assert np.allclose(at_point[:4],[1,0,3,6])
    # Equal model columns remain equal under every paired spatial bootstrap draw.
    matrix = np.array([[1., 1.], [3., 3.], [7., 7.]])
    totals = group_sums(matrix, [(0, 0), (1, 0), (1, 0)])
    draws = np.concatenate(list(paired_block_bootstrap(totals, replicates=200)))
    assert np.array_equal(draws[:, 0], draws[:, 1])
    # A degenerate paired contrast has a zero-width interval, rather than NaN or infinity.
    zero_draws = np.zeros((200, 2, 2))
    lo, hi, critical, se = simultaneous_bands(np.zeros((2, 2)), zero_draws)
    assert np.array_equal(lo, hi) and np.isfinite(lo).all() and critical == 0
    report = {'passed': True, 'checked_at_utc': datetime.now(timezone.utc).isoformat(),
              'core_test_predictions_read': False,
              'checks': ['weighted confusion and IoU', 'independent sklearn Brier/NLL/AP',
                         'invalid support exclusion', 'variable-cost budget skip and stable ties',
                         'whole-unit area integration', 'false-positive companion catches all-positive map',
                         'temperature decision invariance', 'ensemble mean/variance/mutual information',
                         'predicted-negative entropy integration', 'weighted operating points and tied scores',
                         'paired spatial resampling identity',
                         'simultaneous-band degenerate contrast handling'],
              'source_sha256': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in [Path(__file__), ROOT/'experiments/evaluation_metrics.py', ROOT/'experiments/evaluate_frozen_core.py',
                               ROOT/'experiments/operating_points.py']}}
    (ROOT/'reports/evaluation_arithmetic_checks.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
