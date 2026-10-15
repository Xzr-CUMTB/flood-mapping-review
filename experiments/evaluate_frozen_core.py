"""Fixed-protocol core-test metrics, review policies and spatial uncertainty; no plots."""
from pathlib import Path
from datetime import datetime, timezone
import json, sys, time, os
os.environ['MKL_THREADING_LAYER']='SEQUENTIAL'
import numpy as np
import pandas as pd
from scipy.special import expit, logit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.train_baseline import atomic_json
from experiments.infer_frozen_core import check_release, sha
from experiments.evaluation_metrics import (FIELDS, patch_metrics, metric_result, entropy, ProbabilityHistogram,
    review_units, whole_unit_selection, spatial_block_ids, paired_block_bootstrap, simultaneous_bands, tile_sum)

UNIT_FIELDS = ['cost_m2', 'fn_m2', 'fp_m2', 'mean_entropy', 'total_entropy', 'mean_disagreement',
               'fn_probability_density', 'learned_omission_density', 'stacker_omission_density',
               'negative_entropy_density', 'ensemble_variance', 'ensemble_mutual_information',
               'negative_ensemble_variance','negative_ensemble_mutual_information']
RAW_NAMES = ['sar', 'fusion', 'rgb_fusion', 'optical', 'value_gate', 'ordinary_gate', 'stacker',
             'probability_average', 'lower_entropy_expert', 'inverse_entropy_blend', 'terramind_tiny', 'terramind_tiny_random',
             'fusion_ensemble']
CAL_NAMES = ['sar_temperature', 'fusion_temperature', 'value_gate_temperature']
ALL_NAMES = RAW_NAMES+CAL_NAMES


def get_maps(raw, index, temperatures):
    maps = {name: np.asarray(raw[name][index], dtype='float64') for name in RAW_NAMES if name in raw}
    s, f = maps['sar'], maps['fusion']
    hs, hf = entropy(s), entropy(f)
    maps['probability_average'] = (s+f)/2
    maps['lower_entropy_expert'] = np.where(hs <= hf, s, f)
    # Weights proportional to 1/entropy, with finite 1e-7 clipping in entropy().
    maps['inverse_entropy_blend'] = (hf*s+hs*f)/(hs+hf)
    if 'fusion_seed29' in raw and 'fusion_seed43' in raw:
        maps['fusion_ensemble'] = (f+np.asarray(raw['fusion_seed29'][index], dtype='float64')
                                  +np.asarray(raw['fusion_seed43'][index], dtype='float64'))/3.
    for name in ['sar', 'fusion', 'value_gate']:
        p = expit(logit(np.clip(maps[name], 1e-7, 1-1e-7))/temperatures[name]['temperature'])
        if not np.array_equal(p >= .5, maps[name] >= .5):
            raise ValueError('Temperature changed the classification decision')
        maps[name+'_temperature'] = p
    return maps


def group_sums(values, blocks):
    keys, inverse = np.unique(np.asarray(blocks), axis=0, return_inverse=True)
    output = np.zeros((len(keys),)+values.shape[1:], dtype=np.float64)
    np.add.at(output, inverse, values)
    return output


def uncertainty_rows(metrics, frame, masks, protocol):
    rows = []
    comparisons = [('value_gate', 'fusion'), ('value_gate', 'ordinary_gate'),
                   ('value_gate', 'probability_average'), ('value_gate', 'stacker'),
                   ('fusion', 'sar'), ('terramind_tiny', 'fusion'), ('terramind_tiny', 'terramind_tiny_random'),
                   ('fusion_ensemble', 'fusion'), ('fusion_temperature','sar_temperature'),
                   ('value_gate_temperature','fusion_temperature')]
    for region, mask in masks.items():
        subset = frame.loc[mask]
        crs = next(protocol[key]['spatial_block_crs'] for key in ['primary_region', 'secondary_region']
                   if protocol[key]['name'] == region)
        for block_size in [protocol['spatial_uncertainty']['primary_block_size_m'], *protocol['spatial_uncertainty']['sensitivity_block_sizes_m']]:
            blocks = spatial_block_ids(subset.lon.to_numpy(), subset.lat.to_numpy(), crs, block_size)
            totals = group_sums(metrics[:, mask, :].transpose(1, 0, 2), blocks)
            if len(totals) < 2:
                continue
            draws = np.concatenate(list(paired_block_bootstrap(totals,
                protocol['spatial_uncertainty']['bootstrap_replicates'], protocol['spatial_uncertainty']['seed'])))
            area = draws[..., :4].sum(axis=-1)
            statistics = {'flood_iou': draws[..., 0]/np.maximum(draws[..., :3].sum(axis=-1), 1e-20),
                          'fn_km2': draws[..., 2]/1e6, 'fp_km2': draws[..., 1]/1e6,
                          'brier': draws[..., 4]/np.maximum(area, 1e-20),
                          'nll': draws[..., 5]/np.maximum(area, 1e-20),
                          'confident_omission_005':draws[...,6]/np.maximum(draws[...,0]+draws[...,2],1e-20),
                          'confident_omission_01':draws[...,7]/np.maximum(draws[...,0]+draws[...,2],1e-20),
                          'confident_omission_02':draws[...,8]/np.maximum(draws[...,0]+draws[...,2],1e-20)}
            for first, second in comparisons:
                a, b = ALL_NAMES.index(first), ALL_NAMES.index(second)
                point_a = metric_result(metrics[a, mask].sum(axis=0))
                point_b = metric_result(metrics[b, mask].sum(axis=0))
                for metric, values in statistics.items():
                    lo, hi = np.quantile(values[:, a]-values[:, b], [.025, .975])
                    rows.append({'region': region, 'first': first, 'second': second, 'metric': metric,
                                 'difference_first_minus_second': point_a[metric]-point_b[metric],
                                 'ci025': lo, 'ci975': hi, 'block_size_m': block_size, 'occupied_blocks': len(totals),
                                 'interpretation': 'Paired within-region spatial composition uncertainty; not event replication'})
    return rows


def evaluate_review(units, frame, masks, protocol, out):
    records, selected_for_primary, contributions = [], {}, {}
    budgets = protocol['budget_fractions']
    for region, patch_mask in masks.items():
        for m, model in enumerate(ALL_NAMES):
            values = np.asarray(units[m, patch_mask]).reshape(-1, len(UNIT_FIELDS))
            good = values[:, 0] > 0
            values = values[good]
            cost, fn, fp = values[:, :3].T
            policies = {field: values[:, j] for j, field in enumerate(UNIT_FIELDS) if j >= 3 and np.isfinite(values[:, j]).all()}
            policies.update({f'random_seed_{seed}': np.random.default_rng(seed).random(len(cost)) for seed in protocol['random_policy_seeds']})
            policies['oracle_fn_density'] = fn/cost
            for fraction in budgets:
                for policy, score in policies.items():
                    selected, used = whole_unit_selection(score, cost, fraction)
                    residual_fn, residual_fp = float(fn[~selected].sum()/1e6), float(fp[~selected].sum()/1e6)
                    record = {'region': region, 'model': model, 'policy': policy, 'budget_fraction': fraction,
                              'actual_area_fraction': used/cost.sum(), 'selected_units': int(selected.sum()),
                              'initial_fn_km2': fn.sum()/1e6, 'initial_fp_km2': fp.sum()/1e6,
                              'residual_fn_km2': residual_fn, 'residual_fp_km2': residual_fp}
                    for weight in protocol['tradeoff_checks']['lambda_FP']:
                        record['area_loss_lambda_'+str(weight)] = residual_fn+weight*residual_fp
                    records.append(record)
                    if fraction == protocol['primary_budget_fraction']:
                        key = (region, model, policy)
                        # Retain contributions for paired block intervals conditional on observed selection.
                        contribution = np.zeros((patch_mask.sum()*16, 2), dtype=np.float64)
                        contribution[good, 0] = fn*(~selected)
                        contribution[good, 1] = fp*(~selected)
                        contributions[key] = contribution.reshape(-1, 16, 2).sum(axis=1)
                        selected_for_primary['|'.join(key)] = np.packbits(selected)
    table = pd.DataFrame(records)
    table['pareto_efficient_fn_fp'] = False
    # Compare deployable policies at the same region and nominal budget. Oracle uses truth.
    for _, group in table[table.policy.ne('oracle_fn_density')].groupby(['region', 'budget_fraction']):
        endpoints = group[['residual_fn_km2', 'residual_fp_km2']].to_numpy()
        efficient = np.ones(len(group), dtype=bool)
        for j, candidate in enumerate(endpoints):
            dominates = np.all(endpoints <= candidate, axis=1) & np.any(endpoints < candidate, axis=1)
            efficient[j] = not dominates.any()
        table.loc[group.index, 'pareto_efficient_fn_fp'] = efficient
    table.to_csv(out/'review_budget_results.csv', index=False)
    np.savez_compressed(out/'primary_review_selections_packed.npz', **selected_for_primary)
    atomic_json(out/'review_selection_encoding.json', {'encoding': 'numpy.packbits, big bit order',
                'order': 'Region patches in prediction_manifest order; 4x4 row-major units; cost_m2 > 0 units only',
                'budget': protocol['primary_budget_fraction'], 'assumption': protocol['review_assumption']})
    comparisons = [
        ('value_gate', 'mean_entropy', 'fusion', 'mean_entropy'),
        ('value_gate', 'mean_entropy', 'ordinary_gate', 'mean_entropy'),
        ('value_gate', 'learned_omission_density', 'value_gate', 'mean_entropy'),
        ('value_gate', 'learned_omission_density', 'value_gate', 'fn_probability_density'),
        ('value_gate', 'learned_omission_density', 'value_gate', 'stacker_omission_density'),
        ('value_gate', 'learned_omission_density', 'fusion', 'learned_omission_density'),
        ('value_gate', 'learned_omission_density', 'terramind_tiny', 'mean_entropy'),
        ('value_gate', 'learned_omission_density', 'value_gate', 'negative_entropy_density'),
        ('value_gate', 'learned_omission_density', 'value_gate', 'ensemble_mutual_information'),
        ('value_gate', 'learned_omission_density', 'fusion_ensemble', 'ensemble_mutual_information'),
        ('value_gate', 'learned_omission_density', 'value_gate', 'negative_ensemble_mutual_information'),
        ('value_gate', 'learned_omission_density', 'fusion_ensemble', 'negative_ensemble_mutual_information')]
    uncertainty = []
    for region, mask in masks.items():
        subset = frame.loc[mask]
        crs = next(protocol[key]['spatial_block_crs'] for key in ['primary_region', 'secondary_region'] if protocol[key]['name'] == region)
        for a, pa, b, pb in comparisons:
            left, right = contributions[(region, a, pa)], contributions[(region, b, pb)]
            delta = (left-right)/1e6
            for size in [10000, 5000, 20000]:
                blocks = spatial_block_ids(subset.lon.to_numpy(), subset.lat.to_numpy(), crs, size)
                totals = group_sums(delta, blocks)
                if len(totals) < 2:
                    continue
                draws = np.concatenate(list(paired_block_bootstrap(totals, protocol['spatial_uncertainty']['bootstrap_replicates'],
                                                                   protocol['spatial_uncertainty']['seed'])))
                point = delta.sum(axis=0)
                for j, endpoint in enumerate(['residual_fn_km2', 'residual_fp_km2']):
                    lo, hi = np.quantile(draws[:, j], [.025, .975])
                    uncertainty.append({'region': region, 'first_model': a, 'first_policy': pa, 'second_model': b,
                        'second_policy': pb, 'endpoint': endpoint, 'difference_first_minus_second': point[j],
                        'ci025': lo, 'ci975': hi, 'block_size_m': size, 'occupied_blocks': len(totals),
                        'budget_fraction': protocol['primary_budget_fraction'],
                        'interpretation': 'Paired spatial blocks; conditional on fixed fitted models and selected review units'})
    pd.DataFrame(uncertainty).to_csv(out/'review_paired_spatial_intervals.csv', index=False)
    simultaneous = []
    contrasts = protocol['confirmatory_review_contrasts']
    for region, mask in masks.items():
        subset = frame.loc[mask]
        crs = next(protocol[key]['spatial_block_crs'] for key in ['primary_region', 'secondary_region']
                   if protocol[key]['name'] == region)
        differences = np.stack([(contributions[(region, a, pa)]-contributions[(region, b, pb)])/1e6
                               for a, pa, b, pb in contrasts], axis=1)
        size = protocol['spatial_uncertainty']['primary_block_size_m']
        blocks = spatial_block_ids(subset.lon.to_numpy(), subset.lat.to_numpy(), crs, size)
        totals = group_sums(differences, blocks)
        if len(totals) < 2:
            continue
        draws = np.concatenate(list(paired_block_bootstrap(totals,
            protocol['spatial_uncertainty']['bootstrap_replicates'], protocol['spatial_uncertainty']['seed'])))
        point = differences.sum(axis=0)
        lower, upper, critical, se = simultaneous_bands(point, draws)
        for j, (a, pa, b, pb) in enumerate(contrasts):
            for k, endpoint in enumerate(['residual_fn_km2', 'residual_fp_km2']):
                simultaneous.append({'region': region, 'first_model': a, 'first_policy': pa, 'second_model': b,
                    'second_policy': pb, 'endpoint': endpoint, 'difference_first_minus_second': point[j, k],
                    'simultaneous_lower95': lower[j, k], 'simultaneous_upper95': upper[j, k],
                    'bootstrap_se': se[j, k], 'maximum_deviation_critical_value': critical,
                    'family_estimates': len(contrasts)*2, 'block_size_m': size, 'occupied_blocks': len(totals),
                    'budget_fraction': protocol['primary_budget_fraction'],
                    'scope': 'Within-region composition uncertainty, conditional on fixed models and review selections'})
    pd.DataFrame(simultaneous).to_csv(out/'review_simultaneous_intervals.csv', index=False)


def main():
    release_path = ROOT/'configs/core_release_s17.json'
    check_release(release_path)
    protocol = json.loads((ROOT/'configs/evaluation_protocol_v1.json').read_text())
    source = ROOT/'runs/core_s17_predictions'
    complete = json.loads((source/'complete.json').read_text())
    if complete['status'] != 'complete' or complete['release_sha256'] != sha(release_path):
        raise ValueError('Completed predictions from the frozen release required')
    frame = pd.read_csv(source/'prediction_manifest.csv')
    if not frame.event_group.eq(protocol['core_group']).all() or frame.cache_index.duplicated().any():
        raise ValueError('Invalid test prediction manifest')
    masks = {protocol[key]['name']: frame.provisional_site.isin(protocol[key]['provisional_sites']).to_numpy()
             for key in ['primary_region', 'secondary_region']}
    if not all(v.any() for v in masks.values()) or not (np.stack(list(masks.values())).sum(axis=0) == 1).all():
        raise ValueError('Every core patch must map to exactly one declared region')
    calibration = json.loads((ROOT/'runs/oof_calibration_s17/complete.json').read_text())
    temperatures = calibration['temperatures']
    operating_points = calibration['operating_points']
    raw = {name: np.load(source/(name+'.npy'), mmap_mode='r') for name in complete['models']}
    foundation_source = ROOT/'runs/core_s17_foundation_predictions'
    foundation_complete = json.loads((foundation_source/'complete.json').read_text())
    if foundation_complete['status'] != 'complete' or foundation_complete['release_sha256'] != sha(release_path):
        raise ValueError('Completed foundation predictions from this release required')
    foundation_frame = pd.read_csv(foundation_source/'prediction_manifest.csv')
    if foundation_frame.cache_index.tolist() != frame.cache_index.tolist():
        raise ValueError('Foundation and primary test samples differ')
    raw.update({name: np.load(foundation_source/(name+'.npy'), mmap_mode='r')
                for name in ['terramind_tiny', 'terramind_tiny_random']})
    for name, value in raw.items():
        if value.shape != (len(frame), 256, 256) or value.dtype != np.float32:
            raise ValueError('Unexpected prediction array '+name)
    data = ROOT/'data/processed/sen2gf3'
    labels, support, areas = [np.load(data/name, mmap_mode='r') for name in ['labels_uint8.npy', 'support_uint8.npy', 'row_area_m2.npy']]
    out = ROOT/'runs/core_s17_evaluation'
    out.mkdir(parents=True, exist_ok=True)
    if (out/'complete.json').exists():
        if json.loads((out/'complete.json').read_text())['release_sha256'] != sha(release_path):
            raise ValueError('Completed evaluation belongs to another release')
        print('Frozen core evaluation already completed', flush=True)
        return
    metrics = np.lib.format.open_memmap(out/'patch_metric_sums.npy', mode='w+', dtype='float64', shape=(len(ALL_NAMES), len(frame), len(FIELDS)))
    units = np.lib.format.open_memmap(out/'review_units.npy', mode='w+', dtype='float64', shape=(len(ALL_NAMES), len(frame), 16, len(UNIT_FIELDS)))
    histograms = {(region, model): ProbabilityHistogram() for region in masks for model in ALL_NAMES}
    operating_sums = {(scenario,region,model):np.zeros(len(FIELDS)) for scenario in operating_points
                      for region in masks for model in operating_points[scenario]}
    router_features=np.load(source/'router_unit_features.npy',mmap_mode='r')
    router_rows=[]
    start = time.monotonic()
    for j, row in enumerate(frame.itertuples()):
        idx = int(row.cache_index)
        y, valid = np.asarray(labels[idx]) == 1, np.asarray(support[idx]) == 7
        w = np.broadcast_to(np.asarray(areas[idx], dtype='float64')[:, None], (256, 256))*valid
        maps = get_maps(raw, j, temperatures)
        members = np.stack([maps['fusion'], np.asarray(raw['fusion_seed29'][j], dtype='float64'),
                            np.asarray(raw['fusion_seed43'][j], dtype='float64')])
        ensemble_variance = members.var(axis=0)
        ensemble_mi = np.maximum(entropy(maps['fusion_ensemble'])-np.mean(entropy(members), axis=0), 0.)
        region = next(region for region, mask in masks.items() if mask[j])
        for m, name in enumerate(ALL_NAMES):
            p = maps[name]
            metrics[m, j] = patch_metrics(p, y, w)
            histograms[(region, name)].add(p, y, w)
            risk_source = 'risk' if name.startswith('value_gate') else 'fusion_risk' if name in ['fusion', 'fusion_temperature'] else None
            values = review_units(p, y, w, maps['sar'], maps['fusion'],
                np.asarray(raw[risk_source][j]) if risk_source else None, maps['stacker'],
                ensemble_variance=ensemble_variance, ensemble_mi=ensemble_mi)
            for field_index, field in enumerate(UNIT_FIELDS):
                units[m, j, :, field_index] = values.get(field, np.full(16, np.nan))
        for scenario,points in operating_points.items():
            for name,point in points.items():
                operating_sums[(scenario,region,name)] += patch_metrics(maps[name],y,w,threshold=point['threshold'])
        unit_cost=tile_sum(w)
        actual_value=tile_sum(w*((maps['sar']-y)**2-(maps['fusion']-y)**2))
        for k in np.flatnonzero(unit_cost>0):
            predicted,gate,low,high,disagreement=router_features[j,k]
            router_rows.append({'region':region,'sample_id':row.sample_id,'cache_index':idx,'unit_index':int(k),
                'cost_m2':unit_cost[k],'predicted_mean_brier_value':predicted,
                'observed_mean_brier_value':actual_value[k]/unit_cost[k],
                'mean_gate':gate,'gate_below_01_fraction':low,'gate_above_09_fraction':high,
                'mean_expert_disagreement':disagreement})
        if (j+1) % 250 == 0:
            progress = {'status': 'evaluating', 'patches': j+1, 'total': len(frame), 'seconds': time.monotonic()-start}
            atomic_json(out/'status.json', progress)
            print(json.dumps(progress), flush=True)
    metrics.flush(); units.flush()
    atomic_json(out/'array_schema.json', {'models': ALL_NAMES, 'metric_fields': FIELDS, 'unit_fields': UNIT_FIELDS,
        'patch_order': '../core_s17_predictions/prediction_manifest.csv', 'unit_order': '4x4 row-major 64x64 units',
        'spatial_blocks': 'Assign every patch and its review units by patch centroid; sizes 5, 10 and 20 km'})
    results = []
    for region, mask in masks.items():
        for m, model in enumerate(ALL_NAMES):
            extra = histograms[(region, model)].result()
            reliability = extra.pop('reliability')
            pd.DataFrame(reliability).to_csv(out/(region+'_'+model+'_reliability.csv'), index=False)
            results.append({'region': region, 'model': model, **metric_result(metrics[m, mask].sum(axis=0)), **extra})
    pd.DataFrame(results).to_csv(out/'map_metrics.csv', index=False)
    operating_results=[]
    for (scenario,region,model),sums in operating_sums.items():
        point=operating_points[scenario][model]
        result=metric_result(sums)
        # Fixed high-confidence thresholds describe the primary 0.5 decision, so omit them here.
        result={key:value for key,value in result.items() if not key.startswith('confident_omission_')}
        result['false_positive_rate']=sums[1]/(sums[1]+sums[3]) if sums[1]+sums[3]>0 else None
        operating_results.append({'region':region,'scenario':scenario,'model':model,
            'threshold':point['threshold'],'development_target_rate':point['target_rate'],
            'development_achieved_rate':point['achieved_rate'],**result,
            'scope':'Operating points matched on development OOF; test rates may differ under distribution shift'})
    pd.DataFrame(operating_results).to_csv(out/'matched_operating_point_metrics.csv',index=False)
    pd.DataFrame(router_rows).to_csv(out/'router_value_unit_diagnostics.csv',index=False)
    pd.DataFrame(uncertainty_rows(metrics, frame, masks, protocol)).to_csv(out/'map_paired_spatial_intervals.csv', index=False)
    evaluate_review(units, frame, masks, protocol, out)
    result = {'status': 'complete', 'completed_at_utc': datetime.now(timezone.utc).isoformat(),
        'patches': len(frame), 'regions': {k: int(v.sum()) for k, v in masks.items()}, 'models': ALL_NAMES,
        'release_sha256': sha(release_path), 'elapsed_seconds': time.monotonic()-start,
        'human_review_measured': False, 'selection_on_test': False, 'figure_generation': False}
    atomic_json(out/'complete.json', result); atomic_json(out/'status.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
