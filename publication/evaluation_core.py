"""Unchanged CPU evaluation functions extracted from the frozen evaluator.

The original function bodies are retained exactly; publication wrappers supply
frozen sufficient statistics without importing the model-training runtime.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from experiments.evaluation_metrics import (metric_result, whole_unit_selection, spatial_block_ids, paired_block_bootstrap, simultaneous_bands)

def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+"\n",encoding="utf-8")

ALL_NAMES = ['sar', 'fusion', 'rgb_fusion', 'optical', 'value_gate', 'ordinary_gate', 'stacker', 'probability_average', 'lower_entropy_expert', 'inverse_entropy_blend', 'terramind_tiny', 'terramind_tiny_random', 'fusion_ensemble', 'sar_temperature', 'fusion_temperature', 'value_gate_temperature']
UNIT_FIELDS = ['cost_m2', 'fn_m2', 'fp_m2', 'mean_entropy', 'total_entropy', 'mean_disagreement', 'fn_probability_density', 'learned_omission_density', 'stacker_omission_density', 'negative_entropy_density', 'ensemble_variance', 'ensemble_mutual_information', 'negative_ensemble_variance', 'negative_ensemble_mutual_information']

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
