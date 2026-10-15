# Paper result reproduction

Values are recalculated from unrounded frozen CSV records.

The computation entry point is `publication/reproduce_numerical.py`. Map effects
use `evaluation/map_metrics.csv` (region/model keys); review effects and ratios
use `evaluation/review_budget_results.csv` (region/model/policy/budget keys).
Review difference intervals come from `evaluation/review_simultaneous_intervals.csv`.
All paths are relative to the separately verified data package; their complete-file
hashes are in `FILE_MANIFEST.csv`. `publication/reproduce_evaluation.py` independently
rebuilds the underlying metrics, selections and interval tables from the original
patch/unit sufficient statistics. Final effects are rounded after subtraction or
division of full-precision inputs. Relative reduction divides the learned-minus-proxy
residual FN difference by learned-ranking residual FN at the same 10% area budget.

| Region | Effect | Recomputed | Unit | Rounded manuscript value | Status |
|---|---|---:|---|---:|---|
| Beijing_Baoding | Fusion minus SAR IoU | 16.388406988 | percentage points | 16.39 | PASS |
| Beijing_Baoding | Value minus fusion residual FN at 10% H review | 5.12006323988 | km2 | 5.12 | PASS |
| Beijing_Baoding | Learned minus proxy residual FN on fixed value map | 22.4250455928 | km2 | 22.43 | PASS |
| Beijing_Baoding | Relative FN reduction replacing learned ranking | 17.4061614061 | percent | 17.4 | PASS |
| Harbin | Fusion minus SAR IoU | 26.2807840652 | percentage points | 26.28 | PASS |
| Harbin | Value minus fusion residual FN at 10% H review | 5.13228741125 | km2 | 5.13 | PASS |
| Harbin | Learned minus proxy residual FN on fixed value map | 44.597456554 | km2 | 44.6 | PASS |
| Harbin | Relative FN reduction replacing learned ranking | 49.1407531197 | percent | 49.1 | PASS |

## Supplementary table reconciliation

The 398 numerical SI cells were checked against frozen full-precision outputs.
Two cells in the latest Word Table S7 require final-decimal correction:

| Region and ranking | Word manuscript | Frozen-result rounding (km²) |
| --- | ---: | ---: |
| Beijing–Baoding, Value/L residual FN | 128.84 | 128.83 |
| Harbin, Value/P residual FN | 46.15 | 46.16 |

The package retains `manuscript_tables/original_transcription/Table_S7.csv`,
the corrected `manuscript_tables/Table_S7.csv`, and
`MANUSCRIPT_TABLE_CORRECTIONS.json`. The original Word manuscript was not edited.
All eight principal reported effects still reproduce at their printed precision.
No frozen result, model, protocol, figure source or experimental selection changed.
