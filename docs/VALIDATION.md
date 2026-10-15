# Code-export validation

Checks were run on 2026-10-15 using the existing Windows Python 3.9 main
environment against the **exported code directory**. No new research training or
core-data inference was performed. A clean-machine dependency installation was
not exercised.

| Check | Observed result |
| --- | --- |
| Python AST parsing | All 46 Python files parsed |
| Local module closure | No missing `models`, `experiments` or `scripts` modules |
| Export provenance | All 61 copied source/configuration file hashes matched the export manifest |
| Scientific implementation preservation | All copied model and experiment modules byte-identical to the original source |
| `verify_evaluation_arithmetic.py` | Passed 12 groups of arithmetic/selection/calibration/interval checks |
| `verify_evaluation_integration.py` | Passed; 16 prediction variants, 1,824 budget rows and 24 simultaneous-interval rows |
| `verify_frozen_evaluator_io.py` | Passed; six synthetic 256×256 patches, 32 map rows, 12 operating-point rows and 96 router-unit rows |

The arithmetic checks include independent scikit-learn comparisons, invalid
support exclusion, stable ties and variable-cost budget skipping, whole-unit area
integration, temperature decision invariance, ensemble uncertainty, weighted
operating points, and paired spatial resampling.

Test scripts are supplied so the checks can be repeated after installation.
Generated test reports and synthetic fixture arrays are ignored by Git. Synthetic
fixtures are explicitly labelled and must not be used as study evidence.

The source/configuration export changes one existing file:
`run_core_completion.py` no longer invokes project-dashboard/status generation.
Its training, freezing, prediction and numerical evaluation stages are retained.
The fresh-checkout setup helper is new packaging code, separate from the
scientific model and evaluation implementation.
