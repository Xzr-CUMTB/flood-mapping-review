# Reproducing the computational pipeline

Run commands from the repository root. Stages below are **sequential**. Do not
launch concurrent GPU queues. A source export supplies code and configuration,
not pre-completed jobs. The full training/data path was not rerun for this export.

## 1. Environments

The original platform is Windows with a single RTX 5070 Ti and CUDA 12.8 PyTorch.
Queue scripts expect `.venv-experiments/Scripts/python.exe` and
`.venv-foundation/Scripts/python.exe`. Windows `tar` must support RAR extraction;
GNU tar alone is not equivalent. Linux execution would require adapting those
platform paths and archive extraction, and is not claimed as tested here.

The main environment uses Python 3.9; the foundation environment uses Python 3.11.
Create isolated Conda base environments with GDAL/PROJ resources, then the named
project venvs. This setup recipe uses the original runtime structure; a clean
machine install has not been exercised during this source export.

```powershell
conda create -n flood-main -c conda-forge python=3.9 pip rasterio pyproj -y
conda run -n flood-main python -m venv --system-site-packages .venv-experiments
$py = Join-Path $PWD '.venv-experiments/Scripts/python.exe'
& $py -s -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
& $py -s -m pip install -r requirements-main.txt
& $py -s scripts/release_setup.py initialize
& $py -s scripts/release_setup.py freeze-environment --environment experiments

conda create -n flood-foundation -c conda-forge python=3.11 pip rasterio pyproj -y
conda run -n flood-foundation python -m venv --system-site-packages .venv-foundation
$fpy = Join-Path $PWD '.venv-foundation/Scripts/python.exe'
& $py -s scripts/download_torch_wheel.py
& $py -s scripts/setup_foundation_runtime.py
& $py -s scripts/download_terramind.py
```

The foundation setup installs the pinned CUDA PyTorch wheel, torchvision and
TerraTorch 1.2.6 using `configs/foundation_constraints.txt`, checks GPU imports,
and records completion. `foundation_environment.configure()` requires valid
`Library/share/proj/proj.db` and `Library/share/gdal/gdalvrt.xsd` under its base
Conda environment. It also sets local, offline model caches. If those resources
are missing, correct the Conda setup before continuing.

`requirements-main.txt` records selected currently observed packages; original
training freezes were not copied because they contain machine build paths. New
environment freezes are generated locally and are ignored by Git.

## 2. Validate software before core evaluation

```powershell
& $py -s scripts/verify_evaluation_arithmetic.py
& $py -s scripts/verify_evaluation_integration.py
& $py -s scripts/verify_frozen_evaluator_io.py
```

All three checks use synthetic data. Reports under `reports/` bind the tested
source hashes. Retain them for the later freeze gate. Rerun checks if the evaluator
changes **before** freezing; never modify a completed frozen release in place.

## 3. Download and prepare the benchmark

```powershell
& $py -s scripts/download_sen2gf3.py
& $py -s scripts/run_stage1_preparation.py
```

Preparation verifies/extracts the archive, aligns and caches inputs, verifies
serial-versus-threaded cache equality, and audits spatial overlap. Inspect
`reports/` and the prepared manifest before training. See [DATA.md](DATA.md).
Downloaders and preparation are separate from the synthetic software checks.

## 4. Initial development screening and explicit review

```powershell
& $py -s scripts/run_development_baselines.py
```

This performs three initial screening fits. Read
`runs/development_pair/paired_screening.json`, `baseline_and_average_metrics.csv`,
`idealized_review_screen.csv` and `review_units.csv`. Verify data and probability
integrity and the suitability of the fixed protocol; favourable performance is
not a prerequisite. Once you have reviewed those outputs:

```powershell
& $py -s scripts/release_setup.py record-development-review --reviewed
& $py -s scripts/make_crossfit_plan.py
```

The helper requires actual completed development fits, the exact benchmark
manifest and the retained sample count. It records your explicit review and
copies the historical benchmark label/support documentation into `reports/` for
freezing. It never fabricates a training-completion marker. Included plan files
document the fixed protocol; they are not evidence of any completed rerun.

## 5. Execute the prescribed 38 fits and calibration

```powershell
& $py -s scripts/run_nested_experiments.py
& $py -s scripts/run_foundation_experiments.py
& $py -s scripts/run_post_training_checks.py
& $py -s scripts/run_ensemble_controls.py
& $py -s scripts/audit_completed_training.py --require-complete
```

The nested queue contains 32 expert/head fits; foundation controls add 3, the
matched fusion omission head adds 1, and the remaining fusion ensemble members
add 2. Experts use 20 fixed epochs, spatial heads 10; main pipeline seed is 17.
Fusion ensemble seeds 17, 29 and 43 constitute an uncertainty control, not three
replications of the entire pipeline. Calibration uses development OOF predictions.

Queues record status and logs under `runs/*_queue/`. Dependencies must complete
in this order; waiting controllers are not general-purpose schedulers. Never
delete a lock while its owning process is alive. After interruption, inspect the
PID and command line. If complete optimizer and random-state recovery is absent,
archive the interrupted run and repeat it with the original fixed seed; do not
describe this as exact continuation. Keep completed outputs and original evidence.

The inherited storage guards constrain working-set size and require free disk
space. They can stop a large local run even on a larger disk; they are recorded
in the code and provenance documentation. No cleanup of existing results is
performed by this export.

## 6. Freeze, infer, evaluate and verify

Record final runtime package versions after setup and before freezing:

```powershell
& $py -s scripts/release_setup.py freeze-environment --environment experiments
& $fpy -s scripts/release_setup.py freeze-environment --environment foundation
& $py -s scripts/run_core_completion.py
& $py -s scripts/verify_completed_batch.py
```

The controller re-audits all fits, evaluates development folds, checks calibration
and software evidence, creates a new frozen release, runs main and foundation
inference, evaluates the frozen predictions, and generates numeric summaries.
Completion requires matching release hashes and real output records, not merely
an exited training process. The historical release in `provenance/` must **not**
be copied into `configs/core_release_s17.json`.

Inspect `runs/core_s17_evaluation/` for CSV results,
`reports/13_core_computational_results.md` for the numerical report,
`reports/core_result_summary.json`, and `reports/completed_batch_verification.json`.
All are generated locally and ignored by Git. Preserve supportive, null and
opposite-direction results and distinguish raw/calibrated probabilities and
FN/FP outcomes. Figure production is outside this repository.
