# Flood mapping and budgeted review

Research code for comparing satellite flood maps, cross-fitted expert-value routing,
and omission-focused review under a fixed valid-area budget.

[中文说明](README.zh-CN.md) · [Reproduction](docs/REPRODUCING.md) ·
[Data](docs/DATA.md) · [Code provenance](docs/PROVENANCE.md)

## What is included

The computational pipeline contains geospatial preprocessing, event and spatial
split audits, SAR/optical–SAR experts, nested cross-fitting, learned routing and
omission scores, development-only calibration, TerraMind controls, a three-member
fusion ensemble, frozen inference, and area-weighted evaluation with paired spatial
bootstrap intervals. Numeric CSV/JSON/Markdown reports are included in the workflow.
Figure generation, manuscript files, satellite data, trained weights and local run
artifacts are excluded from the repository.

The planned batch has **38 fits**, following three initial development-screening
fits. The released benchmark supplies 21,483 patches; the audited protocol retains
8,786 development patches and 8,233 patches in the frozen 2023 core test. Four
development event groups are separated from the Beijing–Baoding and Harbin core.

## Method in the code

For held-out expert predictions `s` (SAR), `f` (fusion) and reference label `y`,
the value target is `v* = (s-y)^2 - (f-y)^2`. A spatial head predicts a signed value
from inference-available inputs. The fixed router uses `g = sigmoid(tanh(h(z))/0.05)`
and `p = s + g*(f-s)`; unavailable optical support sets `g = 0`.

Review ranks whole 64 × 64 pixel units and selects them within a valid-area budget.
The evaluation applies ideal correction against the released reference inside
selected units. This measures residual FN/FP area in a computational review
simulation; it does not measure human review speed or effectiveness. Map IoU,
probability calibration and review outcomes are reported separately, including
unfavourable and null comparisons.

## Layout

| Directory | Role |
| --- | --- |
| `models/` | FloodUNet, spatial heads and TerraMind adapter |
| `experiments/` | Training, cross-fitting, calibration, inference and evaluation |
| `configs/` | Fixed experiment plans and evaluation protocol |
| `scripts/` | Downloads, preprocessing, serial queues, audits and software checks |
| `provenance/` | Original source hashes and historical release/data-audit records |
| `docs/` | Setup, execution order and interpretation |

## Start here

The original execution platform is **Windows, one NVIDIA GPU, CUDA 12.8**, with
separate Python 3.9 main and Python 3.11 foundation environments. The queue and
archive-extraction scripts retain that platform contract. Follow
[the complete reproduction instructions](docs/REPRODUCING.md) to create the two
environments, download the public benchmark, and run the stages in sequence.

CPU software checks can run without satellite data or model downloads after the
main dependencies are installed:

```powershell
python scripts/release_setup.py initialize
python scripts/verify_evaluation_arithmetic.py
python scripts/verify_evaluation_integration.py
python scripts/verify_frozen_evaluator_io.py
```

These checks generate labelled synthetic fixtures under ignored `runs/software_checks/`.
They are software validation, not new empirical results. See
[validation details](docs/VALIDATION.md) for checks performed on this code export.

The historical release record is kept under `provenance/`, not installed as an
active release. A fresh run must generate its own checkpoints, audit records and
`configs/core_release_s17.json` before core inference. The recorded source hashes
make the relationship between this export and the original experiment inspectable.

## Attribution and licensing

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for data and external software
attribution. No open-source license has yet been designated for the original code
in this repository; public access does not itself grant an open-source license.
Third-party materials retain their respective licenses. No manuscript DOI is
assigned by this repository.
