# Limited review reverses model rankings in satellite flood mapping

Research software for flood-map experts, cross-fitted expert-value routing and
budgeted omission review. This publication candidate contains the original
computational pipeline and the renderers for all six main and ten supplementary
figures. The study evaluates ideal reference-based correction of selected spatial
units; it does not measure human reviewer performance.

[中文说明](README.zh-CN.md) · [Reproduction](docs/REPRODUCING.md) ·
[Figures](docs/FIGURE_REPRODUCTION.md) · [Paper results](docs/PAPER_RESULTS.md) ·
[Data](docs/DATA.md) · [Validation](docs/VALIDATION.md) · [Provenance](docs/PROVENANCE.md)

## Reproduction levels

| Level | Inputs | Execution |
| --- | --- | --- |
| 1: numbers, tables and figures | Verified replication-data archives | CPU, `publication.reproduce_numerical`, `verify_manuscript_tables`, `redraw_figures` |
| 2: evaluation and spatial intervals | Frozen patch/unit sufficient statistics | CPU, `publication.reproduce_evaluation` |
| 3: pixel inference and full training | Original Sen2GF3Floods V2; 13 published model states or new training | GPU; original `scripts/`, `experiments/`, `models/` |

The study retains 17,019 of 21,483 patches: 8,786 development and 8,233 frozen core
patches. The prescribed batch contains 38 fits after three screening fits. All
comparisons, including unfavourable and null results, remain in the data package.

## Publication status and data

This is the `publication/reproducibility-release` review branch. The Harvard
Dataverse deposit has **not been created or uploaded**; the authors deferred that
step on 15 October 2026. There is no replication-dataset DOI or public download
URL to cite yet. Local transport archives and file-level checksums are prepared
separately from Git. The code does not embed a placeholder DOI or credentials.

Once supplied with the verified local archives, use:

```text
python -m pip install -r requirements-publication.txt
python -m publication.prepare_data --archives ARCHIVES --manifest ARCHIVES/UPLOAD_FILE_MANIFEST.csv --output DATA
python -m publication.reproduce_numerical --data DATA --output OUTPUT/numerical
python -m publication.verify_manuscript_tables --data DATA --output OUTPUT/tables
python -m publication.reproduce_evaluation --data DATA --output OUTPUT/evaluation
python -m publication.redraw_figures --data DATA --output OUTPUT/redraw --final --verify-reference
```

Use an empty DATA directory. Python 3.9 and the pinned CPU dependencies reproduce
the original calculation and drawing environment. Times New Roman must be present
for exact manuscript TIFF comparisons. See the reproduction guide for environment
setup and the separate Windows/CUDA full-training path.

## Layout

| Directory | Contents |
| --- | --- |
| `models/`, `experiments/`, `scripts/`, `configs/` | Original scientific pipeline and fixed plans |
| `publication/` | Payload checking, safe extraction, numerical/evaluation reproduction, state loading and figure redraw |
| `paper_figures/` | Original source-data derivation programs and final renderers; data live in the separate archive |
| `provenance/` | Historical release identity and source/export hash manifests |
| `docs/` | Reproduction instructions, numerical claims and measured validation |

## Method and licences

The signed value target is `v*=(s-y)^2-(f-y)^2`. The spatial router uses
`g=sigmoid(tanh(h(z))/0.05)` and `p=(1-g)s+gf`, with `g=0` where optical support
is unavailable. Review ranks whole 64×64-pixel units under a common valid-area
budget. Initial IoU and post-review FN/FP are distinct outcomes.

Code is [Apache-2.0](LICENSE). Study-generated replication data use CC BY 4.0;
published model states use Apache-2.0. Sen2GF3Floods derivatives retain their
CC BY 4.0 attribution; TerraMind-derived states retain upstream notices. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md), [NOTICE](NOTICE) and
[CITATION.cff](CITATION.cff). Data, images, weights and credentials are excluded
from Git. No article DOI or immutable software-archive DOI is asserted here.
