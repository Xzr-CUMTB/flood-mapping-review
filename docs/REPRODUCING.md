# Reproducing the paper

Commands run from the code root. Choose the level needed; figure replay and
evaluation reproduction use real frozen study data and do not retrain a model.
The original experiment release is historical evidence, not a completion marker
that can be installed into a fresh run.

## Data access and preparation

The replication archives are prepared locally. Harvard Dataverse creation/upload
was deferred by the authors; no public deposit URL or DOI is available yet.
Obtain the six transport archives and their actual `UPLOAD_FILE_MANIFEST.csv`.
Use a separate empty DATA directory and a separate OUTPUT directory.

```text
python -m pip install -r requirements-publication.txt
python -m publication.prepare_data --archives ARCHIVES --manifest ARCHIVES/UPLOAD_FILE_MANIFEST.csv --output DATA
```

The extractor verifies every archive size and SHA-256, rejects unsafe paths,
links and repeated members, and verifies the entire extracted payload against
FILE_MANIFEST.csv. It uses no network or account token. The local clean-clone
test validates this route; it is not evidence of a Harvard download.

## Level 1: numerical, table and figure reproduction

```text
python -m publication.verify_payload --data DATA --output OUTPUT/qa
python -m publication.reproduce_numerical --data DATA --output OUTPUT/numerical
python -m publication.verify_manuscript_tables --data DATA --output OUTPUT/tables
python -m publication.redraw_figures --data DATA --output OUTPUT/redraw --final --verify-reference
```

Exact reference redraw uses Python 3.9, the pinned publication dependencies and
Times New Roman. All 16 TIFFs are RGB, 1200 dpi. Rendering is independent of model
training. See [figure mapping](FIGURE_REPRODUCTION.md). The eight principal
claims are computed from unrounded evaluation tables, with explicit denominators
and review intervals. The SI check covers 398 numerical cells; original S7
transcription and two documented decimal corrections are both retained.

## Level 2: evaluation reproduction

```text
python -m publication.reproduce_evaluation --data DATA --output OUTPUT/evaluation
```

The checker verifies 8,233 unique patch IDs, disjoint exhaustive regional masks,
32 map rows, 1,824 budget/strategy rows, 456 packed primary selections and FN/FP
area conservation. It regenerates 480 map, 144 review pointwise and 24 simultaneous
interval rows from the original contributions using the original algorithm and
seed. It does not reconstruct bootstrap draws from interval endpoints.

4096-bin average precision is retained in the original CSV but cannot be rebuilt
from the released sufficient statistics alone. Pixel predictions require the
original public images and the published model states. No full inference or
38-fit retraining was repeated for this publication candidate.

## Level 3: model loading, inference and full training

The tensor-only final states are linked to their original checkpoint hashes in
`model_states/MODEL_STATE_MANIFEST.json`. Check main and foundation architectures
in their respective environments:

```text
python -m publication.load_published_states --data DATA --family main --output OUTPUT/main_states.json
python -m publication.load_published_states --data DATA --family foundation --output OUTPUT/foundation_states.json
```

These commands use strict CPU state loading. They do not generate predictions.
The main environment is Python 3.9/CUDA PyTorch; foundation models require the
separate Python 3.11/TerraTorch environment. The model loading function returns
an eval-mode model for application through the original inference code. Model-only
states do not include optimizer/random-state continuation data.

For full data preparation, fixed training, calibration, freezing, inference and
evaluation follow [FULL_TRAINING.md](FULL_TRAINING.md) sequentially. It preserves
the original Windows paths and single-GPU queue contract. A clean installation
on a new machine and training under another platform have not been tested.
