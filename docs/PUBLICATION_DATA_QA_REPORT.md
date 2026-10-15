# Publication data QA report

Checked on 2026-10-15T18:16:24+08:00. Local package and independent GitHub-clone reproduction: **PASS**.
Harvard dataset creation/upload, remote verification and public DOI: **NOT VERIFIED / NOT PERFORMED**, deferred by the author.
The original Word Table S7 contains two unresolved final-decimal discrepancies;
the released table corrects them and retains the original transcription.

## Version and scope

Paper: Limited review reverses model rankings in satellite flood mapping.
Repository: https://github.com/Xzr-CUMTB/flood-mapping-review.
Branch: publication/reproducibility-release; no merge, tag or software DOI created.
Actually cloned and tested implementation commit: `0b4fec61c433a75761aac638176ae1c2f0931906`.
The clean run used final local transport archives and a reused pinned Python 3.9
environment. It did not test a Dataverse download or a clean-machine installation.
All recorded implementation and rendering hashes remain identical in this candidate.

Original frozen release SHA-256:
`093a72b5ad047aa885d05b382e697192849305165ab63b226f33c1d9160ff242`.
Payload FILE_MANIFEST.csv SHA-256: `f780f998b396b5b9d93581c5f5e9ed520360315d615c4099b1be915c10d1224e`.
The manifest covers 565 files. Its own file and checksum file
bring the extracted payload to 567 files and 844,364,719 bytes.
The checksum text binds FILE_MANIFEST.csv; transport hashes bind both files without
self-referential checksum claims.

## Required checks

| Item | Status | Actual evidence |
| --- | --- | --- |
| 1. Whole-file SHA-256 | PASS | All 565 manifested files checked after extraction; all 567 archive members losslessly checked |
| 2. File readability and schemas | PASS | Full CSV chunk reads, JSON parsing and non-object NPY/NPZ loading; actual files, not filenames alone |
| 3. Package/source accounting | PASS | Copy/source hashes and explicit metadata changes; six archives below the observed 2.5-GB limit |
| 4. All figures have real sources | PASS | 16 figures, 82 unique panels; 96 original mappings merged; every indexed source exists |
| 5. Manuscript numerical consistency | PASS for eight claims and corrected 398 table cells; FAIL for two original Word cells | Original S7 preserved; correction ledger below; frozen results unchanged |
| 6. FN/FP conservation | PASS | Patch/unit totals agree; original whole-unit evaluator reproduces all budget rows and primary selections |
| 7. Budget limits | PASS | All 1,824 actual valid-area fractions at or below their 1/5/10/20% budgets |
| 8. Unique IDs and geographic association | PASS | 8,233 unique core patch IDs, unique cache/prediction indices, 16 fixed unit positions per patch and 129,347 positive-area units |
| 9. No event/region double counting | PASS | Disjoint exhaustive regional masks; 17,019 retained IDs; none of 3,386 overlap pairs has both members retained |
| 10. Statistical intervals | PASS | 480 map pointwise + 144 review pointwise + 24 simultaneous rows reproduced, maximum absolute numeric error 0.0 |
| 11. Private paths/credential patterns | PASS for inspected release content | Public text/metadata scanned; no detected absolute machine paths or credential patterns; private files excluded |
| 12. Empirical/synthetic separation | PASS | Original research contributions only; synthetic software fixtures excluded; historical freeze is not a fresh-run completion marker |

The 38 original training checkpoint audits, 142 frozen file bindings and 13 original
prediction-array whole-file hashes were rechecked. No training, fitted model,
label, split, epoch count, test selection or core prediction was changed.
All 13 exported states were tensor-equal to originals and strictly loaded into
their architectures (11 main, 2 foundation); optimizer/training continuation
state is not redistributed. Code audit parsed 156 Python files
and found no changes to the original scientific export files.

## Independent reproduction

The code was cloned from GitHub into an independent directory. Its public
extractor verified six final local archives, safely extracted all members into a
new dataset directory and verified the payload. Numerical, SI table, full fixed
evaluation and all 16 figure replay commands completed with exit code 0.
All 16 regenerated RGB, 1200-dpi TIFFs matched the original complete-file SHA-256,
not merely appearance or dimensions. No data, result or renderer was substituted.

Evaluation reproduced 32 map rows, 1,824 budget rows and 456 primary packed
selection sets. Bootstrap intervals were regenerated from original patch/unit
contributions with the original algorithm and seed and compared to every
archived interval. Original draw arrays were not persisted; no draw distribution
was reconstructed from endpoint summaries.

## Principal results

Fusion-minus-SAR IoU gains reproduce as 16.39 and 26.28 percentage points.
At 10% mean-entropy review, value routing leaves 5.12 and 5.13 km² more FN than
fusion; regional simultaneous 95% intervals are [1.46, 8.78] and [0.06, 10.21] km².
On the fixed value map, learned-minus-proxy ranking residual FN is 22.43 and
44.60 km², with simultaneous intervals [11.82, 33.03] and [28.12, 61.07] km².
Replacing learned ranking therefore reduces its residual FN by 17.4% and 49.1%,
using learned-ranking residual FN as the denominator. No ratio interval is claimed.
Both regions and all null/opposite controls are retained; inference concerns
within-region spatial uncertainty and ideal reference-based correction.

## Mismatch ledger and remaining manuscript action

| Original Word Table S7 cell | Printed (km²) | Correct rounded frozen result (km²) |
| --- | ---: | ---: |
| Beijing–Baoding Value/L residual FN | 128.84 | 128.83 |
| Harbin Value/P residual FN | 46.15 | 46.16 |

The original Word manuscript is unchanged. The source precision is respectively
128.83395178079223 and 46.1570673854599 km². Corrected publication CSVs, original
transcription and MANUSCRIPT_TABLE_CORRECTIONS.json are all included. The two
0.01-km² printed differences do not alter any of the eight principal effects.
Synchronize those two manuscript cells before submission. No cause of the original
transcription discrepancy is assumed.

## Archive ledger

| Archive | Bytes | SHA-256 |
| --- | ---: | --- |
| 01_documentation_tables_validation.tar.gz | 262,028 | `4c5b290d7f5800275a2ad3072944b9c85329ecd1867d7c1b9109cdc2f1fb1f90` |
| 02_main_figure_source_data.tar.gz | 234,185,865 | `d4541bfbc5a8f7806f85fad1bca3b87aad7902ceaffa4bc46c393bc3e181df06` |
| 03_supplementary_figure_source_data.tar.gz | 32,977,964 | `259dac4c65beea1091ea30b19a1b416b5e972a4d358d02e332f89848c72527a3` |
| 04_frozen_evaluation.tar.gz | 179,311,589 | `b01a37eadc7c9c42790a0e93c519922384218cede01f2d9b1e8f6686a6e659a7` |
| 05_provenance_training_models.tar.gz | 91,124,661 | `93ab6921f3ed6746b326ac4b1a80d93770e0202756c70a49d53c8dab3b654fc5` |
| 06_reference_tiff_masters.tar.gz | 118,557,005 | `d8d94280ed4c0372214e95d76005a00f1ba53afb953425982dbe81e284721818` |

Total transport size: 656,419,112 bytes.
Separate NC Source Data ZIP: 267,306,079 bytes;
SHA-256 `6760d26e2e93d6c0348d4e4957a3daec90440ca704741beb3e50901a30b43ea0`. ZIP integrity and all enclosed source
file hashes were verified. The full original ScienceDB archive is cited rather
than redistributed. Each transport archive has a content-specific README.

## Rights and external state

The author confirmed creator order, affiliations and rights to license original
contributions on 15 October 2026. Research data use CC BY 4.0, original code/model
states Apache-2.0; Sen2GF3Floods and TerraMind attribution is retained. This is
author confirmation, not a separate institutional ownership adjudication.

Harvard dataset: not created. Files uploaded: 0. Dataset URL: none. DOI: none.
Remote file IDs, server sizes/checksums, redownload and anonymous public access:
NOT VERIFIED. No review submission or publication occurred. These steps are
deferred by the latest user instruction, not silently represented as completed.

4096-bin average precision is preserved in original CSVs but not independently
recomputed from sufficient statistics. Pixel-level inference requires original
public inputs plus released states. Full GPU inference, all 38 fits and a fresh
machine environment installation are NOT VERIFIED in this packaging task.
The original algorithms, pinned plans, state loader and full-training instructions
are supplied for that separate reproduction level. Formal Data availability and
an actual replication-dataset citation await a real public deposit.

## Evidence locations

Public code-side reports: provenance/publication_validation/.
Payload-side evidence: validation/, PANEL_DATA_INDEX.csv, DATA_DICTIONARY.csv,
FILE_MANIFEST.csv, CHECKSUMS_SHA256.txt, COPY_PROVENANCE.json,
MANUSCRIPT_TABLE_INDEX.json, MANUSCRIPT_TABLE_CORRECTIONS.json and
model_states/MODEL_STATE_MANIFEST.json. Transport manifests contain actual hashes
and explicitly empty remote fields; there are no invented upload confirmations.
