# Data contract

## Primary benchmark

Sen2GF3Floods, version 2, is obtained from the public Science Data Bank record
[10.57760/sciencedb.25341](https://doi.org/10.57760/sciencedb.25341).
`scripts/download_sen2gf3.py` contains the public download endpoint and implements
range resume, length checking and full-file digest verification.

| File | Expected bytes | MD5 |
| --- | ---: | --- |
| `Sen2GF3Floods.rar` | 4,931,104,688 | `eaca85ef80813b56bda8f4a57e88ea88` |

The archive belongs at `data/raw/sen2gf3/Sen2GF3Floods.rar`. The extraction stage
verifies the archive again and checks member paths before invoking Windows `tar`.
It writes `data/extracted/sen2gf3/Sen2GF3Floods/` containing:

```text
label/label_<id>.tif
gaofen3/after_gf3_hh_<id>.TIF
gaofen3/after_gf3_hv_<id>.TIF
sentinel2/before_s2_<id>.TIF
```

The source identifiers span 1–21,483. Preparation aligns SAR and optical inputs
to each label grid and creates `data/processed/sen2gf3/` with:

- `x_uint8.npy`: HH, HV, red, green, blue and NIR input channels;
- `labels_uint8.npy`, `support_uint8.npy`, `row_area_m2.npy`;
- `manifest.csv`, `preparation_complete.json`, `split_audit_passed.json`.

Labels are preserved. Pixel weights are geodesic area; the main comparison uses
common valid support. Duplicate/overlapping footprints and unassigned geographic
groups are screened before training. Event/site association uses the explicit
geographic rules in `prepare_sen2gf3.py`, not disaster dates inferred from filenames.
The expected retained set is 17,019 patches (8,786 development, 8,233 frozen core).

The original manifest SHA-256 is
`4ec15f03b063b615c06e0d1f585541f910e26b5a9ed4b364fe88cde9ca186757`.
The fresh-checkout review helper checks this fingerprint. If it differs, inspect
dataset version, geospatial libraries, serialization and preprocessing outputs;
do not substitute a different split to bypass the gate.

The target is agreement with the released binary inundation reference. The
historical label-semantics audit is included in `provenance/`; it documents the
benchmark's annotation process and the scope of the reference labels.

## Foundation checkpoint

Run `scripts/download_terramind.py` separately. It retrieves
`TerraMind_v1_tiny.pt` at a pinned upstream revision into
`models/pretrained/terramind_tiny/`, verifies 211,873,402 bytes and SHA-256
`e56ea9ebcd4451078b9ca4893d5cd8a89bbee376ae16829c3e7fbbbc76de0eba`.
The adaptation loads compatible pretrained tensors, including the RGB stem, while
the GF-3 HH/HV and NIR inputs use study-specific stems.

HISEA water data and the China event catalogue were investigated in the broader
project but are not inputs to this frozen main computational batch. Their
downloaders and archives are therefore not included. Event catalogues are not
pixel reference labels.
