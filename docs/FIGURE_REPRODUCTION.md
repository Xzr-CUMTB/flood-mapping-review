# Figure reproduction

All 16 figures use their complete original plotting inputs, rather than values
recovered from image colours. `FIGURE_INDEX.json` binds original renderer paths
and reference TIFF hashes. `PANEL_DATA_INDEX.csv` merges 96 source-mapping records
into 82 unique panels and identifies source files, transformations, statistical
units, denominators, release hashes and captions. The figure-specific manifests
and captions retain the full methodological context for each panel.

```text
python -m publication.redraw_figures --data DATA --output OUTPUT --final --verify-reference
python -m publication.redraw_figures --data DATA --output OUTPUT --figures Fig01,Fig03,S06 --final --verify-reference
```

Final replay uses Times New Roman. `--font Arial` generates an alternative font
version; the deposited reference TIFFs are the Times manuscript masters. Exact
comparison therefore uses the default font. PDF and SVG are emitted as well.

| Figures | Source data | Renderer |
| --- | --- | --- |
| Fig01–Fig06 | `figure_data/FigXX/source_data/` | `paper_figures/main/FigXX/scripts/render_figure.py` |
| S01–S10 | `figure_data/SXX/source_data/` | `paper_figures/supplementary/SXX/scripts/render_sxx.py` |

The wrapper copies inputs and renderers into OUTPUT before executing them.
Original data and reference TIFFs are never overwritten. `figure_reproduction.json`
records source-code hashes, dimensions, DPI, RGB pixel comparison and file hashes.
The recorded original-environment replay matched all 16 complete TIFF SHA-256s.

Historical `derive_*` scripts document how plotting inputs were produced from
the original project tree. They are not required for replay and may expect that
tree. The supported data-package entry point is `publication.redraw_figures`.
Statistical evaluation reconstruction is a separate command and does not change
any data supplied to the figures.
