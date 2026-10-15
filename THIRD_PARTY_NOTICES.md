# Third-party data and software

- **Sen2GF3Floods**: public dataset DOI [10.57760/sciencedb.25341](https://doi.org/10.57760/sciencedb.25341),
  associated Scientific Data article [10.1038/s41597-026-06929-6](https://doi.org/10.1038/s41597-026-06929-6).
  The project source record identifies version 2 and CC BY 4.0. Cite the dataset
  and article, retain attribution and check the publisher's current terms when obtaining data.
  This repository does not redistribute satellite imagery or labels.
- **TerraMind**: [IBM/ESA geospatial TerraMind](https://github.com/IBM/terramind)
  and [TerraMind-1.0-tiny checkpoint](https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-tiny).
  The downloader pins revision `2b5ac0a3ed7dd7e922ccfd595b56607f342df343` and verifies
  SHA-256. The upstream model is distributed under Apache-2.0; pretrained weights
  are downloaded separately and not included here.
- **TerraTorch**: [IBM/terratorch](https://github.com/IBM/terratorch), installed as
  version 1.2.6; external dependency under Apache-2.0. The repository contains the
  study's adapter, not a vendored copy of TerraTorch or TerraMind.
- PyTorch, TorchGeo, NumPy, pandas, SciPy, scikit-learn, Rasterio, GDAL, PROJ,
  pyproj and other installed dependencies retain their upstream licenses.

Dependency installation and data/model downloading are explicit user-run stages.
No account token, API credential or private service is required by the main data downloader.
