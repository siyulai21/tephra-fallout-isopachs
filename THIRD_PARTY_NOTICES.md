# Third-party data and sources

Current maps bundle the [Natural Earth 1:10-million coastline](https://github.com/nvkelso/natural-earth-vector/blob/ace5fed0eaf3c6c03c951e75b439ba8fffbc218e/geojson/ne_10m_coastline.geojson), version 5.0.0 GeoJSON, losslessly compressed as `src/empirical_isopach/assets/ne_10m_coastline.geojson.gz`. [Pinned source, original/compressed SHA-256 and transformation record](src/empirical_isopach/assets/coastline-source.json).

The primary isopachs also colour land and lakes using polygons from the same 1:10-million release. [Pinned land and lake sources, hashes and display simplification](src/empirical_isopach/assets/land-source.json)

Natural Earth data are public domain. see [the terms of use](https://www.naturalearthdata.com/about/terms-of-use/). The scale is 1:10 million, not ten-metre accuracy. Shorelines supply geographic context only and never enter fitting or volume calculation. This attribution does not change ownership or licensing of the original code, measurements or historical figures.


Minoan source workbook: Karstens et al. (2023), [doi:10.1038/s41467-023-38176-3](https://doi.org/10.1038/s41467-023-38176-3), CC BY 4.0. Core-catalogue cross-check: Kutterolf et al. (2021), [doi:10.1594/PANGAEA.937486](https://doi.org/10.1594/PANGAEA.937486), CC BY 4.0. Full attribution and derived-data changes are documented in [the source record](data/sources/minoan/README.md) and [audit](data/audits/Minoan/README.md).

B-Tm supporting workbook: Yang et al. (2021), [doi:10.1007/s00445-021-01487-8](https://doi.org/10.1007/s00445-021-01487-8), downloaded from the [University of Cambridge repository](https://www.repository.cam.ac.uk/handle/1810/330433), which states a [CC BY 4.0 licence](https://creativecommons.org/licenses/by/4.0/). The original workbook is preserved unchanged; derived tables add source-row identities and comparisons with the current input. [Attribution, source checksum and observation qualifications](comparisons/source-revision/input-inventory/b-tm-workbook-review/README.md).
