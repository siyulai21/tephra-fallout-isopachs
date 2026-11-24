# Near-vent shorelines

These packs contain land, lake, lake-island, pond and Antarctic ice-front polygons from GSHHG 2.3.7 by Paul Wessel and Walter H. F. Smith. A local panel uses full-resolution geography at up to 0.3 km per displayed pixel, high resolution at up to 1.5 km per pixel and intermediate resolution above that. All tiers cover the globe. [Source and resolution descriptions](https://www.soest.hawaii.edu/wessel/gshhg/) · [Archived dataset](https://doi.org/10.5281/zenodo.7007502) · [License](LICENSE.TXT)

The build clips the original split WGS84 polygons to ten-degree tiles and stores coordinates at one microdegree. It preserves the four land–water nesting levels and uses the Antarctic ice front. The index identifies the upstream ZIP and the SHA-256 of every pack. [Build script](../../../../benchmarks/build_local_shorelines.py) · [Pack inventory](index.json)
