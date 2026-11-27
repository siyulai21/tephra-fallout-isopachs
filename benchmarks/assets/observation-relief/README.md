# Elevation for the principal observation atlas

The map generator reprojects ice-surface elevations from NOAA NCEI’s ETOPO 2022 into each vent-centred equal-area frame. Land combines a pale elevation tint with northwest hillshade. Natural Earth supplies the coastline and the land and inland-water masks. The thickness legend refers to the site markers. [NOAA source and licence](https://doi.org/10.25921/fd45-gt74)

The worldwide background uses a 2 arc-minute grid made by averaging the NOAA 60 arc-second cells. Overlapping border samples support interpolation at tile seams and the antimeridian. For frames up to 120 km across, the generator uses a native 15 arc-second crop when one covers the frame. The global grid also covers newly added or moved details. The receipt lists the crop bounds, original NOAA GeoTIFFs and hashes for both the source files and the compressed arrays. Elevations are rounded to whole metres and floored at zero for display. [Source receipt](source.json)

To rebuild the pinned assets, put the fourteen named GeoTIFFs from the receipt in one directory and run this command with GDAL and the repository’s Python dependencies installed.

```sh
python -m benchmarks.build_observation_relief /path/to/geotiffs
python -m benchmarks.principal_input_figures
python -m benchmarks.principal_input_figures --verify
```
