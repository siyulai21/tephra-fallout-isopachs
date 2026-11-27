# Elevation for the principal observation atlas

The map generator reprojects surface elevation into each vent-centred equal-area frame. Green lowlands, tan uplands and brown ridges combine with northwest hillshade. Natural Earth supplies the coastline and the land and inland-water masks. The thickness legend refers to the site markers.

| Frame span | Preferred elevation grid |
|---|---|
| Up to 45 km | Copernicus GLO-90, nominal 3 arc-seconds |
| Up to 365 km | NOAA ETOPO 2022, 15 arc-seconds |
| Up to 1,450 km | NOAA ETOPO 2022, 60 arc-seconds |
| Worldwide fallback | NOAA ETOPO 2022, 120 arc-seconds averaged from the 60 arc-second grid |

The preferred crop must cover the entire frame. The renderer tries coarser grids when needed. The offshore tile at 14° N, 119° E is absent from the Copernicus feed. The overlapping crop uses ETOPO wherever Copernicus has no cells. Elevations are rounded to whole metres and floored at zero for display. Overlapping borders support global interpolation at tile seams and the antimeridian. [Source tiles, crop bounds and hashes](source.json)

NOAA ETOPO is distributed under CC0 1.0. [NOAA source and licence](https://doi.org/10.25921/fd45-gt74)

Produced using Copernicus WorldDEM™-90 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved.

The organisations in charge of the Copernicus programme by law or by delegation do not incur any liability for any use of the Copernicus WorldDEM™-90. Subsequent redistribution of the Copernicus-derived arrays and imagery is subject to the obligations in article 6 of the GLO-90 licence. [Copernicus GLO-90 licence](https://dataspace.copernicus.eu/sites/default/files/media/files/2025-06/copernicus_contributing_mission_data_access_v2_cop_dem_licenses.pdf)

To rebuild the pinned assets, put the named GeoTIFFs from the receipt in one directory and run this command with GDAL and the repository’s Python dependencies installed.

```sh
python -m benchmarks.build_observation_relief /path/to/geotiffs
python -m benchmarks.principal_input_figures
python -m benchmarks.principal_input_figures --verify
```
