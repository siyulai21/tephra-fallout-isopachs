# Mazama · Directional ensemble

The principal bulk volume is **277 km³**, the whole-Earth integral of the median-thickness map. The conditional thickness-band range is **188–430 km³**. Its endpoints integrate the pointwise 25th and 75th percentiles of latent thickness over Earth. **Numerical warning on the fitted deposit distribution.**

![Mazama, Directional ensemble. Full deposit, near the vent and thickness versus square root of cumulative area](deposit.png)

Each circle shows a measurement. Bounded sites take the colour of their stated limit. The curve counts every patch at or above each thickness. The 1 cm footprint includes 267 of 300 modeled locations. The thickest displayed line is 4096 cm and encloses 5 fitted locations. The volume includes fallout beyond the displayed contours. [Median thickness grid](median-grid.npz) · [Thickness and cumulative area](thickness-area.csv) · [Map integral, resolution and source hashes](deposit.json)

The principal map uses the geographic continuation evaluated on all registered eruptions. [Geographic holdouts, structural alternatives and numerical checks](extent/report.json) · [Complete contour decisions](extent/support.json.gz) · [Predictions at withheld observations](extent/heldout.csv.gz)

## Earlier continuation and its diagnostics

The following figures and statistics describe the fitted parent and its broader continuation. The principal estimate and its interval appear above.

## Reach from the volcano

![Fixed and uncertain fan directions near the source](azimuth/median-isopachs.png)

Distances are great-circle kilometres to the first outward crossing on each compass ray. The area columns give the entire 10 cm footprint in the corresponding 45° sector, including detached areas. Together they distinguish a long point on a contour from an entire broad lobe.

| Bearing | Fixed 10 cm reach (km) | Uncertain 10 cm reach (km) | Fixed ≥10 cm area (km²) | Uncertain ≥10 cm area (km²) |
|---|---:|---:|---:|---:|
| N | 143.2 | 143.2 | 15,940 | 15,857 |
| NE | 718.9 | 719.5 | 164,101 | 164,314 |
| E | 231.8 | 232.3 | 25,335 | 25,079 |
| SE | 207.5 | 207.4 | 18,149 | 18,149 |
| S | 147.8 | 143.7 | 9,336 | 8,852 |
| SW | 162.9 | 141.4 | 9,657 | 7,781 |
| W | 140.3 | 133.6 | 7,863 | 7,143 |
| NW | 148.8 | 148.7 | 8,219 | 8,073 |

![Probability that a deposit reaches ten centimetres](azimuth/exceedance.png)

The 50% and 90% probability lines answer different questions. At a location on the 90% line, nine in ten model draws reach 10 cm. These are pointwise probabilities conditional on the model. Joint confidence over an entire region requires a different calculation.

## Withheld observations

Each row averages negative log predictive scores over equally weighted folds. Lower scores predict the withheld thicknesses and limits better. The passing-only columns exclude numerical failures. Coverage counts exact withheld thicknesses inside the uncertain-direction 95% predictive interval. A dash marks an unavailable score or coverage.

| Deletion | All estimable folds | Fixed | Uncertain | Passing folds | Fixed, passing | Uncertain, passing | Exact-site 95% coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| Geographic regions | 4 | 0.774 | 0.772 | 2 | 0.866 | 0.875 | 40/46 (87.0%) |
| Source quadrants | 3 | 1.035 | 1.055 | 2 | 0.840 | 0.856 | 42/46 (91.3%) |

[Predictions for every estimable held-out site](azimuth/heldout.csv). The numerical record includes all planned folds, row identities, unscorable directions, fit warnings and phase-integration checks.

## Inspected archives

The principal fit includes the interpreted upper limits below. Each core enters once. Other sites provide an occurrence check. The interval describes the modeled latent thickness at each site.

| Site | Interpreted upper limit (cm) | Median thickness (cm) | Conditional 95% interval (cm) |
|---|---:|---:|---:|
| Fish Lake | 5 | 8.32 | 4.82–14.2 |
| Jenny Lake, Wyoming | 54.5 | 0.7 | 0.273–1.58 |
| Tumalo Lake | 50 | 38.2 | 18–79.4 |
| Round Lake | 14 | 11.1 | 6.47–19.2 |
| Davis Lake | 6 | 5.12 | 2.32–10.4 |
| Bonaparte Meadows | 12 | 2.9 | 0.118–4.32 |
| Twobit Creek | 8 | 2.38 | 0.419–3.73 |
| Lavington | 10 | 1.46 | 0.0362–2.36 |
| Independence Lake | Occurrence only | 1.65 | 0.53–4.7 |
| Lake Cleveland | Occurrence only | 1.6 | 0.508–4.19 |
| Jesmond Bog | 1 | 0.614 | 0.249–1.31 |
| Lofty Lake | 0.2 | 0.462 | 0.146–0.969 |
| Shadow Lake SL11C | 14 | 4.31 | 1.74–9.65 |
| Sunrise Lake UL11D | 8 | 4.24 | 1.79–9.21 |
| Round Lake A17/A18 composite, Washington | 60 | 3.09 | 0.157–4.61 |
| Long Lake LOL14C composite, eastern Cascades | 7 | 5.18 | 1.53–7.69 |
| Cleland Lake F-09 intermediate core | Occurrence only | 2.37 | 1.3–3.59 |
| Patjens Lakes south-shore fen | 4 | 9.41 | 5.8–15.6 |
| Dog Lake Oregon 2016 composite | Occurrence only | 18.7 | 12.3–30.8 |
| White Pine Marsh WPMA 2018 composite | Occurrence only | 17 | 10.9–27.2 |
| Skull Creek dunes, Catlow Valley | 5 | 9.74 | 5.7–17.1 |
| Deep Creek peat pit, North Okanagan | 5 | 1.26 | 0.0477–2.16 |
| Beaver Lake, Linn County, Willamette Valley | 2 | 2.94 | 1.46–6.1 |
| Medicine Lake, B100NC-1 | 13.75 | 11 | 1.73–27.8 |
| Mahoney Lake, deep basin | 63 | 2.05 | 0.0892–3.14 |
| Lodgepole Quarry | 50 | 48.7 | 32.8–77.4 |
| Simon Young Site 24 | 5 | 3.5 | 1.74–6.9 |
| Banff National Park, Saskatchewan Crossing | 25 | 0.936 | 0.0194–1.66 |
| Beverley | 5 | 0.778 | 0.32–1.5 |
| South Saskatchewan River | 5 | 5.43 | 1.04–13.6 |
| Willapa Canyon terrace, Brown Bear BB326-36-PC | 4 | 1.14 | 0.000419–3.19 |
| Thunder Lake, North Cascades, Washington | 119.5 | 1.86 | 0.861–3.64 |
| Ridley Lake, North Cascades, Washington | 111 | 1.62 | 0.826–2.96 |
| Windy Lake, British Columbia | 86 | 2.14 | 0.346–3.38 |
| Thunder Lake, Cariboo Mountains, British Columbia | 32 | 0.463 | 0.0321–1.15 |
| Redmountain Lake, British Columbia | 111 | 0.17 | 0.0482–0.595 |
| Dismal Lake, Idaho | 141 | 10.3 | 4.37–19 |
| Rocky Ridge Lake, Idaho | 161 | 10.3 | 4.7–19.3 |
| Horseshoe Lake, Idaho | 211 | 9.78 | 4.36–18.7 |
| Lower Frozen Lake, British Columbia | 8 | 1.53 | 0.576–3.82 |
| Eagle Lake, Mount Revelstoke, British Columbia | 15 | 1.16 | 0.0104–1.91 |
| Mount Baker RS–15, Pratt Creek upstream of Ridley Creek | Occurrence only | 3.07 | 1.68–5.47 |
| Mount Baker south-flank pits between Easton and Squak glaciers | 30 | 2.94 | 1.69–5.02 |
| Pocket Lake, Mount Baker | 38 | 3.01 | 1.7–5.23 |
| North Fork Whychus Creek, downstream cutbank, Oregon | 66 | 11 | 6.13–21.2 |
| Redfish Lake, RDF07-1A, Idaho | 8 | 2.78 | 1.45–5.84 |
| Redfish Lake, RDF07-4A, Idaho | 8 | 2.78 | 1.45–5.87 |
| Little Redfish Lake, LRFL08-02, Idaho | 8 | 2.79 | 1.42–5.93 |
| Pettit Lake | 6 | 2.72 | 1.52–5.43 |
| Marion Lake | 5 | 1.67 | 0.627–3.77 |
| Surprise Lake | 3.5 | 1.66 | 0.623–3.74 |
| Hager Pond | 105 | 11.6 | 5.31–23.1 |
| Waits Lake | 141 | 12.3 | 6.5–22.8 |
| Huff Lake | 1080 | 11.8 | 5.06–24.9 |
| Fish Lake Valley, Leidy Creek peat core 1 | Occurrence only | 0.959 | 0.341–2.83 |
| Fish Lake Valley, Leidy Creek peat core 7 | Occurrence only | 0.966 | 0.346–2.84 |
| Fish Lake Valley, Furnace Creek hand auger | 1 | 0.836 | 0.273–2.72 |
| Lily Lake | 45 | 12.6 | 7.25–20.4 |
| Dead Horse Lake | Occurrence only | 42 | 25.9–71.8 |
| Carp Lake | 4 | 7.56 | 1.41–11.9 |
| Copper Lake, core COPL1 | 1 | 1.51 | 0.651–2.47 |
| Lower Burstall Lake, core LBSL2 | 5 | 4.26 | 2.44–6.97 |
| Lake Louise, core LL1 | 7 | 1.31 | 0.435–2.25 |
| Sibbald Lake, core SIBL1 | 50 | 2.52 | 1.36–6 |
| Summit Lake, Waterton, core SUML1 | 143 | 8.89 | 4.25–16.5 |
| Begbie Lake, Greater Victoria water supply core | 1 | 1.1 | 0.577–1.88 |
| Sheep Mountain Bog | 42 | 8.34 | 3.46–17.2 |
| Matheson Lake | 313 | 1.23 | 0.649–2.06 |
| Dailey Lake Bog | 345 | 10.5 | 5.41–19.7 |
| Starvation Flat bog | 210 | 11.6 | 5.3–23.9 |
| Prior Lake | 127 | 1.02 | 0.558–1.89 |
| Pike Lake  | 329 | 1.01 | 0.555–1.88 |
| Maltby Lake  | 244 | 0.99 | 0.537–1.89 |
| Upper Kananaskis Lake | 2 | 5.05 | 2.66–9.05 |
| Frederick Lake | 28 | 2.57 | 1.34–6.29 |
| Johnson Lake | 156 | 1.84 | 1–3.19 |
| Nordegg Bridge | 5 | 0.832 | 0.0449–1.54 |
| Hasse Lake | 1 | 0.647 | 0.183–1.27 |
| Lake of the Woods, Cathedral Park core A12 | 4 | 1.93 | 0.214–3.18 |
| Pyramid Lake, Cathedral Park core B7b | 20 | 1.93 | 0.215–3.19 |
| Glacier Lake, Cathedral Park core C5 | 13 | 1.92 | 0.221–3.18 |
| Quiniscoe Lake, Cathedral Park core D8 | 28 | 1.91 | 0.219–3.18 |
| E.L. Smith project, southwest Edmonton | 100 | 0.754 | 0.295–1.44 |
| Goldeye Lake, winter 1981 lake core | 575 | 0.806 | 0.026–1.48 |
| Mt. Kobau Lake near Osoyoos | 10 | 2.18 | 0.105–3.32 |
| Mica Creek, Columbia River peat terrace | 5 | 0.651 | 0.0137–1.32 |
| West Calgary, Harrison high-terrace pit | 5 | 3.09 | 1.34–8.21 |
| Upper Elk Valley terrace, north of Elkford | 10 | 6.44 | 3.34–11.4 |
| Kilpoola Lake, 1996 core 2 | 33 | 2.37 | 0.0894–3.53 |
| Frog Lake, Greater Victoria water supply core | 225 | 1.12 | 0.598–1.92 |
| Swanson Lake, Greater Victoria water supply core | 268 | 1.15 | 0.562–1.96 |
| Worley Lake, Greater Victoria water supply core | 190 | 1.17 | 0.52–2.04 |
| Stowell Lake, Salt Spring Island | 2 | 0.89 | 0.455–1.81 |
| Jenny Lake, JEN13-2E | 4 | 0.703 | 0.275–1.59 |
| Jenny Lake, JEN15-3A | 63 | 0.702 | 0.274–1.59 |
| Jenny Lake, JEN16-1A | 48 | 0.702 | 0.275–1.59 |
| Jenny Lake, JEN16-2 | 54 | 0.705 | 0.276–1.59 |
| On-off Lake core 08-Onoff02 | 82 | 0.237 | 0.0651–0.831 |
| NEL01, small lake near Nelson | 6 | 2.81 | 1.47–4.14 |
| NEL03, subalpine lake near Nelson | Occurrence only | 7.86 | 4.4–13.8 |
| Castor Lake, 2003 recovery | 30 | 2.96 | 0.108–4.34 |
| Foy Lake, core A, drive 4 | 8 | 8.83 | 3.61–16.3 |
| Foy Lake, core C | 109 | 8.83 | 3.61–16.3 |
| Simon Young Site 22 | Occurrence only | 0.995 | 0.168–3.08 |
| Osgood Swamp | Occurrence only | 1.05 | 0.000581–6.78 |
| Simon Young Site 21 | Occurrence only | 1.21 | 0.371–3.6 |
| Paisley Cave | Occurrence only | 23.7 | 15–38.4 |
| Toketee Falls | Occurrence only | 274 | 149–515 |
| Prineville Reservoir | Occurrence only | 28.7 | 12.4–67.2 |
| Simon Young Site 20 | Occurrence only | 0.486 | 0.184–1.35 |
| Mitchell | Occurrence only | 18.7 | 10.8–32 |
| West Stayton | Occurrence only | 3.26 | 1.53–7.09 |
| Hubbard | Occurrence only | 4.25 | 2.35–8.01 |
| Palouse Falls | Occurrence only | 17.1 | 8.73–33.3 |
| Lake Superior | Occurrence only | 0.0109 | 5.07e-05–0.244 |
| Simon Young Site 19 | Occurrence only | 3.71 | 0.491–18.8 |
| Arrow Lake | Occurrence only | 3.5 | 1.99–5.81 |
| Bow Lake | Occurrence only | 3.53 | 2.02–5.86 |
| Galata | Occurrence only | 6.4 | 1.92–15.4 |
| Rithets Bog, Vancouver Island | Occurrence only | 1.05 | 0.582–1.92 |
| Nordans Pond bog | Occurrence only | 0.00666 | 2.07e-05–3.47 |
| Burnaby Lake | Occurrence only | 1.45 | 0.655–3.21 |
| Lower Arrow Lake | Occurrence only | 2.66 | 1.06–4.14 |
| Fraser Canyon | Occurrence only | 1.53 | 0.575–3.8 |
| Friday's Site | Occurrence only | 5.45 | 1.22–14.3 |
| Drynoch Slide Archaeological Site | Occurrence only | 0.796 | 0.355–1.97 |
| Chase | Occurrence only | 0.912 | 0.0717–1.87 |
| Cobb Lake, Kootenay National Park | Occurrence only | 4.91 | 2.46–9.15 |
| Bow River | Occurrence only | 2.23 | 1.29–4.89 |
| Lake Louise | Occurrence only | 1.31 | 0.435–2.25 |
| James Pass | Occurrence only | 1.25 | 0.513–2.12 |
| Goldeye Lake Fen | Occurrence only | 0.806 | 0.026–1.48 |
| Boss Hill near Buffalo Lake | Occurrence only | 1.52 | 0.688–4.32 |
| Upper Pinto Fen | Occurrence only | 0.304 | 0.00445–0.943 |
| Keephills Fen | Occurrence only | 0.613 | 0.127–1.24 |
| GISP2, Greenland | Occurrence only | 3.64e-05 | 1.5e-07–0.0023 |
| Camp Century, Greenland | Occurrence only | 1.84e-05 | 6.05e-14–0.00169 |
| Lake Superior, Isle Royale core | Occurrence only | 0.0267 | 0.000368–0.925 |
| Bloomingdale Bog | Occurrence only | 0.000288 | 4.41e-07–0.0154 |
| Sidney Bog core2 | Occurrence only | 0.000201 | 6.02e-08–0.017 |
| Thin-Ice Pond | Occurrence only | 0.000125 | 2.16e-08–0.0148 |
| Balsam Creek kettle lake, BCK-284 | Occurrence only | 0.00194 | 2.84e-06–0.0782 |
| NGRIP ice core | Occurrence only | 1.65e-05 | 2.76e-11–0.0016 |
| Placentia Bay AI07-10G | Occurrence only | 0.0024 | 4.28e-06–2.26 |
| Redden Springs central wetland | Occurrence only | 1.23 | 0.175–6.3 |
| North Redden Springs | Occurrence only | 1.24 | 0.185–6.27 |
| An Loch Mor, Inis Oirr, MOR-T12 | Occurrence only | 0.002 | 1.19e-07–1.35 |
| Nuangola Lake | Occurrence only | 8.05e-05 | 1.08e-06–0.00404 |
| Pound Cove Bog | Occurrence only | 0.00673 | 2.09e-05–3.48 |
| Southwest Pond Bog | Occurrence only | 0.0065 | 2.01e-05–3.48 |
| Beaver Lake | Occurrence only | 1.12 | 0.364–2.21 |
| Lake Sawyer | Occurrence only | 4.08 | 2.15–7.32 |
| Tarboo Lake | Occurrence only | 2.33 | 0.876–5.76 |
| Bull Run Lake | Occurrence only | 8.03 | 5.35–12 |
| Felker Lake, core FEL01 | Occurrence only | 0.418 | 0.144–1.02 |
| Young Table 1.1 distal site 1 | Occurrence only | 3.28 | 1.53–7.15 |
| Young Table 1.1 distal site 2 | Occurrence only | 4.39 | 2.44–8.26 |
| Young Table 1.1 distal site 17 | Occurrence only | 6.07 | 1.73–15.1 |
| Young Table 1.1 distal site 19 | Occurrence only | 3.83 | 0.547–18.8 |
| Young Table 1.1 distal site 20 | Occurrence only | 0.478 | 0.175–1.28 |
| Young Table 1.1 distal site 21 | Occurrence only | 1.22 | 0.394–3.5 |
| Young Table 1.1 distal site 22 | Occurrence only | 1.01 | 0.149–3.18 |
| Young Table 1.1 distal site 25 | Occurrence only | 0.889 | 0.000178–6.39 |
| Dry Meadow fen | Occurrence only | 911 | 503–2.07e+03 |
| Crooked Meadow fen | Occurrence only | 883 | 498–1.97e+03 |
| FSLR 460 fen | Occurrence only | 847 | 495–1.8e+03 |
| Johnson Meadow fen | Occurrence only | 880 | 505–1.93e+03 |
| Jack Creek at Skookum Butte fen | Occurrence only | 846 | 502–1.76e+03 |
| Kootenai 24LN1125 A-776 BCV3 | Occurrence only | 9.48 | 3.76–20.7 |
| Kootenai canyon, 24LN1020 Jennings | Occurrence only | 9.48 | 3.76–20.7 |
| Sasquatch Lake NEL02 | Occurrence only | 2.84 | 1.55–4.14 |
| On-off Lake core 08-Onoff01 | Occurrence only | 0.237 | 0.0651–0.831 |
| On-off Lake core 08-Onoff03 | Occurrence only | 0.237 | 0.0651–0.831 |
| Friday's Site, UA 2704 | Occurrence only | 5.45 | 1.22–14.3 |
| Bow River ~2 km downstream from Lac des Arcs, UA 2820 | Occurrence only | 2.23 | 1.29–4.89 |
| Bow River ~2 km downstream from Lac des Arcs, UA 2821 | Occurrence only | 2.23 | 1.29–4.89 |
| James Pass, UA 2701 | Occurrence only | 1.25 | 0.513–2.12 |
| James Pass, UA 2702 | Occurrence only | 1.25 | 0.513–2.12 |
| James Pass peel, UA 2851 | Occurrence only | 1.25 | 0.513–2.12 |
| James Pass peel, UA 2854 | Occurrence only | 1.25 | 0.513–2.12 |
| Battle River/Iron Creek valley, UA 2927 | Occurrence only | 2.29 | 0.479–9.99 |
| Battle River/Iron Creek valley, UA 2928 | Occurrence only | 2.28 | 0.481–9.87 |
| Brule Lake , UA 2906 | Occurrence only | 0.379 | 0.00423–1 |
| Maskuta Creek, Hinton, UA 2930 | Occurrence only | 0.382 | 0.00336–0.988 |
| near Grand Cache, UA 2861 | Occurrence only | 0.205 | 0.0104–0.748 |
| near Grand Cache, UA 2858 | Occurrence only | 0.205 | 0.00708–0.761 |
| Boss Hill Site (Buffalo Lake), UA 2963 | Occurrence only | 1.52 | 0.688–4.32 |
| On-Off Lake (Castle Creek Glacier), Jensen S1 literature entry | Occurrence only | 0.237 | 0.0651–0.832 |
| Frenchman River Valley near Val Marie, Jensen S1 literature entry | Occurrence only | 3.89 | 0.866–14 |
| Effingham Inlet, west coast Vancouver Island, Jensen S1 literature entry | Occurrence only | 0.506 | 0.0953–1.25 |
| Lake Stowell, Saltspring Island, Jensen S1 literature entry | Occurrence only | 0.89 | 0.455–1.81 |
| Dog Lake, Kootenay National Park, Jensen S1 literature entry | Occurrence only | 4.62 | 2.21–9.16 |
| Upper Pinto Fen on Highway 40, Jensen S1 literature entry | Occurrence only | 0.302 | 0.0047–0.937 |
| Ponoka buried soil, Ivarson 1972 | Occurrence only | 1.29 | 0.62–2.97 |
| Lower Decker Lake | 300 | 2.74 | 1.46–5.65 |
| Moscow, Idaho | 60 | 11.8 | 5.59–21.1 |
| Lake Washington | 1100 | 3.26 | 1.8–5.73 |

## Checks and reproducibility

The shared contour audit found eligible bands under none of the screened support policies. Audit status is completed. [All contour decisions and deletion failures](azimuth/complete-contour-audit.json)

The spatial integration check passed at 16,384 points per scramble. The map used up to 128 posterior draws per chain and structure. Resolved contour areas passed the split-packet check. The joint-volume quantiles passed the split check.

Full-fit sampling warnings occurred for narrow_missing_fan.

[Numerical report, all sector volumes and sensitivities](azimuth/report.json) · [Complete fields and per-field volumes](azimuth/fields.npz) · [Principal map grids](azimuth/median-grids.npz) · [Near-source comparison grids](azimuth/map-grid.npz) · [Fragment decisions](azimuth/fragment-decisions.csv) · [Bracketed coordinates](azimuth/bracketed-fragments.csv) · [File hashes](azimuth/artifacts.json)

## Related directional studies

- [Earlier fixed-direction ensemble and its separate calibration](ensemble/README.md)
- [Single-structure and proximal-cap sensitivities](single-structure/README.md)
- [Measured-fan continuation and inspected-site evidence](bearing/README.md)
