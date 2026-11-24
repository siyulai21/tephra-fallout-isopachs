# Mazama · Directional ensemble

The principal bulk volume is **332 km³**, the whole-Earth integral of the median-thickness map. The conditional 95% interval for total fallout volume is **257–1.04e+03 km³** across complete posterior draws. **Numerical warning on the fitted deposit distribution.**

![Mazama, Directional ensemble. Full deposit, near the vent and thickness versus square root of cumulative area](deposit.png)

Each circle shows a measurement. Bounded sites take the colour of their stated limit. The curve counts every patch at or above each thickness. The volume also includes thinner fallout beyond the 0.5 cm line. [Thickness and cumulative area](thickness-area.csv) · [Map integral, resolution and source hashes](deposit.json)

The principal map uses the geographic continuation evaluated on all registered eruptions. [Geographic holdouts, structural alternatives and numerical checks](extent/report.json) · [Complete contour decisions](extent/support.json.gz) · [Predictions at withheld observations](extent/heldout.csv.gz)

## Earlier continuation and its diagnostics

The following figures and statistics describe the fitted parent and its broader continuation. The principal estimate and its interval appear above.

## Reach from the volcano

![Fixed and uncertain fan directions near the source](azimuth/median-isopachs.png)

Distances are great-circle kilometres to the first outward crossing on each compass ray. The area columns give the entire 10 cm footprint in the corresponding 45° sector, including detached areas. Together they distinguish a long point on a contour from an entire broad lobe.

| Bearing | Fixed 10 cm reach (km) | Uncertain 10 cm reach (km) | Fixed ≥10 cm area (km²) | Uncertain ≥10 cm area (km²) |
|---|---:|---:|---:|---:|
| N | 361.1 | 361.0 | 80,724 | 80,724 |
| NE | 503.0 | 502.6 | 143,880 | 143,706 |
| E | 212.3 | 212.3 | 22,656 | 22,651 |
| SE | 181.2 | 181.2 | 14,155 | 14,155 |
| S | 162.3 | 158.5 | 10,792 | 10,182 |
| SW | 181.2 | 155.4 | 12,047 | 9,391 |
| W | 157.4 | 148.0 | 9,874 | 8,674 |
| NW | 169.9 | 169.9 | 10,483 | 10,304 |

![Probability that a deposit reaches ten centimetres](azimuth/exceedance.png)

The 50% and 90% probability lines answer different questions. At a location on the 90% line, nine in ten model draws reach 10 cm. These are pointwise probabilities conditional on the model. Joint confidence over an entire region requires a different calculation.

## Withheld observations

Each row averages negative log predictive scores over equally weighted folds. Lower scores predict the withheld thicknesses and limits better. The passing-only columns exclude numerical failures. Coverage counts exact withheld thicknesses inside the uncertain-direction 95% predictive interval. A dash marks an unavailable score or coverage.

| Deletion | All estimable folds | Fixed | Uncertain | Passing folds | Fixed, passing | Uncertain, passing | Exact-site 95% coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| Geographic regions | 4 | 0.860 | 0.874 | 2 | 0.585 | 0.583 | 50/58 (86.2%) |
| Source quadrants | 3 | 0.878 | 0.903 | 2 | 1.028 | 1.043 | 56/58 (96.6%) |

[Predictions for every estimable held-out site](azimuth/heldout.csv). The numerical record includes all planned folds, row identities, unscorable directions, fit warnings and phase-integration checks.

## Inspected archives

The principal fit includes the interpreted upper limits below. Each core enters once. Other sites provide an occurrence check. The interval describes the modeled latent thickness at each site.

| Site | Interpreted upper limit (cm) | Median thickness (cm) | Conditional 95% interval (cm) |
|---|---:|---:|---:|
| Skull Creek dunes, Catlow Valley | 5 | 5.09 | 2.42–10.6 |
| Deep Creek peat pit, North Okanagan | 5 | 6.75 | 4.04–12.7 |
| Medicine Lake, B100NC-1 | 13.75 | 13.4 | 3.64–33 |
| Mahoney Lake, deep basin | 63 | 7.8 | 5.37–14.2 |
| Lodgepole Quarry | 50 | 111 | 67.2–188 |
| Simon Young Site 24 | 5 | 2.98 | 1.32–6.76 |
| Banff National Park, Saskatchewan Crossing | 25 | 5.13 | 2.7–14.6 |
| Beverley | 5 | 4.29 | 1.49–12.9 |
| South Saskatchewan River | 5 | 2.86 | 0.593–11 |
| Jesmond Bog | 1 | 2.6 | 0.921–4.9 |
| Willapa Canyon terrace, Brown Bear BB326-36-PC | 4 | 1.41 | 0.384–4.34 |
| Thunder Lake, North Cascades, Washington | 119.5 | 4.7 | 2.33–9.79 |
| Ridley Lake, North Cascades, Washington | 111 | 4.57 | 2.57–8.28 |
| Redmountain Lake, British Columbia | 111 | 1.44 | 0.494–3.48 |
| Dismal Lake, Idaho | 141 | 6.25 | 2.7–15.6 |
| Rocky Ridge Lake, Idaho | 161 | 5.16 | 1.87–14 |
| Horseshoe Lake, Idaho | 211 | 4.59 | 1.57–13.4 |
| Mount Baker RS–15, Pratt Creek upstream of Ridley Creek | 20 | 4.97 | 2.36–11.2 |
| Mount Baker south-flank pits between Easton and Squak glaciers | 30 | 4.95 | 2.43–10.7 |
| Pocket Lake, Mount Baker | 38 | 5 | 2.4–11.2 |
| Redfish Lake, RDF07-1A, Idaho | 8 | 2.8 | 1.5–5.43 |
| Redfish Lake, RDF07-4A, Idaho | 8 | 2.8 | 1.51–5.43 |
| Little Redfish Lake, LRFL08-02, Idaho | 8 | 2.79 | 1.47–5.42 |
| Pettit Lake | 6 | 2.79 | 1.51–5.29 |
| Hager Pond | 105 | 11.1 | 6.02–20.8 |
| Waits Lake | 141 | 13.8 | 6.93–28.7 |
| Huff Lake | 1080 | 10.2 | 6.08–18.4 |
| Fish Lake Valley, Leidy Creek peat core 1 | 2 | 0.413 | 0.0697–1.71 |
| Fish Lake Valley, Leidy Creek peat core 7 | 2.5 | 0.416 | 0.0707–1.72 |
| Fish Lake Valley, Furnace Creek hand auger | 1 | 0.348 | 0.0543–1.58 |
| Lily Lake | 45 | 9.23 | 5.13–18.1 |
| Dead Horse Lake | 27 | 72.8 | 41.8–135 |
| Carp Lake | 4 | 15.6 | 10–25.1 |
| Simon Young Site 22 | Occurrence only | 0.494 | 0.069–2.01 |
| Osgood Swamp | Occurrence only | 0.45 | 0.00021–5.08 |
| Simon Young Site 21 | Occurrence only | 1.54 | 0.581–5.92 |
| Paisley Cave | Occurrence only | 55.2 | 32.7–98.6 |
| Toketee Falls | Occurrence only | 363 | 198–726 |
| Prineville Reservoir | Occurrence only | 24.5 | 15.5–43 |
| Simon Young Site 20 | Occurrence only | 0.527 | 0.187–1.35 |
| Mitchell | Occurrence only | 19.2 | 11.8–33.5 |
| West Stayton | Occurrence only | 4.6 | 2.19–11 |
| Hubbard | Occurrence only | 4.86 | 2.75–8.91 |
| Palouse Falls | Occurrence only | 20.8 | 10.8–38.2 |
| Lake Superior | Occurrence only | 0.00245 | 6.14e-06–0.128 |
| Simon Young Site 19 | Occurrence only | 1.01 | 0.0145–8.51 |
| Arrow Lake | Occurrence only | 4.67 | 2.56–8.48 |
| Bow Lake | Occurrence only | 4.76 | 2.59–8.69 |
| Galata | Occurrence only | 1.64 | 0.059–12.5 |
| Rithets Bog, Vancouver Island | Occurrence only | 2.52 | 1.21–4.91 |
| Nordans Pond bog | Occurrence only | 2.13e-05 | 3.61e-09–0.153 |
| Burnaby Lake | Occurrence only | 2.18 | 0.979–5.35 |
| Lower Arrow Lake | Occurrence only | 13.9 | 7.23–27.1 |
| Fraser Canyon | Occurrence only | 4.33 | 2.33–7.62 |
| Drynoch Slide Archaeological Site | Occurrence only | 3.94 | 1.7–7.43 |
| Chase | Occurrence only | 5.27 | 3.18–10.2 |
| Cobb Lake, Kootenay National Park | Occurrence only | 9.66 | 3.92–24.5 |
| Goldeye Lake Fen | Occurrence only | 4.89 | 2.25–14.6 |
| Upper Pinto Fen | Occurrence only | 3.11 | 0.951–9.98 |
| Keephills Fen | Occurrence only | 4.27 | 1.66–13.7 |
| GISP2, Greenland | Occurrence only | 0.0338 | 0.000179–3.14 |
| Camp Century, Greenland | Occurrence only | 0.0314 | 0.000403–2.46 |
| Lake Superior, Isle Royale core | Occurrence only | 0.00324 | 8.22e-06–0.249 |
| Bloomingdale Bog | Occurrence only | 8.12e-05 | 1.19e-07–0.00891 |
| Sidney Bog core2 | Occurrence only | 3.64e-05 | 2.34e-08–0.00806 |
| Thin-Ice Pond | Occurrence only | 1.67e-05 | 6.01e-09–0.00735 |
| Balsam Creek kettle lake, BCK-284 | Occurrence only | 0.000341 | 4.85e-07–0.0416 |
| NGRIP ice core | Occurrence only | 0.0266 | 0.000253–3.22 |
| Placentia Bay AI07-10G | Occurrence only | 9.62e-06 | 2.46e-09–0.0808 |
| Redden Springs central wetland | Occurrence only | 1.37 | 0.197–7.74 |
| North Redden Springs | Occurrence only | 1.4 | 0.207–7.74 |
| An Loch Mor, Inis Oirr, MOR-T12 | Occurrence only | 0.0014 | 1.84e-07–0.732 |
| Nuangola Lake | Occurrence only | 7.4e-05 | 2.47e-07–0.00532 |
| Pound Cove Bog | Occurrence only | 2.13e-05 | 3.59e-09–0.153 |
| Southwest Pond Bog | Occurrence only | 2.11e-05 | 3.76e-09–0.152 |
| Beaver Lake | Occurrence only | 1.55 | 0.44–3.74 |
| Lake Sawyer | Occurrence only | 5.73 | 2.63–12.4 |
| Tarboo Lake | Occurrence only | 2.91 | 1.12–8.65 |
| Bull Run Lake | Occurrence only | 14.3 | 9.14–23.5 |
| Felker Lake, core FEL01 | Occurrence only | 1.96 | 0.588–4.53 |
| Young Table 1.1 distal site 1 | Occurrence only | 4.61 | 2.17–11.1 |
| Young Table 1.1 distal site 2 | Occurrence only | 4.97 | 2.86–8.9 |
| Young Table 1.1 distal site 17 | Occurrence only | 1.5 | 0.0612–11.8 |
| Young Table 1.1 distal site 19 | Occurrence only | 1.04 | 0.0138–8.75 |
| Young Table 1.1 distal site 20 | Occurrence only | 0.549 | 0.176–1.35 |
| Young Table 1.1 distal site 21 | Occurrence only | 1.54 | 0.596–5.4 |
| Young Table 1.1 distal site 22 | Occurrence only | 0.503 | 0.0631–2.07 |
| Young Table 1.1 distal site 25 | Occurrence only | 0.364 | 6.15e-05–4.63 |
| Dry Meadow fen | Occurrence only | 815 | 461–1.83e+03 |
| Crooked Meadow fen | Occurrence only | 798 | 460–1.72e+03 |
| FSLR 460 fen | Occurrence only | 779 | 467–1.59e+03 |
| Johnson Meadow fen | Occurrence only | 813 | 474–1.7e+03 |
| Jack Creek at Skookum Butte fen | Occurrence only | 812 | 482–1.67e+03 |
| Kootenai 24LN1125 A-776 BCV3 | Occurrence only | 7.96 | 3.82–16.6 |
| Kootenai canyon, 24LN1020 Jennings | Occurrence only | 7.96 | 3.82–16.6 |

## Checks and reproducibility

The shared contour audit found eligible bands under none of the screened support policies. Audit status is completed. [All contour decisions and deletion failures](azimuth/complete-contour-audit.json)

The spatial integration check passed at 16,384 points per scramble. The map used up to 64 posterior draws per chain and structure. Resolved contour areas passed the split-packet check. The joint-volume quantiles passed the split check.

Full-fit sampling warnings occurred for narrow_fans.

[Numerical report, all sector volumes and sensitivities](azimuth/report.json) · [Complete fields and per-field volumes](azimuth/fields.npz) · [Principal map grids](azimuth/median-grids.npz) · [Near-source comparison grids](azimuth/map-grid.npz) · [Fragment decisions](azimuth/fragment-decisions.csv) · [Bracketed coordinates](azimuth/bracketed-fragments.csv) · [File hashes](azimuth/artifacts.json)

## Related directional studies

- [Earlier fixed-direction ensemble and its separate calibration](ensemble/README.md)
- [Single-structure and proximal-cap sensitivities](single-structure/README.md)
- [Measured-fan continuation and inspected-site evidence](bearing/README.md)
