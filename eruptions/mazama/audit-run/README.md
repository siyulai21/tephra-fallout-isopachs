# Mazama model results on 244 observations

This interim report covers the two regional models and the global nested-isopach model. All three use 58 measurements, 153 lower bounds and 33 upper bounds. Eight observations from Great Basin field sections and lake cores enter this version. The directional and spherical calculations and the 33 source-interpretation studies are in progress. [Reviewed input](../../../data/eruptions/Mazama_reviewed.csv) · [Source audit](../../../data/audits/Mazama/README.md) · [Five-family calculation](https://github.com/siyulai21/tephra-fallout-isopachs/actions/runs/45261422486)

| Method | Domain | Median-map volume (km³) | Conditional 95% interval (km³) | Geographic score ↓ | Withheld measured beds covered |
|---|---|---:|---:|---:|---:|
| GP ensemble | 1,556,082 km² observation hull | 409.47 | 332.36–835.00 | 1.166 | 43/58 (74.1%) |
| Cubic spline | 1,556,082 km² observation hull | 2,138.33 | 1,125.17–4,716.92 | 2.634 | 47/58 (81.0%) |
| Bayesian nested isopachs | Earth, conditional continuation | 270.95 | 186.94–724.35 | 1.011 | 42/58 (72.4%) |

The regional and nested volume integrations passed their numerical checks. All three models undercovered withheld measurements at the nominal 95% level. None admitted a complete contour band under the strict, standard or permissive support policies. [Regional fits and folds](regional-report.json) · [Nested validation](nested-extent.json) · [Nested integration](nested-map.json) · [Regional contour decisions](regional-support.json.gz) · [Nested contour decisions](nested-support.json.gz)

![GP and spline on the common observation hull](regional-isopachs.png)

![Conditional global nested isopachs](nested-deposit.png)

## Yellowstone Lake core detail

The 4–5 mm beds reported by Morgan et al. occur in ten cores at eight positions. Seven positions enter the fit. Two pairs of cores share positions. The irregular bed at the YL16-6A slump toe is tested separately. [Core coordinates, depths and identity](../../../data/audits/Mazama/YELLOWSTONE.md)

![Locations and individual Mazama ash beds at Yellowstone Lake](../../../data/audits/Mazama/yellowstone-cores.png)

[Young’s mapped field sections](../../../data/audits/Mazama/YOUNG_AND_FIELD_SECTIONS.md) · [Earlier 236-observation checkpoint](https://github.com/siyulai21/tephra-fallout-isopachs/tree/f477a71c28e0074d80caa4ebaacff88c1142a645/eruptions/mazama/audit-run)
