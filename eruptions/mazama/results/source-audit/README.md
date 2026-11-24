# Mazama source interpretation tests

[Distal field review, matched volumes and local thicknesses](distal-facies-results.md)

## Sensitivity to the geological interpretation

Each endpoint below is a separate refit in km³. The GP and spline use the fixed footprint of the 26 historical measurements for this comparison. The other three models integrate over Earth. The full table gives each fit’s own conditional interval and numerical checks. [All source refits](source-sensitivity-volumes.csv)

| Family | Comparable principal map | Smallest source-scenario map | Largest source-scenario map | Scenarios with a numerical warning |
|---|---:|---:|---:|---:|
| GP ensemble | 91.2 | 33.7 ([omit-young-proximal](sensitivities/omit-young-proximal/)) | 222.2 ([omit-all-archive-ceilings](sensitivities/omit-all-archive-ceilings/)) | 0/15 |
| Cubic spline | 194.5 | 105.7 ([omit-young-proximal](sensitivities/omit-young-proximal/)) | 395.6 ([omit-all-archive-ceilings](sensitivities/omit-all-archive-ceilings/)) | 8/15 |
| Directional ensemble | 214.5 | 68.8 ([omit-young-proximal](sensitivities/omit-young-proximal/)) | 332.8 ([omit-all-archive-ceilings](sensitivities/omit-all-archive-ceilings/)) | 11/15 |
| Spherical mixture | 240.6 | 163.4 ([omit-young-proximal](sensitivities/omit-young-proximal/)) | 371.2 ([restore-thick-archive-readings](sensitivities/restore-thick-archive-readings/)) | 1/15 |
| Bayesian nested corridors | 175.3 | 103.4 ([omit-young-proximal](sensitivities/omit-young-proximal/)) | 274.6 ([prior-244](sensitivities/prior-244/)) | 4/15 |

The endpoint range describes the tested source choices. Sampling intervals and geographic prediction have separate assessments. [Withheld site predictions](heldout-source-observations.csv) · [Probabilities assigned to withheld upper and lower bounds](heldout-bound-probabilities.csv) · [Site-by-site changes](local-thickness-comparison.csv) · [All contour crossings, compressed CSV](local-crossing-comparison.csv.gz)

[Case home](../../README.md) · [Evidence and calculations](../../../../data/audits/Mazama/BOUND_REVIEW.md) · [Numerical table](source-sensitivity-volumes.csv)

The common input contains 271 observations, including 31 measured beds, 137 lower bounds and 103 upper bounds. The 15 source refits test preservation, ash identity, interval endpoints and shared site positions. Global models refit each source scenario using continuation calibration that excludes Mazama. The regional refits keep the principal chosen settings and compare volumes on the original 26-measurement footprint.

| Method | Scenario | Median-map volume (km³) | Conditional 95% interval (km³) | Draw checks | Map integration |
|---|---|---:|---:|---|---|
| directional | alternate-field-bounds | 215 | 155–661 | False | True |
| directional | before-field-243 | 218 | 157–686 | False | True |
| directional | cathedral-one | 215 | 156–635 | False | True |
| directional | half-capture-ceilings | 220 | 161–638 | True | True |
| directional | jenny-omit | 216 | 151–688 | False | True |
| directional | jenny-one | 214 | 152–633 | True | True |
| directional | omit-all-archive-ceilings | 333 | 262–2.47e+03 | True | True |
| directional | omit-basin-beds | 201 | 151–638 | True | True |
| directional | omit-qualified-identities | 214 | 157–593 | False | True |
| directional | omit-reinterpreted-ceilings | 248 | 192–874 | False | True |
| directional | omit-young-proximal | 68.8 | 57.2–456 | False | True |
| directional | principal | 214 | 154–642 | False | True |
| directional | prior-244 | 332 | 257–1.04e+03 | False | True |
| directional | restore-thick-archive-readings | 323 | 217–934 | False | True |
| directional | restore-williams | 220 | 155–636 | False | True |
| directional | yellowstone-one | 211 | 163–687 | False | True |
| nested-isopachs | alternate-field-bounds | 177 | 105–584 | True | True |
| nested-isopachs | before-field-243 | 196 | 110–757 | True | True |
| nested-isopachs | cathedral-one | 175 | 122–437 | False | True |
| nested-isopachs | half-capture-ceilings | 180 | 108–474 | True | True |
| nested-isopachs | jenny-omit | 181 | 101–576 | True | True |
| nested-isopachs | jenny-one | 187 | 108–555 | True | True |
| nested-isopachs | omit-all-archive-ceilings | 223 | 143–846 | True | True |
| nested-isopachs | omit-basin-beds | 159 | 111–408 | False | True |
| nested-isopachs | omit-qualified-identities | 238 | 112–423 | False | True |
| nested-isopachs | omit-reinterpreted-ceilings | 183 | 111–561 | True | True |
| nested-isopachs | omit-young-proximal | 103 | 61.1–294 | True | True |
| nested-isopachs | principal | 175 | 106–605 | True | True |
| nested-isopachs | prior-244 | 275 | 187–724 | True | True |
| nested-isopachs | restore-thick-archive-readings | 208 | 125–746 | True | True |
| nested-isopachs | restore-williams | 194 | 109–581 | True | True |
| nested-isopachs | yellowstone-one | 237 | 123–724 | False | True |
| spherical-mixture | alternate-field-bounds | 242 | 154–1.28e+03 | True | True |
| spherical-mixture | before-field-243 | 240 | 152–1.38e+03 | True | True |
| spherical-mixture | cathedral-one | 237 | 151–1.42e+03 | True | True |
| spherical-mixture | half-capture-ceilings | 239 | 155–1.39e+03 | True | True |
| spherical-mixture | jenny-omit | 244 | 153–1.97e+03 | True | True |
| spherical-mixture | jenny-one | 242 | 148–1.61e+03 | True | True |
| spherical-mixture | omit-all-archive-ceilings | 329 | 203–2.75e+03 | True | True |
| spherical-mixture | omit-basin-beds | 306 | 158–5.55e+03 | True | True |
| spherical-mixture | omit-qualified-identities | 240 | 152–1.55e+03 | True | True |
| spherical-mixture | omit-reinterpreted-ceilings | 258 | 167–1.32e+03 | True | True |
| spherical-mixture | omit-young-proximal | 163 | 89.6–2.05e+03 | True | True |
| spherical-mixture | principal | 241 | 154–1.58e+03 | True | True |
| spherical-mixture | prior-244 | 352 | 213–9.13e+03 | False | True |
| spherical-mixture | restore-thick-archive-readings | 371 | 186–4.72e+03 | True | True |
| spherical-mixture | restore-williams | 238 | 156–1.45e+03 | True | True |
| spherical-mixture | yellowstone-one | 248 | 157–1.79e+03 | True | True |
| spline | alternate-field-bounds | 195 | 147–1.6e+04 | False | True |
| spline | before-field-243 | 199 | 161–2.05e+04 | True | True |
| spline | cathedral-one | 194 | 147–1.59e+04 | False | True |
| spline | half-capture-ceilings | 195 | 155–1.55e+04 | False | True |
| spline | jenny-omit | 194 | 147–1.59e+04 | False | True |
| spline | jenny-one | 194 | 147–1.59e+04 | False | True |
| spline | omit-all-archive-ceilings | 396 | 444–1.67e+05 | True | True |
| spline | omit-basin-beds | 243 | 186–3.79e+04 | True | True |
| spline | omit-qualified-identities | 196 | 149–1.64e+04 | False | True |
| spline | omit-reinterpreted-ceilings | 208 | 196–2.57e+04 | True | True |
| spline | omit-young-proximal | 106 | 78.9–1.51e+04 | True | True |
| spline | principal | 194 | 147–1.59e+04 | False | True |
| spline | prior-244 | 227 | 215–1.14e+04 | False | True |
| spline | restore-thick-archive-readings | 345 | 241–7e+04 | False | True |
| spline | restore-williams | 224 | 167–2.65e+04 | True | True |
| spline | yellowstone-one | 221 | 167–2.63e+04 | True | True |
| stacked | alternate-field-bounds | 91.1 | 60–202 | True | True |
| stacked | before-field-243 | 92.4 | 60.9–205 | True | True |
| stacked | cathedral-one | 91.3 | 60.2–202 | True | True |
| stacked | half-capture-ceilings | 99.2 | 64.6–221 | True | True |
| stacked | jenny-omit | 91.2 | 60.1–202 | True | True |
| stacked | jenny-one | 91.2 | 60.1–202 | True | True |
| stacked | omit-all-archive-ceilings | 222 | 125–580 | True | True |
| stacked | omit-basin-beds | 115 | 82.7–294 | True | True |
| stacked | omit-qualified-identities | 91.6 | 60.3–202 | True | True |
| stacked | omit-reinterpreted-ceilings | 137 | 84.7–312 | True | True |
| stacked | omit-young-proximal | 33.7 | 21–94 | True | True |
| stacked | principal | 91.2 | 60.1–202 | True | True |
| stacked | prior-244 | 167 | 122–361 | True | True |
| stacked | restore-thick-archive-readings | 162 | 98.8–424 | True | True |
| stacked | restore-williams | 107 | 69.5–254 | True | True |
| stacked | yellowstone-one | 105 | 72.1–250 | True | True |

The principal case home uses its full accepted hull for regional volumes; the two principal regional rows above deliberately use the smaller common footprint. The median-map integral and the posterior distribution of complete-deposit integrals are different quantities. False numerical flags remain provisional; they are not hidden by a successful map integral.

Local checks cover the 271 principal localities and the omitted previous-fit localities, including the YL16-6A slump-toe site. Each has predictions along its bearing and adjacent ±11.25° rays at 0.5, 1, 5 and 10 cm. All contour crossings and their radial-grid brackets are retained. Regional predictions end at each scenario hull. Camp Creek and Skull Creek also retain section-vertex checks.

The full comparison uses identical input hashes and geographic folds. The reports retain geographic and quadrant warnings, supported-contour rejection reasons and the eight spherical prior intervals. Numerical convergence does not establish observational support over Earth.

[before-field-243](sensitivities/before-field-243/) · [alternate-field-bounds](sensitivities/alternate-field-bounds/) · [prior-244](sensitivities/prior-244/) · [omit-reinterpreted-ceilings](sensitivities/omit-reinterpreted-ceilings/) · [half-capture-ceilings](sensitivities/half-capture-ceilings/) · [omit-all-archive-ceilings](sensitivities/omit-all-archive-ceilings/) · [omit-qualified-identities](sensitivities/omit-qualified-identities/) · [jenny-one](sensitivities/jenny-one/) · [jenny-omit](sensitivities/jenny-omit/) · [cathedral-one](sensitivities/cathedral-one/) · [yellowstone-one](sensitivities/yellowstone-one/) · [restore-williams](sensitivities/restore-williams/) · [restore-thick-archive-readings](sensitivities/restore-thick-archive-readings/) · [omit-basin-beds](sensitivities/omit-basin-beds/) · [omit-young-proximal](sensitivities/omit-young-proximal/) · [principal](sensitivities/principal/)

## Reproduction

The reproduction commands run the principal fit and 15 source scenarios. Regional source scenarios retain the principal chosen settings and fixed original common hull.

```sh
python -m src.empirical_isopach.cli compare Mazama --output results/new-mazama
for family in regional directional spherical-mixture nested-isopachs; do
  python -m benchmarks.mazama_sensitivity --output results/new-mazama-sources --principal results/new-mazama --family "$family"
done
python -m benchmarks.mazama_release regional --principal results/new-mazama --scenarios results/new-mazama-sources
python -m benchmarks.mazama_release principal --principal results/new-mazama --scenarios results/new-mazama-sources
```

[Publication checks](publication-checks.json) · [Release identities](release.json) · [Earlier published results](../../../../archive/2026-11-23-mazama-source-audit/README.md)
