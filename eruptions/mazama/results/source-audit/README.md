# Mazama source interpretation tests

## Sensitivity to the geological interpretation

Each endpoint below is a separate refit in km³. The GP and spline use the fixed footprint of the 26 historical measurements for this comparison. The other three models integrate over Earth. The full table gives each fit’s own conditional interval and numerical checks. [All source refits](source-sensitivity-volumes.csv)

| Family | Comparable principal map | Smallest source-scenario map | Largest source-scenario map | Scenarios with a numerical warning |
|---|---:|---:|---:|---:|
| GP ensemble | 178.3 | 60.7 ([omit-bounds](sensitivities/omit-bounds/)) | 204.2 ([great-basin-omit](sensitivities/great-basin-omit/)) | 0/33 |
| Cubic spline | 603.1 | 34.4 ([omit-bounds](sensitivities/omit-bounds/)) | 774.9 ([great-basin-omit](sensitivities/great-basin-omit/)) | 0/33 |
| Directional ensemble | 332.1 | 181.6 ([omit-bounds](sensitivities/omit-bounds/)) | 548.4 ([new-beds-lower](sensitivities/new-beds-lower/)) | 22/33 |
| Spherical mixture | 370.2 | 122.5 ([omit-bounds](sensitivities/omit-bounds/)) | 645.9 ([new-beds-lower](sensitivities/new-beds-lower/)) | 4/33 |
| Bayesian nested corridors | 271.0 | 92.3 ([omit-bounds](sensitivities/omit-bounds/)) | 322.5 ([new-beds-lower](sensitivities/new-beds-lower/)) | 1/33 |

The endpoint range describes the tested source choices. Sampling intervals and geographic prediction have separate assessments. [Withheld site predictions](heldout-source-observations.csv) · [Probabilities assigned to withheld upper and lower bounds](heldout-bound-probabilities.csv) · [Site-by-site changes](local-thickness-comparison.csv) · [All contour crossings, compressed CSV](local-crossing-comparison.csv.gz)

[Case home](../../README.md) · [Evidence and calculations](../../../../data/audits/Mazama/BOUND_REVIEW.md) · [Numerical table](source-sensitivity-volumes.csv)

The common input contains 244 observations, including 58 measured beds, 153 lower bounds and 33 upper bounds. The 33 source refits test preservation, ash identity, interval endpoints and shared site positions. Global models refit each source scenario using continuation calibration that excludes Mazama. The regional refits keep the principal chosen settings and compare volumes on the original 26-measurement footprint.

| Method | Scenario | Median-map volume (km³) | Conditional 95% interval (km³) | Draw checks | Map integration |
|---|---|---:|---:|---|---|
| directional | bachelor-west-alternate | 328 | 255–1.08e+03 | False | True |
| directional | bracket-omit | 340 | 261–1.12e+03 | False | True |
| directional | dated-capture-alternate | 329 | 264–991 | False | True |
| directional | dated-sections-omit | 330 | 257–1.03e+03 | False | True |
| directional | distal-transfer-alternates | 327 | 249–876 | False | True |
| directional | field-range-alternate | 322 | 249–1.12e+03 | False | True |
| directional | field-source-omit | 337 | 266–1.24e+03 | True | True |
| directional | great-basin-alternate | 343 | 255–1.06e+03 | False | True |
| directional | great-basin-omit | 334 | 261–1.21e+03 | False | True |
| directional | half-transfer | 243 | 186–1.36e+03 | False | True |
| directional | kootenai-position-east | 329 | 259–1.09e+03 | True | True |
| directional | lake-capture-alternate | 307 | 241–1.19e+03 | True | True |
| directional | lake-core-alternates | 333 | 258–987 | False | True |
| directional | lake-field-omit | 303 | 248–1.31e+03 | True | True |
| directional | late-source-omit | 328 | 250–1.06e+03 | False | True |
| directional | new-beds-lower | 548 | 443–2.56e+03 | True | True |
| directional | offshore-omit | 330 | 255–1.01e+03 | False | True |
| directional | omit-bounds | 182 | 115–2e+03 | True | True |
| directional | omit-gold | 329 | 256–1.07e+03 | False | True |
| directional | omit-new-records | 189 | 132–2.16e+03 | True | True |
| directional | omit-williams | 352 | 280–1.08e+03 | False | True |
| directional | portage-upper | 328 | 250–1.01e+03 | False | True |
| directional | principal | 332 | 257–1.04e+03 | False | True |
| directional | range-upper | 325 | 251–964 | False | True |
| directional | round-meadow-one | 333 | 255–1.09e+03 | False | True |
| directional | skull-omit | 330 | 253–1.06e+03 | False | True |
| directional | weak-identity-omit | 320 | 248–1.29e+03 | True | True |
| directional | western-field-omit | 313 | 254–1.38e+03 | False | True |
| directional | yellowstone-2c-5mm | 331 | 257–1.08e+03 | False | True |
| directional | yellowstone-one | 324 | 258–1.1e+03 | True | True |
| directional | yellowstone-slump | 328 | 256–1.04e+03 | False | True |
| directional | young-distal-omit | 356 | 265–1.55e+03 | False | True |
| directional | young-section-positions | 333 | 252–1.11e+03 | True | True |
| directional | young-sections-omit | 189 | 140–1.3e+03 | True | True |
| nested-isopachs | bachelor-west-alternate | 282 | 183–923 | True | True |
| nested-isopachs | bracket-omit | 277 | 194–887 | True | True |
| nested-isopachs | dated-capture-alternate | 297 | 191–819 | True | True |
| nested-isopachs | dated-sections-omit | 293 | 189–891 | True | True |
| nested-isopachs | distal-transfer-alternates | 280 | 178–758 | True | True |
| nested-isopachs | field-range-alternate | 302 | 190–838 | True | True |
| nested-isopachs | field-source-omit | 289 | 188–948 | True | True |
| nested-isopachs | great-basin-alternate | 284 | 174–917 | True | True |
| nested-isopachs | great-basin-omit | 265 | 182–823 | True | True |
| nested-isopachs | half-transfer | 170 | 119–481 | True | True |
| nested-isopachs | kootenai-position-east | 273 | 184–741 | True | True |
| nested-isopachs | lake-capture-alternate | 233 | 160–664 | True | True |
| nested-isopachs | lake-core-alternates | 273 | 186–733 | True | True |
| nested-isopachs | lake-field-omit | 235 | 165–637 | True | True |
| nested-isopachs | late-source-omit | 262 | 180–756 | True | True |
| nested-isopachs | new-beds-lower | 323 | 222–945 | True | True |
| nested-isopachs | offshore-omit | 283 | 189–874 | True | True |
| nested-isopachs | omit-bounds | 92.3 | 68.6–387 | True | True |
| nested-isopachs | omit-gold | 273 | 181–746 | True | True |
| nested-isopachs | omit-new-records | 280 | 119–1.74e+03 | False | True |
| nested-isopachs | omit-williams | 258 | 179–789 | True | True |
| nested-isopachs | portage-upper | 276 | 187–803 | True | True |
| nested-isopachs | principal | 271 | 187–724 | True | True |
| nested-isopachs | range-upper | 268 | 178–823 | True | True |
| nested-isopachs | round-meadow-one | 264 | 184–922 | True | True |
| nested-isopachs | skull-omit | 260 | 188–762 | True | True |
| nested-isopachs | weak-identity-omit | 276 | 179–833 | True | True |
| nested-isopachs | western-field-omit | 268 | 177–622 | True | True |
| nested-isopachs | yellowstone-2c-5mm | 274 | 190–751 | True | True |
| nested-isopachs | yellowstone-one | 262 | 181–610 | True | True |
| nested-isopachs | yellowstone-slump | 276 | 182–791 | True | True |
| nested-isopachs | young-distal-omit | 291 | 189–936 | True | True |
| nested-isopachs | young-section-positions | 273 | 190–777 | True | True |
| nested-isopachs | young-sections-omit | 142 | 109–446 | True | True |
| spherical-mixture | bachelor-west-alternate | 341 | 208–7.59e+03 | True | True |
| spherical-mixture | bracket-omit | 399 | 226–9.34e+03 | True | True |
| spherical-mixture | dated-capture-alternate | 365 | 227–8.34e+03 | True | True |
| spherical-mixture | dated-sections-omit | 378 | 216–1.1e+04 | True | True |
| spherical-mixture | distal-transfer-alternates | 356 | 212–2.37e+04 | False | True |
| spherical-mixture | field-range-alternate | 357 | 208–1.04e+04 | True | True |
| spherical-mixture | field-source-omit | 562 | 271–2.82e+04 | True | True |
| spherical-mixture | great-basin-alternate | 362 | 214–9.06e+03 | True | True |
| spherical-mixture | great-basin-omit | 397 | 239–2.71e+04 | False | True |
| spherical-mixture | half-transfer | 211 | 155–1.12e+04 | False | True |
| spherical-mixture | kootenai-position-east | 361 | 216–1.05e+04 | True | True |
| spherical-mixture | lake-capture-alternate | 301 | 198–6.23e+03 | True | True |
| spherical-mixture | lake-core-alternates | 374 | 217–1.09e+04 | True | True |
| spherical-mixture | lake-field-omit | 299 | 205–7.73e+03 | True | True |
| spherical-mixture | late-source-omit | 416 | 211–9.13e+03 | True | True |
| spherical-mixture | new-beds-lower | 646 | 343–1.99e+04 | True | True |
| spherical-mixture | offshore-omit | 352 | 215–9.46e+03 | False | True |
| spherical-mixture | omit-bounds | 122 | 109–907 | True | True |
| spherical-mixture | omit-gold | 354 | 212–6.92e+03 | True | True |
| spherical-mixture | omit-new-records | 229 | 154–3.6e+03 | True | True |
| spherical-mixture | omit-williams | 377 | 218–1.7e+04 | True | True |
| spherical-mixture | portage-upper | 360 | 205–6.64e+03 | True | True |
| spherical-mixture | principal | 370 | 212–7.17e+03 | True | True |
| spherical-mixture | range-upper | 373 | 211–1.07e+04 | True | True |
| spherical-mixture | round-meadow-one | 354 | 214–1.17e+04 | True | True |
| spherical-mixture | skull-omit | 382 | 219–9.73e+03 | True | True |
| spherical-mixture | weak-identity-omit | 377 | 210–5.99e+03 | True | True |
| spherical-mixture | western-field-omit | 423 | 220–9.45e+03 | True | True |
| spherical-mixture | yellowstone-2c-5mm | 368 | 210–7.93e+03 | True | True |
| spherical-mixture | yellowstone-one | 371 | 229–6.15e+03 | True | True |
| spherical-mixture | yellowstone-slump | 367 | 214–1.01e+04 | True | True |
| spherical-mixture | young-distal-omit | 322 | 204–1.14e+04 | True | True |
| spherical-mixture | young-section-positions | 333 | 208–8.69e+03 | True | True |
| spherical-mixture | young-sections-omit | 141 | 120–914 | True | True |
| spline | bachelor-west-alternate | 563 | 276–1.2e+03 | True | True |
| spline | bracket-omit | 676 | 326–1.46e+03 | True | True |
| spline | dated-capture-alternate | 626 | 300–1.36e+03 | True | True |
| spline | dated-sections-omit | 641 | 309–1.4e+03 | True | True |
| spline | distal-transfer-alternates | 625 | 304–1.34e+03 | True | True |
| spline | field-range-alternate | 454 | 224–954 | True | True |
| spline | field-source-omit | 740 | 359–1.62e+03 | True | True |
| spline | great-basin-alternate | 614 | 297–1.32e+03 | True | True |
| spline | great-basin-omit | 775 | 367–1.7e+03 | True | True |
| spline | half-transfer | 452 | 221–970 | True | True |
| spline | kootenai-position-east | 603 | 294–1.29e+03 | True | True |
| spline | lake-capture-alternate | 483 | 248–960 | True | True |
| spline | lake-core-alternates | 627 | 303–1.36e+03 | True | True |
| spline | lake-field-omit | 539 | 274–1.08e+03 | True | True |
| spline | late-source-omit | 646 | 309–1.4e+03 | True | True |
| spline | new-beds-lower | 272 | 188–393 | True | True |
| spline | offshore-omit | 625 | 306–1.34e+03 | True | True |
| spline | omit-bounds | 34.4 | 13.2–108 | True | True |
| spline | omit-gold | 597 | 291–1.27e+03 | True | True |
| spline | omit-new-records | 58.2 | 32.1–108 | True | True |
| spline | omit-williams | 584 | 286–1.24e+03 | True | True |
| spline | portage-upper | 589 | 285–1.26e+03 | True | True |
| spline | principal | 603 | 294–1.29e+03 | True | True |
| spline | range-upper | 544 | 267–1.16e+03 | True | True |
| spline | round-meadow-one | 555 | 269–1.19e+03 | True | True |
| spline | skull-omit | 643 | 310–1.38e+03 | True | True |
| spline | weak-identity-omit | 701 | 330–1.58e+03 | True | True |
| spline | western-field-omit | 461 | 235–943 | True | True |
| spline | yellowstone-2c-5mm | 597 | 292–1.27e+03 | True | True |
| spline | yellowstone-one | 507 | 261–1.02e+03 | True | True |
| spline | yellowstone-slump | 602 | 293–1.29e+03 | True | True |
| spline | young-distal-omit | 569 | 275–1.22e+03 | True | True |
| spline | young-section-positions | 614 | 300–1.31e+03 | True | True |
| spline | young-sections-omit | 89 | 41–213 | True | True |
| stacked | bachelor-west-alternate | 171 | 121–364 | True | True |
| stacked | bracket-omit | 187 | 131–396 | True | True |
| stacked | dated-capture-alternate | 183 | 128–394 | True | True |
| stacked | dated-sections-omit | 183 | 129–396 | True | True |
| stacked | distal-transfer-alternates | 180 | 127–388 | True | True |
| stacked | field-range-alternate | 157 | 114–340 | True | True |
| stacked | field-source-omit | 182 | 126–383 | True | True |
| stacked | great-basin-alternate | 174 | 123–373 | True | True |
| stacked | great-basin-omit | 204 | 139–443 | True | True |
| stacked | half-transfer | 144 | 102–311 | True | True |
| stacked | kootenai-position-east | 178 | 125–381 | True | True |
| stacked | lake-capture-alternate | 158 | 109–317 | True | True |
| stacked | lake-core-alternates | 182 | 128–393 | True | True |
| stacked | lake-field-omit | 171 | 117–337 | True | True |
| stacked | late-source-omit | 184 | 127–398 | True | True |
| stacked | new-beds-lower | 149 | 108–250 | True | True |
| stacked | offshore-omit | 178 | 125–381 | True | True |
| stacked | omit-bounds | 60.7 | 38.3–200 | True | True |
| stacked | omit-gold | 178 | 125–379 | True | True |
| stacked | omit-new-records | 105 | 78.3–156 | True | True |
| stacked | omit-williams | 173 | 120–365 | True | True |
| stacked | portage-upper | 178 | 125–378 | True | True |
| stacked | principal | 178 | 125–381 | True | True |
| stacked | range-upper | 171 | 121–369 | True | True |
| stacked | round-meadow-one | 175 | 123–373 | True | True |
| stacked | skull-omit | 186 | 128–401 | True | True |
| stacked | weak-identity-omit | 178 | 121–390 | True | True |
| stacked | western-field-omit | 169 | 120–341 | True | True |
| stacked | yellowstone-2c-5mm | 178 | 125–378 | True | True |
| stacked | yellowstone-one | 175 | 129–361 | True | True |
| stacked | yellowstone-slump | 178 | 124–378 | True | True |
| stacked | young-distal-omit | 164 | 115–347 | True | True |
| stacked | young-section-positions | 184 | 130–391 | True | True |
| stacked | young-sections-omit | 93.9 | 66.2–242 | True | True |

The principal case home uses its full accepted hull for regional volumes; the two principal regional rows above deliberately use the smaller common footprint. The median-map integral and the posterior distribution of complete-deposit integrals are different quantities. False numerical flags remain provisional; they are not hidden by a successful map integral.

Local checks cover the 244 principal localities and the omitted YL16-6A slump-toe site. Each has predictions along its bearing and adjacent ±11.25° rays at 0.5, 1, 5 and 10 cm. All contour crossings and their radial-grid brackets are retained. Regional predictions end at each scenario hull. Camp Creek and Skull Creek also retain section-vertex checks.

The full comparison uses identical input hashes and geographic folds. The reports retain geographic and quadrant warnings, supported-contour rejection reasons and the eight spherical prior intervals. Numerical convergence does not establish observational support over Earth.

[great-basin-omit](sensitivities/great-basin-omit/) · [great-basin-alternate](sensitivities/great-basin-alternate/) · [dated-sections-omit](sensitivities/dated-sections-omit/) · [dated-capture-alternate](sensitivities/dated-capture-alternate/) · [kootenai-position-east](sensitivities/kootenai-position-east/) · [lake-field-omit](sensitivities/lake-field-omit/) · [lake-capture-alternate](sensitivities/lake-capture-alternate/) · [field-range-alternate](sensitivities/field-range-alternate/) · [young-sections-omit](sensitivities/young-sections-omit/) · [young-section-positions](sensitivities/young-section-positions/) · [young-distal-omit](sensitivities/young-distal-omit/) · [distal-transfer-alternates](sensitivities/distal-transfer-alternates/) · [round-meadow-one](sensitivities/round-meadow-one/) · [offshore-omit](sensitivities/offshore-omit/) · [western-field-omit](sensitivities/western-field-omit/) · [omit-bounds](sensitivities/omit-bounds/) · [half-transfer](sensitivities/half-transfer/) · [portage-upper](sensitivities/portage-upper/) · [range-upper](sensitivities/range-upper/) · [omit-new-records](sensitivities/omit-new-records/) · [new-beds-lower](sensitivities/new-beds-lower/) · [weak-identity-omit](sensitivities/weak-identity-omit/) · [yellowstone-one](sensitivities/yellowstone-one/) · [yellowstone-2c-5mm](sensitivities/yellowstone-2c-5mm/) · [yellowstone-slump](sensitivities/yellowstone-slump/) · [omit-gold](sensitivities/omit-gold/) · [late-source-omit](sensitivities/late-source-omit/) · [skull-omit](sensitivities/skull-omit/) · [field-source-omit](sensitivities/field-source-omit/) · [bracket-omit](sensitivities/bracket-omit/) · [lake-core-alternates](sensitivities/lake-core-alternates/) · [bachelor-west-alternate](sensitivities/bachelor-west-alternate/) · [omit-williams](sensitivities/omit-williams/) · [principal](sensitivities/principal/)

## Reproduction

The reproduction commands run the principal fit and 33 source scenarios. Regional source scenarios retain the principal chosen settings and fixed original common hull.

```sh
python -m src.empirical_isopach.cli compare Mazama --output results/new-mazama
for family in regional directional spherical-mixture nested-isopachs; do
  python -m benchmarks.mazama_sensitivity --output results/new-mazama-sources --principal results/new-mazama --family "$family"
  python -m benchmarks.mazama_phase_sensitivity --output results/new-mazama-phase --principal results/new-mazama --family "$family"
done
python -m benchmarks.mazama_release regional --principal results/new-mazama --scenarios results/new-mazama-sources
python -m benchmarks.mazama_release regional --principal results/new-mazama --scenarios results/new-mazama-phase
python -m benchmarks.mazama_release principal --principal results/new-mazama --scenarios results/new-mazama-sources
```

[Publication checks](publication-checks.json) · [Release identities](release.json) · [Earlier published results](../../../../archive/2026-11-23-mazama-source-audit/README.md)
