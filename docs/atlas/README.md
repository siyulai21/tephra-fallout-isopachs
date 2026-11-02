# Tephra deposit atlas

Open [index.html](index.html) in a browser to explore all four deposits.

Rebuild from the repository root with:

```sh
python -m src.probabilistic_isopach atlas --output results/atlas --map-grid 961
```

Each deposit includes PNG/SVG/PDF maps, GeoJSON contour lines, enclosed-area
tables, volume estimates for three explicit regions and two thinning families,
measurements used, diagnostics, and a full reproducibility report.

The statistical fit is unchanged from method 0.2.0. Dashed isopach sections are
outside the measurement convex hull. The map expands to show closed contours;
unresolved levels are listed, never joined along the frame. Contour areas
subtract holes and are checked against a coarser grid.

Headline volumes are expected **bulk deposit volume in a finite region**.
They are not DRE or total erupted volume. Integrals are refined until consecutive
mean- and median-surface estimates agree within 0.5%, or a failure is reported.
The median-surface integral is not the median of random total volume.
Conditional sampling intervals are secondary diagnostics; parameter uncertainty
and quantile-grid convergence are not established. Domain/family sensitivity is
reported separately and is not a confidence interval.

The original CSV provenance has not been independently audited. No new field
measurements or eruption-physics inputs were introduced. Existing nested
validation is reproduced, including unfavorable results.
