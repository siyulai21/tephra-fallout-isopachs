# Which method does what?

[Project home](../README.md) · [Regional evidence](../comparisons/regional/README.md) · [Source map](../src/README.md)

**The active regional workflow reports two peer methods, with no preferred family.** Each produces its own thickness surface, isopachs, regional volumes and geographic validation.

| Public method | Surface | Automatic choices |
|---|---|---|
| [GP ensemble (`stacked`)](methods/gp-ensemble.md) | Directional trend plus a mixture of Matérn GP residual fields | Geographic predictive weights across trend, length-scale and noise candidates |
| [Trend-regularized cubic spline (`spline`)](methods/cubic-splines.md) | Cubic log-thickness surface with derivative and normalized trend penalties | Joint geographic selection of trend, knots, tension, smoothing, trend strength and noise |

Both use measurements and documented bounds. Neither treats predictions as new observations. The spline does not use a GP residual surface. They share data parsing, probability scoring, regional integration and reporting conventions.

The [global spherical mixture](methods/probabilistic-global.md) is the companion for whole-deposit mass. It fits all accepted fans jointly, normalizes every posterior field over WGS84, and reports conditional credible intervals, seven assumption refits and identical geographic holdouts. A failed local contour gate does not veto this separate global estimate.

```sh
tephra-isopachs compare Minoan --output results/minoan-complete
```

The [regional protocol](methods/regional-comparison.md) fixes the same observation hull, automatic buffer and nested folds. All results are retained; no global extrapolation decides which near-data fit looks better. Automatic settings remove per-eruption manual tuning, not scientific assumptions. Read conditional intervals with held-out coverage and geographic-deletion sensitivity.

## Historical labels

- `selected` is one selected GP configuration, retained for historical reproduction. It is not a third method in the new comparison.
- Individual GP fitters remain necessary internal components of the ensemble.
- The original spline and first fixed-smoothing extension remain in historical/research packages; neither produces new regional reports.
- `eruptions/<slug>/results/` contains current peer-method results. Earlier calculations are available through [Git history](HISTORY.md).

The [validation catalogue](VALIDATION.md) separates current matched comparisons from earlier, less thoroughly tuned spline experiments. None proves universal superiority.
