# Eruption report catalogue

[Project home](../README.md) · [Newcomer guide](../docs/START_HERE.md) · [How to read the outputs](../docs/OUTPUTS.md)

**All eruption reports here use Gaussian-process (GP) models.** No current report is a cubic-spline calculation. `selected` and `stacked` distinguish one GP configuration from a GP ensemble; global/regional describes the integration domain.

Start with an eruption overview or the [current six-eruption comparison](global-fallout/README.md). The headline volume integrates the modeled bulk fallout over the complete Earth, with explicit extrapolation diagnostics.

| Eruption | Measurements / lower bounds | Main model | Current calculations | Earlier analysis |
|---|---:|---|---|---|
| [Mazama](mazama/README.md) | 45 / 0 | GP, exponential trend | [Selected GP](mazama/global/selected/README.md) · [GP ensemble](mazama/global/stacked/README.md) | [Regional history](mazama/regional/README.md) |
| [Tambora](tambora/README.md) | 26 / 0 | GP, power-law trend | [Selected GP](tambora/global/selected/README.md) · [GP ensemble](tambora/global/stacked/README.md) | [Regional history](tambora/regional/README.md) |
| [Aso-4](aso-4/README.md) | 87 / 0 | GP, power-law trend | [Selected GP](aso-4/global/selected/README.md) · [GP ensemble](aso-4/global/stacked/README.md) | [Regional history](aso-4/regional/README.md) |
| [B-Tm](b-tm/README.md) | 84 / 0 | GP, power-law trend | [Selected GP](b-tm/global/selected/README.md) · [GP ensemble](b-tm/global/stacked/README.md) | [Regional history](b-tm/regional/README.md) |
| [YTT](ytt/README.md) | 46 / 0 | GP, exponential trend | [Selected GP](ytt/global/selected/README.md) · [GP ensemble](ytt/global/stacked/README.md) | [Regional history](ytt/regional/README.md) |
| [Minoan](minoan/README.md) | 92 / 9 | GP, power-law trend | [Selected GP](minoan/global/selected/README.md) · [GP ensemble](minoan/global/stacked/README.md) | [Regional history](minoan/regional/README.md) |

## Navigate within an eruption

- `README.md`: current headline, figures, assumptions and interpretation.
- `global/`: current selected GP and GP ensemble, with full numerical exports.
- `regional/`: earlier finite-rectangle calculations and data-selection sensitivities.
- YTT's [archive](ytt/archive/README.md): superseded 43- and 45-record input versions. Its current input has 46 observations.

Minoan's accepted observations combine all Plinian and co-ignimbrite fallout fans in one field; identified turbidites are excluded. Source conflicts and incomplete cores are documented in the [audit](../data/audits/Minoan/README.md).

## Scope and provenance

A complete-Earth integral is conditional on a fitted model, not proof that every deposit sector is observed. Model-family and data-deletion sensitivities are not confidence intervals. Global probability intervals are not estimated. Regional simulation intervals apply only to their recorded rectangles.

Numerical files, maps and source hashes have been preserved during reorganization. Markdown adds method labels and navigation; raw provenance retains original execution paths. The [regional comparison summary](regional-comparison/README.md) and [research validation](../docs/validation/README.md) are historical, not replacements for the current reports.

[Accepted input catalogue](../data/README.md) · [Method comparison](../docs/METHODS.md) · [Reproduce the analyses](../benchmarks/README.md)
