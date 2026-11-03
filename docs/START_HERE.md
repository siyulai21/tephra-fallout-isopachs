# Start here

[Project home](../README.md) · [Documentation](README.md) · [Reports](../reports/README.md)

## What question does this project answer?

After an explosive volcanic eruption, field studies recover ash at scattered locations. This project asks what thickness surface is compatible with those observations, where equal-thickness contours lie, and what volume the fitted surface implies. It also checks where prediction or extrapolation is unreliable.

An **isopach** connects equal deposit thickness. **Fallout** is material deposited from an eruption cloud; the target is not automatically every volcanic product. **Bulk volume** includes deposit pore space. **Dense-rock equivalent (DRE)** needs additional density or porosity evidence.

## Follow one example

The [Minoan report](../reports/minoan/README.md) is a useful entry point:

1. The [source audit](../data/audits/Minoan/README.md) explains how a published table becomes usable observations.
2. The [accepted CSV](../data/eruptions/Minoan_tephra.csv) contains 92 measured thicknesses and nine lower bounds from incomplete cores. All accepted fallout fans enter one field.
3. The [current GP calculation](../reports/minoan/global/selected/README.md) shows maps, geographic holdouts and volume diagnostics.
4. The [regional analysis](../reports/minoan/regional/README.md) preserves earlier finite-region estimates and source-selection sensitivities.

## What happens in a calculation?

```text
Source measurements and geological notes
    ↓ review units, deposit meaning, locations and exclusions
Accepted thicknesses and documented upper/lower bounds
    ↓ fit and evaluate on withheld geographic regions
Gaussian-process thickness surface
    ↓ draw contours and integrate the fitted surface
Maps, bulk fallout volume, failure checks and a reproducible report
```

Missing or unvisited locations are prediction locations. They do not become zero thickness or invented observations. An incomplete core can supply a lower bound; an actual documented detection limit can supply an upper bound.

## Which model am I looking at?

**All eruption calculations under `reports/` use Gaussian-process models.** A directional thinning trend describes broad decay and asymmetry; a spatial covariance model describes local departures. There are no spline knots in this workflow.

| Label | Meaning |
|---|---|
| `selected` | One GP configuration chosen by geographic predictive scores |
| `stacked` | An evaluated ensemble of GP configurations |
| Exponential / power-law | Alternative thinning trends within the GP model |
| `global/` | Current volume calculation over the complete physical Earth |
| `regional/` | Earlier volume calculations within a stated rectangle |
| `archive/` | Superseded input selections or historical material |

The original cubic spline and trend-regularized spline remain available for research. They are not steps inside the report workflow. [Compare the implementations](METHODS.md).

## What should I inspect carefully?

Start with the input audit and predictions for withheld measurements. Then inspect changes between thinning families, included records and omitted geographic groups. Numerical convergence establishes that an integral was calculated consistently; it cannot establish that extrapolated thickness is correct.

The current headline integrates the whole Earth, so choosing a larger map or rectangle does not change it. That removes a numerical-domain choice but leaves scientific assumptions about unobserved deposit. Reports identify large distant contributions and model disagreement. No global probability interval is currently estimated; older regional intervals cannot be attached to the global estimate.

## Where next?

- Read the [eruption catalogue](../reports/README.md) or [combined comparison](../reports/global-fallout/README.md).
- Run an example or submit data using the [quick start](QUICKSTART.md).
- Inspect [model equations](methods/gaussian-process.md), [data rules](../data/README.md), or [source code](../src/README.md).

Read historical results with their recorded input selection, model version and volume domain.
