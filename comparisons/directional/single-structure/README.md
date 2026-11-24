# Single-structure directional sensitivity

The reference model uses a fixed fan geometry. Each eruption’s sensitivity report tests 13 policies for the unsampled deposit and an unresolved proximal cap. Geographic scores for the single-structure, ensemble and measured-fan controls appear together for the current inputs. [Current comparison, including all 300 Mazama localities](../fixed-ensemble/README.md) · [Mazama reference fit and policy diagnostics](../../../eruptions/mazama/results/directional/single-structure/README.md)

## Unsampled sectors at Aso-4

The southeast and south each have two exact readings. The west has none. With the southeast quadrant withheld, a 15 cm offshore observation at 709 km fell above the directional 95% predictive interval of 0.0003–1.95 cm. An unsampled sector can contain a real lobe. [Outermost site by octant](octant-extremes.csv) · [Outermost site by quadrant](quadrant-extremes.csv)

![Conditional sector volume intervals for both Aso-4 fits](aso-4-sectors.png)

## Known-deposit tests

Of 12 fresh trials, 8 passed the numerical checks. Their conditional volume intervals contained the true volume in 4/8. The full trial table includes failures, deliberately missing directions, and the known volumes of the simulated deposits. [All trials](known-volumes.csv) · [Fit records](known-volume-fits.json.gz)

## Reproduce

Run the single-structure and geographic benchmarks with a fresh results directory. The comparison command pairs geographic folds, rebuilds the two interval tables and checks each input identity.

~~~sh
python -m benchmarks.directional --help
python -m benchmarks.directional_replication --help
python -m benchmarks.directional_comparison --help
python -m benchmarks.verify_directional
~~~

[Eruption workflow](../../../docs/NEW_ERUPTION.md) · [Benchmark scripts](../../../benchmarks/README.md)
