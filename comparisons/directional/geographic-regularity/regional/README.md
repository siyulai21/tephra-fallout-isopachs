# Matched regional controls

The GP ensemble and cubic spline were rerun on the same nine accepted inputs, observation hulls and outer geographic folds as the directional comparison. Model selection uses inner training folds. These regional integrals cover the observation hull, not the entire Earth, so their volumes are not estimates of the same area as the directional totals.

[Comparison table](comparison.csv) · [Numerical reports, accepted rows, predictions and inner selections](fits.tar.gz)

All 18 regional volume checks pass. The retained reports include warnings and selection failures; passing integration is not a claim that every attempted model fit converged. The published case controls remain separate from this fresh matched benchmark.

Reproduce with `python -m benchmarks.regional --skip-synthetic --output results/directional-regional`. All input identities and software-source fingerprints are recorded in the parent study's [provenance](../provenance.json).
