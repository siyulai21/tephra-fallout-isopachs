# Replay wrapped angular covariance

From the repository root, install the package with its tested numerical dependencies. Limit BLAS to one thread per worker when fitting several independent eruptions concurrently.

```sh
tephra-isopachs global b-tm --geometry wrapped-angular-covariance --output results/millennium-wrapped
python -m benchmarks.wrapped_angular.verify
python -m benchmarks.wrapped_angular.known --output /tmp/wrapped-known.csv
python -m benchmarks.wrapped_angular.thickness_area --output /tmp/wrapped-thinning --draws
python -m pytest -q tests/test_wrapped_angular.py
```

The public runner refits all geographic blocks, populated compass quadrants and the buffered outer-distance holdout. It repeats the full fit with a three-harmonic spectrum. The verifier checks posterior and observation identities, input row membership, event scores, full-field site replays and all three map integrals. Its comparison uses the pinned input editions at commit `49aaaa448ec2e9ab8f017873a4aba62190ce71fa`. Rerun the existing global families with their own entry points to obtain new comparator fits on a later edition.

The synthetic benchmark fits only the frozen observation rows. It opens the independent known-truth field after fitting, evaluates complete coefficient draws on two signed global quadratures, then reports volume coverage, intersection over union and latent coverage. The thickness–area benchmark reads the signed cumulative areas and in-contour volumes, then refits every admissible segment count, the power law and the Weibull. It reports unsupported ranks and divergent tails. [Scientific results and checks](../../comparisons/wrapped-angular-covariance/evidence.md)
