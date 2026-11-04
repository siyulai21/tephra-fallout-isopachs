# Two methods, matched regional evidence

[Current findings](../../reports/regional-methods/README.md) · [Spline equations](cubic-splines.md) · [GP ensemble](gp-ensemble.md)

The current workflow compares one **GP ensemble** and one **trend-regularized cubic spline** as peers. Per-method tuning does not designate an overall preferred family. Agreement with a desired published eruption volume is never a selection criterion.

## Region and volume estimand

The primary region is the convex hull of all accepted exact and upper/lower-bound locations in the source-centred WGS84 Lambert azimuthal equal-area projection. Pseudo-points are ignored, including their coordinates. The vent is not inserted into the hull. The region can bridge unsampled gaps and exclude unsampled proximal deposits: these are **regional bulk fallout volumes, not complete eruption volumes**.

The spline fitting rectangle is the coordinate bounding box plus 5% of each span. This geometry-only rule defines a numerical domain, not a deposit boundary. Both methods integrate the same hull. The spline refuses predictions beyond its rectangle. No global tails or artificially closed contours enter the comparison.

Expected volume is integral E[T(x)] dx. The integral of pointwise median thickness is also reported; it is not the median of random volume. Centimetres times square kilometres convert to cubic kilometres by 10^-5.

The hull is triangulated about its centroid. Positive Gauss–Duffy quadrature uses orders 8, 12, 20, 32, 48 and 64. Two consecutive changes below 0.1% in expected and median-surface volume are required. Failed convergence remains visible. Numerical agreement cannot establish geological accuracy.

## Geographic validation without manual settings

Four outer folds use deterministic recursive spatial median partitions. Colocated rows stay together. Each outer model tunes on three inner folds; the final all-data model tunes on four. Trend/noise fitting, spline selection, GP fitting, candidate screening and stacking weights are refitted within the appropriate training data.

The buffer is the median nearest-distinct-site distance. It is reduced to half, one quarter or zero only if a training subset would have fewer than six exact measurements at four distinct non-collinear sites. Feasibility uses coordinates and observation kinds, never thickness values. All attempts are recorded. Insufficient data fail rather than reusing withheld responses.

Full survey coordinates supply the common region as known design information: the question is prediction across a **known survey footprint**. Dependence and preferential sampling can remain. Geographic blocking follows concerns described by [Roberts et al. (2017)](https://doi.org/10.1111/ecog.02881); the chosen partition does not guarantee independent test observations.

The primary descriptive score weights outer folds equally. Pooled negative log score, exact-observation RMSE, 95% observation coverage, interval width and paired fold differences are also retained. Bounds contribute event-probability scores, not point errors. Four folds and six development datasets cannot establish a universal winner or statistical significance.

## Correlated volume draws and sensitivity

For GP stacking, one component is drawn for the whole field, followed by a correlated latent surface. For the spline, one coefficient vector determines the whole field. Both use 2,000 draws and seed 1729. Independent observation noise is excluded from latent volume. The ensemble follows [predictive-distribution stacking](https://arxiv.org/abs/1704.02030).

Sampling quadrature increases through orders 4, 6, 8, 10, 12 and 16 until expected volume is within 0.5% of the refined integral. Order-statistic ranges describe Monte Carlo precision of quantiles. These checks do not establish tail-quantile quadrature convergence. Intervals condition on fitted parameters and selection; real and simulated coverage is assessed rather than assumed.

Every outer training model is also integrated on the original hull after complete retuning. Those four geographic-deletion volumes show sensitivity to missing regions, not confidence intervals. They are displayed separately from conditional intervals, particularly when strong spline shrinkage gives misleadingly narrow intervals.

## Known-truth trials

Ten fixed seeds are used for each of four scenarios: smooth radial/directional thinning, a secondary lobe, a missing northeast sampling sector, and censored observations. None is generated from the fitted GP or spline distribution. Each run has 60 training locations, independent 0.25-log-unit observation noise and 400 independent test sites inside its training-site hull.

For censoring, training values below log thickness 1 become upper limits at 1; values above 3 become lower bounds at 3. Test observations remain exact. Both methods see identical inputs, region and automatic buffer. Regional truth is integrated to a tighter 0.001% tolerance. Reports include prediction accuracy, latent coverage, volume errors and conditional volume-interval coverage.

Forty trials were attempted; 38 completed for both methods. Two heavily censored trials lacked enough exact measurements in nested geographic training subsets and remain explicit failures, not silently excluded successes. Metrics describe completed trials.

These are small development stress tests, not an external calibration certificate. Unfavorable outcomes are retained without retuning rules after inspecting test outcomes.

## Reproduce

```sh
python -m benchmarks.regional --output results/regional-methods
```

All six accepted inputs, including combined Plinian/co-ignimbrite Minoan and the audited 46-record YTT selection, enter unchanged. Audits are verified and source/input hashes recorded. Historical selected-GP and global calculations remain separate and are not used to rank regional fits.
