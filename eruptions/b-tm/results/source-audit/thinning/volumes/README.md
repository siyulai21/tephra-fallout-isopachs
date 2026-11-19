# How much do thinning breaks change global B-Tm fallout volume?

These six fits use the same 68 audited observations and the reference directional fan geometry. Each map integral adds the pointwise median thickness over Earth. The median and conditional interval in the next columns come from integrating each coherent deposit separately. Volumes are bulk km³. All disputed DPRK locality summaries are withheld. [Seven source-interpretation sensitivities and principal ensemble](../../README.md) · [Local predictions and held-out geographic scores](../README.md)

| Radial thinning | Median-map integral | Whole-deposit median | Conditional 95% interval | Warnings |
|---|---:|---:|---:|---|
| Published smooth reference | 48.4 | 55.5 | 33.7–134.7 | None |
| One shared break | 48.4 | 56.0 | 33.6–146.6 | None |
| Two shared breaks | 48.6 | 57.1 | 33.8–150.5 | None |
| Exponential kernels | 45.7 | 49.6 | 32.3–125.5 | None |
| Exponential kernels, one break | 45.4 | 49.0 | 31.8–122.9 | None |
| One break, wider slope prior | 48.8 | 57.3 | 33.6–160.6 | None |

The matched smooth, one-break and two-break median maps integrate to 48.4–48.6 km³. The differences are small relative to posterior uncertainty and the numerical precision of the comparison. All three share one fan geometry. The principal six-structure ensemble also represents uncertainty about the presence and width of fans in poorly sampled directions. Its 54.0 km³ median-map estimate has separate draws and diagnostics. A break in the regional thinning rate, a separate fallout fan and the local stratigraphic interpretation of a 7 m formation make different predictions.

## Numerical integration

Each posterior draw retains its four-chain identity, node thicknesses, fan geometry and break distances. In a radial interval with true lower radius a and slope s, the effective radius is w(a) + s(r−a). The equal-area element remains r dr dθ. Splitting background intervals at each fitted break gives an analytic integral on each piece. The footprint and its transition use positive quadrature on the same triangles and rays as the production model. The calculation extends to the antipode. Numerical checks compare all five spatial regions and eight compass sectors at successive quadrature orders. An independent adaptive scalar quadrature checks the radial formula. Setting all slope changes to unity reproduces the production integrator on complete test fields.

The median map uses the production log-radius change of variables and two independent scrambled Sobol sequences. A preliminary map uses 64 draws per chain. The reported map and whole-deposit intervals use every draw. For the full map, the 40-part spatial integral must agree within 0.75% across nested grids and 1% across scrambles. The integrated mean field must agree within 0.5% with the independently integrated mean of the same deposits. Two interleaved posterior subsets must agree within 2.5% with the central integral. The published conditional intervals describe the complete fields. Numerical differences between closely spaced medians should be read alongside their posterior intervals and convergence diagnostics. [Complete numerical diagnostics](diagnostics.json) · [Whole-deposit draws](volume-draws.npz) · [Volume CSV](volumes.csv)

With the matched fits saved in a fresh experiment directory, regenerate these results from the repository root.

```sh
for v in smooth one_break two_breaks exponential broken_exponential wide_prior; do
  python -m benchmarks.btm_thinning_mass --work results/btm-break-check --output results/btm-break-mass --variant "$v"
  python -m benchmarks.btm_thinning_volume_report --work results/btm-break-check --mass results/btm-break-mass --variant "$v"
done
python -m benchmarks.btm_thinning_volume_report --work results/btm-break-check --mass results/btm-break-mass --output results/btm-break-report/volumes
```
