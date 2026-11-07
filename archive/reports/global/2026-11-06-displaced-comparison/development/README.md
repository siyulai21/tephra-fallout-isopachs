> Archived calculation. Links below point to its original repository revision; unmodified Markdown bytes are retained in the adjacent `.original.txt` file.

# Development record and retained failures

[Global results](https://github.com/siyulai21/tephra-fallout-isopachs/blob/b4bbbcbe21477a3cd2b2bdfe8930706987cc22cd/comparisons/global/README.md) · [Final method](https://github.com/siyulai21/tephra-fallout-isopachs/blob/b4bbbcbe21477a3cd2b2bdfe8930706987cc22cd/docs/methods/probabilistic-global.md) · [Final stress tests](https://github.com/siyulai21/tephra-fallout-isopachs/blob/b4bbbcbe21477a3cd2b2bdfe8930706987cc22cd/comparisons/global/stress-tests.md)

The global method changed during development. The final scientific specifications were frozen before the all-eruption run and the fresh-seed stress suite. The earlier tests below helped choose those specifications; they are not independent validation. No published eruption volume was used as a fitting target.

## What the first known-truth trial exposed

All four rows below use the same smooth exponential-thinning development case, seed 0, with true volume **33.5703 km³**. The row intervals describe the respective conditional posterior, including cases with numerical warnings; they must not be treated as equally reliable estimates.

| Development version | Median km³ | Conditional 95% interval km³ | Finding |
|---|---:|---:|---|
| Gaussian/power kernels, diffuse weights | 82.72 | 43.07–1,679.96 | Missed the true volume; zero divergences and R-hat 1.003, but the initial depth-based efficiency veto failed |
| Add exponential core, retain diffuse weights | 79.35 | 39.55–2,175.99 | Still missed the truth; adding the missing thinning shape alone was insufficient |
| Sparse fan/kernel weights, direct volume coordinates | 46.71 | 33.26–572.52 | Truth entered the interval, but R-hat 1.053 and 17 divergences made this short run unreliable |
| Same sparse physical model, centered amplitude coordinates | 48.07 | 33.34–1,875.54 | Zero divergences and R-hat 1.008; the short prototype still fell just below the ESS ≥400 rule (minimum bulk ESS 392) |

Records: [Gaussian/power](https://github.com/siyulai21/tephra-fallout-isopachs/blob/b4bbbcbe21477a3cd2b2bdfe8930706987cc22cd/comparisons/global/development/gaussian-power-smooth.json), [exponential/diffuse](https://github.com/siyulai21/tephra-fallout-isopachs/blob/b4bbbcbe21477a3cd2b2bdfe8930706987cc22cd/comparisons/global/development/exponential-diffuse-smooth.json), [sparse/direct](https://github.com/siyulai21/tephra-fallout-isopachs/blob/b4bbbcbe21477a3cd2b2bdfe8930706987cc22cd/comparisons/global/development/sparse-direct-smooth.json), [centered prototype](https://github.com/siyulai21/tephra-fallout-isopachs/blob/b4bbbcbe21477a3cd2b2bdfe8930706987cc22cd/comparisons/global/development/centered-smooth.json). Their original diagnostic flags are retained. The last two prototypes each used 600 warmup and 800 retained draws per chain; the final studies use more effort and the declared retry protocol.

The changes addressed two distinct problems. An exponential core represents a sharper centre than a Gaussian/power mixture. Sparse Dirichlet weights allow redundant fans or kernel shapes to carry negligible mass, reducing a prior tendency to allocate unseen mass to every component. This concentration remains a modelling choice, so a diffuse-weight alternative is retained for every eruption. Finite upper tails can remain extremely wide.

## A numerical change that preserves the scientific model

The centered sampler uses `a = log V + g(shape)`, where `g` is the average unit-volume log field at training locations. The explicit prior-density correction and unit Jacobian preserve the complete physical posterior. This is not a new data-derived prior.

The [archived-direct versus final joint-density check](https://github.com/siyulai21/tephra-fallout-isopachs/blob/b4bbbcbe21477a3cd2b2bdfe8930706987cc22cd/comparisons/global/development/parameterization-equivalence.json) compares the baseline, diffuse-tail and shifted-volume priors at the same physical state, with exact measurements and both kinds of bounds. All differences are zero to floating-point precision. An independent permanent test reconstructs the full prior-plus-likelihood density in [the numerical tests](https://github.com/siyulai21/tephra-fallout-isopachs/blob/b4bbbcbe21477a3cd2b2bdfe8930706987cc22cd/tests/test_mass_mixture.py).

The centered prototype took 123.8 seconds against 270.4 seconds for the direct prototype, on this development machine and trial. This single timing is not a general speed guarantee. The improvement motivated the final longer runs, not a relaxation of the convergence checks. Maximum tree-depth saturation is now reported as an efficiency warning separately from R-hat, ESS, divergences and energy BFMI; this distinction follows the cited Stan diagnostic guidance in the method document.

## What is preserved

- [All 38 completed exploratory attempt records](https://github.com/siyulai21/tephra-fallout-isopachs/blob/b4bbbcbe21477a3cd2b2bdfe8930706987cc22cd/comparisons/global/development/completed-development-attempts.json.gz), including attempts made before superseded workers were stopped. Partial, interrupted chains were never summarized or combined with completed chains.
- Exact archived source text for [Gaussian/power](https://github.com/siyulai21/tephra-fallout-isopachs/blob/b4bbbcbe21477a3cd2b2bdfe8930706987cc22cd/comparisons/global/development/sources/gaussian-power), [exponential/diffuse](https://github.com/siyulai21/tephra-fallout-isopachs/blob/b4bbbcbe21477a3cd2b2bdfe8930706987cc22cd/comparisons/global/development/sources/exponential-diffuse), [sparse/direct](https://github.com/siyulai21/tephra-fallout-isopachs/blob/b4bbbcbe21477a3cd2b2bdfe8930706987cc22cd/comparisons/global/development/sources/sparse-direct), and the [centered prototype](https://github.com/siyulai21/tephra-fallout-isopachs/blob/b4bbbcbe21477a3cd2b2bdfe8930706987cc22cd/comparisons/global/development/sources/centered-prototype-inference.py.txt).
- [Checksums](https://github.com/siyulai21/tephra-fallout-isopachs/blob/b4bbbcbe21477a3cd2b2bdfe8930706987cc22cd/comparisons/global/development/source-manifest.json) for the source snapshots and selected prototype records. The `.py.txt` snapshots are evidence, not active model implementations.

Full discarded posterior caches remain in the local development archive. The final baseline whole-field samples and every final sensitivity's volume samples are published in each eruption report. The final stress suite uses seeds 10 and 11, records every outcome, and does not reuse the seed-0 trial as independent validation. Twelve small development trials cannot establish universal 95% coverage or rule out model misspecification.
