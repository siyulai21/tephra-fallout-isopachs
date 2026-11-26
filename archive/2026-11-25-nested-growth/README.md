# Nested footprint development archive

The preceding principal packets are retained here from commit [`514572583423c861d5cb5a43e4e6bbd963f42cf8`](https://github.com/siyulai21/tephra-fallout-isopachs/tree/514572583423c861d5cb5a43e4e6bbd963f42cf8). They use the same accepted inputs as the continuous-growth comparison. [File identities](previous-files.json)

The Tambora and YTT packets also remain local controls for the fixed footprint study. Other fixed studies use their previously registered source editions. No historical posterior is relabeled as a new fit.

## Rejected development trials

The [development manifest](development/files.json) signs each compressed trial and every file inside it. These are diagnostic attempts, including failed and interrupted fits, rather than additional principal estimates. Their maps were not cosmetically smoothed. [Contemporaneous visual notes](development/development-visual-notes.md) · [Numerical environment](development/run-environment.json)

| Trial | Reason it was not selected |
|---|---|
| Initial positive area growth | Fixed directions and a restrictive proximal component left overly circular transitions. |
| Expanded fixed-direction catalogue | Added core scales and secondary persistence, but rigid directions and poor sampler movement remained. |
| Joint continuous-direction sampler | The eight-coordinate joint proposal moved too rarely; independent populations disagreed. The principal sampler separates the thinning and angular blocks. |
| Centered proximal core | Removing proximal displacement alone did not resolve Ilopango's northwest continuation. |
| Source-focused prior reweighting | Did not establish a general improvement; importance weights and effective sample sizes remain available. |
| Advecting distal focus | Separating proximal displacement from far focusing did not improve the motivating extrapolation diagnostic. |
| Optional circular proximal core | Partly shortened the northwest extension but restored rounder near contours without a clear improvement in agreement. |

Each tar archive is rooted at its original path relative to `results/nested-adaptive/`. It contains the completed posterior packets, diagnostics, available review plates and exact variant source. The last four variants use a recorded geometry override: their catalogue indices must be interpreted with that archived source, not with the principal field reader. The trial scripts document the overrides. Replaying an early engine requires an isolated checkout with its retained modules; do not substitute today's implementation while retaining an earlier engine identity. Interrupted cases are identified in the status records and were not used as completed comparisons.

[Current method](../../docs/methods/nested-footprints.md) · [Matched review](../../comparisons/nested-isopachs/adaptive-growth/README.md)
