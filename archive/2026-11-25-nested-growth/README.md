# Nested footprint development archive

The preceding principal packets are retained here from commit [`514572583423c861d5cb5a43e4e6bbd963f42cf8`](https://github.com/siyulai21/tephra-fallout-isopachs/tree/514572583423c861d5cb5a43e4e6bbd963f42cf8). They use the same accepted inputs as the continuous-growth comparison. [File identities](previous-files.json)

The Tambora and YTT packets also remain local controls for the fixed footprint study. Other fixed studies use their previously registered source editions. No historical posterior is relabeled as a new fit.

The shared near-vent display rule changes Minoan's directional and spherical windows to match the new nested close-up. Their posterior packets, saved thickness grids and volume integrals are unchanged. [Presentation-only identities](presentation-reframes.json)

## Development trials

The [development manifest](development/files.json) signs each compressed trial and every file inside it. These are diagnostic attempts, including failed and interrupted fits, rather than additional principal estimates. Their maps were not cosmetically smoothed. [Contemporaneous visual notes](development/development-visual-notes.md) · [Numerical environment](development/run-environment.json)

| Trial | Disposition |
|---|---|
| Initial positive area growth | Fixed directions and a restrictive proximal component left overly circular transitions. |
| Expanded fixed-direction catalogue | Added core scales and secondary persistence, but rigid directions and poor sampler movement remained. |
| Joint continuous-direction sampler | The eight-coordinate joint proposal moved too rarely; independent populations disagreed. The principal sampler separates the thinning and angular blocks. |
| Centered proximal core | Removing proximal displacement alone did not resolve Ilopango's northwest continuation. |
| Source-focused prior reweighting | Did not establish a general improvement; importance weights and effective sample sizes remain available. |
| Advecting distal focus | Separating proximal displacement from far focusing did not improve the motivating extrapolation diagnostic. |
| Optional circular proximal core | Partly shortened the northwest extension but restored rounder near contours without a clear improvement in agreement. |
| Local validation attempts | Moved the complete independent validation matrix to a recorded workflow run. Retained all completed and interrupted local attempts; selection was by run, never by score. The historical Pinatubo preview with unconverted source minima is ineligible. |
| Initial map evaluations | The median integral stabilized before both conditional quartile fields did. Retained the earlier Tambora and Aso-4 releases and reviews; the stricter common map check requires joint field and quartile stability. |
| Additional local and thinning-law moves | Reduced disagreement but left YTT's whole-field and B-Tm's fourth-fold site checks unresolved after 2,048 moves. The source, all checkpoints and logs are retained. |
| Joint pilot-proposal experiment | Resolved the motivating YTT population disagreement using exact Metropolis corrections. Retained the development checkpoints and an initial implementation error. The production fallback is separately signed, rerun uniformly on unsuccessful production partitions, and additionally requires stability between successive checkpoints. |

Each tar archive is rooted at its original path relative to `results/nested-adaptive/`. It contains the completed posterior packets, diagnostics, available review plates and exact variant source. The last four variants use a recorded geometry override: their catalogue indices must be interpreted with that archived source, not with the principal field reader. The trial scripts document the overrides. Replaying an early engine requires an isolated checkout with its retained modules; do not substitute today's implementation while retaining an earlier engine identity. Interrupted cases are identified in the status records and were not used as completed comparisons.

## Independent computation receipts

The [run registry](validation-runs.json) identifies each immutable source commit and its assigned partitions. The `validation/` ZIPs preserve the original workflow artifacts, including every posterior, checkpoint, fitting log and environment receipt. Adjacent JSON files retain GitHub's SHA-256, the run identity and a hash for every member. The importer verifies these identities before using a packet. The superseded Pinatubo preview is retained under `validation/ineligible/`; the corrected run fits all reported minima as points. Failed numerical checks remain failures in the collected study even when the workflow completed successfully.

The additional 24,576-particle Pinatubo fourth-fold attempt was precomputed while its preceding budget ran. Both passed; the principal fit retains the first passing 12,288-particle attempt. The larger result is archived as unselected, without choosing between their predictive scores.

[Production run history](development/production-run-history.tar.gz) retains local fitting, refinement, release and verification logs, including interrupted attempts and superseded failures. Its member hashes are in the development manifest. The Pinatubo base run resumed saved completed checkpoints after a memory interruption; the interruption did not change its seed, input or selection rule. [Publication inventory changes](publication-inventory-changes.json) distinguish this revision from pre-existing inventory mismatches and confirm that the accepted CSVs and source audits are unchanged.

[Current method](../../docs/methods/nested-footprints.md) · [Matched review](../../comparisons/nested-isopachs/adaptive-growth/README.md)
