# Complete Aso-4 development records

[Study and reproduction commands](../README.md) · [Source audit](../../../data/audits/Aso-4/README.md)

- [Original-data protocol](study/protocol.json): nine declared source/geometry cases, each with its input, definition, report and full posterior files. The `offset_sd1` case also contains four fully refitted geographic holdouts.
- [Original-posterior influence diagnosis](diagnosis/report.json) and [all 87 approximate deletion checks](diagnosis/single-row-deletion.csv).
- [Coordinate/bound revision protocol](revision/protocol.json), [revised full study](revision/baseline/study.json), [coordinate-only experiment](revision/coordinate_only/report.json), and [matched tighter-prior study](revision/offset_sd1/study.json).
- [Biwa source-conflict protocol and results](conflict/report.json): both primary-table endpoints on the corrected coordinates.
- [Marine-setting protocol and results](marine/report.json): remove all 28 explicitly submarine records under each of two geometry priors.

Every numerical attempt is retained as an immutable JSON record plus a complete `.npz` posterior under its case's posterior folder. Failed early attempts are not pooled into successful draws or hidden. Checksums, input-array signatures, runtime versions, priors, chain counts, draw counts and numerical diagnostics are recorded with each attempt. The selected final attempt is identified in the corresponding fit record.

These source alternatives, omissions and geometry priors have no probability weights. They do not form a joint credible interval. The marine omission changes spatial coverage and is not a fitted transport correction.
