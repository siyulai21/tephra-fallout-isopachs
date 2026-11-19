# Development of the directional estimate

These two exploratory models were each run on all six accepted eruptions with eleven policies, geographic and quadrant deletions, and known-volume trials. The final model was developed after these experiments. All three protocols use the same accepted eruption measurements.

| Exploratory model | Aso-4 median km³ [conditional 95%] | West-sector median fraction | Geographic score | Exact-site coverage |
|---|---:|---:|---:|---:|
| single scale | 1,298 [733–2,297] | 11.7% | 1.991 | 69.9% |
| two scale fixed tail | 872 [508–1,828] | 0.9% | 1.558 | 85.5% |

The first protocol assigned one shared radial scale and allowed directional extensions only from the distal half of the retained triangle vertices. An isolated, thick offshore Aso-4 measurement could increase the scale in every direction. Buffered predictions also deteriorated. The second protocol gave compact proximal and directional components separate shares, let isolated measurements constrain narrow bearings, and restored unresolved site variability in gaps.

Both early reference models fixed the distal power at four. Even with the improved direction rule, the second model's intervals missed the true 33.6 km³ smooth exponential deposit on all four development seeds. Its multi-lobe medians also exceeded the known total. This motivated inferring the distal shape on a common prior, with fixed power and exponential thinnings retained as sensitivities. The final evaluation adds fresh noise seeds and preserves all numerical warnings.

Before the first complete protocol, the initial implementation produced 62 posterior attempt records in four cases. One Aso-4 fit stopped when a vectorized sampler compiled for a different array shape was reused. The implementation now keys its sampler by input shapes. All six cases were restarted in fresh directories for the first complete experiment. [Initial designs, attempt checksums and failure](initial-attempts.json.gz).

The compressed record includes every design, attempt seed and posterior checksum, geometry, numerical diagnostic, integrated volume, directional budget and held-out prediction. The lighter table indexes the same experiments. The source archives overlay the unchanged common code from the project immediately before this study, commit 9fdd740b. Run the archived benchmark for each case or scenario and integrate its saved fields with the archived estimate function. The early protocol used seeds 10 and 11. The fixed-tail protocol also used seeds 40 and 41.

Some fixed-tail integration workers were already running when the source on disk advanced to the final protocol. The record preserves their original on-disk hashes and separately identifies the frozen implementation loaded by those workers. Reintegrating every draw of the Aso-4 and smooth-10 reference fields with the frozen code reproduces every sector and region bit for bit. The published final protocol has consistent source hashes throughout.

[Experiments and outcomes](experiments.csv) · [Full exploratory designs, validations and attempts](experiments.json.gz) · [Frozen first implementation](first-protocol.zip) · [Frozen fixed-tail implementation](fixed-tail-protocol.zip) · [Independent reintegration](reintegration.json) · [Final study](../README.md).
