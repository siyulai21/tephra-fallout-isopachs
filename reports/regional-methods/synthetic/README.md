# Known-truth regional trials

[Comparison and interpretation](../README.md) · [Per-run results](synthetic.json) · [Compact table](comparison.csv)

Forty trials were attempted: 38 completed for both methods; two heavily censored trials lacked sufficient exact observations in nested geographic training subsets. They remain explicit failures in the JSON and attempted-count denominator. Metrics and coverage describe the completed cases.

`selection/` retains candidate tables for every completed final fit. `benchmark-at-execution.py` preserves the exact benchmark driver whose checksum is recorded in the numerical result. The current benchmark has a later summary-table formatting fix for missing trials; numerical fitting modules and simulation definitions are unchanged.

Conditional intervals omit fitted-parameter and selection uncertainty. Poor coverage is evidence of that limitation, not a reason to remove an unfavorable trial.
