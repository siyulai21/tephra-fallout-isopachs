# Peer empirical methods and regional reports

[Source map](../README.md) · [Regional protocol](../../docs/methods/regional-comparison.md) · [Spline derivation](../../docs/methods/cubic-splines.md)

`tephra-isopachs compare` evaluates the GP ensemble and automatic cubic spline without choosing an overall winner.

| File | Responsibility |
|---|---|
| `spline.py` | Cubic basis, integrated penalties, mixed likelihood and conditional coefficient uncertainty |
| `selection.py` | Exhaustive geographic tuning with automatic boundary expansion |
| `validation.py` | Shared automatic buffer, nested splits and prediction/deletion assessment |
| `region.py` | Observation hull, regional integration and joint volume draws |
| `reporting.py` | Provenance, tuning tables, matched maps and reports |
| `cli.py` | CSV-oriented command without manual statistical tuning options |

The GP remains in `probabilistic_isopach/`. This package shares observation semantics and probability/partition utilities but the active spline does not fit a GP. New families should implement the common predictive-distribution contract, use identical outer folds and integrate the same physical region. Never feed one method's predictions into another as training evidence.
