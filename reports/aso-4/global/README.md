# Aso-4: current global calculations

[Eruption overview](../README.md) · [All reports](../../README.md) · [Method definitions](../../../docs/METHODS.md)

Both calculations below use **Gaussian-process models**, the same accepted input selection and the complete physical WGS84 Earth for their headline volume.

| Calculation | Meaning |
|---|---|
| [Selected GP](selected/README.md) | One configuration chosen using geographic validation; the current default |
| [GP ensemble](stacked/README.md) | A weighted combination of GP configurations, evaluated as an alternative |

Regional figures and input sensitivities remain in [regional history](../regional/README.md). Integration over the whole Earth removes a calculation-boundary choice; it does not establish the accuracy of extrapolation or a global probability interval.
