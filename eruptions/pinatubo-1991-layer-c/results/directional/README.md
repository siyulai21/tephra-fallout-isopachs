# Pinatubo 1991 · climactic Layer C · Directional ensemble

The principal bulk volume is **5.78 km³**, the whole-Earth integral of the median-thickness map. The conditional thickness-band range is **4.65–7.26 km³**. Its endpoints integrate the pointwise 25th and 75th percentiles of latent thickness over Earth. **Numerical warning on the fitted deposit distribution.**

![Pinatubo 1991 · climactic Layer C, Directional ensemble. Full deposit, near the vent and thickness versus square root of cumulative area](deposit.png)

Each circle shows a measurement. Bounded sites take the colour of their stated limit. The curve counts every patch at or above each thickness. The 1 cm footprint includes 209 of 235 modeled locations. The thickest displayed line is 32 cm and encloses 5 fitted locations. The volume includes fallout beyond the displayed contours. [Median thickness grid](median-grid.npz) · [Thickness and cumulative area](thickness-area.csv) · [Map integral, resolution and source hashes](deposit.json)

The principal map uses the geographic continuation evaluated on all registered eruptions. [Geographic holdouts, structural alternatives and numerical checks](extent/report.json) · [Complete contour decisions](extent/support.json.gz) · [Predictions at withheld observations](extent/heldout.csv.gz)

## Earlier continuation and its diagnostics

The following figures and statistics describe the fitted parent and its broader continuation. The principal estimate and its interval appear above.

## Reach from the volcano

![Fixed and uncertain fan directions near the source](azimuth/median-isopachs.png)

Distances are great-circle kilometres to the first outward crossing on each compass ray. The area columns give the entire 10 cm footprint in the corresponding 45° sector, including detached areas. Together they distinguish a long point on a contour from an entire broad lobe.

| Bearing | Fixed 10 cm reach (km) | Uncertain 10 cm reach (km) | Fixed ≥10 cm area (km²) | Uncertain ≥10 cm area (km²) |
|---|---:|---:|---:|---:|
| N | 19.3 | 18.2 | 145 | 137 |
| NE | 17.8 | 17.8 | 123 | 123 |
| E | 21.9 | 21.9 | 171 | 171 |
| SE | 22.6 | 22.6 | 205 | 205 |
| S | 31.2 | 31.2 | 431 | 431 |
| SW | 48.1 | 48.4 | 1,111 | 1,111 |
| W | 74.8 | 74.8 | 1,531 | 1,531 |
| NW | 31.2 | 30.9 | 341 | 330 |

![Probability that a deposit reaches ten centimetres](azimuth/exceedance.png)

The 50% and 90% probability lines answer different questions. At a location on the 90% line, nine in ten model draws reach 10 cm. These are pointwise probabilities conditional on the model. Joint confidence over an entire region requires a different calculation.

## Withheld observations

Each row averages negative log predictive scores over equally weighted folds. Lower scores predict the withheld thicknesses and limits better. The passing-only columns exclude numerical failures. Coverage counts exact withheld thicknesses inside the uncertain-direction 95% predictive interval. A dash marks an unavailable score or coverage.

| Deletion | All estimable folds | Fixed | Uncertain | Passing folds | Fixed, passing | Uncertain, passing | Exact-site 95% coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| Geographic regions | 4 | 0.982 | 1.131 | 1 | 0.815 | 0.942 | 146/147 (99.3%) |
| Source quadrants | 4 | 1.430 | 1.383 | 1 | 1.997 | 1.471 | 136/147 (92.5%) |

[Predictions for every estimable held-out site](azimuth/heldout.csv). The numerical record includes all planned folds, row identities, unscorable directions, fit warnings and phase-integration checks.

## Inspected archives

The principal fit includes the interpreted upper limits below. Each core enters once. Other sites provide an occurrence check. The interval describes the modeled latent thickness at each site.

| Site | Interpreted upper limit (cm) | Median thickness (cm) | Conditional 95% interval (cm) |
|---|---:|---:|---:|
| WW-open-01 | Occurrence only | 1.37e-12 | 1.06e-39–2.45e-05 |
| WW-open-02 | Occurrence only | 5.22e-06 | 5.93e-11–0.000842 |
| WW-open-03 | Occurrence only | 1.75e-05 | 4.83e-09–0.00148 |
| WW-open-04 | Occurrence only | 1.92e-06 | 6.34e-35–0.000833 |
| WW-open-05 | Occurrence only | 3.1e-05 | 1.13e-08–0.00208 |
| WW-open-06 | Occurrence only | 3.97e-06 | 8.72e-32–0.0012 |
| WW-open-07 | Occurrence only | 5.75e-05 | 6.11e-08–0.00285 |
| WW-open-08 | Occurrence only | 6.09e-06 | 1.45e-31–0.00163 |
| WW-open-09 | Occurrence only | 4.87e-10 | 7e-29–0.000399 |
| WW-open-10 | Occurrence only | 0.000109 | 1.52e-07–0.0044 |
| WW-open-11 | Occurrence only | 1.79e-05 | 7.45e-28–0.00308 |
| WW-open-12 | Occurrence only | 2.38e-08 | 5.56e-28–0.00135 |
| WW-open-13 | Occurrence only | 2.75e-05 | 1.69e-07–0.000825 |
| WW-open-14 | Occurrence only | 0.000542 | 1.67e-06–0.0116 |
| WW-open-15 | Occurrence only | 1.97e-06 | 1.43e-25–0.00452 |
| WW-open-16 | Occurrence only | 0.000256 | 6.16e-12–0.0133 |
| WW-open-17 | Occurrence only | 0.00104 | 3.8e-06–0.0212 |
| WW-open-18 | Occurrence only | 0.00113 | 7.82e-07–0.0259 |
| WW-open-19 | Occurrence only | 0.00139 | 4.28e-05–0.0182 |
| WW-open-20 | Occurrence only | 1.22e-07 | 2.14e-19–0.00381 |
| WW-open-21 | Occurrence only | 0.000221 | 2.16e-06–0.0137 |
| WW-open-22 | Occurrence only | 0.00038 | 1.03e-05–0.0116 |
| WW-open-23 | Occurrence only | 0.00196 | 7.61e-05–0.0759 |
| WW-open-24 | Occurrence only | 0.00224 | 3.41e-14–0.173 |
| WW-open-25 | Occurrence only | 0.0167 | 0.000155–0.207 |
| WW-open-26 | Occurrence only | 0.0228 | 9.87e-05–0.224 |
| WW-open-27 | Occurrence only | 0.0229 | 0.00198–0.183 |
| WW-open-28 | Occurrence only | 0.0306 | 0.00661–0.165 |
| WW-open-29 | Occurrence only | 0.084 | 0.0157–0.188 |
| WW-open-30 | Occurrence only | 0.0697 | 0.00418–0.189 |
| WW-open-31 | Occurrence only | 0.0352 | 0.000738–0.187 |
| WW-open-32 | Occurrence only | 0.00416 | 0.000164–0.0206 |
| WW-open-33 | Occurrence only | 0.00147 | 6.03e-08–0.00919 |
| WW-open-34 | Occurrence only | 0.00132 | 1.39e-07–0.00922 |
| WW-open-35 | Occurrence only | 0.00083 | 2.34e-06–0.00804 |
| WW-open-36 | Occurrence only | 0.000864 | 7.61e-06–0.00885 |
| WW-open-37 | Occurrence only | 0.000677 | 1.27e-06–0.00689 |
| SO132_10 | Occurrence only | 0.385 | 0.206–0.64 |
| SO132_11 | Occurrence only | 0.974 | 0.67–1.35 |
| SO132_12 | Occurrence only | 1.44 | 1.03–1.9 |
| SO132_12A | Occurrence only | 1.52 | 1.22–2.07 |
| SO132_13 | Occurrence only | 2.74 | 1.77–4.08 |
| SO132_14 | Occurrence only | 2.05 | 1.51–2.57 |
| SO132_15 | Occurrence only | 1.22 | 0.981–2.12 |
| SO132_16 | Occurrence only | 1.11 | 0.692–1.76 |
| SO132_17 | Occurrence only | 0.222 | 0.00874–1.32 |
| SO132_19 | Occurrence only | 0.0578 | 0.0014–0.955 |
| SO132_2 | Occurrence only | 0.0131 | 0.00121–0.037 |
| SO132_20 | Occurrence only | 0.712 | 0.145–2.03 |
| SO132_21 | Occurrence only | 1.99 | 1.42–2.55 |
| SO132_22 | Occurrence only | 2.44 | 1.62–3.64 |
| SO132_23 | Occurrence only | 2.07 | 1.57–2.72 |
| SO132_24 | Occurrence only | 1.7 | 1.21–2.4 |
| SO132_25 | Occurrence only | 2.2 | 1.65–2.91 |
| SO132_26 | Occurrence only | 2.68 | 2.04–3.41 |
| SO132_27 | Occurrence only | 3.11 | 2.37–4.02 |
| SO132_28 | Occurrence only | 3.82 | 2.65–5.53 |
| SO132_29 | Occurrence only | 4.24 | 3.28–5.55 |
| SO132_3 | Occurrence only | 0.0431 | 0.0124–0.0863 |
| SO132_30 | Occurrence only | 3.47 | 2.61–4.42 |
| SO132_31 | Occurrence only | 1.96 | 1.49–2.52 |
| SO132_32 | Occurrence only | 2.47 | 1.74–3.52 |
| SO132_33 | Occurrence only | 2.01 | 1.23–3.32 |
| SO132_34 | Occurrence only | 1.55 | 0.382–4.3 |
| SO132_35 | Occurrence only | 1.92 | 0.0313–6.82 |
| SO132_36 | Occurrence only | 4.49 | 2.63–7.43 |
| SO132_37 | Occurrence only | 4.28 | 2.62–6.76 |
| SO132_37A | Occurrence only | 4.94 | 3.6–7.65 |
| SO132_38 | Occurrence only | 5.38 | 3.46–8.09 |
| SO132_39 | Occurrence only | 4.77 | 1.28–10.1 |
| SO132_4 | Occurrence only | 0.0973 | 0.0185–0.261 |
| SO132_40 | Occurrence only | 7.14 | 3.57–8.98 |
| SO132_41 | Occurrence only | 6.46 | 4.39–9.81 |
| SO132_42 | Occurrence only | 8.34 | 7.14–9.91 |
| SO132_43 | Occurrence only | 8.98 | 7.84–10.4 |
| SO132_44 | Occurrence only | 7.96 | 6.95–9.34 |
| SO132_44A | Occurrence only | 8.43 | 7.36–9.86 |
| SO132_47 | Occurrence only | 0.982 | 0.548–1.7 |
| SO132_48 | Occurrence only | 0.798 | 0.409–1.41 |
| SO132_49 | Occurrence only | 3.72 | 1.26–8.33 |
| SO132_5 | Occurrence only | 0.325 | 0.234–0.536 |
| SO132_50 | Occurrence only | 4.51 | 2.87–7.31 |
| SO132_7 | Occurrence only | 0.765 | 0.584–1.4 |
| SO132_8 | Occurrence only | 1.04 | 0.831–1.54 |
| SO132_9 | Occurrence only | 0.634 | 0.364–0.892 |
| GIK18429 | Occurrence only | 0.00219 | 7.9e-05–0.0383 |
| GIK18430 | Occurrence only | 0.0126 | 0.00116–0.0358 |
| SO140B_10 | Occurrence only | 3.12 | 2.37–4.02 |
| SO140B_11 | Occurrence only | 3.81 | 2.64–5.52 |
| SO140B_12 | Occurrence only | 4.29 | 2.62–6.71 |
| SO140B_13 | Occurrence only | 5.38 | 3.45–8.08 |
| SO140B_14 | Occurrence only | 7.13 | 3.57–8.97 |
| SO140B_23 | Occurrence only | 9.82 | 8.57–11.3 |
| SO140B_24 | Occurrence only | 8.35 | 7.14–9.92 |
| SO140B_25 | Occurrence only | 7.99 | 6.91–9.33 |
| SO140B_3 | Occurrence only | 0.6 | 0.374–0.92 |
| SO140B_4 | Occurrence only | 1.22 | 0.98–2.12 |
| SO140B_5 | Occurrence only | 1.58 | 1.18–2.36 |
| SO140B_6 | Occurrence only | 2.05 | 1.51–2.57 |
| SO140B_7 | Occurrence only | 2.81 | 2.17–3.55 |
| SO140B_8 | Occurrence only | 1.99 | 1.43–2.55 |
| SO140B_9 | Occurrence only | 1.7 | 1.21–2.4 |
| SO114-4 (Wetzel 404) | Occurrence only | 5.39 | 3.38–8.26 |
| GIK17920-1 | Occurrence only | 7.14 | 3.57–8.98 |
| GIK17921-1 | Occurrence only | 8.35 | 7.14–9.92 |
| GIK17922-1 | Occurrence only | 1.84 | 1.27–2.47 |
| GIK17924-1 | Occurrence only | 2.14e-08 | 7.97e-28–0.00137 |
| GIK17925-2 | Occurrence only | 2.84e-10 | 1.64e-29–0.000309 |
| GIK17926-2 | Occurrence only | 1.03e-06 | 5.23e-26–0.00352 |
| GIK17927-1 | Occurrence only | 0.00109 | 2.58e-14–0.141 |
| GIK17928-2 | Occurrence only | 7.4e-08 | 4.94e-20–0.00358 |
| GIK17929-1 | Occurrence only | 4.35e-06 | 2.69e-11–0.000752 |
| GIK17930-1 | Occurrence only | 1.44e-05 | 2.48e-09–0.0014 |
| GIK17931-1 | Occurrence only | 2.68e-05 | 6.71e-09–0.00199 |
| GIK17932-1 | Occurrence only | 4.13e-05 | 1.75e-08–0.00258 |
| GIK17933-2 | Occurrence only | 0.000135 | 2.16e-07–0.00484 |
| GIK17934-1 | Occurrence only | 0.00054 | 1.65e-06–0.0114 |
| GIK17935-2 | Occurrence only | 0.00081 | 2.56e-06–0.0159 |
| GIK17936-1 | Occurrence only | 0.000665 | 3.73e-07–0.0217 |
| GIK17937-1 | Occurrence only | 1.15e-05 | 7.42e-31–0.00266 |
| GIK17938-1 | Occurrence only | 4.5e-06 | 6.26e-34–0.00157 |
| GIK17939-1 | Occurrence only | 2.46e-06 | 1.63e-35–0.00111 |
| GIK17940-1 | Occurrence only | 1.54e-06 | 1.17e-36–0.00088 |
| GIK17941-1 | Occurrence only | 1.01e-12 | 7.95e-40–2.18e-05 |
| GIK17942-1 | Occurrence only | 2.39e-05 | 1.21e-07–0.000858 |
| GIK17943-1 | Occurrence only | 0.000176 | 1.31e-12–0.00986 |
| GIK17944-1 | Occurrence only | 8.78e-05 | 1.26e-06–0.00329 |
| GIK17945-1 | Occurrence only | 0.000169 | 1.18e-06–0.0125 |
| GIK17946-1 | Occurrence only | 0.000314 | 7.83e-06–0.0105 |
| GIK17947-2 | Occurrence only | 0.00137 | 3.54e-05–0.0196 |
| GIK17948-1 | Occurrence only | 0.0169 | 0.000138–0.202 |
| GIK17949-1 | Occurrence only | 0.00174 | 4.65e-05–0.0602 |
| GIK17950-1 | Occurrence only | 0.0265 | 0.00509–0.156 |
| GIK17951-1 | Occurrence only | 0.0189 | 0.0012–0.168 |
| GIK17952-2 | Occurrence only | 0.016 | 5.59e-05–0.179 |
| GIK17953-3 | Occurrence only | 0.756 | 0.482–1.14 |
| GIK17953-2 | Occurrence only | 0.759 | 0.491–1.18 |
| GIK17954-1 | Occurrence only | 0.0969 | 0.0539–0.195 |
| GIK17955-1 | Occurrence only | 0.129 | 0.0783–0.22 |
| GIK17956-1 | Occurrence only | 0.142 | 0.0916–0.255 |
| GIK17957-1 | Occurrence only | 0.0691 | 0.00533–0.182 |
| GIK17958-1 | Occurrence only | 0.115 | 0.0613–0.202 |
| GIK17959-1 | Occurrence only | 0.0853 | 0.0183–0.186 |
| GIK17960-1 | Occurrence only | 0.0339 | 0.000362–0.171 |
| GIK17961-1 | Occurrence only | 0.00387 | 0.000168–0.0191 |
| GIK17962-1 | Occurrence only | 0.00139 | 5.59e-08–0.0088 |
| GIK17963-2 | Occurrence only | 0.000717 | 9.04e-06–0.00823 |
| GIK17964-1 | Occurrence only | 0.000612 | 1.28e-06–0.00641 |
| GIK17965-1 | Occurrence only | 0.000687 | 6.11e-06–0.0077 |
| SO114_2 | Occurrence only | 4.49 | 2.62–7.44 |
| SO114_8 | Occurrence only | 8.98 | 7.84–10.4 |

## Checks and reproducibility

The shared contour audit found eligible bands under none of the screened support policies. Audit status is completed. [All contour decisions and deletion failures](azimuth/complete-contour-audit.json)

The spatial integration check passed at 8,192 points per scramble. The map used up to 64 posterior draws per chain and structure. Resolved contour areas passed the split-packet check. The joint-volume quantiles passed the split check.

Full-fit sampling warnings occurred for narrow_fans.

[Numerical report, all sector volumes and sensitivities](azimuth/report.json) · [Complete fields and per-field volumes](azimuth/fields.npz) · [Principal map grids](azimuth/median-grids.npz) · [Near-source comparison grids](azimuth/map-grid.npz) · [Fragment decisions](azimuth/fragment-decisions.csv) · [Bracketed coordinates](azimuth/bracketed-fragments.csv) · [File hashes](azimuth/artifacts.json)

## Related directional studies

- [Earlier fixed-direction ensemble and its separate calibration](ensemble/README.md)
- [Single-structure and proximal-cap sensitivities](single-structure/README.md)
- [Measured-fan continuation and inspected-site evidence](bearing/README.md)
