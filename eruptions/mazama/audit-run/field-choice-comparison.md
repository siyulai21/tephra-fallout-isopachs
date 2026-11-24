# How the distal source decisions affect modelled volume

All fifteen geological alternatives completed on GitHub. Their 75 family-specific results are shown here. The GP and cubic spline cover the fixed hull of 26 historical measurements in this comparison. The other three methods integrate over Earth, conditional on their continuations. Volumes are in bulk km³. The principal regional common-hull values independently replay the official selected settings. The larger current-hull regional integrals appear on the case page. [Current-hull estimates and maps](README.md) · [Unrounded volumes, conditional intervals, flags and GitHub artifact hashes](field-choice-volumes.csv) · [Source definitions](../../../benchmarks/mazama_sensitivity.py)

| Source interpretation | GP ensemble, common hull | Cubic spline, common hull | Directional, Earth | Spherical mixture, Earth | Nested isopachs, Earth |
|---|---:|---:|---:|---:|---:|
| principal | 91.17 | 194.46 † | 214.46 † | pending | 175.29 |
| alternate-field-bounds | 91.15 | 194.54 † | 215.05 † | 241.60 | 176.99 |
| before-field-243 | 92.35 | 198.63 | 218.46 † | 239.61 | 196.31 |
| cathedral-one | 91.34 | 194.28 † | 214.90 † | 236.53 | 174.80 † |
| half-capture-ceilings | 99.16 | 195.34 † | 219.50 | 238.81 | 179.77 |
| jenny-omit | 91.23 | 194.47 † | 216.12 † | 243.67 | 180.56 |
| jenny-one | 91.18 | 194.46 † | 213.61 | 242.17 | 187.49 |
| omit-all-archive-ceilings | 222.17 | 395.55 | 332.80 | 329.40 | 222.66 |
| omit-basin-beds | 115.28 | 243.04 | 201.17 | 305.71 | 158.69 † |
| omit-qualified-identities | 91.59 | 195.67 † | 214.36 † | 239.78 | 238.20 † |
| omit-reinterpreted-ceilings | 137.18 | 207.78 | 247.59 † | 257.95 | 183.42 |
| omit-young-proximal | 33.71 | 105.68 | 68.81 † | 163.40 | 103.37 |
| prior-244 | 166.69 | 226.62 † | 331.96 † | 352.19 † | 274.64 |
| restore-thick-archive-readings | 161.68 | 344.80 † | 322.84 † | 371.17 | 208.35 |
| restore-williams | 106.50 | 224.49 | 220.45 † | 237.78 | 193.99 |
| yellowstone-one | 104.58 | 221.25 | 211.31 † | 248.05 | 237.24 † |

A dagger marks an unpassed fit, draw or integration check. Geographic prediction has its own checks. All source inputs matched the registered SHA-256 values, all downloaded packets matched GitHub’s digests and every family artifact matched its report. [Geologic evidence by site](../../../data/audits/Mazama/distal-review/FIELD_EVIDENCE.md) · [Withheld regional site predictions](regional-withheld-by-site.csv)

Restoring the earlier thick readings at nine archives raises both regional integrals on their fixed comparison footprint. It also raises the directional median-map integral from about 214 to 323 km³ and the nested integral from about 175 to 208 km³. The directional fits retain numerical warnings. The alternate Johnson, Kilpoola, Foy and Greater Victoria field bounds have a much smaller effect on both of those global families. These comparisons change groups of evidence decisions together. A single site’s contribution cannot be recovered by subtracting these volumes.
