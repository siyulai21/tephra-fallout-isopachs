# Mazama principal observations and thickness intervals

The accepted input contains **300 localities**. It adds 29 localities to the preceding 271-location audit. Each row supplies one observation to every family. Coordinates are WGS84 longitude and latitude and thicknesses are centimetres. [Principal CSV](../../../eruptions/Mazama_primary_ranges.csv) · [All 872 locality and evidence decisions](locality-decisions.csv) · [Source readings and figures](source-notes.md) · [Input and source hashes](provenance.json)

| Principal observation | Localities | Model input |
|---|---:|---|
| Point thickness | 46 | `Thickness` |
| Two-sided interval | 29 | `Lower_cm` and `Upper_cm` |
| Lower limit | 129 | `Limit_cm` with `Kind=lower_limit` |
| Upper limit | 96 | `Limit_cm` with `Kind=upper_limit` |

## Where Buckland’s classification changes inclusion

All **71 P-coded S1 localities** have an explicit decision. Twenty-six already had a quantitative row under the same ID. Twenty-three previously omitted rows now enter the principal input. Gold Lake and Creston/Wilbur have a quantitative row under a different ID. Of the 20 remaining omissions, 16 lack an interpretable thickness for the target at the specified position. Four have numerical values with a specific field conflict. [All 71 S1 primary decisions](buckland-primary.csv) · [Original S1 transcription](../buckland2020-293-localities.csv)

| Omitted numeric entry | Field evidence |
|---|---|
| Muir Creek, 850 cm | The roadcut exposes lapillistone and overlying stratified sandstone. The charcoal context has a flow interpretation. A fall-only thickness cannot be isolated. |
| Carbon River, 5 cm | The cited three-inch value is a park-wide maximum. The section lacks a defensible local thickness. |
| Williams Lake Fen, 46 cm | Two horizons are separated by 9 cm of lake sediment. Their combined report does not resolve the climactic bed. |
| Goldeye Lake Fen, 2 cm | The investigator describes a mixed 20 cm low-organic interval. The published profile conflicts with a discrete 2 cm bed. |

South Ice Cave (68 cm), Three Creek (37 cm), Breitenbush (17 cm), Tepee Lake (10 cm) and Squeah Lake (2 cm) enter on Buckland’s P classification. Their audit distinguishes the compiler-supplied thickness from direct verification of the original bed. At Moscow the observer reports enhancement of a 60 cm package and an estimated original thickness of about 30 cm. The fitted conditional ceiling is 60 cm. Distance alone did not remove or admit a site. [Field and source decisions](buckland-primary.csv)

The full crosswalk also includes all 21 S1 rows labelled B (both primary and remobilised). Separately identified airfall already contributes at Wildcat Lake (1 cm basal component) and the Skagit delta (2 cm basal bed under R2001-KSNEA). At Cooley Lake, the hole repeatedly filled and could not penetrate the ash. The 252 cm entry cannot be assigned to an intact primary thickness. At Stampede, the newly recovered field classifications do resolve primary beds. [All catalogue classes and source decisions](locality-decisions.csv)

## How the ranges enter

The GP, cubic spline, directional ensemble, spherical mixture and nested corridors fit the probability that the log-thickness observation lies between the two endpoints. For a Gaussian observation with log mean μ and standard deviation σ this is Φ((log U − μ)/σ) − Φ((log L − μ)/σ). Mixtures integrate this event within each component before combining probabilities. Geographic training and held-out scoring use the same event. Each interval has one row and one pair of coordinates. A quoted range supplies no independent estimate of log standard deviation.

Examples include 1.27–1.905 cm at Portage Inlet, 15–30 cm wet thickness at Newman Lake, 0.4–0.5 cm in Yellowstone core YL16-2C, and 2–5 cm in Mount Baker section RS–5. At Creston the identified auger is retained once. At the Cypress Hills H, Q and Stampede sites, intervals represent variation between named primary soundings. The transcription provides the individual thicknesses and survey coordinates. [Range review](range-review.csv) · [Cypress Hills sections](cypress-sections.csv)

The audit also records ranges from mixed packages, partial eruptive units, multiple cores with documented focusing and studies reporting only a group-level facies envelope. These retain a one-sided constraint or an omission where only one endpoint or neither endpoint transfers to **original local fallout**. The CSV columns `ReportedLower_cm` and `ReportedUpper_cm` retain the observed values independently of the model columns. A bed depth pair such as 177–200 cm at Dog Lake measures a single 23 cm layer. [Quantity and locality decisions](locality-decisions.csv) · [Interpretation protocol](../../../../docs/methods/sites-without-measured-thickness.md)

All five published fits use these 300 localities on matched geographic holdouts. The case page gives the regional and whole-Earth estimates, their thickness bands and the numerical qualifications. [Maps and results](../../../../eruptions/mazama/README.md)
