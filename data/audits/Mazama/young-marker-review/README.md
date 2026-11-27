# Young’s Mazama field sections

The contributor’s six annotated map crops identify the centres of **12 overprinted field symbols**. Registering them to Young’s Appendix 1d and joining them to the thickness column of Appendix 1c restores three independent sections and corrects the positions of two existing observations. Young’s numbered Appendix sections now supply 106 of the 304 principal Mazama localities. The 25 distant sites in Young’s Table 1.1 use a separate numbering system. [Numbered appendix map](appendix-index.png) · [Contributor marks beside the original print](source-comparison.png) · [All 138 Appendix section decisions](section-register.csv)

## Sections with climactic fallout thickness

| Section | WGS84 latitude | WGS84 longitude | Young’s upper pumice, cm | Principal decision |
|---|---:|---:|---:|---|
| 95 | 43.145777 | −121.692090 | 358 | Add. Section 15 retains its independently tabulated 194 cm, 5.55 km away on the registered plate. |
| 97 | 43.232371 | −121.792281 | 425 | Move the existing reading 4.61 km from the dot for section 100. Section 100 has clast sizes only. |
| 116 | 42.979377 | −122.100973 | 1,200 | Refine the position by 27 m on the registered map. Section 116 supplies its own reading. |
| 134 | 43.461211 | −122.070585 | 75 | Add. Section 133 retains its independently tabulated 70 cm, 5.39 km away on the registered plate. |
| 150 | 42.981157 | −122.095544 | 2,000+ | Add. The north-wall section is about 0.49 km from section 116 on the plate. Young marks the exposed thickness as a minimum. |

All five values come from Appendix 1c, PDF p. 320. The target is total climactic airborne fallout at about 7.6 ka. Each thickness measures the preserved Plinian Upper Pumice Unit and supplies a geological lower limit if that pumice fell in place. Missing co-ignimbrite ash, compaction and erosion can reduce the preserved thickness. Imported pumice would compromise the transfer. The source table retains the lower-limit designation and Young’s plus sign is recorded for section 150. Under the repository’s fitting rule, each value enters once as a point at the reported minimum. [Appendix transcription](../young1990-appendix1.csv) · [Reported dataset](../../../eruptions/Mazama_young_centroids.csv) · [Fitting dataset](../../../eruptions/Mazama_young_centroids_minimum_fit.csv)

## Reading the marked sections

| Contributor crop | Appendix 1d PDF page | Marked sections | Field-table evidence |
|---|---:|---|---|
| [1](clip-01.png) | 321 | 141, 142, 143 | Appendix 1a records Llao Rock pumice of 35+, 70 and 90 cm respectively. Those readings concern the precursor eruption. |
| [2](clip-02.png) | 321 | 110, 154 | Appendix 1a records Llao Rock and Cleetwood pumice. The 2,000 cm at 154 is Lower Cleetwood. |
| [3](clip-03.png) | 323 | 112, 116, 150, 111 | Sections 116 and 150 enter separately. At 111 and 112 the upper-pumice thickness cells are blank. The layer 1 pumice/lithic entries of 40/47 and 50/20+ cm describe clast diameters. Areal coverage and packing are unknown. |
| [4](clip-04.png) | 324 | 134 | The marked dot overlaps the printed 133. The unmarked dot to the left locates 133. |
| [5](clip-05.png) | 324 | 97 | The marked dot overlaps the printed 100. The unmarked dot to the left locates 100. |
| [6](clip-06.png) | 324 | 95 | The marked dot overlaps the printed 16. The separate dark symbols locate 16 to the left and 15 to the right. |

Sections 110, 111, 112, 141, 142, 143 and 154 therefore have reviewed positions and no numerical constraint on climactic bulk thickness from these tables. The separate 83/84 overprint is outside the supplied crops. Neither has a tabulated Upper Pumice Unit thickness. The section register retains their earlier map qualification through the original picks and both section entries. [Previous map picks](../young1990-map-picks.csv) · [Primary-source audit and table search](../YOUNG_AND_FIELD_SECTIONS.md)

## Position and reproduction

The complete 363-page Lancaster thesis from the contributor’s copy matches the archived SHA-256. Re-extracting PDF pp. 313–320 reproduces all 295 field-table lines byte for byte. The thesis supplies the section IDs, unit, thickness, units and qualifier. Appendix 1d supplies the missing positions. The printed map scales, north arrows, vent registration and the common section 106 anchor on p. 321 follow the preceding field audit. [Thesis catalogue](https://ethos.bl.uk/OrderDetails.do?uin=uk.bl.ethos.315541) · [Source-file identity](../../../sources/mazama/source-manifest.csv) · [Geographic controls](../young1990-georeferencing.json)

The crop registration matches surviving black print after masking the coloured annotations. Refitting the left and right portions of the print separately shifted a marked centre by at most 0.12 pixel on the 128-dpi page. That measures transfer from crop to page. Alternative map origins move these sites by about 3 km in absolute WGS84 position. The geographic digits preserve the computation. Distances between dots on the same plate share its scale and orientation. [Crop transforms and checksums](clip-registration.json) · [Old and new pixels, geographic positions and alternative origins](section-register.csv)

```sh
python data/audits/Mazama/young-marker-review/check_source_plate.py "/path/to/Young-1990.pdf"
python -m benchmarks.mazama_young_centroids
python -m benchmarks.principal_input_figures --verify mazama
```

The 304-site source contains 47 measurements, 132 geological lower limits, 96 upper limits and 29 two-sided ranges. The published model results use the 300-site input. The intervening 301-site Bolan Lake edition and its field evidence have an immutable revision. [Current input figures](../../../../eruptions/mazama/observations/README.md) · [Release identities](provenance.json) · [301-site release](https://github.com/siyulai21/tephra-fallout-isopachs/tree/580afd31154844b33a211f2a9ccb22b4c74d7c2f/eruptions/mazama)
