# Sources for the Ilopango Tierra Blanca Joven fallout

## Pedrazzi et al. (2019)

*The Ilopango Tierra Blanca Joven (TBJ) eruption, El Salvador: Volcano-stratigraphy and physical characterization of the major Holocene event of Central America.* The preserved file is the publisher's original Supplementary Material 1. Sheet `Hoja1` lists site IDs, locations and the thickness of each mapped unit in centimetres. Keep the workbook with its original attribution. [Paper](https://doi.org/10.1016/j.jvolgeores.2019.03.006) · [Publisher's workbook](https://ars.els-cdn.com/content/image/1-s2.0-S0377027318303706-mmc1.xls) · [Preserved workbook](pedrazzi-supplement-1.xls)

The source audit reads the original spreadsheet cells. Columns C and D give north latitude and west longitude in degrees and decimal minutes. Column P gives the measured thickness of the upper co-ignimbrite fallout, Unit G. Column H includes other parts of the TBJ sequence, including deposits from pyroclastic density currents. [Source-to-input ledger](../../audits/Ilopango-TBJ/selection.csv)

## Smith et al. (2020)

*The magnitude and impact of the 431 CE Tierra Blanca Joven eruption of Ilopango, El Salvador.* The authors give the coordinates of cores OC001 and MES001 and report a 3 cm bed at each in Methods, “Ash Dispersal Modeling”. They attribute these readings to Neff and Sheets and use them for the co-ignimbrite dispersal calculation. The CSV transcribes the two numerical records at the precision printed in that paragraph. [Methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC7584997/) · [Two core records](smith-2020-cores.csv)

Page 7 of the accepted Oxford manuscript gives the two thicknesses and their coordinates. The ice-core methods also locate a third site, TUNU2013 in Greenland, where glass chemistry identifies TBJ ash. The audit records it as an eruption-level occurrence. The optional replay checks the two measured beds and the Greenland coordinates against the manuscript text. The SHA-256 of that original PDF is `1a01f05e868a02974c067802c1c5e3b6d1f2d431147201efeef225b6a2ba6c78`. [Download the Oxford manuscript](https://ora.ox.ac.uk/objects/uuid%3Aebeebaeb-712f-450c-9239-a9bd65851e00/files/shq37vn92g)

Kutterolf et al. correlate marine bed C1 with TBJ. Pedrazzi et al. place the associated offshore measurements in column P. Offshore thickness can reflect bed preservation and the distal correlation alone cannot resolve individual eruption pulses. The regional calculations use the authors’ assignment. [Marine stratigraphy, Table 1](https://doi.org/10.1029/2007GC001631)

The modelling vent at 13.672° N, 89.053° W is the Smithsonian Global Volcanism Program's reference for Ilopango caldera. The erupting vents and the parts of the density currents that lofted ash could have occupied a wider area. [Ilopango reference location](https://volcano.si.edu/volcano.cfm?vn=343060) · [Deposit interpretation](https://doi.org/10.1016/j.jvolgeores.2019.03.006)

[Source and accepted-input checksums](../../audits/Ilopango-TBJ/provenance.json) · [Selection and coverage](../../audits/Ilopango-TBJ/README.md)
