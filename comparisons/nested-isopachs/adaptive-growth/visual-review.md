# Visual and geological review

Each assessment uses the final published median grid at matched proximal, intermediate and full-deposit scales. The paired plate locates both inner viewports in the full deposit. The separate release plate shows the actual supported and conditional contour styling. Shape descriptors and site residuals accompany each plate; they diagnose the field and do not adjust it.

## Mazama

[Three-scale comparison](review/mazama/scales.png) · [Site evidence](review/mazama/sites.csv.gz) · [Contour descriptors](review/mazama/morphology.csv)

The small circular summit contour gives way to a turning proximal structure, with a northeastern main lobe and a smaller southeastern shoulder already evident in the 256–128 cm geometry. The intermediate and full maps preserve this concavity and gradual change of spacing instead of replacing it with a smooth oval. The northeastern thin reach shortens, but the southwestern conditional extension grows farther into the Pacific. The observed southern branch remains connected to the main field. There are no crossing successive levels or detached rings in the displayed median contours.

Southeastern Y1990-A082 and B2024-UKR improve, as does the two-sided FLV93-C7 interval. The very thick proximal Y1990-A118 is underestimated more strongly, and the northwestern P1999-BEAVER-OR upper bound and B2020-052 measurement worsen. The new geography therefore does not resolve every upwind conflict. Measured log RMSE improves slightly from 0.739 to 0.737, median absolute log error from 0.416 to 0.355, and bounded negative log probability from 0.779 to 0.756. Median upper-limit violations fall from 11 to 10 while interval violations rise from 21 to 24. Geographic score worsens from 1.182 to 1.380 despite passing numerical checks on all four folds; this is a predictive tradeoff, not a convergence warning.

## Tambora

[Three-scale comparison](review/tambora/scales.png) · [Site evidence](review/tambora/sites.csv.gz) · [Contour descriptors](review/tambora/morphology.csv)

The former nearly circular 64 cm contour becomes a mildly eccentric, continuously connected inner contour. Its boundary eccentricity rises from 0.07 to 0.46; the 32–1 cm contours gradually change orientation rather than keeping one expanding ellipse. The full view retains a broad western and northwestern dispersal envelope, with the small proximal contour embedded coherently in the larger field. Every displayed level has one connected component, no radial re-entry and positive spacing from the next contour. The broad south and east margins remain conditional.

Only four measured sites lie within 100 km: CSV rows 2–5. The eastern row 4 is still substantially overpredicted; its signed log error falls from 1.52 to 1.43. The method does not manufacture a sharp nearby edge to fit that one disagreement. Overall measured log RMSE improves from 0.621 to 0.575, while the geographic negative log score worsens from 1.382 to 1.520. The matched regional controls and all fold scores remain in the study.

The first numerical map used 512 posterior draws and contained small angular ripples. The final shared quartile-field check selected 4,096 draws; the ripples diminish without modifying the field or smoothing any contour. The earlier map and review are retained in the development archive.

## Aso-4

[Three-scale comparison](review/aso-4/scales.png) · [Site evidence](review/aso-4/sites.csv.gz) · [Contour descriptors](review/aso-4/morphology.csv)

This is the weakest geometric change. The 64 cm proximal contour remains nearly circular, with a small southward offset, and the 32–8 cm contours retain a broadly concentric relationship. They connect smoothly to the outer field without crossings or detached rings. The outer 1–2 cm contours become somewhat less southward, but still extend thousands of kilometres into poorly sampled western and southern directions. Their broad curvature is a conditional model result, not a mapped deposit edge. The actual deposit plate starts at 2 cm; the matched review also shows the 1 cm continuation. Neither justifies a precise western cutoff.

Northern rows 85 and 88 improve slightly, while row 49 (1 cm, 436 km north) remains substantially overpredicted. Southeastern row 37 (15 cm at 709 km) remains underpredicted. These contrasting observations help explain why a simple narrow northeastern fan is inadequate, but do not establish the opposite side of a full perimeter. Measured log RMSE changes from 0.677 to 0.678 and median absolute log error from 0.401 to 0.407. The sole upper limit, row 43, receives a slightly higher event probability. Geographic score changes from 1.364 to 1.372, and the independent survey transfer worsens. The larger full fit passes the site, field-population and mapped-quartile checks; numerical stability does not remove this geological uncertainty.

## B-Tm

[Three-scale comparison](review/b-tm/scales.png) · [Site evidence](review/b-tm/sites.csv.gz) · [Contour descriptors](review/b-tm/morphology.csv)

The proximal contours now turn and tighten towards the northwest instead of carrying thick ash far upwind as nearly parallel rounded fronts. Their eccentricity and unequal spacing evolve continuously into the eastward fan; the full view shows the compact proximal structure as part of that fan rather than as a separate circular cap. The distal axis bends slightly southeast, and the thinner southern and eastern contours extend farther. This is a redistribution of supported and conditional thickness, not uniform shrinkage of the deposit. The contours remain connected and ordered, without artificial lobe seams.

The Quanyang interpreted upper bound improves from a log event probability of −5.545 to −0.607, and Longgang from −0.989 to −0.019. Neither is exceeded by the final median. The other two upper limits remain satisfied. Across all bounds the negative log probability improves from 1.641 to 0.166. Median absolute measured log error improves from 0.522 to 0.420, while measured RMSE worsens from 0.759 to 0.788, including greater overprediction at Oga-san. Geographic score is essentially unchanged (1.357 to 1.362). The joint posterior fallback resolves the fourth fold's numerical disagreement; its failed particle refinements remain in the record. The independent three-site survey score improves, but that separate revised training fit retains a numerical warning at the common auxiliary ceiling.

## Minoan

[Three-scale comparison](review/minoan/scales.png) · [Site evidence](review/minoan/sites.csv.gz) · [Contour descriptors](review/minoan/morphology.csv)

The 16 cm contour now tilts southeast near the vent before the lower contours turn toward an approximately eastward distal axis. The joint geometry remains connected, with orderly widening and no artificial lobe junction. The proximal contour is visibly part of that turning field in the full map. North–south spreading contracts, but the eastward 1 cm continuation becomes longer and the westward thin contours also extend. These unsampled margins remain conditional; this case does not demonstrate a universal reduction in upwind reach.

The 2 cm northern locality at row 63 and the 0.2 cm southeastern locality at row 94 improve. Conversely, the 1 cm western observation at row 5 (104 km) is overpredicted more strongly, while the nearby 10 cm western observation at row 4 (82 km) improves. The field still misses that sharp local thickness contrast. Measured log RMSE changes from 0.989 to 0.984, median absolute log error worsens from 0.519 to 0.531, and geographic score improves from 1.711 to 1.675. The observations contain no upper limits or intervals.

## Ilopango TBJ

[Three-scale comparison](review/ilopango-tbj/scales.png) · [Site evidence](review/ilopango-tbj/sites.csv.gz) · [Contour descriptors](review/ilopango-tbj/morphology.csv)

The nearly concentric proximal circles become eccentric, approximately north–south contours. Their orientation changes continuously toward the northwest–southeast outer envelope. The 256 cm summit and successive 128, 64 and 32 cm contours stay connected and smoothly spaced; the full view no longer attaches a round core abruptly to an oblique outer shape. The northwestern 1 cm continuation becomes substantially longer, however. It is a remaining extrapolation weakness, not a newly established deposit boundary.

Eastern records SM1-P38 and SM1-P30 improve, and the two-sided SM1-P63 interval receives a less extreme prediction. The thin northern SM1-P118 measurement and northeastern SM1-P116 interval worsen. The latter remains a strong conflict between the fitted field and bounded evidence. Measured log RMSE improves from 1.021 to 1.011, while median absolute log error worsens from 0.537 to 0.596 and the bounded negative log probability worsens from 1.522 to 1.572. Median interval violations fall from 11 to 10. Geographic score changes from 1.761 to 1.788. Centered-core, circular-core, altered-focus and prior-reweighting trials did not establish a general solution and remain in the archive; no case-specific cutoff was imposed.

## Sakurajima Satsuma

[Three-scale comparison](review/sakurajima-satsuma/scales.png) · [Site evidence](review/sakurajima-satsuma/sites.csv.gz) · [Contour descriptors](review/sakurajima-satsuma/morphology.csv)

The small 64 cm proximal contour becomes eccentric and aligns gradually with the larger northwestern–southeastern envelope. The 32, 16 and 8 cm contours have smooth, unequal spacing rather than one rigidly expanding oval. In the full picture the southern 1 cm reach contracts from roughly 500 to 350 km, while the northwestern reach expands. The near contours remain geometrically coherent within that full field; the method does not reproduce every east–west local thickness contrast.

Northern GSJ-1F-02 and southeastern S2-T41 improve; northern GSJ-1B-03 and southwestern GSJ-1B-33 worsen. Both upper limits, GSJ-1B-31 and GSJ-1D-26, receive higher event probabilities. Measured log RMSE improves from 0.497 to 0.483, bounded negative log probability from 0.258 to 0.220, and geographic score from 1.092 to 0.963. Median absolute log error increases from 0.330 to 0.374. The final map check uses 4,096 draws.

## YTT

[Three-scale comparison](review/ytt/scales.png) · [Site evidence](review/ytt/sites.csv.gz) · [Contour descriptors](review/ytt/morphology.csv)

The northern conditional extension shortens substantially, while the southern thin continuation grows. The 16 cm contour is less circular and turns northwest into the larger western shoulder. The 8–1 cm contours preserve that shoulder and progressively change their eccentricity and spacing; they do not create separate rings or an abrupt join between the northern and western lobes. The inner 16 cm contour remains coherent within the full-deposit picture. The ±100 km proximal panel contains no eligible contour or observation in either model: the data do not resolve a caldera-scale fallout profile. The published ±800 km near view shows the five surrounding observations and the conditional 16 cm contour, without inventing a proximal summit contour.

Northwestern rows 36, 37 and 41 at roughly 4,070–4,090 km receive less excessive predictions, including the 0.065 cm measurement at row 41. Southern row 2 (5 cm at 1,038 km) is overpredicted more strongly, and northeastern row 8 (2 cm at 1,759 km) also worsens. Measured log RMSE improves from 0.536 to 0.526 and median absolute log error from 0.291 to 0.278. There are no upper limits or intervals in the accepted table. Geographic score worsens from 1.707 to 1.759. The full fit and one geographic fold required the shared joint posterior fallback; all four geographic fits now pass. The remaining large northern and western conditional margins are not established deposit edges.

## Pinatubo Layer C

[Three-scale comparison](review/pinatubo-1991-layer-c/scales.png) · [Site evidence](review/pinatubo-1991-layer-c/sites.csv.gz) · [Contour descriptors](review/pinatubo-1991-layer-c/morphology.csv)

The western and southern parts of the preceding field met at an abrupt shoulder, visible in both the 8–4 cm intermediate contours and the thin full-deposit outline. Positive area growth replaces that shoulder with a continuously turning field. The eastern 16 cm reach tightens and the long southern 1 cm extension contracts, while western marine reach grows. Boundary eccentricity increases from 0.38 to 0.72 at 16 cm and from 0.48 to 0.86 at 1 cm. The inner axis turns from about 50° to 77° across those levels. Successive contours remain connected and nested, with unequal but coherent spacing; the near-vent geometry belongs visibly to the westward full field rather than forming a separate round cap. The comparatively elliptical distal contours should not be read as a surveyed perimeter.

Eastern P7-062 (0.7 cm at 45 km) and the northeastern K-01 upper limit improve substantially, as do western WW-17955-1 and W04-SO-132-28. Northwestern W09-47 and southwestern W09-33 worsen. The W04-SO-140-25 and W04-SO-132-29 intervals receive higher event probabilities, while WW-17921-1, H01-25 and W04-SO-132-07 worsen. This remains a limitation in adapting to the marine thickness structure: all 18 interval medians lie outside their reported ranges, compared with 13 previously, even though the aggregate bounded-event score improves.

Measured log RMSE improves from 0.503 to 0.433, median absolute log error from 0.318 to 0.219, and bounded negative log probability from 1.317 to 1.103. Median upper-limit violations fall from 11 to 5. Geographic score improves from 1.419 to 1.350, with 83.1% coverage by nominal 95% observation intervals; the uncertainty remains undercalibrated. All four geographic fits pass. The full fit required 2,048 fixed-target moves and two successive passing checkpoints; the final mapped quartile check passes at 2,048 draws. Neither numerical agreement nor the improved average score resolves every conflicting marine record.
