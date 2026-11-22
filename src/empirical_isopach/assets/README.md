# Packaged model and map inputs

The [directional calibration packet](directional-calibration.json) contains the six structures’ held-out log scores, accepted input identities and fold exclusions. It lets the assistant analyse a new eruption with weights learned from other cases. [Model protocol](../../../docs/methods/directional-ensemble.md).

## Offline coastline

`ne_10m_coastline.geojson.gz` is the losslessly compressed Natural Earth **1:10-million** coastline. This is a cartographic scale, not a ten-metre accuracy claim. [Pinned URL, version, licence and both hashes](coastline-source.json).

The original coordinates are unchanged. The renderer projects and clips them for context; coastlines never enter the likelihood, observation hull, support decision or volume calculation. The package includes the asset so plots need no network access. [Third-party notices](../../../THIRD_PARTY_NOTICES.md).

The primary map bases use Natural Earth land and inland lakes at the same scale. [Sources, hashes and simplification](land-source.json)
