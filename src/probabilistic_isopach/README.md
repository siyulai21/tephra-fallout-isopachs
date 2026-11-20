# Earth geometry and compatibility imports

Both global models and regional maps use the WGS84 equal-area geometry in this directory. The global fit caches record the hash of the entire geometry module. It also contains the earlier GP integration functions. [Earth-domain implementation](global_volume.py) · [Active model families](../empirical_isopach/models/README.md)

The module command forwards explicit `compare`, `global` and `methods` requests to the active workflow. The compatibility imports re-export the GP, shared data utilities or plotting style used by earlier callers. The current implementations live in the [regional GP family](../empirical_isopach/models/gaussian_process/README.md), [shared core](../empirical_isopach/core/README.md) and [figure tools](../empirical_isopach/visualization/README.md).

Use `tephra-isopachs compare` to calculate an eruption. The [release history](../../CHANGELOG.md#earlier-results) links the former GP-only command and its output conventions.
