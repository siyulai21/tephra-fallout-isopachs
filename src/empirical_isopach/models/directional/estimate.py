"""Summarize coherent posterior fields and retain their directional mass budgets."""
from pathlib import Path
import json

import numpy as np

from .geometry import COMPASS, Geometry, polar
from .integration import PARTS, integrate_checked
from .study import digest, write_json


def interval(a):
    return dict(zip(("lower95", "median", "upper95"), np.quantile(a, [.025, .5, .975]).tolist()))


def estimate(data, latitude, policy, folder):
    import arviz as az
    folder = Path(folder)
    attempts = [json.loads(p.read_text()) for p in sorted(folder.glob("attempt-*.json"))]
    last = attempts[-1]
    path = folder/last["file"]
    if digest(path) != last["posterior_sha256"]:
        raise ValueError("Posterior checksum mismatch")
    geometry = Geometry(data, latitude, policy)
    design = json.loads((folder/"design.json").read_text())
    if design["geometry"] != geometry.describe():
        raise ValueError("Integration geometry differs from fitting geometry")
    code_files = (Path(__file__), Path(__file__).with_name("integration.py"))
    identity = {"integration_source_sha256": {p.name: digest(p) for p in code_files},
                "posterior_sha256": last["posterior_sha256"], "design_sha256": digest(folder/"design.json")}
    outfile = folder/"integral.json"
    massfile = folder/"mass.npz"
    if outfile.exists():
        result = json.loads(outfile.read_text())
        if identity == result["signature"] and digest(massfile) == result["mass_sha256"]:
            return result
    with np.load(path) as saved:
        draws = dict(saved)
    mass, idx, check = integrate_checked(draws, geometry, n_per_chain=draws["source"].shape[1])
    total = mass.sum(axis=(-1, -2))
    chains = az.from_dict(posterior={"log_volume": np.log(total)})
    rh = float(az.rhat(chains).log_volume)
    bulk = float(az.ess(chains, method="bulk").log_volume)
    tail = float(az.ess(chains, method="tail").log_volume)
    volume_check = {"rhat": rh, "bulk_ess": bulk, "tail_ess": tail,
                    "log_quantile_mcse": [float(az.mcse(chains, method="quantile", prob=p).log_volume)
                                          for p in (.025, .5, .975)],
                    "passed": bool(rh < 1.01 and min(bulk, tail) >= 400)}
    r, theta = polar(data.xy, geometry.axis)
    compass = np.floor((theta+np.pi/8)/(np.pi/4)).astype(int) % 8
    exact_radius = polar(geometry.mesh.xy, geometry.axis)[0]
    budget = {}
    for j, label in enumerate(COMPASS):
        in_sector = (compass == j) & (r > 0)
        exact = in_sector & ~data.censored
        budget[label] = {"exact": int(exact.sum()), "bounds": int((in_sector & data.censored).sum()),
            "exact_row_ids": data.row_ids[exact].tolist(),
            "distal_half_exact": int((exact & (r >= np.median(exact_radius))).sum()),
            "farthest_exact_chord_km": float(r[exact].max()) if exact.any() else None,
            "footprint_km2": check["area_by_compass_km2"][0][j],
            "volume": interval(mass[..., j].sum(-1)),
            "fraction": interval(mass[..., j].sum(-1)/total),
            "parts": {k: interval(mass[..., i, j]) for i, k in enumerate(PARTS)}}
    with massfile.with_suffix(".tmp").open("wb") as stream:
        np.savez_compressed(stream, mass=mass, draw_index=idx)
    massfile.with_suffix(".tmp").replace(massfile)
    result = {"signature": identity, "mass_sha256": digest(massfile), "attempts": attempts,
              "posterior_diagnostics_passed": last["diagnostics"]["passed"],
              "volume_diagnostics": volume_check, "quadrature": check,
              "passed": bool(volume_check["passed"] and check["passed"] and last["diagnostics"]["passed"]),
              "volume_km3": interval(total), "part_names": PARTS, "compass": budget,
              "vertex_at_source_rows": data.row_ids[r == 0].tolist(),
              "parts": {k: {"volume_km3": interval(mass[..., i, :].sum(-1)),
                            "fraction": interval(mass[..., i, :].sum(-1)/total)} for i, k in enumerate(PARTS)}}
    write_json(outfile, result)
    return result

