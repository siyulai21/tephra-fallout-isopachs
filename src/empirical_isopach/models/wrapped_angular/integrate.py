"""Positive quadrature of three pointwise thickness maps and joint fallout realizations."""

import numpy as np
from src.probabilistic_isopach.global_volume import EarthDomain, earth_cells
from .source import excess_volume

PROBABILITIES = (0.25, 0.5, 0.75)
RULES = ((12, 72, 0.5), (20, 144, 0.25))


def integrate(deposit, xy, latitude, *, rules=RULES, prior_power=12):
    domain = EarthDomain.wgs84(latitude)
    last = float(domain.rho(xy).max())
    edges = np.unique([0.0, 3.0, 30.0, 300.0, last, domain.rho_max])
    if last > domain.rho_max:
        raise ValueError("Station lies outside the equal-area Earth")
    records = []
    for order, angles, phase in rules:
        layers, draws, surface = [], [], 0.0
        for points, area in earth_cells(domain, edges, order, angles, 8.0, phase):
            maps = np.zeros(3)
            realization = np.zeros(deposit.bank(points[:1]).shape[:2])
            for start in range(0, len(points), 96):
                s = slice(start, start + 96)
                log_thickness = deposit.quantiles(
                    points[s],
                    PROBABILITIES,
                    prior_power=prior_power,
                    prior_seed=67957 if len(records) == 0 else 81119,
                )
                maps += np.exp(log_thickness) @ area[s] * 1.0e-5
                realization += np.exp(deposit.bank(points[s])) @ area[s] * 1.0e-5
            layers.append(maps)
            draws.append(realization)
            surface += area.sum()
        if not np.isclose(surface, np.pi * domain.rho_max**2, rtol=5.0e-6):
            raise ArithmeticError("Quadrature does not cover the physical Earth")
        regional = np.sum(draws, axis=0)
        total = regional[:, :, None]
        if deposit.params is not None:
            params = deposit.params
            extra = excess_volume(
                params.reshape(-1, 5),
                np.broadcast_to(deposit.centres[:, :, None], params.shape[:-1]).ravel(),
                deposit.source,
                2.0,
                order * 2,
            )
            total = total + extra.reshape(params.shape[:-1])
        layers = np.array(layers)
        records.append(
            dict(
                radial_order=order,
                bearings=angles,
                phase=phase,
                earth_area_km2=float(surface),
                volumes_km3=layers.sum(axis=0).tolist(),
                annulus_quartile_volumes_km3=layers.tolist(),
                draw_interval95_km3=np.quantile(total, [0.025, 0.975]).tolist(),
                draw_population_medians_km3=np.median(total, axis=(1, 2)).tolist(),
                coherent_draws_km3=total.tolist(),
            )
        )
    v = np.asarray([r["volumes_km3"] for r in records])
    error = np.abs(v[-1] - v[-2]) / np.maximum(v[-1], 1.0e-300)
    return dict(
        probabilities=list(PROBABILITIES),
        volumes_km3=v[-1].tolist(),
        passed=bool(np.all(error < 0.01)),
        relative_refinement=error.tolist(),
        annulus_edges_km=edges.tolist(),
        farthest_station_radius_km=last,
        rules=records,
    )


def joint_volumes(deposit, data, latitude, counts=(256, 1024, 4096, 16384)):
    """Independently refine the interval of whole-field integrals, including conditional local peaks."""
    from .source import coordinates, proposals, log_shape, amplitude_proposal, conditional_weights

    domain = EarthDomain.wgs84(latitude)
    edges = np.unique([0.0, 3.0, 30.0, 300.0, float(domain.rho(data.xy).max()), domain.rho_max])
    near = deposit.local(data.xy)
    local = data.subset(near)
    probabilities = [0.025, 0.5, 0.975]
    history = []
    for count in counts:
        bank = deposit.regional.coherent(count, 72821)
        extra = np.zeros((2, count))
        extra_error = 0.0
        source = deposit.source
        if source is not None and source["area_km2"] is not None:
            loc = coordinates(local.xy, source)
            reg = bank(np.vstack((source["centroid_km"], local.xy)))
            for pop, fields in enumerate(reg):
                prior, branch = proposals(12, 39191 + pop, with_branch=True)
                unit = prior.copy()
                unit[:, 0] = 0.0
                shape = log_shape(unit, loc, 2.0)
                rng = np.random.default_rng(np.random.SeedSequence([37549, pop, count]))
                selected = prior[rng.integers(len(prior), size=count)].copy()
                if near.any():
                    for j, row in enumerate(fields):
                        params, ratio = amplitude_proposal(prior, branch, row[1:], row[0], shape, local)
                        weights = conditional_weights(
                            row[1:], row[0], shape + params[:, None, 0], params, local, ratio
                        )
                        index = np.searchsorted(np.exp(weights).cumsum(), rng.random()).clip(
                            max=len(params) - 1
                        )
                        selected[j] = params[index]
                extra[pop] = excess_volume(selected, fields[:, 0], source, 2.0, 40)
                low = excess_volume(selected, fields[:, 0], source, 2.0, 24)
                extra_error = max(extra_error, float(np.max(abs(low - extra[pop]))))
        values = []
        for order, angles, phase in RULES:
            volume = extra.copy()
            for xy, area in earth_cells(domain, edges, order, angles, 8.0, phase):
                for i in range(0, len(xy), 256):
                    sl = slice(i, i + 256)
                    volume += np.exp(bank(xy[sl])) @ area[sl] * 1.0e-5
            values.append(volume)
        quantiles = np.quantile(values[-1], probabilities)
        by_population = np.quantile(values[-1], probabilities, axis=1)
        split = np.ptp(by_population, axis=1) / quantiles
        quadrature = float(np.max(np.abs(values[0] / values[-1] - 1.0)))
        change = np.abs(quantiles / np.array(history[-1]["quantiles_km3"]) - 1.0) if history else None
        passed = bool(
            change is not None
            and np.all(change < [0.12, 0.06, 0.12])
            and np.all(split < [0.15, 0.08, 0.15])
            and quadrature < 0.01
        )
        history.append(
            dict(
                regional_draws_per_population=count,
                quantiles_km3=quantiles.tolist(),
                population_quantiles_km3=by_population.T.tolist(),
                split_relative_difference=split.tolist(),
                sample_refinement=change.tolist() if change is not None else None,
                maximum_field_quadrature_relative_difference=quadrature,
                maximum_source_refinement_km3=extra_error,
            )
        )
        if passed:
            break
    return dict(
        probabilities=probabilities,
        quantiles_km3=quantiles.tolist(),
        passed=passed,
        refinement=history,
        final_draws_km3=values[-1].tolist(),
    )
