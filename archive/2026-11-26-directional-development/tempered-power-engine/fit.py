"""Bayesian averaging of smooth directional fields with exact and censored site evidence."""

from itertools import product

import numpy as np
from scipy.special import logsumexp, ndtr

from src.probabilistic_isopach.global_volume import EarthDomain
from .conditioning import condition, corrected
from .geometry import LENGTHS, NOISE, RANGES, TAILS, SPREAD, LOCAL_SD, LocalBasis, design, log_mean
from .posterior import Component, DirectionalField
from .compression import compression

SEED = 173071
KEEP_MASS = 0.9995


def fit(data, latitude, *, seed=SEED, power=8, shared_fraction=0.5):
    data = data.canonical()
    axis = EarthDomain.wgs84(latitude).axis_scale
    reach = np.empty(0)
    local = {v: LocalBasis.fit(data.xy, axis, v, shared_fraction) for v in LENGTHS}
    radii = np.r_[0.0, np.geomspace(4.0, 12000.0, 10)]
    angles = 2 * np.pi * np.arange(12) / 12
    probes = np.column_stack(
        ((axis * np.outer(radii, np.sin(angles))).ravel(), (np.outer(radii, np.cos(angles)) / axis).ravel())
    )
    checks = np.vstack((data.xy, probes))
    designs, training, screening, states = {}, {}, [], []
    for scale, complexity, length in product(RANGES, (0, 1), LENGTHS):
        B, d = design(data.xy, axis, scale, complexity, local[length])
        n = local[length].averaging.shape[0]
        resolved = np.sum(B[:, -n:] ** 2, axis=1) if n else 0.0
        rim2 = np.maximum(
            0.0, 1.0 - (EarthDomain.wgs84(latitude).rho(data.xy) / EarthDomain.wgs84(latitude).rho_max) ** 2
        )
        remainder = np.maximum(0.0, (LOCAL_SD * rim2) ** 2 - resolved)
        key = (scale, complexity, length)
        training[key] = (B, d, remainder)
        designs[key] = np.vstack((B, design(probes, axis, scale, complexity, local[length])[0]))
        for tail, noise, spread in product(TAILS, NOISE, SPREAD):
            state = condition(B, log_mean(d, tail, spread), data, np.sqrt(noise**2 + remainder))
            i = len(states)
            corrections = [corrected(state, 6, seed + 16 * i + j + 8)[2] for j in range(2)]
            screening.append(state["logz"] + logsumexp(corrections) - np.log(2.0))
            # Screening needs evidence only. Keeping 576 dense posterior
            # covariances makes an otherwise rank-adaptive fit need gigabytes.
            summary = {k: state[k] for k in ("logz", "ep_residual", "iterations", "control_variate")}
            states.append((key, tail, noise, spread, summary))
    log_weights = np.asarray(screening)
    p = np.exp(log_weights - logsumexp(log_weights))
    order = np.argsort(-p)
    # Retain a wider screen for the high-resolution evidence correction.
    keep = order[: 1 + np.searchsorted(np.cumsum(p[order]), 0.99999)]
    corrected_log = log_weights.copy()
    independent_log = np.tile(log_weights, (2, 1))
    detail, components = {}, []
    for i in keep:
        key, tail, noise, spread, state = states[i]
        B, d, remainder = training[key]
        state = condition(B, log_mean(d, tail, spread), data, np.sqrt(noise**2 + remainder))
        states[i] = (key, tail, noise, spread, state)
        delta = 0.0
        for exponent in range(power, min(12, power + 3) + 1):
            sims = [corrected(state, exponent, seed + 16 * int(i) + j + 1) for j in range(2)]
            corr = np.array([s[2] for s in sims])
            delta = float(np.ptp(corr))
            if delta <= 0.10 and min(s[3] for s in sims) >= 0.14:
                break
        corrected_log[i] = state["logz"] + logsumexp(corr) - np.log(2.0)
        independent_log[:, i] = state["logz"] + corr
        detail[i] = (sims, exponent, delta)
    p = np.exp(corrected_log - logsumexp(corrected_log))
    order = np.array(sorted(keep, key=lambda i: -p[i]))
    keep2 = order[: 1 + np.searchsorted(np.cumsum(p[order]), KEEP_MASS)]
    retained = p[keep2].sum()
    independent_mass = np.exp(independent_log - logsumexp(independent_log[:, keep2], axis=1)[:, None])
    records = []
    reference = []
    for i, (key, tail, noise, spread, state) in enumerate(states):
        row = dict(
            scale_km=key[0],
            evolving=bool(key[1]),
            local_length=key[2],
            tail=tail,
            noise=noise,
            spread=spread,
            log_evidence=float(corrected_log[i]),
            probability=float(p[i]),
            retained=bool(i in keep2),
            ep_residual=state["ep_residual"],
            ep_iterations=state["iterations"],
        )
        records.append(row)
        if i not in keep2:
            continue
        sims, exponent, delta = detail[i]
        row.update(
            scramble_log_evidence_difference=delta,
            relative_ess=min(s[3] for s in sims),
            points_per_scramble=2**exponent if len(state["X"]) else 0,
        )
        if len(state["X"]):
            parts, err, minimum = compression(state, sims, designs[key], 4 if power <= 8 else 12)
            row["minimum_corrected_covariance_ratio"] = minimum
        else:
            parts, err = [(1.0, state["mean"], state["cov"])], 0.0
        row["max_compression_cdf_error"] = err
        check_matrix = designs[key]
        offset = log_mean(design(checks, axis, key[0], key[1], local[key[2]])[1], tail, spread)
        reference.append((independent_mass[:, i], check_matrix, offset, state, sims))
        for w, m, c in parts:
            components.append(
                Component(
                    p[i] * w / retained,
                    key[0],
                    key[1],
                    key[2],
                    tail,
                    noise,
                    spread,
                    m,
                    np.linalg.cholesky(c),
                )
            )
    important = [r for r in records if r["retained"]]
    weighted = lambda name: float(sum(r["probability"] * r[name] for r in important) / retained)
    diagnostics = dict(
        shared_fraction=shared_fraction,
        retained_mass=float(retained),
        states=records,
        conditional_components=len(components),
        weighted_cdf_error=weighted("max_compression_cdf_error"),
        weighted_scramble_difference=weighted("scramble_log_evidence_difference"),
        weighted_relative_ess=weighted("relative_ess"),
        proposal_residual=max(r["ep_residual"] for r in important),
    )
    field, validation = reduce_field(latitude, local, components, reach, checks)
    dist = field.distribution(checks)
    thresholds = np.array([dist.quantile(q) for q in (0.025, 0.25, 0.5, 0.75, 0.975)])
    corrected_cdf = np.zeros((2, *thresholds.shape))
    for mass, B, offset, state, sims in reference:
        sd = np.sqrt(np.maximum(1e-12, ((B @ state["complete_cov"]) * B).sum(1)))
        for rep, (centers, rw, _, _) in enumerate(sims):
            means = offset[:, None] + B @ centers.T
            for j, level in enumerate(thresholds):
                if state.get("control_variate"):
                    s0 = np.sqrt(np.maximum(1e-12, ((B @ state["cov"]) * B).sum(1)))
                    base = ndtr((level - offset - B @ state["mean"]) / s0)
                    cdf = base + ndtr((level[:, None] - means) / sd[:, None]) @ (rw - 1 / len(rw))
                else:
                    cdf = ndtr((level[:, None] - means) / sd[:, None]) @ rw
                corrected_cdf[rep, j] += mass[rep] * cdf
    diagnostics.update(validation)
    err = corrected_cdf.mean(axis=0) - np.array([0.025, 0.25, 0.5, 0.75, 0.975])[:, None]
    diagnostics["max_pooled_cdf_error"] = float(abs(err).max())
    diagnostics["rms_pooled_cdf_error"] = float(np.sqrt(np.mean(err**2)))
    diagnostics["max_independent_cdf_difference"] = float(np.ptp(corrected_cdf, axis=0).max())
    diagnostics["passed"] = bool(
        diagnostics["max_pooled_cdf_error"] < 0.035
        and diagnostics["max_independent_cdf_difference"] < 0.06
        and diagnostics["weighted_scramble_difference"] < 0.10
        and diagnostics["weighted_relative_ess"] > 0.14
        and diagnostics["proposal_residual"] < 1e-4
        and retained >= KEEP_MASS - 1e-10
        and all(r.get("minimum_corrected_covariance_ratio", 1.0) > 0 for r in important)
    )
    field.subspaces = {
        key: np.linalg.qr(training[key][0].T, mode="reduced")[0]
        for key in {(c.scale, c.evolving, c.length) for c in field.components}
    }
    return field, diagnostics


def reduce_field(latitude, local, parts, reach, probes):
    """Stratify component mass and compare the actual marginal CDFs against the complete mixture."""
    from dataclasses import replace

    exact = DirectionalField(latitude, local, parts, reach)
    checks = exact.distribution(probes)
    obs = exact.distribution(probes, observation=True)
    quantiles = (0.025, 0.25, 0.5, 0.75, 0.975)
    limits = [[d.quantile(q) for q in quantiles] for d in (checks, obs)]
    original = len(parts)
    error = 0.0
    result = exact
    for count in (48, 96, 192, 384, 768):
        if count >= original:
            result = exact
            error = 0.0
            break
        positions = (np.arange(count) + 0.5) / count
        ids, ns = np.unique(np.searchsorted(exact.weights.cumsum(), positions), return_counts=True)
        weights = ns / count
        errors = []
        for d, thresholds in zip((checks, obs), limits, strict=True):
            for q, at in zip(quantiles, thresholds, strict=True):
                cdf = weights @ ndtr((at - d.means[ids]) / np.sqrt(d.variances[ids]))
                errors.append(abs(cdf - q).max())
        error = float(max(errors))
        if error <= 0.009:
            result = DirectionalField(
                latitude,
                local,
                [replace(parts[i], weight=w) for i, w in zip(ids, weights, strict=True)],
                reach,
            )
            break
        result = exact
    if result is exact:
        error = 0.0
    return result, dict(
        original_components=original,
        published_components=len(result.components),
        component_reduction_max_cdf_error=error,
        probe_count=len(probes),
    )
