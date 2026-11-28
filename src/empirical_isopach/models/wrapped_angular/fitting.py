"""Separate the mapped source-local process from broad regional thinning."""

import json
from hashlib import sha256
from importlib.resources import files
import numpy as np
from scipy.special import logsumexp, stdtr
from .field import RegionalField, Deposit
from .regional import Posterior
from .source import coordinates, proposals, log_shape, amplitude_proposal, conditional_weights, predictive


def evidence_fingerprint(data):
    """Identity of the numeric likelihood and coordinates, independent of reported-minimum provenance."""
    data = data.canonical()
    h = sha256()
    for values in (
        data.xy,
        data.log_value,
        data.log_sd,
        data.censored,
        data.lower_censored,
        data.interval_log_upper,
    ):
        h.update(np.ascontiguousarray(values, dtype="<f8").tobytes())
    return h.hexdigest()


def source_for(meta, data=None):
    """Use independently mapped moments only when the fitting edition and registered vent match."""
    ledger = files("src.empirical_isopach").joinpath("assets/wrapped-source-areas.json")
    for row in json.loads(ledger.read_text()):
        if np.array_equal(row["vent_lonlat"], meta["vent_lonlat"]):
            if row["input_sha256"] == meta["input_sha256"] or (
                data is not None and evidence_fingerprint(data) == row["numeric_evidence_sha256"]
            ):
                return {key: value for key, value in row.items() if key != "numeric_evidence_sha256"}
    return None


def source_mask(xy, source):
    if source is None or source["area_km2"] is None:
        return np.zeros(len(xy), bool)
    return np.linalg.norm(coordinates(xy, source), axis=1) < 2.0


def fit(
    data, latitude, source=None, target=None, *, covariance="wrapped", bound_power=12, refined_source=True
):
    """Each source-section update conditions on a whole realization and preserves its regional weight."""
    if np.any(data.lower_censored & ~data.interval):
        raise ValueError("Lower-only records must first use the public reported-minimum fitting policy")
    target = data if target is None else target
    inner = source_mask(data.xy, source)
    if inner.all():
        raise ValueError("The source partition leaves no exterior stations to identify the regional field")
    outer = data.subset(~inner)
    posterior = Posterior.fit(outer, latitude, power=bound_power, covariance=covariance)
    regional = RegionalField.from_fit(posterior)
    bank = regional.coherent()
    bed = posterior.distribution(target.xy, True, target.log_sd)
    local = source_mask(target.xy, source)
    scores = dict(
        row_id=target.row_ids,
        kind=np.where(target.interval, "interval", np.where(target.censored, "upper", "point")),
        source_neighborhood=local,
        log_score=bed.score(target),
        cdf=bed.cdf(target.log_value),
        primary_median_cm=np.exp(regional.distribution(target.xy).quantile(0.5)),
        bed_median_cm=np.exp(bed.quantile(0.5)),
    )
    diagnostics = dict(
        **posterior.detail,
        covariance=covariance,
        source_rows=data.row_ids[inner].tolist(),
        exterior_rows=outer.row_ids.tolist(),
    )
    deposit = Deposit(regional, source, bank)
    if source is None or source["area_km2"] is None:
        return deposit, posterior, scores, diagnostics
    local_data, held = data.subset(inner), target.subset(local)
    locations, held_locations = coordinates(local_data.xy, source), coordinates(held.xy, source)
    n_local = len(local_data.xy)
    values = bank(np.vstack([source["centroid_km"], local_data.xy, held.xy]))
    centres = values[:, :, 0]
    n = values.shape[1]
    power, count = (14, 25) if refined_source else (11, 12)
    chosen = np.empty((2, n, count, 5))
    log_score = np.empty((2, n, len(held.xy)))
    log_cdf = np.empty_like(log_score)
    ess = np.empty((2, n))
    for k in range(2):
        prior, branch = proposals(power, 31657 + 7919 * (k + 1), with_branch=True)
        unit = prior.copy()
        unit[:, 0] = 0.0
        shape = log_shape(unit, locations, 2.0)
        test_shape = log_shape(unit, held_locations, 2.0)
        rng = np.random.default_rng(np.random.SeedSequence([31657, k, 149]))
        for j, row in enumerate(values[k]):
            z = row[1 : n_local + 1]
            params, ratio = (
                amplitude_proposal(prior, branch, z, row[0], shape, local_data)
                if refined_source
                else (prior, 0.0)
            )
            weights = conditional_weights(z, row[0], shape + params[:, None, 0], params, local_data, ratio)
            ess[k, j] = np.exp(-logsumexp(2 * weights))
            idx = np.searchsorted(np.exp(weights).cumsum(), (np.arange(count) + rng.random(count)) / count)
            chosen[k, j] = params[idx.clip(max=len(params) - 1)]
            if local.any():
                log_score[k, j], log_cdf[k, j] = predictive(
                    weights, row[1 + n_local :], row[0], test_shape + params[:, None, 0], params, held
                )
    deposit = Deposit(regional, source, bank, chosen, centres, prior_only=not n_local)
    if local.any():
        scores["log_score"][local] = logsumexp(log_score, axis=(0, 1)) - np.log(2 * n)
        scores["cdf"][local] = np.exp(logsumexp(log_cdf, axis=(0, 1)) - np.log(2 * n))
        scores["primary_median_cm"][local] = np.exp(deposit.quantiles(held.xy, (0.5,))[0])
        law = SourceBeds(deposit, held.xy, held.log_sd)
        scores["bed_median_cm"][local] = np.exp(law.quantile(0.5))
    diagnostics.update(
        source_proposals_per_region=2**power,
        conditional_source_draws=count,
        minimum_source_ess=float(ess.min()),
        median_source_ess=float(np.median(ess)),
    )
    return deposit, posterior, scores, diagnostics


class SourceBeds:
    """Predict preserved source-section thickness with its own Student-t scatter."""

    def __init__(self, deposit, xy, log_sd=0.0):
        params = deposit.params.reshape(-1, 5)
        count = deposit.params.shape[2]
        shape = log_shape(params, coordinates(xy, deposit.source), 2.0)
        peak = np.repeat(deposit.centres.ravel(), count)[:, None] + shape
        region = np.repeat(deposit.bank(xy).reshape(-1, len(xy)), count, axis=0)
        self.location = np.logaddexp(region, peak)
        self.scale = np.hypot(params[:, 4, None], log_sd)

    def cdf(self, y):
        return stdtr(4, (y - self.location) / self.scale).mean(axis=0)

    def quantile(self, p):
        from scipy.special import stdtrit

        ends = self.location + self.scale * stdtrit(4, p)
        lo, hi = ends.min(axis=0), ends.max(axis=0)
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            low = self.cdf(mid) < p
            lo, hi = np.where(low, mid, lo), np.where(low, hi, mid)
        return 0.5 * (lo + hi)
