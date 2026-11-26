"""Gaussian control-variate compression of corrected bounded-event posteriors."""

import numpy as np
from scipy.special import logsumexp, ndtr, ndtri
from scipy.stats import norm


def compression(state, sims, checks, max_groups=4):
    """Correct proposal moments, using its analytic moments as control variates.

    When the correction weights are constant the result is exactly the Gaussian
    proposal, independent of Monte Carlo sample covariance error. A separate CDF
    correction checks the Gaussian approximation at off-survey probes as well.
    """
    mixing = np.exp(np.array([s[2] for s in sims]) - logsumexp([s[2] for s in sims]))
    beta = np.concatenate([s[0] for s in sims])
    delta = np.concatenate([w * (s[1] - 1 / len(s[1])) for w, s in zip(mixing, sims, strict=True)])
    factor = state["control_factor"]
    if not factor.shape[1]:
        return [(1.0, state["mean"], state["cov"])], 0.0, 1.0
    z = (beta - state["mean"]) @ state["control_recovery"]
    completion = state["complete_cov"]
    shift = delta @ z
    relative = np.eye(len(shift)) + (z.T * delta) @ z - np.outer(shift, shift)
    ev, vectors = np.linalg.eigh((relative + relative.T) / 2)
    minimum = float(ev.min())
    # A nonpositive estimate is retained as a failed numerical attempt. The
    # caller must refine it; this floor only makes that attempt serializable.
    covariance = completion + factor @ ((vectors * np.maximum(ev, 1e-8)) @ vectors.T) @ factor.T
    mean = state["mean"] + factor @ shift
    projected = checks @ mean
    sd = np.sqrt(np.maximum(1e-12, ((checks @ covariance) * checks).sum(1)))
    reference_sd = np.sqrt(np.maximum(1e-12, ((checks @ state["cov"]) * checks).sum(1)))
    reference_mean = checks @ state["mean"]
    mapped = checks @ beta.T
    quantiles = np.array([0.025, 0.25, 0.5, 0.75, 0.975])
    values = projected + ndtri(quantiles)[:, None] * sd
    conditional_sd = np.sqrt(np.maximum(1e-20, ((checks @ completion) * checks).sum(1)))
    reference = np.array(
        [
            ndtr((at - reference_mean) / reference_sd)
            + ndtr((at[:, None] - mapped) / conditional_sd[:, None]) @ delta
            for at in values
        ]
    )
    weights = np.concatenate([w * s[1] for w, s in zip(mixing, sims, strict=True)])
    transformed = checks @ factor
    axes = np.empty((0, len(shift)))

    def group(ix, lo, hi):
        mass0 = float(np.prod(ndtr(hi) - ndtr(lo)))
        if len(lo):
            normz = ndtr(hi) - ndtr(lo)
            first = (norm.pdf(lo) - norm.pdf(hi)) / normz
            lp = np.zeros_like(lo)
            hp = np.zeros_like(hi)
            np.multiply(lo, norm.pdf(lo), out=lp, where=np.isfinite(lo))
            np.multiply(hi, norm.pdf(hi), out=hp, where=np.isfinite(hi))
            second = 1 + (lp - hp) / normz - first * first
            m0 = first @ axes
            c0 = np.eye(len(shift)) + (axes.T * (second - 1)) @ axes
        else:
            m0 = np.zeros(len(shift))
            c0 = np.eye(len(shift))
        correction = delta[ix]
        mass = mass0 + correction.sum()
        if mass <= 1e-8:
            return None
        m = (mass0 * m0 + correction @ z[ix]) / mass
        c = (mass0 * (c0 + np.outer(m0, m0)) + (z[ix].T * correction) @ z[ix]) / mass - np.outer(m, m)
        c = (c + c.T) / 2
        smallest = float(np.linalg.eigvalsh(c).min())
        if smallest <= 1e-8:
            return None
        cm = reference_mean + transformed @ m
        csd = np.sqrt(np.maximum(1e-12, ((transformed @ c) * transformed).sum(1) + conditional_sd**2))
        cdf = mass * ndtr((values - cm) / csd)
        return dict(mass=mass, m=m, cov=c, minimum=smallest, ix=ix, lo=lo, hi=hi, cdf=cdf)

    first = group(np.arange(len(beta)), np.empty(0), np.empty(0))
    if first is None:
        return [(1.0, mean, covariance)], 1.0, minimum
    groups = [first]
    for _ in range(max_groups):
        discrepancy = sum(g["cdf"] for g in groups) - reference
        probe = np.unravel_index(abs(discrepancy).argmax(), discrepancy.shape)
        error = float(abs(discrepancy[probe]))
        if error <= 0.012 or len(groups) >= max_groups:
            break
        eligible = [j for j, g in enumerate(groups) if len(g["ix"]) >= 32 and g["mass"] > 0.002]
        if not eligible:
            break
        sign = np.sign(discrepancy[probe])

        def mismatch(j):
            g = groups[j]
            ix = g["ix"]
            actual = weights[ix] @ ndtr((values[probe] - mapped[probe[1], ix]) / conditional_sd[probe[1]])
            return sign * (g["cdf"][probe] - actual)

        which = max(eligible, key=mismatch)
        direction = transformed[probe[1]].copy()
        direction /= max(np.linalg.norm(direction), 1e-15)
        orthogonal = direction - axes.T @ (axes @ direction)
        if np.linalg.norm(orthogonal) > 0.2:
            axes = np.vstack((axes, orthogonal / np.linalg.norm(orthogonal)))
            for g in groups:
                g["lo"] = np.r_[g["lo"], -np.inf]
                g["hi"] = np.r_[g["hi"], np.inf]
            dim = len(axes) - 1
        else:
            dim = int(np.argmax(abs(axes @ direction)))
        parent = groups[which]
        split = ndtri((ndtr(parent["lo"][dim]) + ndtr(parent["hi"][dim])) / 2)
        ix = parent["ix"]
        lower = (z[ix] @ axes[dim]) <= split
        hi = parent["hi"].copy()
        hi[dim] = split
        lo = parent["lo"].copy()
        lo[dim] = split
        children = [group(ix[lower], parent["lo"].copy(), hi), group(ix[~lower], lo, parent["hi"].copy())]
        if any(c is None for c in children):
            break
        groups[which : which + 1] = children
    discrepancy = sum(g["cdf"] for g in groups) - reference
    error = float(abs(discrepancy).max())
    parts = [
        (g["mass"], state["mean"] + factor @ g["m"], completion + factor @ g["cov"] @ factor.T)
        for g in groups
    ]
    return parts, error, min(g["minimum"] for g in groups)
