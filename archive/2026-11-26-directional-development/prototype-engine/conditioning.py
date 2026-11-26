"""Gaussian evidence and expectation propagation proposals for documented bounded events."""

import numpy as np
from scipy.linalg import cho_solve
from scipy.special import log_ndtr, ndtri, ndtri_exp, logsumexp
from scipy.stats import qmc

from ....core.likelihood import censored_terms, normal_log_event


def gaussian(precision, linear):
    chol = np.linalg.cholesky(precision)
    cov = cho_solve((chol, True), np.eye(len(chol)), check_finite=False)
    return cov @ linear, cov, float(np.log(np.diag(chol)).sum())


def condition(B, offset, data, noise):
    """Condition in the bounded-site subspace and complete the Gaussian field.

    Extra basis coefficients with no bounded-site influence do not enlarge the
    EP solve or the importance integral. Coincident events remain separate
    likelihood factors, even when their latent values have deficient rank.
    """
    measurable = ~data.censored
    error = np.hypot(data.log_sd, noise)
    y = data.log_value-offset
    A = B[measurable]/error[measurable, None]
    z = y[measurable]/error[measurable]
    h = A.T@z
    if len(A) < B.shape[1]:
        factor = np.linalg.cholesky(np.eye(len(A))+A@A.T)
        solved = cho_solve((factor, True), A, check_finite=False)
        base_mean = A.T@cho_solve((factor, True), z, check_finite=False)
        base_cov = np.eye(B.shape[1])-A.T@solved
        base_logz = (-.5*(len(z)*np.log(2*np.pi)+z@cho_solve((factor, True), z, check_finite=False))
                     -np.log(error[measurable]).sum()-np.log(factor.diagonal()).sum())
    else:
        base_mean, base_cov, logdet = gaussian(np.eye(B.shape[1])+A.T@A, h)
        base_logz = (-.5*(len(z)*np.log(2*np.pi)+z@z)-np.log(error[measurable]).sum()
                     +.5*h@base_mean-logdet)
    base_cov = (base_cov+base_cov.T)/2
    X = B[~measurable]
    sd, target = error[~measurable], y[~measurable]
    lower = data.lower_censored[~measurable]
    hi = (data.interval_log_upper-offset)[~measurable]
    mu0 = X@base_mean
    cross = base_cov@X.T
    S = X@cross
    eigenvalues, eigenvectors = np.linalg.eigh((S+S.T)/2)
    active = np.flatnonzero(eigenvalues > max(1e-14, eigenvalues.max(initial=0.)*1e-12))[::-1]
    root = np.sqrt(eigenvalues[active])
    F = eigenvectors[:, active]*root
    completion = (cross@eigenvectors[:, active])/root
    tau, eta, cs = np.zeros((3, len(X)))
    residual = 0.
    for iteration in range(64):
        linear = F.T@(eta-tau*mu0)
        mean_z, cov_z, logdet = gaussian(np.eye(len(active))+(F.T*tau)@F, linear)
        if not len(X):
            break
        mu = mu0+F@mean_z
        variance = np.maximum(1e-10, np.sum((F@cov_z)*F, axis=1))
        cavity = 1./np.maximum(1e-10, 1./variance-tau)
        center = cavity*(mu/variance-eta)
        loss, g, curv = censored_terms(center, target, np.sqrt(cavity+sd**2), lower, hi)
        denom = np.maximum(1e-10, 1-cavity*curv)
        proposed = curv/denom
        nat = (center*curv-g)/denom
        c = (-loss-.5*np.log(denom)
             -.5*((center-cavity*g)**2/(cavity*denom)-center**2/cavity))
        residual = max(np.max(abs(proposed-tau)/(1+abs(proposed))),
                       np.max(abs(nat-eta)/(1+abs(nat))), np.max(abs(c-cs)/(1+abs(c))))
        if residual < 3e-6 or iteration == 63:
            break
        tau += .55*(proposed-tau)
        eta += .55*(nat-eta)
        cs += .55*(c-cs)
    logz = base_logz+cs.sum()+eta@mu0-.5*tau@(mu0*mu0)+.5*linear@mean_z-logdet
    mean = base_mean+completion@mean_z
    cov = base_cov+completion@(cov_z-np.eye(len(active)))@completion.T
    gain = cho_solve((np.linalg.cholesky(S+np.diag(sd**2)), True), cross.T, check_finite=False).T
    complete_cov = base_cov-gain@cross.T
    event_cov = F@cov_z@F.T
    eigenvalues, eigenvectors = np.linalg.eigh((event_cov+event_cov.T)/2)
    active = np.flatnonzero(eigenvalues > max(1e-14, eigenvalues.max(initial=0.)*1e-12))[::-1]
    event_factor = eigenvectors[:, active]*np.sqrt(eigenvalues[active])
    recovery = X.T@(eigenvectors[:, active]/np.sqrt(eigenvalues[active]))
    control_factor = cov@recovery
    conditional_cov = cov-control_factor@control_factor.T
    return dict(mean=mean, cov=(cov+cov.T)/2, h=h, X=X, sd=sd,
                target=target, lower=lower, hi=hi, tau=tau, eta=eta, cs=cs,
                complete_cov=(conditional_cov+conditional_cov.T)/2,
                measurement_complete_cov=(complete_cov+complete_cov.T)/2,
                control_variate="bounded" if len(X) else False,
                control_factor=control_factor, control_recovery=recovery,
                complete_gain=gain, base_mean=base_mean, base_event_mean=mu0,
                logz=float(logz), ep_residual=float(residual), iterations=iteration+1,
                event_factor=event_factor)


def bounded_normal(a, b, u):
    """Inverse truncated-normal CDF, using survival probabilities in the right tail."""
    right = a > 0
    lo = np.where(right, -b, a)
    hi = np.where(right, -a, b)
    q = np.logaddexp(np.log1p(-u) + log_ndtr(lo), np.log(u) + log_ndtr(hi))
    return np.where(right, -1.0, 1.0) * ndtri_exp(q)


def corrected(state, power, seed):
    """Importance correction of both evidence and the conditional law; no event is a point value."""
    if state.get("control_variate") == "bounded":
        factor = state["event_factor"]
        rng = qmc.Sobol(factor.shape[1], scramble=True, seed=seed)
        z = ndtri(np.clip(rng.random_base2(power), 1e-14, 1-1e-14))
        f = state["X"]@state["mean"]+z@factor.T
        event = normal_log_event(state["target"], f, state["sd"], np.ones(len(factor), bool),
                                 state["lower"], state["hi"])
        ratio = (event+.5*f*f*state["tau"]-f*state["eta"]-state["cs"]).sum(axis=1)
        normalizer = logsumexp(ratio)
        weights = np.exp(ratio-normalizer)
        centers = state["mean"]+z@state["control_factor"].T
        return centers, weights, float(normalizer-np.log(len(z))), float(1/(len(z)*(weights@weights)))
    if state.get("control_variate"):
        from .residual import corrected as residual_corrected
        return residual_corrected(state, power, seed)
    mean = state["mean"]
    X = state["X"]
    if not len(X):
        return mean[None, :], np.ones(1), 0.0, 1.0
    sd, y, lower, hi = (state[k] for k in ("sd", "target", "lower", "hi"))
    factor = state["event_factor"]
    dimension = factor.shape[1]
    rng = qmc.Sobol(dimension + len(X), scramble=True, seed=seed)
    u = np.clip(rng.random_base2(power), 1e-14, 1.0 - 1e-14)
    f = X @ mean + ndtri(u[:, :dimension]) @ factor.T
    events = normal_log_event(y, f, sd, np.ones(len(y), bool), lower, hi)
    ratio = (events + 0.5 * f * f * state["tau"] - f * state["eta"] - state["cs"]).sum(axis=1)
    norm = logsumexp(ratio)
    w = np.exp(ratio - norm)
    two = np.isfinite(hi)
    lower_edge = np.where(lower | two, y, -np.inf)
    upper_edge = np.where(two, hi, np.where(lower, np.inf, y))
    completed = f + sd * bounded_normal((lower_edge - f) / sd, (upper_edge - f) / sd, u[:, dimension:])
    centers = state["base_mean"]+(completed-state["base_event_mean"])@state["complete_gain"].T
    return centers, w, float(norm - np.log(len(u))), float(1.0 / (len(w) * (w @ w)))
