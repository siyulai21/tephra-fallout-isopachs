"""Public volume estimates use the integral of the pointwise median thickness field."""


def retain_endpoints(summary):
    """Remove a central estimate from a volume distribution. Keep precision and units."""
    if isinstance(summary, dict):
        if {"lower95", "upper95"} <= summary.keys():
            for key in ("median", "mean"):
                summary.pop(key, None)
        for item in summary.values():
            retain_endpoints(item)
    elif isinstance(summary, list):
        for item in summary:
            retain_endpoints(item)


def directional_intervals(report):
    for block in (report["volume"], report["volume"].get("archive_conditioned", {})):
        if "coherent_volume_km3" in block:
            retain_endpoints(block["coherent_volume_km3"])


def spherical_intervals(report):
    for fit in report["fits"].values():
        for key in ("volume", "parent_volume", "bearing_width_sensitivities_km3", "enhanced_hidden_fan_km3"):
            retain_endpoints(fit.get(key))
    bearing = report["bearing"]
    retain_endpoints(bearing.get("parent_reference_km3"))
    for alternative in bearing.get("reach_sensitivity", {}).values():
        retain_endpoints(alternative["volume"])
    for part in report["mass_partition"]["parts"].values():
        retain_endpoints(part.get("volume_km3"))
    report["estimate_source"] = "deposit.json#median_map_volume_km3"


def endpoints(values):
    return [values[0], values[-1]]


def corridor_intervals(report):
    """Accept an original fit record or the current endpoint-only publication."""
    report.pop("conditional_volume_km3", None)
    if "volume_km3" not in report:
        return
    report["conditional_volume_interval95_km3"] = endpoints(report.pop("volume_km3"))
    report["reference_volume_interval95_km3"] = endpoints(report.pop("reference_volume_km3"))
    report["sector_volume_interval95_km3"] = [endpoints(v) for v in report.pop("sector_volume_km3")]
    report["population_volume_intervals95_km3"] = [endpoints(v) for v in report.pop("volume_population_quantiles")]
    prior = report["prior_sensitivity"]
    for scenario in (*prior["broad_prior_scenarios"].values(), *prior["conditional_by_law"]):
        values = scenario.pop("volume_km3", None)
        scenario["conditional_volume_interval95_km3"] = endpoints(values) if values is not None else None
    for check in report["quadrature"]:
        check["conditional_volume_interval95_km3"] = endpoints(check.pop("volume_km3"))


def whole_area_estimates(report, mapped):
    """Attach the broad sensitivity's median-map integral and retain volume interval endpoints."""
    report["median_map_volume_km3"] = mapped["median_map_volume_km3"]
    report["estimate_source"] = "median-field.json#median_map_volume_km3"
    if "conditional_volume_km3" not in report:
        return
    report["conditional_volume_interval95_km3"] = endpoints(report.pop("conditional_volume_km3"))
    report["slab_volume_intervals95_km3"] = [endpoints(q) for q in report.pop("slab_volume_km3")]
    report["volume_intervals95_by_tail_km3"] = {
        tail: endpoints(q) if q is not None else None for tail, q in report.pop("volume_conditional_on_tail").items()
    }
    for prior in report["prior_sensitivity"].values():
        prior["volume_interval95_km3"] = endpoints(prior.pop("volume_km3"))
    terminal = report.get("terminal_continuation")
    if terminal:
        terminal["unrestricted_volume_interval95_km3"] = endpoints(terminal.pop("unrestricted_volume_km3"))
        terminal["transfer_multiplier_intervals95_km3"] = {
            scale: endpoints(q) for scale, q in terminal.pop("transfer_multiplier_km3").items()
        }


def regional_estimates(report):
    """Keep the median-map estimate and the interval endpoints in each regional result."""
    def mapped(integral):
        integral.pop("expected_km3", None)
        integral["volume_statistic"] = "Integral of pointwise median thickness"
        integral.pop("median_surface_note", None)

    for method, item in report["methods"].items():
        mapped(item["regional_volume"])
        joint = item["joint_volume"]
        joint["conditional_quantiles_km3"].pop("0.5", None)
        joint.get("quantile_monte_carlo_90_ranges_km3", {}).pop("0.5", None)
        joint.pop("sampling_mean_km3", None)
        for deletion in report["validation"]["methods"][method]["selections"]:
            mapped(deletion["regional_deletion_volume"])
