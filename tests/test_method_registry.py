"""The model catalogue drives fitting and rendering, including future families."""
import json

import numpy as np
import pytest

from src.empirical_isopach.models import registry
from src.empirical_isopach.core.catalog import resolve_dataset
from src.empirical_isopach.reporting import compare_dataset, write_readme


def test_public_families_and_historical_imports(capsys):
    from src.empirical_isopach.cli import main
    from src.probabilistic_isopach.model import SpatialModel as old_gp
    from src.empirical_isopach.models.gaussian_process.model import SpatialModel
    from src.empirical_isopach.spline import TrendRegularizedSpline as old_spline
    from src.empirical_isopach.models.cubic_spline.model import TrendRegularizedSpline
    assert old_gp is SpatialModel
    assert old_spline is TrendRegularizedSpline
    assert registry.active_methods() == ("stacked", "spline")
    main(["methods"])
    output = capsys.readouterr().out
    assert "GP ensemble" in output and "Trend-regularized cubic spline" in output
    with pytest.raises(ValueError, match="Unknown empirical method"):
        registry.method_spec("unregistered")


def test_third_family_reaches_folds_volumes_figures_and_benchmark(tmp_path, monkeypatch):
    from benchmarks.regional import summarize
    calls = []

    class FixedField:
        """Test-only correlated Gaussian field; never a scientific registry entry."""
        def __init__(self, level):
            self.level = level

        def predict_log(self, xy, observation=False, log_sd=None, covariance=False):
            mean = np.full(len(xy), self.level)
            if covariance:
                return mean, np.full((len(xy), len(xy)), .04)
            variance = np.full(len(xy), .04 + (.1 if observation else 0))
            if observation and log_sd is not None:
                variance += np.asarray(log_sd)**2
            return mean, variance

        def describe(self):
            return {"test_only": True, "level": self.level}

    def fitter(key, level):
        def fit(data, bounds, folds, buffer_km):
            calls.append((key, set(data.row_ids)))
            return FixedField(level), {"candidates": [
                {"candidate": {"level": level}, "fold_scores": [1.]*folds, "failures": []}]}
        return fit

    methods = {key: registry.Method(label, fitter(key, level), "Dispatch test only")
               for key, label, level in (("stacked", "Test GP", 1.),
                                         ("spline", "Test spline", 1.1),
                                         ("future", "Test third family", 1.2))}
    monkeypatch.setattr(registry, "REGISTRY", methods)
    output = tmp_path/"tambora"
    report = compare_dataset(resolve_dataset("Tambora"), output, draws=100, progress=lambda _: None)
    assert tuple(report["methods"]) == ("stacked", "spline", "future")
    assert len(calls) == 15  # three methods, four outer fits plus all-data fits
    assert len(report["validation"]["paired_fold_score_differences"]) == 3
    for fold in range(4):
        assert calls[fold][1] == calls[4+fold][1] == calls[8+fold][1]
    for key in methods:
        result = report["methods"][key]
        assert result["regional_volume"]["converged"]
        assert result["joint_volume"]["expected_grid_converged"]
        assert (output/result["selection"]["candidate_table"]).is_file()
    for filename in ("isopachs.png", "diagnostics.png", "heldout_predictions.csv"):
        assert (output/filename).stat().st_size > 100
    assert "Test third family" in (output/"README.md").read_text()
    summarize(tmp_path)
    assert "Test third family" in (tmp_path/"README.md").read_text()
    assert "future" in (tmp_path/"comparison.csv").read_text()

    # Old reports remain readable if their method is later removed/renamed.
    methods.pop("future")
    saved = json.loads((output/"report.json").read_text())
    write_readme(output, "Tambora", saved)
    assert "Test third family" in (output/"README.md").read_text()
