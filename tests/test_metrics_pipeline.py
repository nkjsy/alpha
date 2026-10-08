import numpy as np
import pandas as pd

from alpha.metrics import deflated_sharpe_ratio, performance_summary, probabilistic_sharpe_ratio
from alpha.pipeline import run_research
from alpha.validation import ExperimentLog, walk_forward_splits


def test_dsr_penalizes_more_trials():
    r = pd.Series(np.random.default_rng(0).normal(0.0006, 0.01, 1000))
    assert deflated_sharpe_ratio(r, 1) > deflated_sharpe_ratio(r, 50) > deflated_sharpe_ratio(r, 1000)
    assert abs(deflated_sharpe_ratio(r, 1) - probabilistic_sharpe_ratio(r)) < 1e-12


def test_performance_summary_basic():
    r = pd.Series([0.01, -0.02, 0.03, 0.0] * 100)
    s = performance_summary(r)
    assert s["max_drawdown"] < 0 and s["ann_vol"] > 0


def test_walk_forward_no_overlap():
    d = pd.bdate_range("2010-01-01", "2020-12-31")
    sp = walk_forward_splits(d, train_years=3, test_years=1, embargo_days=21)
    assert len(sp) == 8
    for s in sp:
        assert s.train[1] < s.test[0]


def test_experiment_log_counts(tmp_path):
    log = ExperimentLog(tmp_path / "e.jsonl")
    log.log("a", {"x": 1}, {"sharpe": 1.0})
    log.log("b", {"x": 2}, {"sharpe": 0.5})
    assert log.n_trials() == 2
    assert len(log.trial_sharpes()) == 2


def test_pipeline_end_to_end(market, tmp_path):
    cfg = {
        "name": "t",
        "holdout_start": str(market.dates[700].date()),
        "factors": [{"name": "momentum"}, {"name": "low_vol"}],
        "preprocess": {"neutralize_sector": True},
        "combine": {"method": "icir", "window": 126, "min_periods": 60},
        "portfolio": {"freq": "ME", "horizon": 21, "n_long": 0.2, "n_short": 0.2},
    }
    res = run_research(cfg, data=market, out_dir=tmp_path)
    assert (tmp_path / "report.txt").exists()
    assert res.stats["n_trials"] == 1
    assert "out_of_sample" in res.stats
    assert res.factor_summary.loc["momentum(lookback=252,skip=21)", "ic_mean"] > 0
