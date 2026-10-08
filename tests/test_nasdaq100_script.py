import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


def _load():
    path = Path(__file__).resolve().parents[1] / "scripts" / "nasdaq100_momentum.py"
    spec = importlib.util.spec_from_file_location("nasdaq100_momentum", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_run_scenarios_on_synthetic(market):
    mod = _load()
    close = market.close.iloc[:, :40]
    bench = close.mean(axis=1)
    rebal = pd.date_range(close.index[0], close.index[-1], freq="ME")
    membership = pd.DataFrame(
        [(d, t) for d in rebal for t in close.columns[:30]], columns=["month_end_date", "ticker"]
    )
    summary, navs, cov = mod.run_scenarios(close, bench, membership, start=str(close.index[300].date()), n_trials=10)
    assert len(summary) == 7
    assert summary.loc["S1 原版回测复现", "annual_cost"] == 0
    assert summary.loc["S4 +成本5bp/不ffill", "annual_cost"] > 0
    assert np.isfinite(summary["sharpe"]).all()
    assert (cov <= 1).all()
