import importlib.util
from pathlib import Path

from alpha.data import make_synthetic_market


def test_mine_script_runs_on_synthetic(tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts" / "mine_factors.py"
    spec = importlib.util.spec_from_file_location("mine_factors", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    m = make_synthetic_market(n_stocks=100, n_days=252 * 7, seed=5, momentum_strength=0.3, start="2014-01-02")
    res, hold, bts, n_trials = mod.run(m, "2015-01-01", "2019-06-01", min_t=2.0, log_path=tmp_path / "e.jsonl")
    assert len(res.table) >= 20
    assert n_trials == len(res.table)
    assert "基准 11-1 动量" in bts
    assert set(hold.index) == set(res.selected)


def test_mine_script_industry_and_sub_universe(tmp_path):
    import numpy as np
    import pandas as pd

    path = Path(__file__).resolve().parents[1] / "scripts" / "mine_factors.py"
    spec = importlib.util.spec_from_file_location("mine_factors", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    m = make_synthetic_market(n_stocks=120, n_days=252 * 7, seed=6, momentum_strength=0.3, start="2014-01-02")
    rng = np.random.default_rng(0)
    m.fields["book_value"] = m.close * 0 + rng.uniform(1, 2, size=m.close.shape[1])
    m.fields["market_cap"] = m.close * 10
    groups = pd.DataFrame([list(rng.choice(["A", "B", "C"], size=m.close.shape[1]))] * len(m.dates),
                          index=m.dates, columns=m.close.columns)
    sub = m.universe.copy()
    sub.iloc[:, 60:] = False
    res, hold, bts, n_trials = mod.run(m, "2015-01-01", "2019-06-01", min_t=2.0, log_path=tmp_path / "e.jsonl",
                                       groups=groups, sub_universe=sub)
    assert "book_to_market|ind" in res.table.index
    assert {"ic_incr_sub", "t_incr_sub"} <= set(res.table.columns)
    assert res.table["t_incr_sub"].notna().any()
