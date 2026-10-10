import numpy as np
import pandas as pd

from alpha.data import make_synthetic_market
from alpha.factors import build_factor
from alpha.mining import benjamini_hochberg, build_candidates, mine_factors, oriented
from alpha.portfolio import rebalance_schedule


def test_bh_monotone_and_bounds():
    p = pd.Series([0.001, 0.01, 0.03, 0.5, 0.9], index=list("abcde"))
    q = benjamini_hochberg(p)
    assert (q >= p).all() and (q <= 1).all()
    assert q["a"] == 0.005


def test_mining_finds_planted_signal_and_rejects_noise():
    m = make_synthetic_market(n_stocks=150, n_days=252 * 6, seed=3, momentum_strength=0.35)
    rng = np.random.default_rng(0)
    noise = {f"noise{i}": pd.DataFrame(rng.normal(size=m.close.shape), index=m.dates, columns=m.tickers)
             for i in range(10)}
    cands = build_candidates([{"name": "momentum", "lookback": 252, "skip": 21}, "low_vol"], m)
    cands.update(noise)
    reb = rebalance_schedule(m.dates, "ME")
    reb = reb[(reb > m.dates[300]) & (reb < m.dates[-30])]
    res = mine_factors(m, cands, reb, m.forward_returns(21), min_t=2.5)
    assert "momentum(lookback=252,skip=21)" in res.selected
    assert not any(s.startswith("noise") for s in res.selected)
    assert set(res.table.index) == set(cands)


def test_incremental_ic_removes_baseline_duplicate():
    m = make_synthetic_market(n_stocks=150, n_days=252 * 6, seed=4, momentum_strength=0.35)
    mom = build_factor("momentum").compute(m)
    reb = rebalance_schedule(m.dates, "ME")
    reb = reb[(reb > m.dates[300]) & (reb < m.dates[-30])]
    # 候选是基准的单调变换：原始 IC 显著，但增量 IC 应接近 0
    res = mine_factors(m, {"mom_copy": mom * 2 + 1}, reb, m.forward_returns(21), baseline={"mom": mom})
    row = res.table.loc["mom_copy"]
    assert abs(row["t_stat"]) > 2
    assert abs(row["t_incr"]) < 1.5
    assert res.selected == []
    o = oriented(res.factors, res.table, ["mom_copy"])
    assert o["mom_copy"].shape == mom.shape


def test_seasonality_uses_same_month_prior_years():
    idx = pd.bdate_range("2010-01-01", "2016-12-31")
    rng = np.random.default_rng(1)
    close = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0, 0.01, (len(idx), 2)), axis=0), index=idx, columns=["A", "B"])
    from alpha.data.market import MarketData

    sig = build_factor({"name": "seasonality", "years": 3, "min_years": 3}).compute(MarketData(close=close))
    mc = close.groupby(close.index.to_period("M")).last()
    mret = mc / mc.shift(1) - 1
    # 2015 年 6 月的信号 = 2012/2013/2014 年 7 月收益的均值
    expected = mret.loc[[pd.Period("2012-07"), pd.Period("2013-07"), pd.Period("2014-07")], "A"].mean()
    assert sig.loc["2015-06-15", "A"] == expected
