import numpy as np
import pandas as pd

from alpha.factors import build_factor
from alpha.portfolio import rebalance_schedule
from alpha.strategies import regime_switch_targets, trend_regime


def test_trend_regime_uses_past_only():
    idx = pd.bdate_range("2020-01-01", periods=300)
    px = pd.Series(np.linspace(100, 200, 300), index=idx)
    r = trend_regime(px, 200)
    assert not r.iloc[:199].any() and r.iloc[199:].all()
    px2 = px.copy()
    px2.iloc[250:] = 1.0
    pd.testing.assert_series_equal(trend_regime(px2, 200).iloc[:250], r.iloc[:250])


def test_regime_switch_picks_and_counts(market):
    score = build_factor({"name": "momentum", "lookback": 252, "skip": 21}).compute(market)
    reb = rebalance_schedule(market.dates, "ME")
    regime = pd.Series(False, index=market.dates)
    regime.iloc[600:700] = True
    t = regime_switch_targets(score, market.universe, regime, reb, n_on=3, n_off=10)
    n_names = (t > 0).sum(axis=1)
    on_days = t.index[(t.index >= market.dates[600]) & (t.index < market.dates[700])]
    assert (n_names.loc[on_days] == 3).all()
    off_days = t.index[t.index >= market.dates[750]]
    assert (n_names.loc[off_days] == 10).all()
    invested = t[t.sum(axis=1) > 0]
    assert np.allclose(invested.sum(axis=1), 1)
    assert (t.loc[t.index >= market.dates[300]].sum(axis=1) > 0).all()  # 动量预热期过后应满仓
    # risk-on 每日重排：所选股票就是当天得分最高的 3 只
    d = on_days[-1]
    best = score.loc[d].where(market.universe.loc[d]).nlargest(3).index
    assert set(t.loc[d][t.loc[d] > 0].index) == set(best)


def test_monthly_mode_only_changes_on_rebalance_or_regime_flip(market):
    score = build_factor("momentum").compute(market)
    reb = rebalance_schedule(market.dates, "ME")
    regime = pd.Series(False, index=market.dates)
    t = regime_switch_targets(score, market.universe, regime, reb, off_rerank="monthly")
    assert set(t.index) <= set(reb)
