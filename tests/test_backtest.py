import numpy as np
import pandas as pd
import pytest

from alpha.backtest import CostModel, run_backtest
from alpha.portfolio import build_weights, rebalance_schedule

ZERO = CostModel(0, 0, 0, 0)


def _prices():
    idx = pd.bdate_range("2021-01-01", periods=6)
    return pd.DataFrame({"A": [100, 110, 121, 121, 133.1, 133.1], "B": [100, 100, 90, 90, 90, 99]}, index=idx, dtype=float)


def test_lag_and_drift_exact():
    px = _prices()
    w = pd.DataFrame({"A": [1.0], "B": [0.0]}, index=[px.index[0]])
    # lag=1：第 1 天收盘建仓，只吃到第 2 天及以后的收益
    r = run_backtest(w, px, lag=1, costs=ZERO)
    assert r.returns.iloc[0] == 0
    assert r.nav.iloc[-1] == pytest.approx(133.1 / 110)
    r0 = run_backtest(w, px, lag=0, costs=ZERO)
    assert r0.nav.iloc[-1] == pytest.approx(133.1 / 100)


def test_costs_charged_on_turnover():
    px = _prices()
    w = pd.DataFrame({"A": [1.0, 0.0], "B": [0.0, 1.0]}, index=[px.index[0], px.index[2]])
    c = CostModel(commission_bps=10, spread_bps=0, impact_bps=0, borrow_bps_annual=0)
    net = run_backtest(w, px, lag=0, costs=c)
    gross = run_backtest(w, px, lag=0, costs=ZERO)
    # 建仓 1.0 + 换仓约 2.0（卖 A 买 B），每单位 10bp
    expected = gross.nav.iloc[-1] * (1 - 0.001) * (1 - 0.002)
    assert net.nav.iloc[-1] == pytest.approx(expected, rel=1e-5)
    assert net.turnover.iloc[1] == pytest.approx(1.0, rel=1e-2)


def test_long_short_market_neutral_zero_when_identical():
    idx = pd.bdate_range("2021-01-01", periods=50)
    rng = np.random.default_rng(0)
    path = 100 * np.cumprod(1 + rng.normal(0, 0.01, 50))
    px = pd.DataFrame({"A": path, "B": path}, index=idx)
    w = pd.DataFrame({"A": [1.0], "B": [-1.0]}, index=[idx[0]])
    r = run_backtest(w, px, lag=0, costs=ZERO)
    assert np.allclose(r.returns, 0, atol=1e-12)


def test_delisting_closes_position():
    px = _prices()
    px.loc[px.index[3]:, "B"] = np.nan
    w = pd.DataFrame({"A": [0.5], "B": [0.5]}, index=[px.index[0]])
    r = run_backtest(w, px, lag=0, costs=ZERO, delisting_return=-1.0)
    assert r.weights["B"].iloc[-1] == 0
    # 第 3 天 B 退市，按 -100% 计：亏掉 B 的持仓市值 0.45
    assert r.returns.loc[px.index[3]] == pytest.approx(-0.45 / (0.605 + 0.45))


def test_build_weights_sums(market):
    reb = rebalance_schedule(market.dates, "ME")
    score = market.close.pct_change(63, fill_method=None).where(market.universe)
    w = build_weights(score, reb, n_long=0.2, n_short=0.2, max_weight=0.1)
    active = w[(w != 0).any(axis=1)]
    assert np.allclose(active.clip(lower=0).sum(axis=1), 1)
    assert np.allclose(active.clip(upper=0).sum(axis=1), -1)
    assert active.abs().max().max() <= 0.1 + 1e-12


def test_rebalance_schedule_month_end():
    d = pd.bdate_range("2021-01-01", "2021-04-30")
    r = rebalance_schedule(d, "ME")
    assert list(r.strftime("%Y-%m-%d")) == ["2021-01-29", "2021-02-26", "2021-03-31", "2021-04-30"]
