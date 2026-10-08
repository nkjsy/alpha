import numpy as np
import pandas as pd
import pytest

from alpha.data.market import MarketData
from alpha.factors import FACTOR_REGISTRY, build_factor

PRICE_FACTORS = ["momentum", "reversal", "low_vol", "max_ret", "size", "amihud", "low_beta"]


@pytest.mark.parametrize("name", PRICE_FACTORS)
def test_no_lookahead(market, name):
    """篡改 t 之后的数据，t 及之前的因子值不应改变。"""
    f = build_factor(name)
    base = f.compute(market)
    t = market.dates[600]
    close = market.close.copy()
    close.loc[close.index > t] *= np.random.default_rng(0).uniform(0.5, 2.0, close.loc[close.index > t].shape)
    vol = market.volume.copy()
    vol.loc[vol.index > t] *= 3
    mcap = market.fields["market_cap"].copy()
    mcap.loc[mcap.index > t] *= 5
    tampered = MarketData(close=close, volume=vol, universe=market.universe, fields={"market_cap": mcap})
    after = f.compute(tampered)
    pd.testing.assert_frame_equal(base.loc[:t], after.loc[:t])


def test_momentum_definition():
    idx = pd.bdate_range("2020-01-01", periods=300)
    close = pd.DataFrame({"A": np.arange(1, 301, dtype=float)}, index=idx)
    m = build_factor({"name": "momentum", "lookback": 252, "skip": 21}).compute(MarketData(close=close))
    assert m["A"].iloc[251] != m["A"].iloc[251]  # NaN before warm-up
    assert m["A"].iloc[252] == pytest.approx(close["A"].iloc[231] / close["A"].iloc[0] - 1)


def test_registry_and_unknown():
    assert set(PRICE_FACTORS) <= set(FACTOR_REGISTRY)
    with pytest.raises(KeyError):
        build_factor("does_not_exist")


def test_align_fundamental_respects_availability():
    from alpha.factors.fundamental import align_fundamental

    dates = pd.bdate_range("2021-01-01", periods=30)
    rec = pd.DataFrame({"ticker": ["A"], "available_date": ["2021-01-09"], "book": [10.0]})  # 周六披露
    out = align_fundamental(rec, dates, pd.Index(["A"]), "book", lag_days=1)
    first = out["A"].first_valid_index()
    # 周六 -> 周一 2021-01-11，再滞后 1 个交易日 -> 2021-01-12
    assert first == pd.Timestamp("2021-01-12")
