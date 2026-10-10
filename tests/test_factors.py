import numpy as np
import pandas as pd
import pytest

from alpha.data.market import MarketData
from alpha.factors import FACTOR_REGISTRY, build_factor

PRICE_FACTORS = [
    "momentum", "reversal", "low_vol", "max_ret", "size", "amihud", "low_beta",
    "high52w", "resid_momentum", "sharpe_momentum", "info_continuity", "low_idio_vol", "low_skew",
    "ma_ratio", "overnight_momentum", "intraday_reversal", "abnormal_volume", "seasonality",
    "industry_momentum",
]


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
    opn = market.open.copy()
    opn.loc[opn.index > t] *= 0.7
    tampered = MarketData(close=close, open=opn, volume=vol, universe=market.universe, sector=market.sector,
                          fields={"market_cap": mcap})
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


FUNDAMENTAL_FACTORS = [
    "book_to_market", "earnings_yield", "sales_to_price", "cfo_yield", "gross_profitability",
    "operating_profitability", "roe", "low_accruals", "low_asset_growth", "low_net_issuance",
    "low_leverage", "sue", "sales_growth", "quality_composite",
]
FIELDS = ["book_value", "market_cap", "earnings_ttm", "revenue_ttm", "cfo_ttm", "gross_profit_ttm",
          "operating_income_ttm", "assets", "shares", "liabilities", "sue"]


@pytest.mark.parametrize("name", FUNDAMENTAL_FACTORS)
def test_fundamental_no_lookahead(market, name):
    rng = np.random.default_rng(1)
    fields = {k: pd.DataFrame(rng.lognormal(0, 1, market.close.shape), index=market.dates, columns=market.tickers)
              for k in FIELDS}
    base = build_factor(name).compute(MarketData(close=market.close, fields=fields))
    t = market.dates[600]
    tampered = {k: v.copy() for k, v in fields.items()}
    for v in tampered.values():
        v.loc[v.index > t] *= 3
    after = build_factor(name).compute(MarketData(close=market.close, fields=tampered))
    pd.testing.assert_frame_equal(base.loc[:t], after.loc[:t])
    assert base.loc[t:].notna().any().any()
