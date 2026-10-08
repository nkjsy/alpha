"""端到端的未来函数检查：篡改某日之后的所有数据，该日及之前的信号和目标仓位必须完全不变。"""

import numpy as np
import pandas as pd
import pytest

from alpha.data.market import MarketData
from alpha.data.universe import snapshot_mask
from alpha.pipeline import run_research


def _tamper(market: MarketData, t: pd.Timestamp, seed: int = 0) -> MarketData:
    rng = np.random.default_rng(seed)
    after = market.close.index > t

    def scramble(df, lo=0.3, hi=3.0):
        df = df.copy()
        df.loc[after] = df.loc[after] * rng.uniform(lo, hi, df.loc[after].shape)
        return df

    return MarketData(
        close=scramble(market.close),
        volume=scramble(market.volume),
        universe=market.universe,
        sector=market.sector,
        fields={"market_cap": scramble(market.fields["market_cap"])},
    )


@pytest.mark.parametrize("method", ["equal", "icir"])
def test_pipeline_targets_ignore_future(market, tmp_path, method):
    cfg = {
        "factors": [{"name": "momentum"}, {"name": "low_vol"}, {"name": "size"}, {"name": "low_beta"}],
        "universe": {"min_dollar_volume": 1e5},
        "preprocess": {"neutralize_sector": True, "neutralize_size": True},
        "combine": {"method": method, "window": 126, "min_periods": 60},
        "portfolio": {"freq": "ME", "horizon": 21, "n_long": 0.2, "n_short": 0.2},
    }
    t = market.dates[700]
    a = run_research(cfg, data=market, out_dir=tmp_path / "a")
    b = run_research(cfg, data=_tamper(market, t), out_dir=tmp_path / "b")
    pd.testing.assert_frame_equal(a.score.loc[:t], b.score.loc[:t])
    pd.testing.assert_frame_equal(a.targets.loc[:t], b.targets.loc[:t])
    # 目标仓位在 t 之后确实发生了变化，说明篡改生效、测试不是空转
    assert not a.targets.loc[t:].equals(b.targets.loc[t:])


def test_in_sample_stats_ignore_holdout(market, tmp_path):
    """篡改留出期价格，样本内绩效与单因子评估必须不变。"""
    holdout = market.dates[700]
    cfg = {
        "holdout_start": str(holdout.date()),
        "factors": [{"name": "momentum"}, {"name": "reversal"}],
        "combine": {"method": "icir", "window": 126, "min_periods": 60},
        "portfolio": {"freq": "ME", "horizon": 21, "n_long": 0.2},
    }
    a = run_research(cfg, data=market, out_dir=tmp_path / "a")
    b = run_research(cfg, data=_tamper(market, holdout - pd.Timedelta(days=1)), out_dir=tmp_path / "b")
    assert a.stats["in_sample"] == b.stats["in_sample"]
    pd.testing.assert_frame_equal(a.factor_summary, b.factor_summary)
    assert a.stats["out_of_sample"] != b.stats["out_of_sample"]


def test_forward_returns_keep_delisted():
    idx = pd.bdate_range("2021-01-01", periods=6)
    close = pd.DataFrame({"A": [10, 11, 12, 13, 14, 15], "B": [10, 10, 8, np.nan, np.nan, np.nan]}, index=idx, dtype=float)
    md = MarketData(close=close)
    f = md.forward_returns(horizon=2, lag=1, delisting_return=-0.5)
    # B：第 0 天信号，第 1 天 10 买入，第 3 天前退市，最后价格 8 -> -20%，再叠加 -50%
    assert f.loc[idx[0], "B"] == pytest.approx(0.8 * 0.5 - 1)
    # 第 3 天之后 B 已无价格，不应产生收益
    assert np.isnan(f.loc[idx[3], "B"])
    # 数据末尾（未来不足 lag+horizon）保持 NaN，而不是被当成退市
    assert f.iloc[-3:].isna().all().all()


def test_snapshot_mask_not_used_early():
    dates = pd.bdate_range("2021-01-01", "2021-03-31")
    snaps = pd.DataFrame({
        "month_end_date": ["2021-01-31", "2021-02-28", "2021-02-28"],
        "ticker": ["A", "A", "B"],
    })
    m = snapshot_mask(snaps, dates)
    # 1 月 31 日是周日 -> 1 月快照从 1 月 29 日（周五）生效
    assert not m.loc[:"2021-01-28"].any().any()
    assert m.loc["2021-01-29", "A"]
    # B 只在 2 月末快照中出现，2 月 26 日（2 月最后交易日）之前不能出现
    assert not m.loc[:"2021-02-25", "B"].any()
    assert m.loc["2021-02-26":, "B"].all()
