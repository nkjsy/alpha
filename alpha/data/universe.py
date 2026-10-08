from __future__ import annotations

import pandas as pd


def membership_mask(
    membership: pd.DataFrame, dates: pd.DatetimeIndex, tickers: pd.Index | None = None
) -> pd.DataFrame:
    """由成分股区间表生成 point-in-time 股票池。

    membership 列：ticker, start, end（end 为空表示至今仍在池中）。
    同一 ticker 可以有多段区间（被剔除后又重新纳入）。
    """
    m = membership.copy()
    m["start"] = pd.to_datetime(m["start"])
    m["end"] = pd.to_datetime(m["end"]).fillna(pd.Timestamp.max)
    cols = tickers if tickers is not None else pd.Index(sorted(m["ticker"].unique()))
    mask = pd.DataFrame(False, index=dates, columns=cols)
    for row in m.itertuples(index=False):
        if row.ticker not in mask.columns:
            continue
        in_range = (dates >= row.start) & (dates <= row.end)
        mask.loc[in_range, row.ticker] = True
    return mask


def liquidity_mask(
    dollar_volume: pd.DataFrame, window: int = 63, min_dollar_volume: float = 5e6
) -> pd.DataFrame:
    """按过去 window 日平均成交额过滤，只使用历史数据。"""
    adv = dollar_volume.rolling(window, min_periods=window // 2).mean()
    return adv >= min_dollar_volume
