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


def snapshot_mask(
    snapshots: pd.DataFrame,
    dates: pd.DatetimeIndex,
    tickers: pd.Index | None = None,
    date_col: str = "month_end_date",
    ticker_col: str = "ticker",
) -> pd.DataFrame:
    """由定期成分快照（如每月末成分表）生成每日股票池。

    快照日期 D 的成分从“D 当天或之前的最后一个交易日”开始生效，并沿用到下一个快照。
    这样月末快照不会被提前用到当月的其它交易日（月中调入的股票不会提前出现）。
    """
    s = snapshots[[date_col, ticker_col]].dropna().copy()
    s[date_col] = pd.to_datetime(s[date_col])
    cols = tickers if tickers is not None else pd.Index(sorted(s[ticker_col].unique()))
    mask = pd.DataFrame(False, index=dates, columns=cols)
    snap_dates = sorted(s[date_col].unique())
    for i, d in enumerate(snap_dates):
        pos = dates.searchsorted(d, side="right") - 1  # D 当天或之前的最后一个交易日
        if pos < 0:
            continue
        end = dates.searchsorted(snap_dates[i + 1], side="right") - 1 if i + 1 < len(snap_dates) else len(dates)
        members = [t for t in s.loc[s[date_col] == d, ticker_col] if t in mask.columns]
        if end > pos:
            mask.iloc[pos:end, mask.columns.get_indexer(members)] = True
    return mask
