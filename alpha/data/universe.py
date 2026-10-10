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


def liquid_pool(dollar_volume: pd.DataFrame, n: int, window: int = 63, start: str | None = None) -> list[str]:
    """月末按过去 window 日平均成交额排名，返回“曾进入前 n 名”的股票，用来缩小待下载基本面的范围。"""
    adv = dollar_volume.rolling(window, min_periods=window // 2).mean()
    me = adv.groupby(adv.index.to_period("M")).tail(1)
    if start is not None:
        me = me[me.index >= pd.Timestamp(start)]
    ranks = me.rank(axis=1, ascending=False)
    return sorted(ranks.columns[(ranks <= n).any()])


def cap_rank_membership(
    market_cap: pd.DataFrame,
    n: int = 1000,
    rank_month: int = 5,
    start: str | None = None,
) -> pd.DataFrame:
    """仿罗素1000：每年 rank_month 月最后一个交易日按市值取前 n 名，从下个月末的快照起生效，持有一年。

    只用排名日当天已知的市值（市值字段本身已按财报披露日对齐），返回每月末快照
    (month_end_date, ticker)，交给 snapshot_mask 使用。
    """
    mcap = market_cap.dropna(how="all")
    month_last = mcap.groupby(mcap.index.to_period("M")).tail(1).index
    rank_dates = [d for d in month_last if d.month == rank_month]
    rows = []
    for d in month_last:
        prior = [r for r in rank_dates if r.to_period("M") < d.to_period("M")]
        if not prior:
            continue
        if start is not None and d < pd.Timestamp(start):
            continue
        top = mcap.loc[prior[-1]].dropna().nlargest(n).index
        rows.extend((d, t) for t in top)
    return pd.DataFrame(rows, columns=["month_end_date", "ticker"])
