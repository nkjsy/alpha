"""基本面因子。

基本面数据必须先按“可获得日期”（财报披露日 + 缓冲）对齐到交易日并前向填充，
再放进 MarketData.fields。用 align_fundamental 完成这一步，避免前视偏差。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha.data.market import MarketData
from alpha.factors.base import Factor, register


def align_fundamental(
    records: pd.DataFrame,
    dates: pd.DatetimeIndex,
    tickers: pd.Index,
    value_col: str,
    available_col: str = "available_date",
    lag_days: int = 1,
    max_staleness: int = 400,
) -> pd.DataFrame:
    """把长表基本面记录（ticker, available_date, value）转成交易日宽表。

    available_date 应为财报公开日期（如 SEC filing date），而不是财报期末日。
    lag_days 为额外缓冲交易日；超过 max_staleness 个交易日未更新则置空。
    """
    r = records[["ticker", available_col, value_col]].dropna().copy()
    r[available_col] = pd.to_datetime(r[available_col])
    wide = r.pivot_table(index=available_col, columns="ticker", values=value_col, aggfunc="last")
    wide = wide.reindex(columns=tickers)
    # 把披露日映射到下一个交易日（含当天），再平移 lag_days
    pos = dates.searchsorted(wide.index, side="left")
    keep = pos < len(dates)
    wide = wide[keep]
    wide.index = dates[pos[keep]]
    wide = wide.groupby(level=0).last()
    out = wide.reindex(dates).shift(lag_days)
    return out.ffill(limit=max_staleness)


@register
class FieldFactor(Factor):
    """直接把 MarketData.fields 中的某个字段当因子，sign=-1 表示越小越好。"""

    name = "field"

    def __init__(self, field: str, sign: int = 1, log: bool = False) -> None:
        super().__init__(field=field, sign=sign, log=log)
        self.field, self.sign, self.log = field, sign, log

    def compute(self, data: MarketData) -> pd.DataFrame:
        x = data.field(self.field)
        if self.log:
            x = np.log(x.where(x > 0))
        return self.sign * x


@register
class BookToMarket(Factor):
    """价值：账面价值 / 市值。需要 fields['book_value'] 与 fields['market_cap']。"""

    name = "book_to_market"

    def compute(self, data: MarketData) -> pd.DataFrame:
        bv = data.field("book_value")
        mc = data.field("market_cap")
        return (bv / mc.where(mc > 0)).where(bv > 0)


@register
class EarningsYield(Factor):
    """盈利收益率：TTM 净利润 / 市值。需要 fields['earnings_ttm'] 与 fields['market_cap']。"""

    name = "earnings_yield"

    def compute(self, data: MarketData) -> pd.DataFrame:
        e = data.field("earnings_ttm")
        mc = data.field("market_cap")
        return e / mc.where(mc > 0)
