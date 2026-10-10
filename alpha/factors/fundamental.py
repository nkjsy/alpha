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


def _ratio(num: pd.DataFrame, den: pd.DataFrame, positive_den: bool = True) -> pd.DataFrame:
    return num / den.where(den > 0) if positive_den else num / den.replace(0, np.nan)


@register
class SalesToPrice(Factor):
    """价值：TTM 收入 / 市值。"""

    name = "sales_to_price"

    def compute(self, data: MarketData) -> pd.DataFrame:
        return _ratio(data.field("revenue_ttm"), data.field("market_cap"))


@register
class CashFlowYield(Factor):
    """价值：TTM 经营现金流 / 市值。"""

    name = "cfo_yield"

    def compute(self, data: MarketData) -> pd.DataFrame:
        return _ratio(data.field("cfo_ttm"), data.field("market_cap"))


@register
class GrossProfitability(Factor):
    """盈利能力：TTM 毛利 / 总资产（Novy-Marx 2013）。"""

    name = "gross_profitability"

    def compute(self, data: MarketData) -> pd.DataFrame:
        return _ratio(data.field("gross_profit_ttm"), data.field("assets"))


@register
class OperatingProfitability(Factor):
    """盈利能力：TTM 营业利润 / 账面权益（Fama-French 2015 RMW 的简化版）。"""

    name = "operating_profitability"

    def compute(self, data: MarketData) -> pd.DataFrame:
        return _ratio(data.field("operating_income_ttm"), data.field("book_value"))


@register
class ROE(Factor):
    """盈利能力：TTM 净利润 / 账面权益。"""

    name = "roe"

    def compute(self, data: MarketData) -> pd.DataFrame:
        return _ratio(data.field("earnings_ttm"), data.field("book_value"))


@register
class LowAccruals(Factor):
    """盈利质量：−(净利润 − 经营现金流) / 总资产，应计越低越好（Sloan 1996）。"""

    name = "low_accruals"

    def compute(self, data: MarketData) -> pd.DataFrame:
        return -_ratio(data.field("earnings_ttm") - data.field("cfo_ttm"), data.field("assets"))


@register
class LowAssetGrowth(Factor):
    """投资：总资产同比增速取负（Cooper, Gulen & Schill 2008）。"""

    name = "low_asset_growth"

    def __init__(self, lag: int = 252) -> None:
        super().__init__(lag=lag)
        self.lag = lag

    def compute(self, data: MarketData) -> pd.DataFrame:
        a = data.field("assets")
        return -(a / a.shift(self.lag).where(a.shift(self.lag) > 0) - 1)


@register
class LowNetIssuance(Factor):
    """净发行：股本同比变化取负，回购多、增发少的公司未来表现更好（Pontiff & Woodgate 2008）。"""

    name = "low_net_issuance"

    def __init__(self, lag: int = 252) -> None:
        super().__init__(lag=lag)
        self.lag = lag

    def compute(self, data: MarketData) -> pd.DataFrame:
        s = data.field("shares")
        return -np.log(s / s.shift(self.lag).where(s.shift(self.lag) > 0))


@register
class LowLeverage(Factor):
    """财务杠杆：负债 / 总资产取负。"""

    name = "low_leverage"

    def compute(self, data: MarketData) -> pd.DataFrame:
        return -_ratio(data.field("liabilities"), data.field("assets"))


@register
class EarningsSurprise(Factor):
    """盈余惊喜 SUE（季节性随机游走），用于捕捉财报后漂移 PEAD（Bernard & Thomas 1989）。

    披露日以 10-Q/10-K 的 SEC filed 日期为准，通常晚于业绩发布会，因此偏保守。
    """

    name = "sue"

    def compute(self, data: MarketData) -> pd.DataFrame:
        return data.field("sue")


@register
class SalesGrowth(Factor):
    """成长：TTM 收入同比增速。"""

    name = "sales_growth"

    def __init__(self, lag: int = 252) -> None:
        super().__init__(lag=lag)
        self.lag = lag

    def compute(self, data: MarketData) -> pd.DataFrame:
        r = data.field("revenue_ttm")
        return r / r.shift(self.lag).where(r.shift(self.lag) > 0) - 1


@register
class QualityComposite(Factor):
    """综合质量：毛利/资产、营业利润/权益、ROE、低净发行、低应计的截面 z 分数均值（Asness 等 QMJ 思路）。

    单个质量指标噪声大，合成后更稳定；缺失的分项按剩余分项平均。
    """

    name = "quality_composite"

    COMPONENTS = ("gross_profitability", "operating_profitability", "roe", "low_net_issuance", "low_accruals")

    def compute(self, data: MarketData) -> pd.DataFrame:
        from alpha.factors.base import build_factor
        from alpha.preprocess import apply_universe, winsorize_mad, zscore

        parts = []
        for name in self.COMPONENTS:
            try:
                x = build_factor(name).compute(data)
            except KeyError:
                continue
            parts.append(zscore(winsorize_mad(apply_universe(x, data.universe))))
        if not parts:
            raise KeyError("quality_composite 需要基本面字段")
        total = sum(p.fillna(0) for p in parts)
        count = sum(p.notna().astype(int) for p in parts)
        return (total / count.replace(0, np.nan)).where(count > 0)
