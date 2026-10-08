"""基于价量的经典因子。所有窗口单位为交易日。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha.data.market import MarketData
from alpha.factors.base import Factor, register


@register
class Momentum(Factor):
    """动量：过去 lookback 天到过去 skip 天的累计收益。

    默认 252/21 即经典的 12-1 月动量；lookback=231, skip=0 等可复现 11-1 之类的变体。
    """

    name = "momentum"

    def __init__(self, lookback: int = 252, skip: int = 21) -> None:
        super().__init__(lookback=lookback, skip=skip)
        if skip >= lookback:
            raise ValueError("skip 必须小于 lookback")
        self.lookback, self.skip = lookback, skip

    def compute(self, data: MarketData) -> pd.DataFrame:
        c = data.close
        return c.shift(self.skip) / c.shift(self.lookback) - 1.0


@register
class Reversal(Factor):
    """短期反转：过去 window 天收益取负。"""

    name = "reversal"

    def __init__(self, window: int = 21) -> None:
        super().__init__(window=window)
        self.window = window

    def compute(self, data: MarketData) -> pd.DataFrame:
        return -(data.close / data.close.shift(self.window) - 1.0)


@register
class LowVolatility(Factor):
    """低波动：过去 window 天日收益标准差取负。"""

    name = "low_vol"

    def __init__(self, window: int = 63) -> None:
        super().__init__(window=window)
        self.window = window

    def compute(self, data: MarketData) -> pd.DataFrame:
        return -data.returns.rolling(self.window, min_periods=int(self.window * 0.8)).std()


@register
class MaxReturn(Factor):
    """彩票效应：过去 window 天最大单日收益取负（Bali, Cakici & Whitelaw 2011）。"""

    name = "max_ret"

    def __init__(self, window: int = 21) -> None:
        super().__init__(window=window)
        self.window = window

    def compute(self, data: MarketData) -> pd.DataFrame:
        return -data.returns.rolling(self.window, min_periods=int(self.window * 0.8)).max()


@register
class Size(Factor):
    """规模：-log(市值)。没有 market_cap 字段时用平均成交额做代理。"""

    name = "size"

    def __init__(self, proxy_window: int = 63) -> None:
        super().__init__(proxy_window=proxy_window)
        self.proxy_window = proxy_window

    def compute(self, data: MarketData) -> pd.DataFrame:
        if "market_cap" in data.fields:
            mc = data.fields["market_cap"]
        else:
            dv = data.dollar_volume
            if dv is None:
                raise ValueError("size 因子需要 market_cap 或 volume")
            mc = dv.rolling(self.proxy_window, min_periods=self.proxy_window // 2).mean()
        return -np.log(mc.where(mc > 0))


@register
class Amihud(Factor):
    """Amihud 非流动性：mean(|r| / 成交额)，越不流动值越大（流动性溢价）。"""

    name = "amihud"

    def __init__(self, window: int = 21) -> None:
        super().__init__(window=window)
        self.window = window

    def compute(self, data: MarketData) -> pd.DataFrame:
        dv = data.dollar_volume
        if dv is None:
            raise ValueError("amihud 因子需要 volume")
        illiq = data.returns.abs() / dv.where(dv > 0)
        return np.log(illiq.rolling(self.window, min_periods=self.window // 2).mean() * 1e9)


@register
class Beta(Factor):
    """低贝塔：对等权市场收益的滚动贝塔取负（Frazzini & Pedersen BAB 思路）。"""

    name = "low_beta"

    def __init__(self, window: int = 252) -> None:
        super().__init__(window=window)
        self.window = window

    def compute(self, data: MarketData) -> pd.DataFrame:
        r = data.returns.where(data.universe)
        m = r.mean(axis=1)
        mp = int(self.window * 0.8)
        cov = r.rolling(self.window, min_periods=mp).cov(m)
        var = m.rolling(self.window, min_periods=mp).var()
        return -cov.div(var, axis=0)
