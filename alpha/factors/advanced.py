"""文献中的进阶价量因子。

所有因子只用 t 日收盘及以前的数据；需要“市场收益”的地方用当日股票池等权平均收益，
它在 t 日收盘时已知。参考文献附在各类的 docstring 里。
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from alpha.data.market import MarketData
from alpha.factors.base import Factor, register


def market_return(data: MarketData) -> pd.Series:
    """股票池等权日收益。"""
    return data.returns.where(data.universe).mean(axis=1)


def rolling_beta(r: pd.DataFrame, m: pd.Series, window: int) -> pd.DataFrame:
    mp = int(window * 0.8)
    cov = r.rolling(window, min_periods=mp).cov(m)
    var = m.rolling(window, min_periods=mp).var()
    return cov.div(var, axis=0)


def residual_returns(data: MarketData, beta_window: int = 252) -> pd.DataFrame:
    """r_i,s - beta_i,s * m_s。beta_i,s 用截至 s 的滚动窗口估计。"""
    r = data.returns
    m = market_return(data)
    beta = rolling_beta(r, m, beta_window)
    return r - beta.mul(m, axis=0)


@register
class High52w(Factor):
    """距 52 周高点：close / 过去 window 天最高收盘价（George & Hwang 2004）。"""

    name = "high52w"

    def __init__(self, window: int = 252) -> None:
        super().__init__(window=window)
        self.window = window

    def compute(self, data: MarketData) -> pd.DataFrame:
        c = data.close
        return c / c.rolling(self.window, min_periods=int(self.window * 0.8)).max()


@register
class ResidualMomentum(Factor):
    """残差动量：剔除市场暴露后的累计特质收益 / 特质波动（Blitz, Huij & Martens 2011）。

    比原始动量更少受市场反转冲击，回撤更小。
    """

    name = "resid_momentum"

    def __init__(self, lookback: int = 252, skip: int = 21, beta_window: int = 252) -> None:
        super().__init__(lookback=lookback, skip=skip, beta_window=beta_window)
        self.lookback, self.skip, self.beta_window = lookback, skip, beta_window

    def compute(self, data: MarketData) -> pd.DataFrame:
        e = residual_returns(data, self.beta_window).shift(self.skip)
        n = self.lookback - self.skip
        mp = int(n * 0.8)
        return e.rolling(n, min_periods=mp).sum() / e.rolling(n, min_periods=mp).std()


@register
class SharpeMomentum(Factor):
    """风险调整动量：形成期日收益均值 / 标准差（Rachev et al. 2007 等）。"""

    name = "sharpe_momentum"

    def __init__(self, lookback: int = 252, skip: int = 21) -> None:
        super().__init__(lookback=lookback, skip=skip)
        self.lookback, self.skip = lookback, skip

    def compute(self, data: MarketData) -> pd.DataFrame:
        r = data.returns.shift(self.skip)
        n = self.lookback - self.skip
        mp = int(n * 0.8)
        return r.rolling(n, min_periods=mp).mean() / r.rolling(n, min_periods=mp).std()


@register
class InfoDiscreteness(Factor):
    """信息连续性 “frog in the pan”（Da, Gurun & Warachka 2014）。

    ID = sign(形成期收益) × (%下跌日 − %上涨日)。返回 −ID：值越大，收益越是由许多
    小幅同向变动积累而成，此类动量更持久。
    """

    name = "info_continuity"

    def __init__(self, lookback: int = 252, skip: int = 21) -> None:
        super().__init__(lookback=lookback, skip=skip)
        self.lookback, self.skip = lookback, skip

    def compute(self, data: MarketData) -> pd.DataFrame:
        r = data.returns.shift(self.skip)
        n = self.lookback - self.skip
        mp = int(n * 0.8)
        valid = r.notna().astype(float).rolling(n, min_periods=mp).sum()
        pos = (r > 0).astype(float).where(r.notna()).rolling(n, min_periods=mp).sum() / valid
        neg = (r < 0).astype(float).where(r.notna()).rolling(n, min_periods=mp).sum() / valid
        pret = data.close.shift(self.skip) / data.close.shift(self.lookback) - 1
        return -(np.sign(pret) * (neg - pos))


@register
class IdioVol(Factor):
    """低特质波动：剔除市场后的残差波动取负（Ang, Hodrick, Xing & Zhang 2006）。"""

    name = "low_idio_vol"

    def __init__(self, window: int = 63, beta_window: int = 252) -> None:
        super().__init__(window=window, beta_window=beta_window)
        self.window, self.beta_window = window, beta_window

    def compute(self, data: MarketData) -> pd.DataFrame:
        e = residual_returns(data, self.beta_window)
        return -e.rolling(self.window, min_periods=int(self.window * 0.8)).std()


@register
class Skewness(Factor):
    """低偏度：日收益偏度取负，彩票型股票未来收益偏低（Boyer, Mitton & Vorkink 2010）。"""

    name = "low_skew"

    def __init__(self, window: int = 252) -> None:
        super().__init__(window=window)
        self.window = window

    def compute(self, data: MarketData) -> pd.DataFrame:
        return -data.returns.rolling(self.window, min_periods=int(self.window * 0.8)).skew()


@register
class MARatio(Factor):
    """趋势：close / window 日均线 − 1（Han, Zhou & Zhu 2016 的单均线版本）。"""

    name = "ma_ratio"

    def __init__(self, window: int = 200) -> None:
        super().__init__(window=window)
        self.window = window

    def compute(self, data: MarketData) -> pd.DataFrame:
        c = data.close
        return c / c.rolling(self.window, min_periods=int(self.window * 0.8)).mean() - 1


@register
class OvernightMomentum(Factor):
    """隔夜动量：过去 window 天隔夜收益 log(open_t / close_{t-1}) 之和（Lou, Polk & Skouras 2019）。

    需要 open。t 日开盘价在 t 日收盘前已知。
    """

    name = "overnight_momentum"

    def __init__(self, window: int = 252) -> None:
        super().__init__(window=window)
        self.window = window

    def compute(self, data: MarketData) -> pd.DataFrame:
        if data.open is None:
            raise ValueError("overnight_momentum 需要 open")
        on = np.log(data.open / data.close.shift(1))
        return on.rolling(self.window, min_periods=int(self.window * 0.8)).sum()


@register
class IntradayReversal(Factor):
    """日内收益反转：过去 window 天日内收益 log(close/open) 之和取负（Lou, Polk & Skouras 2019）。"""

    name = "intraday_reversal"

    def __init__(self, window: int = 21) -> None:
        super().__init__(window=window)
        self.window = window

    def compute(self, data: MarketData) -> pd.DataFrame:
        if data.open is None:
            raise ValueError("intraday_reversal 需要 open")
        intra = np.log(data.close / data.open)
        return -intra.rolling(self.window, min_periods=int(self.window * 0.8)).sum()


@register
class AbnormalVolume(Factor):
    """成交量异动：log(近 short 天均量 / 近 long 天均量)。方向由数据决定（关注度效应）。"""

    name = "abnormal_volume"

    def __init__(self, short: int = 21, long: int = 252) -> None:
        super().__init__(short=short, long=long)
        self.short, self.long = short, long

    def compute(self, data: MarketData) -> pd.DataFrame:
        if data.volume is None:
            raise ValueError("abnormal_volume 需要 volume")
        v = data.volume.where(data.volume > 0)
        s = v.rolling(self.short, min_periods=int(self.short * 0.8)).mean()
        l_ = v.rolling(self.long, min_periods=int(self.long * 0.8)).mean()
        return np.log(s / l_)


@register
class Seasonality(Factor):
    """收益季节性：过去 years 年中“下一个月”同月份的平均收益（Heston & Sadka 2008）。

    t 日所在月份为 M，信号是 R(M+1−12k) 的均值，k=1..years。用到的月份最晚在
    11 个月前就已结束，不含当月未完成的数据。适合月末调仓。
    """

    name = "seasonality"

    def __init__(self, years: int = 5, min_years: int = 3) -> None:
        super().__init__(years=years, min_years=min_years)
        self.years, self.min_years = years, min_years

    def compute(self, data: MarketData) -> pd.DataFrame:
        c = data.close
        per = c.index.to_period("M")
        month_close = c.groupby(per).last()
        mret = month_close / month_close.shift(1) - 1
        lags = [mret.shift(12 * k - 1) for k in range(1, self.years + 1)]
        stacked = np.stack([x.to_numpy() for x in lags])
        cnt = np.sum(np.isfinite(stacked), axis=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            avg = np.nanmean(np.where(np.isfinite(stacked), stacked, np.nan), axis=0)
        sig = pd.DataFrame(np.where(cnt >= self.min_years, avg, np.nan), index=mret.index, columns=c.columns)
        out = sig.reindex(per)
        out.index = c.index
        return out.where(c.notna())


@register
class IndustryMomentum(Factor):
    """行业动量：所属行业的等权动量（Moskowitz & Grinblatt 1999）。需要 sector。"""

    name = "industry_momentum"

    def __init__(self, lookback: int = 126, skip: int = 0) -> None:
        super().__init__(lookback=lookback, skip=skip)
        self.lookback, self.skip = lookback, skip

    def compute(self, data: MarketData) -> pd.DataFrame:
        sec = data.sector_frame()
        if sec is None:
            raise ValueError("industry_momentum 需要 sector")
        c = data.close
        mom = (c.shift(self.skip) / c.shift(self.lookback) - 1).where(data.universe)
        out = pd.DataFrame(np.nan, index=c.index, columns=c.columns)
        for g in pd.unique(sec.to_numpy().ravel()):
            if pd.isna(g):
                continue
            m = sec == g
            avg = mom.where(m).mean(axis=1)
            out = out.mask(m, pd.DataFrame(np.repeat(avg.to_numpy()[:, None], c.shape[1], 1), index=c.index, columns=c.columns))
        return out
