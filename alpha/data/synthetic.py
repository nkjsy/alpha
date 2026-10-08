from __future__ import annotations

import numpy as np
import pandas as pd

from alpha.data.market import MarketData


def make_synthetic_market(
    n_stocks: int = 200,
    n_days: int = 252 * 6,
    seed: int = 0,
    momentum_strength: float = 0.10,
    n_sectors: int = 8,
    start: str = "2015-01-02",
) -> MarketData:
    """生成带有已知因子结构的模拟市场，用于测试和演示。

    植入的 alpha：股票有缓慢变化的隐藏“漂移”，过去收益高的股票未来收益也偏高
    （即动量有效）。momentum_strength 是隐藏年化漂移的横截面标准差。另含市场、行业因子与特质噪声，
    并随机安排股票上市/退市以测试股票池处理。
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
    tickers = [f"S{i:04d}" for i in range(n_stocks)]
    sectors = rng.integers(0, n_sectors, n_stocks)

    # 隐藏漂移：AR(1)，持续性高，让过去收益对未来有预测力
    drift = np.zeros((n_days, n_stocks))
    phi = 0.995
    sigma_stat = momentum_strength / 252
    sigma_d = sigma_stat * np.sqrt(1 - phi**2)
    d = rng.normal(0, sigma_stat, n_stocks)
    for t in range(n_days):
        d = phi * d + rng.normal(0, sigma_d, n_stocks)
        drift[t] = d

    mkt = rng.normal(0.0003, 0.011, n_days)
    sec = rng.normal(0, 0.006, (n_days, n_sectors))
    beta = rng.uniform(0.6, 1.4, n_stocks)
    vol = rng.uniform(0.012, 0.03, n_stocks)
    idio = rng.normal(0, 1, (n_days, n_stocks)) * vol
    rets = drift + mkt[:, None] * beta + sec[:, sectors] + idio

    close = 20 * np.exp(np.cumsum(np.log1p(np.clip(rets, -0.5, 1.0)), axis=0))
    close = pd.DataFrame(close, index=dates, columns=tickers)

    # 随机上市/退市
    listed = pd.DataFrame(True, index=dates, columns=tickers)
    for j in range(n_stocks):
        r = rng.random()
        if r < 0.15:
            listed.iloc[: rng.integers(1, n_days // 2), j] = False
        elif r < 0.30:
            listed.iloc[rng.integers(n_days // 2, n_days) :, j] = False
    close = close.where(listed)

    shares = pd.Series(rng.lognormal(18, 1, n_stocks), index=tickers)
    volume = pd.DataFrame(
        rng.lognormal(0, 0.3, (n_days, n_stocks)), index=dates, columns=tickers
    ) * (shares * 0.005)
    volume = volume.where(listed)
    market_cap = close * shares

    return MarketData(
        close=close,
        open=close.shift(1) * (1 + rng.normal(0, 0.003, close.shape)),
        volume=volume,
        universe=listed,
        sector=pd.Series([f"SEC{s}" for s in sectors], index=tickers),
        fields={"market_cap": market_cap},
    )
