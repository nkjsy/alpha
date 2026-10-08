"""多因子合成。

关键：基于 IC 的加权只能使用“已经实现”的 IC。t 日信号对应的 IC 要到
t + lag + horizon 日收盘才知道，因此权重必须相应滞后。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha.preprocess import zscore


def equal_weight(factors: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """等权合成（因子需已标准化）。缺失视为 0，全部缺失保持 NaN。"""
    stack = list(factors.values())
    total = sum(f.fillna(0) for f in stack)
    count = sum(f.notna().astype(int) for f in stack)
    return zscore(total.where(count > 0))


def weighted(factors: dict[str, pd.DataFrame], weights: pd.DataFrame) -> pd.DataFrame:
    """按时变权重合成。weights: index 日期，columns 因子名。"""
    idx = next(iter(factors.values())).index
    w = weights.reindex(idx).ffill()
    total = None
    count = None
    for name, f in factors.items():
        term = f.fillna(0).mul(w[name], axis=0)
        total = term if total is None else total + term
        c = f.notna().astype(int)
        count = c if count is None else count + c
    return zscore(total.where(count > 0))


def realized_ic_frame(
    factors: dict[str, pd.DataFrame], fwd_returns: pd.DataFrame, method: str = "spearman"
) -> pd.DataFrame:
    from alpha.evaluate import information_coefficient

    return pd.DataFrame({k: information_coefficient(v, fwd_returns, method=method) for k, v in factors.items()})


def ic_weights(
    ic: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    horizon: int,
    lag: int = 1,
    window: int = 252,
    min_periods: int = 126,
    mode: str = "ic",
    long_only_weights: bool = False,
) -> pd.DataFrame:
    """根据历史 IC 计算因子权重，已经处理好可获得时间。

    ic        日度 IC（index 为信号日），来自 realized_ic_frame
    calendar  完整交易日历
    mode      'ic'：权重 ∝ 滚动平均 IC；'icir'：权重 ∝ 滚动 IC 均值 / 标准差
    返回的权重 w.loc[t] 只依赖于在 t 日收盘前已经实现的 IC。
    """
    ic = ic.reindex(calendar)
    # 信号日 s 的 IC 在 s + lag + horizon 日收盘实现，t 日可用的最晚信号日为 t - lag - horizon
    known = ic.shift(lag + horizon)
    mean = known.rolling(window, min_periods=min_periods).mean()
    if mode == "icir":
        std = known.rolling(window, min_periods=min_periods).std()
        raw = mean / std.replace(0, np.nan)
    elif mode == "ic":
        raw = mean
    else:
        raise ValueError(f"未知 mode {mode!r}")
    if long_only_weights:
        raw = raw.clip(lower=0)
    denom = raw.abs().sum(axis=1).replace(0, np.nan)
    return raw.div(denom, axis=0)


def combine(
    factors: dict[str, pd.DataFrame],
    method: str = "equal",
    fwd_returns: pd.DataFrame | None = None,
    horizon: int = 21,
    lag: int = 1,
    window: int = 252,
    min_periods: int = 126,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """返回 (合成因子, 权重)。method: equal / ic / icir。"""
    if method == "equal" or len(factors) == 1:
        return equal_weight(factors), None
    if fwd_returns is None:
        raise ValueError(f"{method} 合成需要 fwd_returns")
    calendar = next(iter(factors.values())).index
    ic = realized_ic_frame(factors, fwd_returns)
    w = ic_weights(ic, calendar, horizon, lag, window, min_periods, mode=method)
    # 权重尚不可用的早期用等权补齐，避免样本前段丢失
    eq = pd.DataFrame(1.0 / len(factors), index=calendar, columns=list(factors))
    w = w.where(w.notna().all(axis=1), eq, axis=0) if not w.empty else eq
    return weighted(factors, w), w
