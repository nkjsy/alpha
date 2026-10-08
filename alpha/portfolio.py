"""由合成得分构建目标权重。"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rebalance_schedule(dates: pd.DatetimeIndex, freq: str = "ME", offset: int = 0) -> pd.DatetimeIndex:
    """调仓日：每个周期的最后一个交易日（freq 如 'W-FRI'、'ME'、'QE'），offset 往前挪 n 日。"""
    s = pd.Series(dates, index=dates)
    last = s.groupby(s.index.to_period(freq.replace("ME", "M").replace("QE", "Q"))).last()
    if offset:
        pos = dates.get_indexer(last.values) - offset
        last = pd.Series(dates[np.clip(pos, 0, len(dates) - 1)])
    return pd.DatetimeIndex(last.values)


def _cap_weights(w: pd.Series, max_weight: float | None) -> pd.Series:
    if max_weight is None or w.empty:
        return w
    total = w.sum()
    for _ in range(50):
        over = w > max_weight * total
        if not over.any():
            break
        excess = (w[over] - max_weight * total).sum()
        w[over] = max_weight * total
        under = ~over
        if w[under].sum() <= 0:
            break
        w[under] += excess * w[under] / w[under].sum()
    return w


def _side(scores: pd.Series, n: int, weighting: str, max_weight: float | None) -> pd.Series:
    if n <= 0 or scores.empty:
        return pd.Series(dtype=float)
    pick = scores.nlargest(n)
    if weighting == "equal":
        w = pd.Series(1.0, index=pick.index)
    elif weighting == "score":
        w = pick - pick.min() + 1e-6
    elif weighting == "inv_vol":
        raise ValueError("inv_vol 需要通过 build_weights 的 vol 参数传入")
    else:
        raise ValueError(f"未知 weighting {weighting!r}")
    w = _cap_weights(w / w.sum(), max_weight)
    return w / w.sum()


def build_weights(
    score: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    n_long: int | float = 0.2,
    n_short: int | float = 0,
    weighting: str = "equal",
    max_weight: float | None = None,
    vol: pd.DataFrame | None = None,
    long_gross: float = 1.0,
    short_gross: float = 1.0,
) -> pd.DataFrame:
    """在每个调仓日选股并给出目标权重。

    n_long / n_short  整数为股票只数，小于 1 的小数为截面比例
    weighting         equal / score / inv_vol（需传 vol，如滚动波动率）
    返回宽表：多头权重和 = long_gross，空头权重和 = -short_gross。
    """
    out = pd.DataFrame(0.0, index=rebalance_dates, columns=score.columns)
    for d in rebalance_dates:
        s = score.loc[d].dropna()
        if s.empty:
            continue
        nl = int(round(n_long * len(s))) if 0 < n_long < 1 else int(n_long)
        ns = int(round(n_short * len(s))) if 0 < n_short < 1 else int(n_short)
        if weighting == "inv_vol":
            v = vol.loc[d].reindex(s.index)
            lw = (1 / v[s.nlargest(nl).index]).dropna()
            sw = (1 / v[s.nsmallest(ns).index]).dropna() if ns else pd.Series(dtype=float)
            lw = _cap_weights(lw / lw.sum(), max_weight) if len(lw) else lw
            sw = _cap_weights(sw / sw.sum(), max_weight) if len(sw) else sw
            lw = lw / lw.sum() if len(lw) else lw
            sw = sw / sw.sum() if len(sw) else sw
        else:
            lw = _side(s, nl, weighting, max_weight)
            sw = _side(-s, ns, weighting, max_weight)
        if len(lw):
            out.loc[d, lw.index] = lw * long_gross
        if len(sw):
            out.loc[d, sw.index] -= sw * short_gross
    return out
