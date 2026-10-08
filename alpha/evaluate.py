"""单因子评估：IC、分组收益、衰减、换手。"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _align(a: pd.DataFrame, b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    a, b = a.align(b, join="inner")
    both = a.notna() & b.notna()
    return a.where(both), b.where(both)


def _row_corr(a: pd.DataFrame, b: pd.DataFrame, min_obs: int) -> pd.Series:
    a = a.sub(a.mean(axis=1), axis=0)
    b = b.sub(b.mean(axis=1), axis=0)
    num = (a * b).sum(axis=1)
    den = np.sqrt((a**2).sum(axis=1) * (b**2).sum(axis=1))
    ic = num / den.replace(0, np.nan)
    n = a.notna().sum(axis=1)
    return ic.where(n >= min_obs)


def information_coefficient(
    factor: pd.DataFrame, fwd_returns: pd.DataFrame, method: str = "spearman", min_obs: int = 20
) -> pd.Series:
    """逐日截面相关系数。method: 'spearman'（Rank IC）或 'pearson'。"""
    f, r = _align(factor, fwd_returns)
    if method == "spearman":
        f, r = f.rank(axis=1), r.rank(axis=1)
    return _row_corr(f, r, min_obs).dropna()


def ic_summary(ic: pd.Series, horizon: int = 1) -> dict[str, float]:
    """IC 统计。horizon>1 时日度 IC 序列重叠，t 值用 Newey-West 调整。"""
    ic = ic.dropna()
    n = len(ic)
    mean, std = ic.mean(), ic.std()
    se = _newey_west_se(ic, lags=max(horizon - 1, 0)) if n > 2 else np.nan
    return {
        "ic_mean": mean,
        "ic_std": std,
        "icir": mean / std if std > 0 else np.nan,
        "t_stat": mean / se if se and se > 0 else np.nan,
        "hit_rate": (ic > 0).mean(),
        "n": n,
    }


def _newey_west_se(x: pd.Series, lags: int) -> float:
    x = x.to_numpy() - x.mean()
    n = len(x)
    gamma0 = x @ x / n
    s = gamma0
    for k in range(1, min(lags, n - 1) + 1):
        w = 1 - k / (lags + 1)
        s += 2 * w * (x[k:] @ x[:-k]) / n
    return float(np.sqrt(max(s, 0) / n))


def quantile_returns(
    factor: pd.DataFrame, fwd_returns: pd.DataFrame, n_quantiles: int = 5
) -> pd.DataFrame:
    """每日按因子分 n 组，返回各组等权平均未来收益（列 1..n，n 为因子最高组）。"""
    f, r = _align(factor, fwd_returns)
    pct = f.rank(axis=1, pct=True)
    q = np.ceil(pct * n_quantiles).clip(1, n_quantiles)
    out = {}
    for k in range(1, n_quantiles + 1):
        out[k] = r.where(q == k).mean(axis=1)
    res = pd.DataFrame(out).dropna(how="all")
    res["long_short"] = res[n_quantiles] - res[1]
    return res


def ic_decay(
    factor: pd.DataFrame,
    close: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 5, 10, 21, 63),
    lag: int = 1,
    method: str = "spearman",
) -> pd.DataFrame:
    """不同持有期下的 IC，用来确定合适的调仓频率。"""
    rows = {}
    for h in horizons:
        fwd = close.shift(-(lag + h)) / close.shift(-lag) - 1
        ic = information_coefficient(factor, fwd, method=method)
        rows[h] = ic_summary(ic, horizon=h)
    return pd.DataFrame(rows).T.rename_axis("horizon")


def rank_autocorrelation(factor: pd.DataFrame, lag: int = 21) -> pd.Series:
    """因子排名的 lag 日自相关；越低意味着换手越高。"""
    r = factor.rank(axis=1)
    return _row_corr(*_align(r, r.shift(lag)), min_obs=20).dropna()


def factor_correlation(factors: dict[str, pd.DataFrame], method: str = "spearman") -> pd.DataFrame:
    """因子两两之间的平均截面相关，用于识别冗余因子。"""
    names = list(factors)
    m = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            c = information_coefficient(factors[a], factors[b], method=method).mean()
            m.loc[a, b] = m.loc[b, a] = c
    return m


def evaluate_factor(
    factor: pd.DataFrame,
    fwd_returns: pd.DataFrame,
    horizon: int,
    rebalance_dates: pd.DatetimeIndex | None = None,
    n_quantiles: int = 5,
) -> dict:
    """一站式评估。若给定 rebalance_dates，则只在调仓日取样（与实际交易一致）。"""
    f = factor if rebalance_dates is None else factor.reindex(rebalance_dates)
    r = fwd_returns if rebalance_dates is None else fwd_returns.reindex(rebalance_dates)
    eff_h = horizon if rebalance_dates is None else 1
    ic = information_coefficient(f, r)
    summary = ic_summary(ic, horizon=eff_h)
    q = quantile_returns(f, r, n_quantiles)
    summary["ls_mean"] = q["long_short"].mean()
    mono = q[list(range(1, n_quantiles + 1))].mean()
    summary["monotonicity"] = mono.rank().corr(pd.Series(range(1, n_quantiles + 1), index=mono.index))
    return {"summary": summary, "ic": ic, "quantiles": q}
