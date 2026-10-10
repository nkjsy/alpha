"""横截面预处理：股票池掩码、去极值、标准化、中性化。

所有函数逐日（逐行）处理，只在当天截面内使用信息，不会产生时间维度的前视。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def apply_universe(df: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    """池外股票置为 NaN。必须在其它截面处理之前调用。"""
    return df.where(universe.reindex_like(df).fillna(False).astype(bool))


def winsorize_mad(df: pd.DataFrame, n: float = 5.0) -> pd.DataFrame:
    """中位数绝对偏差去极值：截断到 median ± n * 1.4826 * MAD。

    超过一半取同一个值时 MAD 为 0（如大多数股票没有内部人买入），此时不截断，否则整个截面会被压成常数。
    """
    med = df.median(axis=1)
    mad = (df.sub(med, axis=0).abs().median(axis=1) * 1.4826).replace(0, np.nan)
    lo, hi = (med - n * mad).fillna(-np.inf), (med + n * mad).fillna(np.inf)
    return df.clip(lower=lo, upper=hi, axis=0)


def winsorize_quantile(df: pd.DataFrame, lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
    lo = df.quantile(lower, axis=1)
    hi = df.quantile(upper, axis=1)
    return df.clip(lower=lo, upper=hi, axis=0)


def zscore(df: pd.DataFrame) -> pd.DataFrame:
    mu = df.mean(axis=1)
    sd = df.std(axis=1).replace(0, np.nan)
    return df.sub(mu, axis=0).div(sd, axis=0)


def rank_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """截面排序后映射到 [-0.5, 0.5]，对异常值最稳健。"""
    r = df.rank(axis=1, pct=True)
    n = df.notna().sum(axis=1)
    # 把 (0,1] 平移居中，使均值为 0
    return r.sub((n + 1) / (2 * n), axis=0)


def neutralize(
    df: pd.DataFrame,
    groups: pd.DataFrame | None = None,
    exposures: dict[str, pd.DataFrame] | None = None,
    min_obs: int = 10,
) -> pd.DataFrame:
    """逐日 OLS 取残差：df ~ 行业哑变量 + 连续暴露（如 log 市值）。

    groups    行业宽表（字符串），为空时只回归连续暴露（含截距）
    exposures {名称: 宽表}，建议先做 zscore
    """
    exposures = exposures or {}
    out = pd.DataFrame(np.nan, index=df.index, columns=df.columns)
    expo_arrays = {k: v.reindex_like(df).to_numpy(dtype=float) for k, v in exposures.items()}
    grp = groups.reindex_like(df).to_numpy(dtype=object) if groups is not None else None
    y_all = df.to_numpy(dtype=float)

    for i in range(len(df)):
        y = y_all[i]
        ok = np.isfinite(y)
        for arr in expo_arrays.values():
            ok &= np.isfinite(arr[i])
        if grp is not None:
            g = grp[i]
            ok &= pd.notna(g)
        if ok.sum() < min_obs:
            continue
        X = [np.ones(ok.sum())]
        for arr in expo_arrays.values():
            X.append(arr[i][ok])
        if grp is not None:
            dummies = pd.get_dummies(pd.Series(g[ok]), drop_first=True, dtype=float)
            if dummies.shape[1]:
                X.extend(dummies.to_numpy().T)
        X = np.column_stack(X)
        beta, *_ = np.linalg.lstsq(X, y[ok], rcond=None)
        resid = np.full_like(y, np.nan)
        resid[ok] = y[ok] - X @ beta
        out.iloc[i] = resid
    return out


def fill_missing(df: pd.DataFrame, universe: pd.DataFrame, value: float = 0.0) -> pd.DataFrame:
    """池内但因子缺失的股票填一个中性值（标准化后用 0）。"""
    u = universe.reindex_like(df).fillna(False).astype(bool)
    return df.where(~u | df.notna(), value).where(u)


def group_demean(df: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    """逐日减去所在行业的均值（行业中性），无行业的股票置为 NaN。"""
    g = groups.reindex_like(df)
    out = pd.DataFrame(np.nan, index=df.index, columns=df.columns)
    for val in pd.unique(g.to_numpy().ravel()):
        if pd.isna(val):
            continue
        m = g == val
        x = df.where(m)
        out = out.where(~m, x.sub(x.mean(axis=1), axis=0))
    return out


def standard_pipeline(
    raw: pd.DataFrame,
    universe: pd.DataFrame,
    winsor: str | None = "mad",
    normalize: str = "zscore",
    groups: pd.DataFrame | None = None,
    exposures: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """常用组合：股票池 -> 去极值 -> 标准化 -> （中性化 -> 再标准化）。"""
    x = apply_universe(raw, universe)
    if winsor == "mad":
        x = winsorize_mad(x)
    elif winsor == "quantile":
        x = winsorize_quantile(x)
    x = rank_normalize(x) if normalize == "rank" else zscore(x)
    if groups is not None or exposures:
        x = zscore(neutralize(x, groups=groups, exposures=exposures))
    return x
