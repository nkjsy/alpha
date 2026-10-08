"""绩效指标与多重检验校正。"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def performance_summary(returns: pd.Series, periods_per_year: int = 252) -> dict[str, float]:
    r = returns.dropna()
    if len(r) < 2:
        return {k: np.nan for k in ("ann_return", "ann_vol", "sharpe", "sortino", "max_drawdown", "calmar")}
    nav = (1 + r).cumprod()
    years = len(r) / periods_per_year
    ann_ret = nav.iloc[-1] ** (1 / years) - 1 if nav.iloc[-1] > 0 else -1.0
    ann_vol = r.std() * np.sqrt(periods_per_year)
    downside = r[r < 0].std() * np.sqrt(periods_per_year)
    dd = nav / nav.cummax() - 1
    mdd = dd.min()
    return {
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": r.mean() / r.std() * np.sqrt(periods_per_year) if r.std() > 0 else np.nan,
        "sortino": r.mean() * periods_per_year / downside if downside > 0 else np.nan,
        "max_drawdown": mdd,
        "calmar": ann_ret / -mdd if mdd < 0 else np.nan,
    }


def probabilistic_sharpe_ratio(returns: pd.Series, sr_benchmark: float = 0.0) -> float:
    """PSR（Bailey & López de Prado 2012）：真实夏普 > 基准的概率。

    sr_benchmark 为与 returns 同频（非年化）的夏普。
    """
    r = returns.dropna()
    n = len(r)
    sr = r.mean() / r.std()
    skew = stats.skew(r)
    kurt = stats.kurtosis(r, fisher=False)
    denom = np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr**2)
    return float(stats.norm.cdf((sr - sr_benchmark) * np.sqrt(n - 1) / denom))


def expected_max_sharpe(n_trials: int, sr_std: float) -> float:
    """在 n_trials 个真实夏普为 0 的策略中，样本夏普最大值的期望。"""
    if n_trials <= 1:
        return 0.0
    g = 0.5772156649
    z1 = stats.norm.ppf(1 - 1 / n_trials)
    z2 = stats.norm.ppf(1 - 1 / (n_trials * np.e))
    return float(sr_std * ((1 - g) * z1 + g * z2))


def deflated_sharpe_ratio(returns: pd.Series, n_trials: int, trial_sharpes: list[float] | None = None) -> float:
    """DSR（Bailey & López de Prado 2014）：校正“试了很多变体挑最好的”带来的选择偏差。

    n_trials       研究过程中实际尝试过的变体数（参数、因子组合、调仓频率……都算）
    trial_sharpes  各变体的（非年化）夏普，用于估计其离散程度；缺省用 1/sqrt(T) 近似
    返回值是概率，通常要求 > 0.95 才认为策略不是过拟合的产物。
    """
    r = returns.dropna()
    if trial_sharpes is not None and len(trial_sharpes) > 1:
        sr_std = float(np.std(trial_sharpes, ddof=1))
    else:
        sr_std = 1 / np.sqrt(len(r))
    sr0 = expected_max_sharpe(n_trials, sr_std)
    return probabilistic_sharpe_ratio(r, sr0)


def bootstrap_sharpe_ci(
    returns: pd.Series, n_boot: int = 2000, block: int = 21, alpha: float = 0.05, seed: int = 0,
    periods_per_year: int = 252,
) -> tuple[float, float]:
    """块自助法给出年化夏普的置信区间（保留自相关结构）。"""
    r = returns.dropna().to_numpy()
    n = len(r)
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    out = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n - block + 1, n_blocks)
        sample = np.concatenate([r[s : s + block] for s in starts])[:n]
        sd = sample.std(ddof=1)
        out[i] = sample.mean() / sd * np.sqrt(periods_per_year) if sd > 0 else np.nan
    lo, hi = np.nanpercentile(out, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)
