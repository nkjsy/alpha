"""批量因子挖掘：在研究样本上评估一组候选因子，做多重检验校正，按增量贡献贪心挑选。

流程：
1. 每个候选因子（含不同参数）都经过同样的预处理，在“研究期调仓日”上计算 Rank IC；
2. 对已有的基准因子（如现有的动量）做截面正交化，得到增量 IC：只有基准解释不了的部分才算新信息；
3. 对所有候选的 p 值做 Benjamini-Hochberg 校正，同时给出 Harvey-Liu-Zhu 建议的 |t|>3 门槛；
4. 按增量 t 值从高到低贪心入选，跳过与已入选因子相关性过高的候选。

每个候选都记一次试验（ExperimentLog），参数网格越大，后续 Deflated Sharpe 惩罚越重。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from alpha.data.market import MarketData
from alpha.evaluate import information_coefficient, quantile_returns, rank_autocorrelation
from alpha.factors import build_factor
from alpha.preprocess import neutralize, standard_pipeline, zscore


@dataclass
class MiningResult:
    table: pd.DataFrame                      # 每个候选一行，按 |t_incr| 排序
    factors: dict[str, pd.DataFrame]         # 预处理后的候选因子（全样本）
    selected: list[str] = field(default_factory=list)
    corr: pd.DataFrame | None = None         # 候选两两平均截面相关（研究期）


def _t_stat(x: pd.Series) -> float:
    x = x.dropna()
    if len(x) < 3 or x.std() == 0:
        return np.nan
    return float(x.mean() / x.std() * np.sqrt(len(x)))


def benjamini_hochberg(p: pd.Series) -> pd.Series:
    """BH 校正后的 q 值（控制错误发现率）。"""
    p = p.dropna()
    n = len(p)
    if n == 0:
        return p
    order = p.sort_values()
    ranks = np.arange(1, n + 1)
    q = (order.to_numpy() * n / ranks)
    q = np.minimum.accumulate(q[::-1])[::-1].clip(max=1.0)
    return pd.Series(q, index=order.index).reindex(p.index)


def build_candidates(specs: list[str | dict], data: MarketData) -> dict[str, pd.DataFrame]:
    """计算原始因子值；缺数据的因子（如没有 open）跳过并提示。"""
    out = {}
    for spec in specs:
        f = build_factor(spec)
        try:
            out[f.label] = f.compute(data)
        except (ValueError, KeyError) as e:
            print(f"跳过 {f.label}：{e}")
    return out


def mine_factors(
    data: MarketData,
    candidates: dict[str, pd.DataFrame],
    research_dates: pd.DatetimeIndex,
    fwd_returns: pd.DataFrame,
    baseline: dict[str, pd.DataFrame] | None = None,
    universe: pd.DataFrame | None = None,
    groups: pd.DataFrame | None = None,
    n_quantiles: int = 5,
    max_corr: float = 0.6,
    min_t: float = 3.0,
    periods_per_year: int = 12,
) -> MiningResult:
    """评估并挑选因子。

    research_dates  研究期调仓日（未来收益窗口须在研究期内，见 pipeline 的隔离处理）
    fwd_returns     与调仓周期匹配的未来收益（如 horizon=21、lag=1）
    baseline        已有因子（原始值），增量 IC 相对它们计算；它们本身不参与挑选
    min_t / max_corr 入选门槛：增量 |t| 下限与和已入选因子的最大相关
    """
    universe = universe if universe is not None else data.universe
    proc = {k: standard_pipeline(v, universe, groups=groups) for k, v in candidates.items()}
    base = {k: standard_pipeline(v, universe, groups=groups) for k, v in (baseline or {}).items()}
    fr = fwd_returns.reindex(research_dates)

    rows = {}
    for name, x in proc.items():
        xr = x.reindex(research_dates)
        ic = information_coefficient(xr, fr)
        q = quantile_returns(xr, fr, n_quantiles)
        mono = q[list(range(1, n_quantiles + 1))].mean()
        half = len(ic) // 2
        row = {
            "ic_mean": ic.mean(),
            "icir": ic.mean() / ic.std() if ic.std() > 0 else np.nan,
            "t_stat": _t_stat(ic),
            "ic_first_half": ic.iloc[:half].mean(),
            "ic_second_half": ic.iloc[half:].mean(),
            "ls_ann": q["long_short"].mean() * periods_per_year,
            "monotonicity": mono.rank().corr(pd.Series(range(1, n_quantiles + 1), index=mono.index)),
            "rank_autocorr": rank_autocorrelation(xr, lag=1).mean(),
            "coverage": (xr.notna() & universe.reindex(research_dates).fillna(False)).sum(axis=1).mean()
            / universe.reindex(research_dates).sum(axis=1).mean(),
            "n": len(ic),
        }
        if base:
            # 对基准因子做截面回归取残差，残差的 IC 即增量信息
            resid = zscore(neutralize(xr, exposures={k: v.reindex(research_dates) for k, v in base.items()}))
            ic_incr = information_coefficient(resid, fr)
            row["ic_incr"] = ic_incr.mean()
            row["t_incr"] = _t_stat(ic_incr)
            row["incr_first_half"] = ic_incr.iloc[: len(ic_incr) // 2].mean()
            row["incr_second_half"] = ic_incr.iloc[len(ic_incr) // 2 :].mean()
            row["corr_baseline"] = max(
                abs(information_coefficient(xr, b.reindex(research_dates)).mean()) for b in base.values()
            )
        else:
            row["ic_incr"], row["t_incr"], row["corr_baseline"] = row["ic_mean"], row["t_stat"], np.nan
            row["incr_first_half"], row["incr_second_half"] = row["ic_first_half"], row["ic_second_half"]
        rows[name] = row

    table = pd.DataFrame(rows).T
    table["p_value"] = 2 * stats.norm.sf(table["t_incr"].abs().astype(float))
    table["q_bh"] = benjamini_hochberg(table["p_value"])
    # 前后两半研究期的增量 IC 同号，才认为不是某一段行情偶然造成的
    table["sign_stable"] = np.sign(table["incr_first_half"]) == np.sign(table["incr_second_half"])
    table = table.reindex(table["t_incr"].abs().sort_values(ascending=False).index)

    corr = pd.DataFrame(np.eye(len(proc)), index=list(proc), columns=list(proc))
    names = list(proc)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            c = information_coefficient(proc[a].reindex(research_dates), proc[b].reindex(research_dates)).mean()
            corr.loc[a, b] = corr.loc[b, a] = c

    selected: list[str] = []
    for name, row in table.iterrows():
        if not (abs(row["t_incr"]) >= min_t and row["q_bh"] <= 0.05 and row["sign_stable"]):
            continue
        if any(abs(corr.loc[name, s]) > max_corr for s in selected):
            continue
        selected.append(name)
    table["selected"] = table.index.isin(selected)
    return MiningResult(table=table, factors=proc, selected=selected, corr=corr)


def oriented(factors: dict[str, pd.DataFrame], table: pd.DataFrame, names: list[str]) -> dict[str, pd.DataFrame]:
    """按研究期增量 IC 的符号调整方向（负的取反），供合成使用。"""
    return {n: factors[n] * (1 if table.loc[n, "ic_incr"] >= 0 else -1) for n in names}
