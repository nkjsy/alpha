"""配置驱动的端到端研究流程：因子 -> 预处理 -> 评估 -> 合成 -> 组合 -> 回测 -> 报告。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from alpha.backtest import BacktestResult, CostModel, run_backtest
from alpha.combine import combine
from alpha.data.loaders import market_from_long_csv, market_from_yfinance
from alpha.data.market import MarketData
from alpha.data.synthetic import make_synthetic_market
from alpha.data.universe import liquidity_mask
from alpha.evaluate import evaluate_factor, factor_correlation
from alpha.factors import build_factor
from alpha.metrics import bootstrap_sharpe_ci, deflated_sharpe_ratio
from alpha.portfolio import build_weights, rebalance_schedule
from alpha.preprocess import standard_pipeline, zscore
from alpha.validation import ExperimentLog


@dataclass
class ResearchResult:
    factor_summary: pd.DataFrame
    factor_corr: pd.DataFrame
    combo_weights: pd.DataFrame | None
    backtest: BacktestResult
    oos_backtest: BacktestResult | None
    stats: dict[str, Any] = field(default_factory=dict)
    score: pd.DataFrame | None = None
    targets: pd.DataFrame | None = None


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_data(cfg: dict) -> MarketData:
    src = cfg.get("source", "synthetic")
    if src == "synthetic":
        return make_synthetic_market(**cfg.get("synthetic", {}))
    if src == "csv":
        return market_from_long_csv(cfg["prices"], cfg.get("membership"), cfg.get("sectors"))
    if src == "yfinance":
        return market_from_yfinance(cfg["tickers"], cfg["start"], cfg.get("end"), membership_csv=cfg.get("membership"))
    raise ValueError(f"未知数据源 {src!r}")


def run_research(cfg: dict, data: MarketData | None = None, out_dir: str | Path | None = None) -> ResearchResult:
    data = data if data is not None else load_data(cfg.get("data", {}))

    uni_cfg = cfg.get("universe", {})
    universe = data.universe
    if uni_cfg.get("min_dollar_volume") and data.dollar_volume is not None:
        universe = universe & liquidity_mask(data.dollar_volume, min_dollar_volume=uni_cfg["min_dollar_volume"])

    # 样本外留出：研究阶段的选择全部基于 holdout_start 之前
    holdout = cfg.get("holdout_start")
    holdout_ts = pd.Timestamp(holdout) if holdout else None

    pp = cfg.get("preprocess", {})
    groups = data.sector_frame() if pp.get("neutralize_sector") else None
    exposures = None
    if pp.get("neutralize_size") and "market_cap" in data.fields:
        exposures = {"log_mcap": zscore(np.log(data.fields["market_cap"].where(universe)))}

    pf = cfg.get("portfolio", {})
    horizon = int(pf.get("horizon", 21))
    lag = int(cfg.get("execution", {}).get("lag", 1))
    rebal = rebalance_schedule(data.dates, pf.get("freq", "ME"))
    fwd = data.forward_returns(horizon=horizon, lag=lag)
    # 研究期调仓日：未来收益窗口（t+lag 到 t+lag+horizon）必须整段落在留出期之前
    if holdout_ts is not None:
        first_holdout = data.dates.searchsorted(holdout_ts)
        end_pos = data.dates.get_indexer(rebal) + lag + horizon
        res_dates = rebal[end_pos < first_holdout]
    else:
        res_dates = rebal

    factors: dict[str, pd.DataFrame] = {}
    rows = {}
    for spec in cfg["factors"]:
        f = build_factor(spec)
        raw = f.compute(data)
        x = standard_pipeline(
            raw, universe, winsor=pp.get("winsorize", "mad"), normalize=pp.get("normalize", "zscore"),
            groups=groups, exposures=exposures,
        )
        factors[f.label] = x
        ev = evaluate_factor(x, fwd, horizon, rebalance_dates=res_dates, n_quantiles=pf.get("n_quantiles", 5))
        rows[f.label] = ev["summary"]

    factor_summary = pd.DataFrame(rows).T
    corr = factor_correlation({k: v.reindex(res_dates) for k, v in factors.items()})

    comb_cfg = cfg.get("combine", {})
    score, w = combine(
        factors, method=comb_cfg.get("method", "equal"), fwd_returns=fwd, horizon=horizon, lag=lag,
        window=comb_cfg.get("window", 252), min_periods=comb_cfg.get("min_periods", 126),
    )

    targets = build_weights(
        score, rebal, n_long=pf.get("n_long", 0.2), n_short=pf.get("n_short", 0),
        weighting=pf.get("weighting", "equal"), max_weight=pf.get("max_weight"),
    )
    cost = CostModel(**cfg.get("costs", {}))
    bt = run_backtest(targets, data.close, lag=lag, costs=cost)

    oos = None
    if holdout_ts is not None:
        oos_targets = targets[targets.index >= holdout_ts]
        if len(oos_targets):
            oos = run_backtest(oos_targets, data.close, lag=lag, costs=cost)
        # 样本内回测的价格也必须截止在留出期之前，否则最后一次调仓后的持仓会一直漂移进留出期
        close_is = data.close[data.close.index < holdout_ts]
        is_targets = targets[targets.index < holdout_ts]
        is_targets = is_targets[data.dates.get_indexer(is_targets.index) + lag < len(close_is)]
        bt_is = run_backtest(is_targets, close_is, lag=lag, costs=cost)
    else:
        bt_is = bt

    log = ExperimentLog(Path(out_dir or cfg.get("output_dir", "output")) / "experiments.jsonl")
    is_summary = bt_is.summary()
    log.log(cfg.get("name", "experiment"), cfg, is_summary)
    n_trials = log.n_trials()
    stats = {
        "in_sample": is_summary,
        "n_trials": n_trials,
        "deflated_sharpe": deflated_sharpe_ratio(bt_is.returns, n_trials, log.trial_sharpes() or None),
        "sharpe_ci_95": bootstrap_sharpe_ci(bt_is.returns, n_boot=500),
    }
    if oos is not None:
        stats["out_of_sample"] = oos.summary()

    result = ResearchResult(factor_summary, corr, w, bt, oos, stats, score=score, targets=targets)
    if out_dir or cfg.get("output_dir"):
        write_report(result, Path(out_dir or cfg["output_dir"]))
    return result


def _fmt(d: dict) -> str:
    return "\n".join(f"  {k:<18}{v: .4f}" if isinstance(v, (int, float, np.floating)) else f"  {k:<18}{v}" for k, v in d.items())


def format_report(r: ResearchResult) -> str:
    lines = ["== 单因子评估（研究样本，调仓日取样） ==", r.factor_summary.round(4).to_string(), ""]
    lines += ["== 因子相关性 ==", r.factor_corr.round(2).to_string(), ""]
    lines += ["== 组合回测（样本内，扣成本） ==", _fmt(r.stats["in_sample"]), ""]
    lines += [
        f"累计试验次数 n_trials = {r.stats['n_trials']}",
        f"Deflated Sharpe（>0.95 才算显著）= {r.stats['deflated_sharpe']:.3f}",
        f"夏普 95% 块自助置信区间 = ({r.stats['sharpe_ci_95'][0]:.2f}, {r.stats['sharpe_ci_95'][1]:.2f})",
        "",
    ]
    if "out_of_sample" in r.stats:
        lines += ["== 留出样本（只看一次！） ==", _fmt(r.stats["out_of_sample"]), ""]
    return "\n".join(lines)


def write_report(r: ResearchResult, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    r.factor_summary.to_csv(out / "factor_summary.csv")
    r.factor_corr.to_csv(out / "factor_corr.csv")
    if r.combo_weights is not None:
        r.combo_weights.to_csv(out / "combo_weights.csv")
    pd.DataFrame({"net": r.backtest.returns, "gross": r.backtest.gross_returns, "cost": r.backtest.costs,
                  "nav": r.backtest.nav}).to_csv(out / "backtest_daily.csv")
    r.backtest.turnover.to_csv(out / "turnover.csv", header=["turnover"])
    (out / "report.txt").write_text(format_report(r), encoding="utf-8")
