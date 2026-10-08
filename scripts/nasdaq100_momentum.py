"""纳指100 11-1 动量 + QQQ 均线择时：原版复现 vs 修正版。

迁移自本地 fin/main_momentum_11_1_regime_immediate.py（回测）和
fin/main_momentum_11_1_live.py（实盘信号）。逐步打开各项修正，看每一项对结果的影响：

  S1 原版回测复现    同日收盘成交、零成本、价格 ffill、每天恢复等权、两种状态都用月末排名
  S2 实盘逻辑        risk-on 时每天按最新收盘重排 Top3（与 live 脚本一致），其余同 S1
  S3 + T+1 成交      信号日下一交易日收盘成交
  S4 + 交易成本      单边 5bp；不再 ffill 价格（退市按最后价格清仓）；只在持仓变化时调仓
  S5 成本 10bp       S4 的成本敏感性
  S6 无择时基准      每月 Top10、T+1、5bp，用来判断择时是否真的有贡献

用法（需要能访问 Yahoo Finance 的网络，pip install yfinance）：
  python scripts/nasdaq100_momentum.py --membership path/to/nasdaq100_monthly_constituents_backtest_2010_2026.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from alpha.backtest import CostModel, run_backtest
from alpha.data.universe import snapshot_mask
from alpha.factors import build_factor
from alpha.data.market import MarketData
from alpha.metrics import deflated_sharpe_ratio, performance_summary
from alpha.portfolio import build_weights, rebalance_schedule
from alpha.strategies import regime_switch_targets, trend_regime

LOOKBACK, SKIP, TOP_ON, TOP_OFF, MA_DAYS = 231, 21, 3, 10, 200
ZERO_COST = CostModel(0, 0, 0, 0)


def normalize_ticker(t: str) -> str:
    return str(t).strip().upper().replace(".", "-")


def load_membership(path: str | Path, start: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["month_end_date"] = pd.to_datetime(df["month_end_date"], errors="coerce")
    df = df.dropna(subset=["month_end_date", "ticker"])
    df = df[df["month_end_date"] >= pd.Timestamp(start)].copy()
    df["ticker"] = df["ticker"].map(normalize_ticker)
    return df[["month_end_date", "ticker"]].drop_duplicates()


def download_prices(tickers: list[str], cache: Path, refresh: bool = False) -> pd.DataFrame:
    """yfinance 复权收盘价宽表，带本地缓存。"""
    if cache.exists() and not refresh:
        return pd.read_pickle(cache)
    import yfinance as yf

    frames = []
    for i in range(0, len(tickers), 50):
        chunk = tickers[i : i + 50]
        raw = yf.download(chunk, period="max", interval="1d", auto_adjust=True, progress=False, group_by="column")
        close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]].rename(columns={"Close": chunk[0]})
        frames.append(close)
        print(f"下载 {min(i + 50, len(tickers))}/{len(tickers)}")
    close = pd.concat(frames, axis=1)
    close.index = pd.to_datetime(close.index).tz_localize(None)
    close = close.loc[:, ~close.columns.duplicated()].sort_index()
    cache.parent.mkdir(parents=True, exist_ok=True)
    close.to_pickle(cache)
    return close


def coverage_report(universe_all: pd.DataFrame, close: pd.DataFrame, rebal: pd.DatetimeIndex) -> pd.Series:
    """每个调仓日，历史成分里有价格数据的比例。缺失的基本是退市/被收购的股票（幸存者偏差）。"""
    u = universe_all.reindex(index=rebal)
    has_px = close.reindex(index=rebal, columns=u.columns).notna()
    return (u & has_px).sum(axis=1) / u.sum(axis=1)


def run_scenarios(
    close: pd.DataFrame,
    bench: pd.Series,
    membership: pd.DataFrame,
    start: str = "2010-01-01",
    n_trials: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    close = close.sort_index()
    dates = close.index
    universe_all = snapshot_mask(membership, dates)  # 包含没有价格数据的成分
    universe = universe_all.reindex(columns=close.columns, fill_value=False)
    rebal = rebalance_schedule(dates, "ME")
    rebal = rebal[rebal >= pd.Timestamp(start)]
    regime = trend_regime(bench.reindex(dates).ffill(), MA_DAYS)
    mom = build_factor({"name": "momentum", "lookback": LOOKBACK + SKIP, "skip": SKIP})

    # 原版：整张价格表 ffill，退市股价格冻结
    close_ff = close.ffill()
    score_ff = mom.compute(MarketData(close=close_ff))
    score = mom.compute(MarketData(close=close))
    uni_ff = universe & close_ff.notna()
    uni = universe & close.notna()

    def targets(sc, un, on, reweight):
        return regime_switch_targets(sc, un, regime, rebal, TOP_ON, TOP_OFF, on_rerank=on, off_rerank="monthly",
                                     reweight_daily=reweight)

    c5, c10 = CostModel(1, 2, 2, 0), CostModel(2, 4, 4, 0)
    runs = {
        "S1 原版回测复现": run_backtest(targets(score_ff, uni_ff, "monthly", True), close_ff, lag=0, costs=ZERO_COST),
        "S2 实盘逻辑": run_backtest(targets(score_ff, uni_ff, "daily", True), close_ff, lag=0, costs=ZERO_COST),
        "S3 +T+1成交": run_backtest(targets(score_ff, uni_ff, "daily", True), close_ff, lag=1, costs=ZERO_COST),
    }
    t_fix = targets(score, uni, "daily", False)
    runs["S4 +成本5bp/不ffill"] = run_backtest(t_fix, close, lag=1, costs=c5)
    runs["S5 成本10bp"] = run_backtest(t_fix, close, lag=1, costs=c10)
    base = build_weights(score.where(uni), rebal, n_long=TOP_OFF)
    runs["S6 无择时Top10"] = run_backtest(base, close, lag=1, costs=c5)

    common_start = max(r.returns.index[0] for r in runs.values())
    rows, navs = {}, {}
    for name, r in runs.items():
        ret = r.returns.loc[common_start:]
        s = performance_summary(ret)
        years = len(ret) / 252
        s["annual_turnover"] = r.turnover.loc[common_start:].sum() / years
        s["annual_cost"] = r.costs.loc[common_start:].sum() / years
        s["dsr"] = deflated_sharpe_ratio(ret, n_trials)
        rows[name] = s
        navs[name] = (1 + ret).cumprod()
    qqq = bench.reindex(dates).pct_change(fill_method=None).loc[common_start:].dropna()
    rows["QQQ 买入持有"] = {**performance_summary(qqq), "annual_turnover": 0.0, "annual_cost": 0.0,
                          "dsr": deflated_sharpe_ratio(qqq, 1)}
    navs["QQQ 买入持有"] = (1 + qqq).cumprod()
    return pd.DataFrame(rows).T, pd.DataFrame(navs), coverage_report(universe_all, close, rebal)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--membership", required=True, help="nasdaq100_monthly_constituents_backtest_2010_2026.csv")
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--n-trials", type=int, default=12,
                   help="研究中实际试过的变体数（Top3/5/10、月度/即时择时、均线长度……），用于 Deflated Sharpe")
    p.add_argument("--cache", default="data/cache/nasdaq100_close.pkl")
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--out", default="output/nasdaq100_momentum")
    args = p.parse_args()

    membership = load_membership(args.membership, args.start)
    tickers = sorted(membership["ticker"].unique().tolist() + ["QQQ"])
    close = download_prices(tickers, Path(args.cache), args.refresh)
    bench = close.pop("QQQ")
    summary, navs, cov = run_scenarios(close, bench, membership, args.start, args.n_trials)

    pd.set_option("display.width", 200)
    print(f"\n成分股价格覆盖率：均值 {cov.mean():.1%}，最低 {cov.min():.1%}（{cov.idxmin():%Y-%m}）")
    missing = sorted(set(membership["ticker"]) - set(close.columns[close.notna().any()]))
    print(f"完全没有价格数据的历史成分 {len(missing)} 只：{', '.join(missing[:30])}{' ...' if len(missing) > 30 else ''}")
    print(f"\n回测区间 {navs.index[0]:%Y-%m-%d} ~ {navs.index[-1]:%Y-%m-%d}，n_trials = {args.n_trials}\n")
    fmt = summary[["ann_return", "ann_vol", "sharpe", "max_drawdown", "annual_turnover", "annual_cost", "dsr"]]
    print(fmt.round(3).to_string())

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "summary.csv")
    navs.to_csv(out / "nav.csv")
    cov.to_csv(out / "coverage.csv", header=["coverage"])
    print(f"\n结果已写入 {out}/")


if __name__ == "__main__":
    main()
