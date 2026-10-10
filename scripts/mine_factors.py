"""纳指100历史成分上的因子挖掘（价量 + SEC 基本面）。

研究期（holdout_start 之前）上批量评估候选因子，相对现有 11-1 动量计算增量 IC，
经 BH 多重检验校正与相关性去重后挑选；最后只在留出期检验一次入选因子与组合。

用法（需要能访问 Yahoo Finance 和 sec.gov，pip install yfinance）：
  python scripts/mine_factors.py --membership C:/money/fin/nasdaq/nasdaq100_monthly_constituents_backtest_2010_2026.csv \
      --sec-user-agent "你的名字 你的邮箱"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from alpha.backtest import CostModel, run_backtest
from alpha.combine import equal_weight
from alpha.data.loaders import download_yahoo_ohlcv, market_from_yahoo_raw
from alpha.data.market import MarketData
from alpha.data.sec import build_fields, fetch_records
from alpha.data.universe import snapshot_mask
from alpha.evaluate import information_coefficient
from alpha.metrics import deflated_sharpe_ratio, performance_summary
from alpha.mining import build_candidates, mine_factors, oriented
from alpha.portfolio import build_weights, rebalance_schedule
from alpha.preprocess import standard_pipeline
from alpha.validation import ExperimentLog

BASELINE = {"name": "momentum", "lookback": 252, "skip": 21}  # 现有策略的 11-1 动量

CANDIDATES: list[dict] = [
    {"name": "momentum", "lookback": 126, "skip": 21},
    {"name": "momentum", "lookback": 252, "skip": 126},   # 中期动量（Novy-Marx 2012）
    {"name": "resid_momentum", "lookback": 252, "skip": 21},
    {"name": "resid_momentum", "lookback": 126, "skip": 21},
    {"name": "sharpe_momentum", "lookback": 252, "skip": 21},
    {"name": "info_continuity", "lookback": 252, "skip": 21},
    {"name": "high52w", "window": 252},
    {"name": "ma_ratio", "window": 50},
    {"name": "ma_ratio", "window": 200},
    {"name": "reversal", "window": 5},
    {"name": "reversal", "window": 21},
    {"name": "intraday_reversal", "window": 21},
    {"name": "overnight_momentum", "window": 21},
    {"name": "overnight_momentum", "window": 252},
    {"name": "low_vol", "window": 21},
    {"name": "low_vol", "window": 63},
    {"name": "low_idio_vol", "window": 21},
    {"name": "low_idio_vol", "window": 63},
    {"name": "low_beta", "window": 252},
    {"name": "max_ret", "window": 21},
    {"name": "low_skew", "window": 252},
    {"name": "abnormal_volume", "short": 21, "long": 252},
    {"name": "amihud", "window": 21},
    {"name": "size"},
    {"name": "seasonality", "years": 5, "min_years": 3},
    {"name": "seasonality", "years": 10, "min_years": 5},
    # 基本面（SEC XBRL，按披露日对齐）
    {"name": "book_to_market"},
    {"name": "earnings_yield"},
    {"name": "sales_to_price"},
    {"name": "cfo_yield"},
    {"name": "gross_profitability"},
    {"name": "operating_profitability"},
    {"name": "roe"},
    {"name": "low_accruals"},
    {"name": "low_asset_growth"},
    {"name": "low_net_issuance"},
    {"name": "low_leverage"},
    {"name": "sales_growth"},
    {"name": "sue"},
    {"name": "quality_composite"},
]


def load_membership(path: str | Path, start: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["month_end_date"] = pd.to_datetime(df["month_end_date"], errors="coerce")
    df = df.dropna(subset=["month_end_date", "ticker"])
    df = df[df["month_end_date"] >= pd.Timestamp(start)].copy()
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper().str.replace(".", "-", regex=False)
    return df[["month_end_date", "ticker"]].drop_duplicates()


def research_and_holdout_dates(dates: pd.DatetimeIndex, start: str, holdout: str, lag: int, horizon: int):
    reb = rebalance_schedule(dates, "ME")
    reb = reb[reb >= pd.Timestamp(start)]
    first_h = dates.searchsorted(pd.Timestamp(holdout))
    end_pos = dates.get_indexer(reb) + lag + horizon
    research = reb[end_pos < first_h]
    holdout_dates = reb[(reb >= pd.Timestamp(holdout)) & (end_pos < len(dates))]
    return reb, research, holdout_dates


def backtest_split(score: pd.DataFrame, close: pd.DataFrame, reb: pd.DatetimeIndex, holdout: str, top_n: int,
                   cost: CostModel) -> dict[str, dict]:
    h = pd.Timestamp(holdout)
    targets = build_weights(score, reb, n_long=top_n)
    is_t = targets[targets.index < h]
    close_is = close[close.index < h]
    is_t = is_t[close.index.get_indexer(is_t.index) + 1 < len(close_is)]
    bt_is = run_backtest(is_t, close_is, lag=1, costs=cost)
    bt_oos = run_backtest(targets[targets.index >= h], close, lag=1, costs=cost)
    return {"research": bt_is.summary(), "holdout": bt_oos.summary(), "_is_returns": bt_is.returns}


def run(data: MarketData, start: str, holdout: str, horizon: int = 21, lag: int = 1, top_n: int = 10,
        min_t: float = 3.0, max_corr: float = 0.6, log_path: str | Path = "output/factor_mining/experiments.jsonl"):
    reb, research, hold = research_and_holdout_dates(data.dates, start, holdout, lag, horizon)
    fwd = data.forward_returns(horizon=horizon, lag=lag)
    cands = build_candidates(CANDIDATES, data)
    base_raw = build_candidates([BASELINE], data)
    res = mine_factors(data, cands, research, fwd, baseline=base_raw, min_t=min_t, max_corr=max_corr)

    log = ExperimentLog(log_path)
    for name, row in res.table.iterrows():
        log.log(f"mine:{name}", {"factor": name, "holdout": holdout}, {"t_incr": row["t_incr"], "ic_mean": row["ic_mean"]})

    # 留出期：只检验入选因子
    hold_rows = {}
    for name in res.selected:
        ic = information_coefficient(res.factors[name].reindex(hold), fwd.reindex(hold))
        hold_rows[name] = {"holdout_ic": ic.mean(), "holdout_t": ic.mean() / ic.std() * np.sqrt(len(ic)) if len(ic) > 2 else np.nan,
                           "n": len(ic)}
    holdout_table = pd.DataFrame(hold_rows).T

    # 组合：基准动量 vs 基准 + 入选因子等权
    base_proc = standard_pipeline(base_raw[next(iter(base_raw))], data.universe)
    cost = CostModel(1, 2, 2, 0)
    bts = {"基准 11-1 动量": backtest_split(base_proc, data.close, reb, holdout, top_n, cost)}
    if res.selected:
        combo = equal_weight({"base": base_proc, **oriented(res.factors, res.table, res.selected)})
        bts["基准 + 入选因子"] = backtest_split(combo, data.close, reb, holdout, top_n, cost)
    n_trials = log.n_trials()
    for v in bts.values():
        v["research"]["dsr"] = deflated_sharpe_ratio(v.pop("_is_returns"), n_trials)
    return res, holdout_table, bts, n_trials


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--membership", required=True)
    p.add_argument("--start", default="2011-01-01", help="研究期起点（之前一年用于因子预热）")
    p.add_argument("--holdout", default="2022-01-01")
    p.add_argument("--min-t", type=float, default=3.0)
    p.add_argument("--max-corr", type=float, default=0.6)
    p.add_argument("--top-n", type=int, default=10)
    p.add_argument("--cache", default="data/cache/nasdaq100_yahoo_raw.pkl")
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--no-sec", action="store_true", help="不下载 SEC 基本面，只挖价量因子")
    p.add_argument("--sec-user-agent", default="alpha-research research@example.com",
                   help="SEC 要求的 User-Agent，格式“名字 邮箱”")
    p.add_argument("--sec-cache", default="data/cache/sec")
    p.add_argument("--out", default="output/factor_mining")
    args = p.parse_args()

    membership = load_membership(args.membership, "2010-01-01")
    tickers = sorted(membership["ticker"].unique())
    raw = download_yahoo_ohlcv(tickers, args.cache, args.refresh)
    adj = raw["adj_close"].dropna(how="all", axis=1)
    universe = snapshot_mask(membership, adj.index, adj.columns)
    fields = {}
    if not args.no_sec:
        records = fetch_records(adj.columns, args.sec_user_agent, args.sec_cache)
        fields = build_fields(records, adj.index, adj.columns, raw["close"], raw.get("splits"))
        cov = fields["market_cap"].notna() & universe if "market_cap" in fields else universe & False
        print(f"SEC 基本面覆盖：{len(records)}/{len(adj.columns)} 只有数据；"
              f"调仓日池内有市值的比例 {(cov.sum(axis=1) / universe.sum(axis=1)).loc['2011':].mean():.1%}")
    data = market_from_yahoo_raw(raw, universe=universe, fields=fields)

    out = Path(args.out)
    res, hold, bts, n_trials = run(data, args.start, args.holdout, top_n=args.top_n, min_t=args.min_t,
                                   max_corr=args.max_corr, log_path=out / "experiments.jsonl")
    pd.set_option("display.width", 220)
    cols = ["ic_mean", "t_stat", "ic_incr", "t_incr", "q_bh", "sign_stable", "corr_baseline", "rank_autocorr", "ls_ann", "selected"]
    print(f"\n== 研究期 {args.start} ~ {args.holdout}（{int(res.table['n'].max())} 个月），相对 11-1 动量的增量 ==")
    print(res.table[cols].astype({"q_bh": float}).round(3).to_string())
    print(f"\n入选（|t_incr|≥{args.min_t}、q≤0.05、前后半段同号、互相关≤{args.max_corr}）：{res.selected or '无'}")
    if len(hold):
        print("\n== 留出期 IC（只看一次） ==")
        print(hold.round(3).to_string())
    print(f"\n== Top{args.top_n} 纯多头组合，T+1、单边 5bp；累计试验 {n_trials} 次 ==")
    for name, v in bts.items():
        r, h = v["research"], v["holdout"]
        print(f"{name:<14} 研究期 年化 {r['ann_return']:.1%} 夏普 {r['sharpe']:.2f} 回撤 {r['max_drawdown']:.1%} DSR {r['dsr']:.2f}"
              f" | 留出期 年化 {h['ann_return']:.1%} 夏普 {h['sharpe']:.2f} 回撤 {h['max_drawdown']:.1%}")

    out.mkdir(parents=True, exist_ok=True)
    res.table.to_csv(out / "candidates.csv")
    res.corr.to_csv(out / "candidate_corr.csv")
    hold.to_csv(out / "holdout_ic.csv")
    pd.DataFrame({f"{k}|{p}": v[p] for k, v in bts.items() for p in ("research", "holdout")}).to_csv(out / "portfolio.csv")
    print(f"\n结果已写入 {out}/")


if __name__ == "__main__":
    main()
