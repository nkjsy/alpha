"""近似罗素1000的历史成分表（按市值排名，每年 5 月末重排、6 月起生效）。

罗素官方的历史成分要付费，这里用免费数据近似：
1. 候选池 = SEC 登记的 NYSE/Nasdaq 现存公司 + 可选的额外成分表（如标普500历史成分，补回部分已退市大公司）；
2. 用 Yahoo 价格和成交额先粗筛：月末过去 63 日平均成交额曾进入前 1500 名的股票；
3. 对粗筛结果用 SEC 披露的股本（按披露日对齐）乘当时价格得到市值，每年 5 月末取前 1000 名。

局限：候选池缺少“既不在标普500、又已经退市”的公司，存在一定幸存者偏差；
外国公司（报 IFRS、无 us-gaap 股本）没有市值，会被排除，这一点和罗素的规则大体一致。

用法（需要能访问 Yahoo 和 sec.gov）：
  python scripts/build_r1000_universe.py --extra-membership data/sp500_monthly_constituents.csv \
      --sec-user-agent "你的名字 你的邮箱"
输出 data/r1000_monthly_constituents.csv，以及给 mine_factors.py 复用的价格缓存 data/cache/r1000_yahoo_raw.pkl。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from alpha.data.loaders import download_yahoo_ohlcv
from alpha.data.sec import build_fields, fetch_listed_companies, fetch_records
from alpha.data.universe import cap_rank_membership, liquid_pool


def build(raw: dict[str, pd.DataFrame], records: dict[str, pd.DataFrame], n: int, start: str) -> pd.DataFrame:
    close = raw["close"].astype(float)
    tickers = close.columns
    fields = build_fields(records, close.index, tickers, close, raw.get("splits"))
    if "market_cap" not in fields:
        raise RuntimeError("SEC 记录里没有股本数据，无法计算市值")
    return cap_rank_membership(fields["market_cap"], n=n, start=start)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--pool-n", type=int, default=1500, help="按成交额粗筛保留的名次")
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--price-start", default="2008-01-01")
    p.add_argument("--extra-membership", default=None, help="额外并入候选池的成分表（month_end_date,ticker）")
    p.add_argument("--sec-user-agent", default="alpha-research research@example.com")
    p.add_argument("--sec-cache", default="data/cache/sec")
    p.add_argument("--pool-cache", default="data/cache/pool_yahoo_raw.pkl")
    p.add_argument("--yahoo-out", default="data/cache/r1000_yahoo_raw.pkl")
    p.add_argument("--out", default="data/r1000_monthly_constituents.csv")
    args = p.parse_args()

    listed = fetch_listed_companies(args.sec_user_agent, args.sec_cache)
    pool = set(listed["ticker"])
    extra = None
    if args.extra_membership:
        extra = pd.read_csv(args.extra_membership)
        extra["ticker"] = extra["ticker"].astype(str).str.strip().str.upper().str.replace(".", "-", regex=False)
        pool |= set(extra["ticker"])
    pool = sorted(t for t in pool if t and t != "NAN")
    print(f"候选池 {len(pool)} 只（交易所挂牌 {len(listed)}）")

    raw = download_yahoo_ohlcv(pool, args.pool_cache, start=args.price_start, dtype="float32")
    dv = (raw["close"].astype(float) * raw["volume"].astype(float))
    liquid = liquid_pool(dv, args.pool_n, start=pd.Timestamp(args.start) - pd.DateOffset(months=8))
    print(f"成交额曾进前 {args.pool_n} 名：{len(liquid)} 只")
    sub = {k: v.reindex(columns=liquid).astype(float) for k, v in raw.items()}

    records = fetch_records(liquid, args.sec_user_agent, args.sec_cache)
    mem = build(sub, records, args.n, args.start)
    members = sorted(mem["ticker"].unique())
    per_month = mem.groupby("month_end_date").size()
    print(f"成分表 {mem['month_end_date'].min():%Y-%m} ~ {mem['month_end_date'].max():%Y-%m}，"
          f"每月 {per_month.min()}~{per_month.max()} 只，共出现 {len(members)} 只")
    if extra is not None:
        e = extra.assign(month_end_date=pd.to_datetime(extra["month_end_date"]))
        e = e[e["month_end_date"] >= mem["month_end_date"].min()]
        hit = e.merge(mem.assign(month_end_date=pd.to_datetime(mem["month_end_date"])), on=["month_end_date", "ticker"])
        print(f"额外成分表的月度记录中，有 {len(hit) / max(len(e), 1):.1%} 也在近似罗素1000里")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    mem.to_csv(args.out, index=False, date_format="%Y-%m-%d")
    keep = {k: v.reindex(columns=members) for k, v in sub.items()}
    Path(args.yahoo_out).parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(keep, args.yahoo_out)
    print(f"已写入 {args.out} 和 {args.yahoo_out}")


if __name__ == "__main__":
    main()
