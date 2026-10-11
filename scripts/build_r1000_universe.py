"""近似罗素1000的历史成分表（按市值排名，每年 5 月末重排、6 月起生效）。

罗素官方的历史成分要付费，这里用免费数据近似：
1. 候选池 = SEC 登记的 NYSE/Nasdaq 现存公司 + 可选的额外成分表（如标普500历史成分，补回部分已退市大公司）；
2. 用 Yahoo 价格和成交额先粗筛：月末过去 63 日平均成交额曾进入前 1500 名的股票；
3. 对粗筛结果用 SEC 披露的股本（按披露日对齐）乘当时价格得到市值，每年 5 月末取前 1000 名。

补救措施：
- 多类股公司的股本依次退用 加权平均股本 -> 净利润 / EPS 反推（见 alpha/data/sec.py）；
- 重组成新控股公司、SEC 映射换了 CIK 的（如 XOM），用 Form 4 记录里的旧 CIK 补回历史财报；
- 大批下载超时丢掉的价格用小批次重下；
- 2009–2010 年 XBRL 尚未普及、有市值的股票不足 1000 只时，用平均成交额补足；
- 额外成分表（标普500）里的股票当月一定纳入。

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

from alpha.data.insider import fetch_insider_events
from alpha.data.loaders import download_yahoo_ohlcv, fill_missing_yahoo
from alpha.data.sec import (build_fields, company_names, fetch_listed_companies, fetch_records, fetch_ticker_map,
                            historical_ciks, same_company)
from alpha.data.universe import cap_rank_membership, liquid_pool


# 重组时改了名、名字对不上的同一公司，手动确认后放这里
SAME_COMPANY_CIKS = {"GOOG": [1288776], "GOOGL": [1288776]}


def merge_old_ciks(cand: dict[str, list[int]], first_px: pd.Series, first_filed: dict[str, pd.Timestamp],
                   names, allow: dict[str, list[int]] | None = None) -> dict[str, list[int]]:
    """决定哪些旧 CIK 的财报并入当前 ticker。

    条件一：股价历史比当前 CIK 的首份财报早一年以上（否则是新公司上市后才有财报，旧 CIK 属于别人）；
    条件二：旧 CIK 的公司名（含曾用名）和当前 CIK 的相似，排除“ticker 被另一家公司重新使用”的情况
    （如 B 曾是 Barnes Group、现在是 Barrick；GAP 曾是 A&P 超市、现在是 Gap）。
    names(ticker 或 CIK) 返回公司名列表；allow 里的组合不看名字直接合并。
    """
    allow = allow or {}
    out: dict[str, list[int]] = {}
    for t, ciks in cand.items():
        f0, p0 = first_filed.get(t), first_px.get(t)
        if f0 is None or pd.isna(f0) or p0 is None or pd.isna(p0) or not p0 < f0 - pd.Timedelta(days=365):
            continue
        cur = names(t)
        keep = [c for c in ciks if c in allow.get(t, []) or same_company(names(c), cur)]
        if keep:
            out[t] = keep
    return out


def build(raw: dict[str, pd.DataFrame], records: dict[str, pd.DataFrame], n: int, start: str,
          always: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    close = raw["close"].astype(float)
    tickers = close.columns
    fields = build_fields(records, close.index, tickers, close, raw.get("splits"))
    if "market_cap" not in fields:
        raise RuntimeError("SEC 记录里没有股本数据，无法计算市值")
    adv = (close * raw["volume"].astype(float)).rolling(63, min_periods=21).mean()
    # 只用有 SEC 财报的公司补位（排除 SPY 之类的基金/ETF，它们成交额很大但不该进股票池）
    adv = adv[[t for t in adv.columns if t in records and len(records[t])]]
    mem = cap_rank_membership(fields["market_cap"], n=n, start=start, fallback=adv, always=always)
    return mem, fields["market_cap"]


def overlap_report(extra: pd.DataFrame, mem: pd.DataFrame, mcap: pd.DataFrame) -> None:
    """额外成分表（标普500）里有市值的比例，反映市值数据的覆盖质量（强制纳入之前）。"""
    e = extra.assign(month=pd.to_datetime(extra["month_end_date"]).dt.to_period("M"))
    e = e[e["month"] >= pd.to_datetime(mem["month_end_date"]).min().to_period("M")]
    last = mcap.groupby(mcap.index.to_period("M")).last()
    has = last.notna()
    e = e[e["month"].isin(has.index)]
    e["has_cap"] = [bool(has.at[m, t]) if t in has.columns else False for m, t in zip(e["month"], e["ticker"])]
    by_year = e.groupby(e["month"].dt.year)["has_cap"].mean()
    print("额外成分表中有市值的比例（按年）：" + "，".join(f"{y} {v:.0%}" for y, v in by_year.items()))
    recent = e[e["month"] == e["month"].max()]
    miss = sorted(recent.loc[~recent["has_cap"], "ticker"])
    print(f"最新一期额外成分表里没有市值的（已强制纳入）{len(miss)} 只：{', '.join(miss)}")


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
    n_before = raw["close"].dropna(how="all", axis=1).shape[1]
    raw, still = fill_missing_yahoo(raw, pool, start=args.price_start, dtype="float32")
    if raw["close"].dropna(how="all", axis=1).shape[1] > n_before:
        pd.to_pickle(raw, args.pool_cache)
    print(f"有价格的 {raw['close'].dropna(how='all', axis=1).shape[1]} 只，仍缺 {len(still)} 只（多为已退市）")
    dv = (raw["close"].astype(float) * raw["volume"].astype(float))
    liquid = liquid_pool(dv, args.pool_n, start=pd.Timestamp(args.start) - pd.DateOffset(months=8))
    print(f"成交额曾进前 {args.pool_n} 名：{len(liquid)} 只")
    sub = {k: v.reindex(columns=liquid).astype(float) for k, v in raw.items()}

    cmap = fetch_ticker_map(args.sec_user_agent, args.sec_cache)
    events = fetch_insider_events(2009, raw["close"].index[-1], args.sec_user_agent, args.sec_cache)
    cand = {t: c for t, c in historical_ciks(events, cmap).items() if t in set(liquid)}
    base = fetch_records(liquid, args.sec_user_agent, args.sec_cache)
    first_px = sub["close"].apply(lambda s: s.first_valid_index())
    first_filed = {t: r["filed"].min() for t, r in base.items() if r is not None and not r.empty}
    titles = dict(zip(listed["ticker"], listed["name"]))

    def names(x):
        if isinstance(x, str):
            return [titles.get(x, "")] + company_names(cmap[x], args.sec_user_agent, args.sec_cache)
        return company_names(x, args.sec_user_agent, args.sec_cache)

    old = merge_old_ciks(cand, first_px, first_filed, names, SAME_COMPANY_CIKS)
    for t in sorted(set(cand) - set(old)):
        if first_filed.get(t) is not None and first_px.get(t) is not None and \
                first_px[t] < first_filed[t] - pd.Timedelta(days=365):
            print(f"  {t} 名字对不上、不合并：现 {names(t)[:2]}，旧 {[names(c)[:1] for c in cand[t]]}")
    print(f"Form 4 里换过 CIK 的 {len(cand)} 只，其中判定为同一公司重组、合并旧 CIK 财报的 {len(old)} 只："
          f"{', '.join(sorted(old))}")
    records = fetch_records(liquid, args.sec_user_agent, args.sec_cache, extra_ciks=old)
    always = None
    if extra is not None:
        always = extra[extra["ticker"].isin(sub["close"].dropna(how="all", axis=1).columns)]
    mem, mcap = build(sub, records, args.n, args.start, always=always)
    members = sorted(mem["ticker"].unique())
    per_month = mem.groupby("month_end_date").size()
    print(f"成分表 {mem['month_end_date'].min():%Y-%m} ~ {mem['month_end_date'].max():%Y-%m}，"
          f"每月 {per_month.min()}~{per_month.max()} 只，共出现 {len(members)} 只")
    if extra is not None:
        overlap_report(extra, mem, mcap)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    mem.to_csv(args.out, index=False, date_format="%Y-%m-%d")
    keep = {k: v.reindex(columns=members) for k, v in sub.items()}
    if "high" not in keep:  # 旧版缓存没有高低价，qlib 的 Alpha158 需要
        hl = download_yahoo_ohlcv(members, Path(args.yahoo_out).with_name("r1000_high_low.pkl"),
                                  start=args.price_start, fields=("high", "low"))
        keep.update({k: hl[k].reindex(index=keep["close"].index, columns=members).astype(float) for k in ("high", "low")})
    Path(args.yahoo_out).parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(keep, args.yahoo_out)
    print(f"已写入 {args.out} 和 {args.yahoo_out}")


if __name__ == "__main__":
    main()
