"""SEC EDGAR XBRL 财报数据（companyfacts API），按 point-in-time 方式整理。

为什么用它：免费、覆盖 2009 年以后所有美国上市公司的 10-K/10-Q，且每个数值都带有
`filed`（提交日期），可以精确知道“哪天市场能看到这个数”。

防前视的处理：
- 同一报告期被多次披露（后续财报里的对比数、10-K/A 重述）时，只保留**首次披露**的值，
  可获得日 = 首次 filed 日期。用重述后的值回填历史属于未来信息。
- 流量项（收入、利润、现金流）用最近 4 个单季相加得到 TTM，Q4 由年报减去前三季推出，
  可获得日取 4 个季度中最晚的 filed。
- 对齐到交易日时，若新披露的报告期反而早于已披露的最新报告期（迟交的旧数据），忽略它。

SEC 要求请求带 User-Agent（含联系方式），限速约 10 次/秒。
"""

from __future__ import annotations

import gzip
import json
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from alpha.factors.fundamental import align_fundamental

TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

# 字段 -> (类型, 按优先级排列的 (taxonomy, tag, unit))
CONCEPTS: dict[str, tuple[str, list[tuple[str, str, str]]]] = {
    "revenue": ("flow", [
        ("us-gaap", "Revenues", "USD"),
        ("us-gaap", "RevenueFromContractWithCustomerExcludingAssessedTax", "USD"),
        ("us-gaap", "RevenueFromContractWithCustomerIncludingAssessedTax", "USD"),
        ("us-gaap", "SalesRevenueNet", "USD"),
    ]),
    "cost_of_revenue": ("flow", [
        ("us-gaap", "CostOfRevenue", "USD"),
        ("us-gaap", "CostOfGoodsAndServicesSold", "USD"),
        ("us-gaap", "CostOfGoodsSold", "USD"),
    ]),
    "gross_profit": ("flow", [("us-gaap", "GrossProfit", "USD")]),
    "operating_income": ("flow", [("us-gaap", "OperatingIncomeLoss", "USD")]),
    "net_income": ("flow", [
        ("us-gaap", "NetIncomeLoss", "USD"),
        ("us-gaap", "ProfitLoss", "USD"),
    ]),
    "cfo": ("flow", [
        ("us-gaap", "NetCashProvidedByUsedInOperatingActivities", "USD"),
        ("us-gaap", "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations", "USD"),
    ]),
    "eps": ("flow", [
        ("us-gaap", "EarningsPerShareDiluted", "USD/shares"),
        ("us-gaap", "EarningsPerShareBasic", "USD/shares"),
    ]),
    "assets": ("instant", [("us-gaap", "Assets", "USD")]),
    "equity": ("instant", [
        ("us-gaap", "StockholdersEquity", "USD"),
        ("us-gaap", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest", "USD"),
    ]),
    "liabilities": ("instant", [("us-gaap", "Liabilities", "USD")]),
    "shares": ("instant", [
        ("dei", "EntityCommonStockSharesOutstanding", "shares"),
        ("us-gaap", "CommonStockSharesOutstanding", "shares"),
    ]),
}

QUARTER_DAYS = (75, 105)
ANNUAL_DAYS = (340, 390)


# ----------------------------------------------------------------------------- 下载

def _get_json(url: str, user_agent: str, retries: int = 3) -> dict:
    import urllib.request

    req = urllib.request.Request(url, headers={"User-Agent": user_agent, "Accept-Encoding": "gzip"})
    for i in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                raw = r.read()
                if r.headers.get("Content-Encoding") == "gzip":
                    raw = gzip.decompress(raw)
                return json.loads(raw)
        except Exception as e:  # pragma: no cover - network
            if getattr(e, "code", None) == 404:
                raise
            if i == retries - 1:
                raise
            time.sleep(2 * (i + 1))
    raise RuntimeError("unreachable")


def fetch_ticker_map(user_agent: str, cache_dir: str | Path) -> dict[str, int]:
    """当前上市公司的 ticker -> CIK。已退市公司不在其中，可用 extra_map 补充。"""
    f = Path(cache_dir) / "company_tickers.json"
    if f.exists():
        data = json.loads(f.read_text())
    else:  # pragma: no cover - network
        data = _get_json(TICKER_MAP_URL, user_agent)
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(data))
    return {v["ticker"].upper().replace(".", "-"): int(v["cik_str"]) for v in data.values()}


def extract_records(facts: dict) -> pd.DataFrame:
    """从 companyfacts JSON 中只抽出 CONCEPTS 需要的记录。"""
    rows = []
    f = facts.get("facts", {})
    for field, (_kind, tags) in CONCEPTS.items():
        for prio, (tax, tag, unit) in enumerate(tags):
            items = f.get(tax, {}).get(tag, {}).get("units", {}).get(unit, [])
            for it in items:
                if "filed" not in it or "end" not in it or it.get("val") is None:
                    continue
                rows.append((field, prio, tag, it.get("start"), it["end"], float(it["val"]), it["filed"], it.get("form", "")))
    df = pd.DataFrame(rows, columns=["field", "prio", "tag", "start", "end", "val", "filed", "form"])
    for c in ("start", "end", "filed"):
        df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def fetch_records(
    tickers: Iterable[str], user_agent: str, cache_dir: str | Path, extra_map: dict[str, int] | None = None,
    refresh: bool = False, sleep: float = 0.12,
) -> dict[str, pd.DataFrame]:
    """逐个 ticker 下载并抽取记录，结果按 ticker 缓存为小 pickle。"""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    cmap = fetch_ticker_map(user_agent, cache)
    cmap.update({k.upper(): v for k, v in (extra_map or {}).items()})
    out: dict[str, pd.DataFrame] = {}
    missing = []
    for i, t in enumerate(tickers):
        f = cache / f"{t}.pkl"
        if f.exists() and not refresh:
            out[t] = pd.read_pickle(f)
            continue
        cik = cmap.get(t)
        if cik is None:
            missing.append(t)
            continue
        try:  # pragma: no cover - network
            facts = _get_json(FACTS_URL.format(cik=cik), user_agent)
        except Exception as e:  # pragma: no cover - network
            print(f"SEC 下载失败 {t}（CIK {cik}）：{e}")
            continue
        rec = extract_records(facts)
        rec.to_pickle(f)
        out[t] = rec
        time.sleep(sleep)
        if (i + 1) % 25 == 0:
            print(f"SEC {i + 1}")
    if missing:
        print(f"SEC 找不到 CIK 的 ticker {len(missing)} 只（多为已退市）：{', '.join(missing)}")
    return out


# ----------------------------------------------------------------------------- 整理

def first_filed(rec: pd.DataFrame) -> pd.DataFrame:
    """同一报告期（start, end）只保留首次披露的值。"""
    rec = rec.sort_values("filed")
    key = ["start", "end"] if rec["start"].notna().any() else ["end"]
    return rec.drop_duplicates(subset=key, keep="first")


def _duration(rec: pd.DataFrame) -> pd.Series:
    return (rec["end"] - rec["start"]).dt.days


def quarterly(rec: pd.DataFrame) -> pd.DataFrame:
    """单季序列（含由年报推出的 Q4），列 end, val, filed。"""
    rec = first_filed(rec.dropna(subset=["start"]))
    d = _duration(rec)
    q = rec[(d >= QUARTER_DAYS[0]) & (d <= QUARTER_DAYS[1])][["start", "end", "val", "filed"]]
    a = rec[(d >= ANNUAL_DAYS[0]) & (d <= ANNUAL_DAYS[1])]
    derived = []
    for row in a.itertuples(index=False):
        inside = q[(q["start"] >= row.start - pd.Timedelta(days=7)) & (q["end"] <= row.end - pd.Timedelta(days=60))]
        if len(inside) == 3 and not ((q["end"] - row.end).abs() <= pd.Timedelta(days=7)).any():
            derived.append((inside["end"].max() + pd.Timedelta(days=1), row.end, row.val - inside["val"].sum(),
                            max(row.filed, inside["filed"].max())))
    if derived:
        q = pd.concat([q, pd.DataFrame(derived, columns=["start", "end", "val", "filed"])], ignore_index=True)
    return q.sort_values("end").drop_duplicates("end", keep="first").reset_index(drop=True)


def annual(rec: pd.DataFrame) -> pd.DataFrame:
    rec = first_filed(rec.dropna(subset=["start"]))
    d = _duration(rec)
    return rec[(d >= ANNUAL_DAYS[0]) & (d <= ANNUAL_DAYS[1])][["end", "val", "filed"]].sort_values("end")


def ttm(rec: pd.DataFrame) -> pd.DataFrame:
    """TTM：连续 4 个单季之和；没有季报的期间（如只交年报的公司）用年报值。"""
    q = quarterly(rec)
    rows = []
    for i in range(3, len(q)):
        w = q.iloc[i - 3 : i + 1]
        if (w["end"].iloc[-1] - w["end"].iloc[0]).days <= 300:
            rows.append((w["end"].iloc[-1], w["val"].sum(), w["filed"].max()))
    out = pd.DataFrame(rows, columns=["end", "val", "filed"])
    a = annual(rec)
    if len(out):
        a = a[~a["end"].isin(out["end"])]
    return pd.concat([out, a], ignore_index=True).sort_values("end").reset_index(drop=True)


def instant(rec: pd.DataFrame) -> pd.DataFrame:
    """时点项（资产、权益、股本），每个 end 取首次披露。"""
    r = rec.sort_values("filed").drop_duplicates(subset=["end"], keep="first")
    return r[["end", "val", "filed"]].sort_values("end").reset_index(drop=True)


def sue(rec: pd.DataFrame, window: int = 8, min_obs: int = 4) -> pd.DataFrame:
    """标准化未预期盈余（季节性随机游走）：(EPS_q − EPS_{q−4}) / 过去 window 个同类差值的标准差。"""
    q = quarterly(rec).set_index("end")
    rows = []
    diffs: list[float] = []
    for end, row in q.iterrows():
        prev = q[(q.index >= end - pd.Timedelta(days=380)) & (q.index <= end - pd.Timedelta(days=350))]
        if prev.empty:
            continue
        d = row["val"] - prev["val"].iloc[-1]
        hist = diffs[-window:]
        if len(hist) >= min_obs and np.std(hist, ddof=1) > 0:
            rows.append((end, d / np.std(hist, ddof=1), row["filed"]))
        diffs.append(d)
    return pd.DataFrame(rows, columns=["end", "val", "filed"])


def _combine_tags(rec: pd.DataFrame, builder) -> pd.DataFrame:
    """按优先级对各个 tag 分别构建序列，再按 end 合并（同一 end 取高优先级 tag）。"""
    parts = []
    for prio, g in rec.groupby("prio"):
        s = builder(g)
        if len(s):
            parts.append(s.assign(prio=prio))
    if not parts:
        return pd.DataFrame(columns=["end", "val", "filed"])
    allp = pd.concat(parts).sort_values(["end", "prio"])
    return allp.drop_duplicates("end", keep="first")[["end", "val", "filed"]].reset_index(drop=True)


def point_in_time_series(records: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """每个字段一张长表：ticker, end, val, available_date。"""
    builders = {"flow": ttm, "instant": instant}
    out: dict[str, list[pd.DataFrame]] = {}
    for t, rec in records.items():
        for field, g in rec.groupby("field"):
            kind = CONCEPTS[field][0]
            s = _combine_tags(g, builders[kind])
            name = f"{field}_ttm" if kind == "flow" else field
            if len(s):
                out.setdefault(name, []).append(s.assign(ticker=t))
            if field == "eps":
                s2 = _combine_tags(g, sue)
                if len(s2):
                    out.setdefault("sue", []).append(s2.assign(ticker=t))
    return {k: pd.concat(v, ignore_index=True).rename(columns={"filed": "available_date"}) for k, v in out.items()}


def _drop_stale_periods(df: pd.DataFrame) -> pd.DataFrame:
    """按可获得日排序后，若某条记录的报告期早于此前已披露的最新报告期，丢弃它。"""
    df = df.sort_values(["ticker", "available_date", "end"])
    prev_max = df.groupby("ticker")["end"].transform(lambda s: s.cummax().shift(1))
    return df[prev_max.isna() | (df["end"] >= prev_max)]


def split_factor_after(splits: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """F(t) = t 之后所有拆股比例的乘积，用于把历史股本换算成与复权价一致的单位。"""
    s = splits.reindex(dates).fillna(0.0)
    s = s.where(s > 0, 1.0)
    rev_cum = s.iloc[::-1].cumprod().iloc[::-1]
    return rev_cum.shift(-1).fillna(1.0)


def build_fields(
    records: dict[str, pd.DataFrame],
    dates: pd.DatetimeIndex,
    tickers: pd.Index,
    close_split_adj: pd.DataFrame,
    splits: pd.DataFrame | None = None,
    lag_days: int = 1,
    max_staleness: int = 300,
    sue_staleness: int = 63,
) -> dict[str, pd.DataFrame]:
    """生成交易日宽表字段，可直接放进 MarketData.fields。

    close_split_adj  只做拆股调整、不做分红调整的收盘价（yfinance auto_adjust=False 的 Close）。
                     市值 = 该价格 × 换算后的股本，等价于“当时的原始价格 × 当时的股本”。
    splits           拆股比例宽表（拆股日为比例，如 4.0），用于换算股本单位
    """
    series = point_in_time_series(records)
    fields: dict[str, pd.DataFrame] = {}
    for name, df in series.items():
        df = df[df["ticker"].isin(tickers)]
        if name == "shares" and splits is not None:
            fac = split_factor_after(splits.reindex(columns=tickers).fillna(0.0), dates)
            pos = dates.searchsorted(df["end"].to_numpy(), side="right") - 1
            ok = pos >= 0
            col = tickers.get_indexer(df["ticker"])
            mult = np.ones(len(df))
            mult[ok] = fac.to_numpy()[pos[ok], col[ok]]
            df = df.assign(val=df["val"].to_numpy() * mult)
        df = _drop_stale_periods(df)
        fields[name] = align_fundamental(
            df, dates, tickers, "val", lag_days=lag_days,
            max_staleness=sue_staleness if name == "sue" else max_staleness,
        )
    if "shares" in fields:
        fields["market_cap"] = close_split_adj.reindex(index=dates, columns=tickers) * fields["shares"]
    if "equity" in fields:
        fields["book_value"] = fields["equity"]
    if "net_income_ttm" in fields:
        fields["earnings_ttm"] = fields["net_income_ttm"]
    if "gross_profit_ttm" not in fields and {"revenue_ttm", "cost_of_revenue_ttm"} <= fields.keys():
        fields["gross_profit_ttm"] = fields["revenue_ttm"] - fields["cost_of_revenue_ttm"]
    elif {"gross_profit_ttm", "revenue_ttm", "cost_of_revenue_ttm"} <= fields.keys():
        fields["gross_profit_ttm"] = fields["gross_profit_ttm"].combine_first(
            fields["revenue_ttm"] - fields["cost_of_revenue_ttm"])
    return fields
