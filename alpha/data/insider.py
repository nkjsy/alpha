"""内部人交易（Form 4），来自 SEC 的 Insider Transactions Data Sets（季度 zip，2006 年起）。

只用公开市场买入（交易代码 P）和卖出（S）：期权行权、授予、赠与等不反映内部人的看法。
可获得日 = Form 4 的提交日（FILING_DATE），再顺延 lag_days 个交易日，避免用到当天收盘前看不到的信息。
只取原始 Form 4，不用 4/A 修正（修正可能在很久以后才提交）。
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

FORM345_URL = "https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/{y}q{q}_form345.zip"


def parse_sec_date(s: pd.Series) -> pd.Series:
    """SEC 数据集里的日期多为 31-MAR-2023，也兼容 2023-03-31。"""
    out = pd.to_datetime(s, format="%d-%b-%Y", errors="coerce")
    miss = out.isna() & s.notna()
    if miss.any():
        out[miss] = pd.to_datetime(s[miss], errors="coerce")
    return out


def _read_tsv(z: zipfile.ZipFile, name: str, usecols: list[str]) -> pd.DataFrame:
    names = {Path(n).name.upper(): n for n in z.namelist()}
    with z.open(names[name]) as f:
        return pd.read_csv(f, sep="\t", usecols=lambda c: c in usecols, dtype=str, quoting=3,
                           encoding="latin-1", on_bad_lines="skip")


def parse_form345(data: bytes | str | Path) -> pd.DataFrame:
    """一个季度 zip -> 交易事件 (cik, symbol, filed, owner, code, shares, value)。"""
    z = zipfile.ZipFile(io.BytesIO(data) if isinstance(data, bytes) else data)
    sub = _read_tsv(z, "SUBMISSION.TSV", ["ACCESSION_NUMBER", "FILING_DATE", "DOCUMENT_TYPE", "ISSUERCIK",
                                         "ISSUERTRADINGSYMBOL"])
    sub = sub[sub["DOCUMENT_TYPE"].str.strip() == "4"]
    own = _read_tsv(z, "REPORTINGOWNER.TSV", ["ACCESSION_NUMBER", "RPTOWNERCIK"])
    own = own.drop_duplicates("ACCESSION_NUMBER")
    tr = _read_tsv(z, "NONDERIV_TRANS.TSV", ["ACCESSION_NUMBER", "TRANS_CODE", "TRANS_SHARES", "TRANS_PRICEPERSHARE"])
    tr = tr[tr["TRANS_CODE"].str.strip().isin(["P", "S"])]
    df = tr.merge(sub, on="ACCESSION_NUMBER").merge(own, on="ACCESSION_NUMBER", how="left")
    shares = pd.to_numeric(df["TRANS_SHARES"], errors="coerce")
    price = pd.to_numeric(df["TRANS_PRICEPERSHARE"], errors="coerce")
    out = pd.DataFrame({
        "cik": pd.to_numeric(df["ISSUERCIK"], errors="coerce"),
        "symbol": df["ISSUERTRADINGSYMBOL"].fillna("").str.strip().str.upper().str.replace(".", "-", regex=False),
        "filed": parse_sec_date(df["FILING_DATE"]),
        "owner": df["RPTOWNERCIK"].fillna(df["ACCESSION_NUMBER"]),
        "code": df["TRANS_CODE"].str.strip(),
        "shares": shares,
        "value": shares * price,
    })
    return out.dropna(subset=["filed", "shares"]).reset_index(drop=True)


def fetch_insider_events(start_year: int, end: pd.Timestamp, user_agent: str, cache_dir: str | Path) -> pd.DataFrame:
    """逐季度下载并解析，解析结果按季度缓存为 pickle。"""
    from alpha.data.sec import get_bytes

    cache = Path(cache_dir) / "form345"
    cache.mkdir(parents=True, exist_ok=True)
    frames = []
    for y in range(start_year, end.year + 1):
        for q in range(1, 5):
            if pd.Timestamp(year=y, month=3 * q, day=1) > end:
                break
            f = cache / f"{y}q{q}.pkl"
            if not f.exists():  # pragma: no cover - network
                try:
                    ev = parse_form345(get_bytes(FORM345_URL.format(y=y, q=q), user_agent, timeout=300))
                except Exception as e:
                    print(f"Form 4 {y}q{q} 下载/解析失败：{e}")
                    continue
                ev.to_pickle(f)
                print(f"Form 4 {y}q{q}：{len(ev)} 笔")
            frames.append(pd.read_pickle(f))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def insider_fields(
    events: pd.DataFrame,
    dates: pd.DatetimeIndex,
    tickers: pd.Index,
    cik_to_ticker: dict[int, str] | None = None,
    window: int = 126,
    lag_days: int = 1,
) -> dict[str, pd.DataFrame]:
    """生成交易日宽表：

    insider_net_ratio  过去 window 日（按可获得日）(买入金额 - 卖出金额) / (买入 + 卖出)，无交易记 0
    insider_buyers     过去 window 日内有公开市场买入的内部人数（同一人多日买入会重复计数）
    数据集开始之前的日期为 NaN。
    """
    ev = events.copy()
    start = ev["filed"].min() if len(ev) else dates[-1]
    cmap = cik_to_ticker or {}
    ev["ticker"] = ev["cik"].map(lambda c: cmap.get(int(c)) if pd.notna(c) else None)
    ev["ticker"] = ev["ticker"].fillna(ev["symbol"])
    ev = ev[ev["ticker"].isin(tickers)]
    pos = dates.searchsorted(ev["filed"].to_numpy(), side="left") + lag_days
    ev = ev.assign(pos=pos)[pos < len(dates)]

    def wide(df: pd.DataFrame, col: str, agg: str) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(0.0, index=dates, columns=tickers)
        w = df.pivot_table(index="pos", columns="ticker", values=col, aggfunc=agg)
        w.index = dates[w.index]
        return w.reindex(index=dates, columns=tickers).fillna(0.0)

    buys, sells = ev[ev["code"] == "P"], ev[ev["code"] == "S"]
    bv = wide(buys.assign(v=buys["value"].abs()), "v", "sum").rolling(window, min_periods=1).sum()
    sv = wide(sells.assign(v=sells["value"].abs()), "v", "sum").rolling(window, min_periods=1).sum()
    nb = wide(buys.drop_duplicates(["ticker", "pos", "owner"]).assign(one=1.0), "one", "sum")
    nb = nb.rolling(window, min_periods=1).sum()
    tot = bv + sv
    ratio = ((bv - sv) / tot.where(tot > 0)).fillna(0.0)
    before = dates < start
    ratio.loc[before] = np.nan
    nb.loc[before] = np.nan
    return {"insider_net_ratio": ratio, "insider_buyers": nb}
