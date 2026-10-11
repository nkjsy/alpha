"""机构持仓（13F），来自 SEC 的 Form 13F Data Sets（2013 年第 2 季度起）。

因子思路（Chen, Hong & Stein 2002）：持有某只股票的机构比例（breadth）上升，说明更多机构看好、
卖空约束下价格尚未充分反映，后续收益偏高。

防前视的处理：
- 只统计原始 13F-HR（不含 13F-HR/A 修正），且只统计在截止日（季末 + 45 天，周末顺延，这里放宽到 47 天）前提交的；
- 某季度的数据在截止日之后的下一个交易日才算可获得。

13F 用 CUSIP 标识股票，这里用 SEC 卖空交割失败（fails-to-deliver）数据里的 CUSIP-代码对照表转换成 ticker，
同一 CUSIP 取最近一次出现的代码（公司改名换代码时 CUSIP 通常不变，最近的代码就是现在的 ticker）。
"""

from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from alpha.data.insider import _read_tsv, parse_sec_date

FORM13F_PAGE = "https://www.sec.gov/data-research/sec-markets-data/form-13f-data-sets"
FTD_URL = "https://www.sec.gov/files/data/fails-deliver-data/cnsfails{ym}{half}.zip"
DEADLINE_DAYS = 47


def _cusip8(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip().str.upper().str.zfill(9).str[:8]


def parse_ftd(data: bytes | str | Path) -> pd.DataFrame:
    """一个 fails-to-deliver zip -> (date, cusip8, symbol)。"""
    z = zipfile.ZipFile(io.BytesIO(data) if isinstance(data, bytes) else data)
    with z.open(z.namelist()[0]) as f:
        df = pd.read_csv(f, sep="|", dtype=str, encoding="latin-1", on_bad_lines="skip")
    df.columns = [c.strip().upper() for c in df.columns]
    out = pd.DataFrame({
        "date": pd.to_datetime(df["SETTLEMENT DATE"], format="%Y%m%d", errors="coerce"),
        "cusip8": _cusip8(df["CUSIP"]),
        "symbol": df["SYMBOL"].fillna("").str.strip().str.upper().str.replace(r"[./]", "-", regex=True),
    })
    return out[(out["symbol"] != "") & out["date"].notna()].drop_duplicates(["cusip8", "symbol"], keep="last")


def cusip_map(ftd: pd.DataFrame) -> dict[str, str]:
    """cusip8 -> 最近一次出现的代码。"""
    last = ftd.sort_values("date").drop_duplicates("cusip8", keep="last")
    return dict(zip(last["cusip8"], last["symbol"]))


def parse_13f(data: bytes | str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """一个 13F 数据集 zip -> (持有人数 (period, cusip8, n_holders), 申报机构数 (period, n_filers))。

    只计按时提交的原始 13F-HR；同一机构同一季度对同一 CUSIP 的多行只算一次；不含期权（PUTCALL）和债券（PRN）。
    """
    z = zipfile.ZipFile(io.BytesIO(data) if isinstance(data, bytes) else data)
    sub = _read_tsv(z, "SUBMISSION.TSV", ["ACCESSION_NUMBER", "FILING_DATE", "SUBMISSIONTYPE", "CIK", "PERIODOFREPORT"])
    sub = sub[sub["SUBMISSIONTYPE"].str.strip() == "13F-HR"].copy()
    sub["filed"] = parse_sec_date(sub["FILING_DATE"])
    sub["period"] = parse_sec_date(sub["PERIODOFREPORT"])
    sub = sub[sub["filed"] <= sub["period"] + pd.Timedelta(days=DEADLINE_DAYS)]
    sub = sub.drop_duplicates(["CIK", "period"], keep="first")[["ACCESSION_NUMBER", "CIK", "period"]]

    names = {Path(n).name.upper(): n for n in z.namelist()}
    keep = []
    with z.open(names["INFOTABLE.TSV"]) as f:
        reader = pd.read_csv(f, sep="\t", dtype=str, quoting=3, encoding="latin-1", on_bad_lines="skip", chunksize=500_000,
                             usecols=lambda c: c in {"ACCESSION_NUMBER", "CUSIP", "SSHPRNAMTTYPE", "PUTCALL"})
        acc = set(sub["ACCESSION_NUMBER"])
        for ch in reader:
            ch = ch[ch["ACCESSION_NUMBER"].isin(acc)]
            ch = ch[(ch["SSHPRNAMTTYPE"].fillna("SH").str.strip() == "SH") & ch["PUTCALL"].fillna("").str.strip().eq("")]
            keep.append(pd.DataFrame({"ACCESSION_NUMBER": ch["ACCESSION_NUMBER"], "cusip8": _cusip8(ch["CUSIP"])}))
    info = pd.concat(keep, ignore_index=True) if keep else pd.DataFrame(columns=["ACCESSION_NUMBER", "cusip8"])
    info = info.drop_duplicates().merge(sub, on="ACCESSION_NUMBER")
    holders = info.groupby(["period", "cusip8"]).size().rename("n_holders").reset_index()
    filers = sub.groupby("period").size().rename("n_filers").reset_index()
    return holders, filers


def list_13f_zips(user_agent: str) -> list[str]:  # pragma: no cover - network
    from alpha.data.sec import get_bytes

    html = get_bytes(FORM13F_PAGE, user_agent).decode("utf-8", "ignore")
    links = sorted(set(re.findall(r'href="([^"]+_form13f\.zip)"', html)))
    return [l if l.startswith("http") else "https://www.sec.gov" + l for l in links]


def fetch_13f(user_agent: str, cache_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """下载全部 13F 数据集，按文件缓存汇总结果；返回合并后的 (holders, filers)。"""
    from alpha.data.sec import get_bytes

    cache = Path(cache_dir) / "form13f"
    cache.mkdir(parents=True, exist_ok=True)
    try:
        urls = list_13f_zips(user_agent)
    except Exception as e:  # pragma: no cover - network
        print(f"13F 数据集列表获取失败：{e}，只用已缓存的文件")
        urls = []
    for url in urls:  # pragma: no cover - network
        f = cache / (Path(url).stem + ".pkl")
        if f.exists():
            continue
        try:
            h, n = parse_13f(get_bytes(url, user_agent, timeout=600))
        except Exception as e:
            print(f"13F {Path(url).name} 下载/解析失败：{e}")
            continue
        pd.to_pickle((h, n), f)
        print(f"13F {Path(url).name}：{len(n)} 个季度，{len(h)} 行")
    hs, ns = [], []
    for f in sorted(cache.glob("*.pkl")):
        h, n = pd.read_pickle(f)
        hs.append(h)
        ns.append(n)
    if not hs:
        return pd.DataFrame(columns=["period", "cusip8", "n_holders"]), pd.DataFrame(columns=["period", "n_filers"])
    # 一个季度的申报可能分散在相邻两个数据集文件里，相加即可（原始 13F-HR 每家机构每季度只有一份）
    holders = pd.concat(hs).groupby(["period", "cusip8"], as_index=False)["n_holders"].sum()
    filers = pd.concat(ns).groupby("period", as_index=False)["n_filers"].sum()
    return holders, filers


def fetch_cusip_map(start: str, end: pd.Timestamp, user_agent: str, cache_dir: str | Path) -> dict[str, str]:
    """下载每半月一份的 fails-to-deliver 文件，生成 cusip8 -> ticker 对照表。"""
    from alpha.data.sec import get_bytes

    cache = Path(cache_dir) / "ftd"
    cache.mkdir(parents=True, exist_ok=True)
    frames = []
    for m in pd.period_range(start, end, freq="M"):
        for half in ("a", "b"):
            f = cache / f"{m.strftime('%Y%m')}{half}.pkl"
            if not f.exists():  # pragma: no cover - network
                try:
                    df = parse_ftd(get_bytes(FTD_URL.format(ym=m.strftime("%Y%m"), half=half), user_agent))
                except Exception as e:
                    if getattr(e, "code", None) != 404:
                        print(f"FTD {m}{half} 下载失败：{e}")
                    continue
                df.to_pickle(f)
            frames.append(pd.read_pickle(f))
    return cusip_map(pd.concat(frames, ignore_index=True)) if frames else {}


def breadth_fields(
    holders: pd.DataFrame,
    filers: pd.DataFrame,
    cmap: dict[str, str],
    dates: pd.DatetimeIndex,
    tickers: pd.Index,
    lag_days: int = 1,
    max_staleness: int = 130,
) -> dict[str, pd.DataFrame]:
    """inst_breadth_chg：持有该股的机构占全部申报机构比例的季度变化，截止日后可获得。"""
    h = holders.assign(ticker=holders["cusip8"].map(cmap))
    h = h[h["ticker"].isin(tickers)]
    h = h.groupby(["period", "ticker"], as_index=False)["n_holders"].sum()
    h = h.merge(filers, on="period")
    h["breadth"] = h["n_holders"] / h["n_filers"]
    wide = h.pivot_table(index="period", columns="ticker", values="breadth").sort_index()
    full = wide.reindex(pd.DatetimeIndex(sorted(filers["period"].unique())))
    # 上一季度没出现视为 0 个机构持有（新上市或新进入），本季度没出现同理
    full = full.fillna(0.0)
    chg = full.diff().iloc[1:]
    chg = chg.where(full.iloc[1:] + full.shift().iloc[1:] > 0)
    avail = chg.index + pd.Timedelta(days=DEADLINE_DAYS)
    pos = dates.searchsorted(avail, side="right") + lag_days - 1
    ok = pos < len(dates)
    chg = chg[ok]
    chg.index = dates[np.maximum(pos[ok], 0)]
    chg = chg.groupby(level=0).last()
    out = chg.reindex(index=dates, columns=tickers).ffill(limit=max_staleness)
    return {"inst_breadth_chg": out}
