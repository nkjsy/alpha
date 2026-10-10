"""从维基百科重建标普500每月末历史成分（当前成分 + 成分变更记录倒推）。

输出与纳指100成分文件相同的格式（month_end_date, ticker），可直接给 mine_factors.py 的 --membership。

局限：维基百科的变更表在 2000 年代后比较完整，但个别变更可能缺漏；退市/改名股票在 Yahoo
可能取不到价格。正式研究建议用 CRSP、Norgate 等专业数据核对。

用法：python scripts/build_sp500_membership.py --start 2010-01-01 --out data/sp500_monthly_constituents.csv
"""

from __future__ import annotations

import argparse
import io
import urllib.request

import pandas as pd

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
# 变更表有时被拆到单独页面
HISTORY_URL = "https://en.wikipedia.org/wiki/Historical_components_of_the_S%26P_500"


def normalize(t) -> str | None:
    if t is None or (isinstance(t, float) and pd.isna(t)):
        return None
    t = str(t).strip().upper().replace(".", "-")
    return t or None


def _read(url: str) -> list[pd.DataFrame]:
    req = urllib.request.Request(url, headers={"User-Agent": "alpha-research/0.1"})
    html = urllib.request.urlopen(req, timeout=60).read().decode("utf-8")
    return pd.read_html(io.StringIO(html))


def _flat(cols) -> str:
    return " ".join(str(x) for c in cols for x in (c if isinstance(c, tuple) else (c,))).lower()


def _is_changes(t: pd.DataFrame) -> bool:
    text = _flat(t.columns)
    return "added" in text and "removed" in text and "date" in text


def fetch_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    """返回 (当前成分表, 变更表)。按列名识别表格，主页面找不到变更表时去历史成分页面找。"""
    tables = _read(WIKI_URL)
    current = next(t for t in tables if any(str(c).lower() in ("symbol", "ticker") for c in t.columns))
    changes = next((t for t in tables if _is_changes(t)), None)
    if changes is None:
        changes = next((t for t in _read(HISTORY_URL) if _is_changes(t)), None)
    if changes is None:
        raise RuntimeError("在维基百科页面上找不到标普500成分变更表，页面结构可能又变了")
    return current, changes


def parse_changes(changes: pd.DataFrame) -> pd.DataFrame:
    """把维基百科变更表整理成 date, added, removed 三列。"""
    df = changes.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [" ".join(dict.fromkeys(str(x) for x in c)).strip() for c in df.columns]
    cols = {c.lower(): c for c in df.columns}
    date_col = next(cols[k] for k in cols if "date" in k)
    add_col = next(cols[k] for k in cols if k.startswith("added") and ("ticker" in k or "symbol" in k))
    rem_col = next(cols[k] for k in cols if k.startswith("removed") and ("ticker" in k or "symbol" in k))
    out = pd.DataFrame({
        "date": pd.to_datetime(df[date_col], errors="coerce"),
        "added": df[add_col].map(normalize),
        "removed": df[rem_col].map(normalize),
    })
    return out.dropna(subset=["date"])


def reconstruct(current: list[str], changes: pd.DataFrame, start: str, end: str | None = None) -> pd.DataFrame:
    """从当前成分出发，按时间倒序撤销变更，得到每个月末的成分。

    某月末 M 的成分 = 当前成分撤销所有生效日 > M 的变更（撤销：去掉 added，加回 removed）。
    """
    members = {normalize(t) for t in current} - {None}
    ch = changes.sort_values("date", ascending=False).reset_index(drop=True)
    month_ends = pd.date_range(start, end or pd.Timestamp.today().normalize(), freq="ME")[::-1]
    rows = []
    i = 0
    for m in month_ends:
        while i < len(ch) and ch.loc[i, "date"] > m:
            added, removed = ch.loc[i, "added"], ch.loc[i, "removed"]
            if isinstance(added, str):
                members.discard(added)
            if isinstance(removed, str):
                members.add(removed)
            i += 1
        rows.extend((m.date(), t) for t in sorted(members))
    out = pd.DataFrame(rows, columns=["month_end_date", "ticker"])
    return out.sort_values(["month_end_date", "ticker"]).reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--out", default="data/sp500_monthly_constituents.csv")
    args = p.parse_args()
    current, changes = fetch_tables()
    sym_col = next(c for c in current.columns if str(c).lower() in ("symbol", "ticker"))
    df = reconstruct(current[sym_col].tolist(), parse_changes(changes), args.start)
    sizes = df.groupby("month_end_date").size()
    print(f"{df['month_end_date'].min()} ~ {df['month_end_date'].max()}，每月成分数 {sizes.min()}~{sizes.max()}，"
          f"历史上出现过的股票 {df['ticker'].nunique()} 只")
    df.to_csv(args.out, index=False)
    print(f"已写入 {args.out}")


if __name__ == "__main__":
    main()
