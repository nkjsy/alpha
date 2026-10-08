from __future__ import annotations

from pathlib import Path

import pandas as pd

from alpha.data.market import MarketData
from alpha.data.universe import membership_mask

PRICE_FIELDS = ("open", "close", "volume")


def load_long_csv(path: str | Path, date_col: str = "date", ticker_col: str = "ticker") -> dict[str, pd.DataFrame]:
    """读取长表 CSV（每行一只股票一天），返回 {字段: 宽表}。

    期望至少有 date, ticker, close 三列；close 需为复权价。
    """
    df = pd.read_csv(path, parse_dates=[date_col])
    df = df.rename(columns={date_col: "date", ticker_col: "ticker"})
    value_cols = [c for c in df.columns if c not in ("date", "ticker")]
    wide = {}
    for c in value_cols:
        wide[c] = df.pivot_table(index="date", columns="ticker", values=c, aggfunc="last").sort_index()
    return wide


def load_membership_csv(path: str | Path) -> pd.DataFrame:
    """读取成分股区间表：ticker,start,end。"""
    return pd.read_csv(path)


def market_from_long_csv(
    prices_csv: str | Path,
    membership_csv: str | Path | None = None,
    sector_csv: str | Path | None = None,
) -> MarketData:
    """从本地 CSV 组装 MarketData。

    sector_csv 列：ticker,sector。
    价格 CSV 中除 open/close/volume 外的数值列（如 market_cap）进入 fields。
    """
    wide = load_long_csv(prices_csv)
    if "close" not in wide:
        raise ValueError("价格文件缺少 close 列")
    close = wide.pop("close")
    open_ = wide.pop("open", None)
    volume = wide.pop("volume", None)
    universe = None
    if membership_csv is not None:
        universe = membership_mask(load_membership_csv(membership_csv), close.index, close.columns)
    sector = None
    if sector_csv is not None:
        sector = pd.read_csv(sector_csv).set_index("ticker")["sector"]
    return MarketData(close=close, open=open_, volume=volume, universe=universe, sector=sector, fields=wide)


def market_from_yfinance(
    tickers: list[str],
    start: str,
    end: str | None = None,
    cache_dir: str | Path = "data/cache",
    membership_csv: str | Path | None = None,
) -> MarketData:
    """用 yfinance 下载复权日线（需要 pip install yfinance）。

    注意：yfinance 只有当前仍上市的股票，退市股缺失会造成幸存者偏差。
    正式回测请使用含退市股的数据源（CRSP、Norgate、Sharadar 等）。
    """
    try:
        import yfinance as yf
    except ImportError as e:  # pragma: no cover
        raise ImportError("请先 pip install yfinance") from e

    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    key = f"yf_{abs(hash((tuple(sorted(tickers)), start, end)))}.pkl"
    f = cache / key
    if f.exists():
        raw = pd.read_pickle(f)
    else:  # pragma: no cover - network
        raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False, group_by="column")
        raw.to_pickle(f)
    close = raw["Close"]
    universe = None
    if membership_csv is not None:
        universe = membership_mask(load_membership_csv(membership_csv), close.index, close.columns)
    return MarketData(close=close, open=raw.get("Open"), volume=raw.get("Volume"), universe=universe)
