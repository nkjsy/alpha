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


def download_yahoo_ohlcv(
    tickers: list[str], cache: str | Path, refresh: bool = False, chunk: int = 50, start: str | None = None,
    fields: tuple[str, ...] = ("open", "close", "adj_close", "volume", "splits"), dtype: str | None = None,
) -> dict[str, pd.DataFrame]:
    """分批下载 yfinance 日线，返回 {字段: 宽表}，并缓存到 pickle。

    字段：open / close（只做拆股调整）、adj_close（拆股 + 分红调整，用于算收益）、volume、splits。
    市值要用 close（不含分红调整）乘股本，否则历史市值会被未来分红压低。
    start 为空时下载全部历史；股票很多时指定 start、只取需要的 fields 以节省内存。
    """
    cache = Path(cache)
    if cache.exists() and not refresh:
        return pd.read_pickle(cache)
    import yfinance as yf

    cols = {"open": "Open", "close": "Close", "adj_close": "Adj Close", "volume": "Volume", "splits": "Stock Splits"}
    cols = {k: v for k, v in cols.items() if k in fields}
    period = {"start": start} if start else {"period": "max"}
    parts: dict[str, list[pd.DataFrame]] = {k: [] for k in cols}
    for i in range(0, len(tickers), chunk):
        sub = tickers[i : i + chunk]
        raw = yf.download(sub, **period, interval="1d", auto_adjust=False, actions=True, progress=False,
                          group_by="column")
        for field, col in cols.items():
            if isinstance(raw.columns, pd.MultiIndex):
                df = raw[col] if col in raw.columns.get_level_values(0) else pd.DataFrame(index=raw.index)
            else:
                df = raw[[col]].rename(columns={col: sub[0]}) if col in raw.columns else pd.DataFrame(index=raw.index)
            parts[field].append(df.astype(dtype) if dtype and field != "splits" else df)
        print(f"下载 {min(i + chunk, len(tickers))}/{len(tickers)}")
    out = {}
    for field, frames in parts.items():
        df = pd.concat(frames, axis=1)
        df.index = pd.to_datetime(df.index).tz_localize(None)
        out[field] = df.loc[:, ~df.columns.duplicated()].sort_index()
    cache.parent.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(out, cache)
    return out


def market_from_yahoo_raw(raw: dict[str, pd.DataFrame], universe: pd.DataFrame | None = None,
                          fields: dict[str, pd.DataFrame] | None = None) -> MarketData:
    """把 download_yahoo_ohlcv 的结果组装成 MarketData：收益用分红复权价，开盘价按同一比例复权。"""
    adj = raw["adj_close"].dropna(how="all", axis=1)
    ratio = adj / raw["close"].reindex_like(adj)
    return MarketData(close=adj, open=raw["open"].reindex_like(adj) * ratio, volume=raw["volume"].reindex_like(adj),
                      universe=universe, fields=fields or {})
