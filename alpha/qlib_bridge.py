"""与 Microsoft qlib 的桥接：把本框架的数据写成 qlib 的二进制格式，再用 qlib 的表达式引擎、
因子库（Alpha158）和模型。

分工：
- qlib 负责通用部分：数据存储、表达式因子、机器学习模型；
- 本框架保留 qlib 没有的部分：按披露日对齐的 SEC 财报 / 内部人 / 13F 字段、历史成分股池、
  增量 IC + 多重检验 + Deflated Sharpe 的防过拟合检验。

写出的目录结构与 qlib 的 dump_bin 一致：
  calendars/day.txt                    交易日
  instruments/<market>.txt             代码<TAB>开始<TAB>结束（按成分快照切成多段）
  features/<code小写>/<字段>.day.bin    float32 小端，第一个数是该股票在日历中的起始下标

价格字段按 qlib 的约定写复权价：$open/$high/$low/$close 为分红拆股复权价，$factor = 复权价 / 原始价，
$volume 为复权后的成交量（原始成交量 / factor），$vwap 用 (高 + 低 + 收) / 3 近似。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from alpha.data.market import MarketData

_WIN_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(10)), *(f"LPT{i}" for i in range(10))}


def code_to_fname(code: str) -> str:
    return f"_qlib_{code}" if code.upper() in _WIN_RESERVED else code


def qlib_price_fields(raw: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """download_yahoo_ohlcv 的结果（需含 high/low）-> qlib 约定的复权价格字段。"""
    adj = raw["adj_close"].dropna(how="all", axis=1).astype(float)
    close = raw["close"].reindex_like(adj).astype(float)
    factor = adj / close
    out = {"close": adj, "factor": factor}
    for f in ("open", "high", "low"):
        if f in raw:
            out[f] = raw[f].reindex_like(adj).astype(float) * factor
    if "volume" in raw:
        out["volume"] = raw["volume"].reindex_like(adj).astype(float) / factor
    if {"high", "low"} <= out.keys():
        out["vwap"] = (out["high"] + out["low"] + out["close"]) / 3
    return out


def membership_spans(universe: pd.DataFrame) -> pd.DataFrame:
    """每日股票池布尔宽表 -> (code, start, end) 连续区间，供 instruments 文件使用。"""
    rows = []
    u = universe.fillna(False).astype(bool)
    dates = u.index
    for code in u.columns:
        x = u[code].to_numpy()
        if not x.any():
            continue
        d = np.diff(np.concatenate([[0], x.astype(int), [0]]))
        starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1
        rows.extend((code, dates[s], dates[e]) for s, e in zip(starts, ends))
    return pd.DataFrame(rows, columns=["code", "start", "end"])


def dump_qlib(fields: dict[str, pd.DataFrame], out_dir: str | Path,
              markets: dict[str, pd.DataFrame] | None = None) -> Path:
    """把 {字段: 宽表} 写成 qlib 目录。markets 为 {市场名: 每日股票池布尔宽表}，另外总会写 all.txt。"""
    out = Path(out_dir)
    dates = None
    for df in fields.values():
        dates = df.index if dates is None else dates.union(df.index)
    dates = pd.DatetimeIndex(dates).sort_values()
    (out / "calendars").mkdir(parents=True, exist_ok=True)
    (out / "instruments").mkdir(parents=True, exist_ok=True)
    pd.Series(dates.strftime("%Y-%m-%d")).to_csv(out / "calendars" / "day.txt", index=False, header=False)

    codes = sorted(set().union(*[df.columns for df in fields.values()]))
    spans_all = []
    for code in codes:
        fdir = out / "features" / code_to_fname(code).lower()
        first = last = None
        for name, df in fields.items():
            if code not in df.columns:
                continue
            s = df[code].reindex(dates)
            valid = s.notna().to_numpy()
            if not valid.any():
                continue
            i0, i1 = int(valid.argmax()), len(valid) - 1 - int(valid[::-1].argmax())
            fdir.mkdir(parents=True, exist_ok=True)
            np.hstack([[i0], s.to_numpy()[i0 : i1 + 1]]).astype("<f").tofile(fdir / f"{name.lower()}.day.bin")
            if name == "close":
                first, last = dates[i0], dates[i1]
        if first is not None:
            spans_all.append((code, first, last))

    def write_market(name: str, spans: pd.DataFrame) -> None:
        spans = spans.assign(start=pd.to_datetime(spans["start"]).dt.strftime("%Y-%m-%d"),
                             end=pd.to_datetime(spans["end"]).dt.strftime("%Y-%m-%d"))
        spans.to_csv(out / "instruments" / f"{name.lower()}.txt", sep="\t", index=False, header=False)

    write_market("all", pd.DataFrame(spans_all, columns=["code", "start", "end"]))
    for name, u in (markets or {}).items():
        write_market(name, membership_spans(u.reindex(index=dates).fillna(False)))
    return out


def dump_market_data(data: MarketData, raw: dict[str, pd.DataFrame], out_dir: str | Path,
                     market: str = "universe", extra_fields: list[str] | None = None) -> Path:
    """MarketData（含 SEC/另类数据字段）+ 原始价格 -> qlib 目录。extra_fields 为空时写出全部 fields。"""
    fields = qlib_price_fields(raw)
    names = extra_fields if extra_fields is not None else list(data.fields)
    for n in names:
        if n in data.fields:
            fields[n.lower()] = data.fields[n].astype(float)
    markets = {market: data.universe} if data.universe is not None else {}
    return dump_qlib(fields, out_dir, markets)


def init_qlib(provider_uri: str | Path) -> None:
    import qlib
    from qlib.constant import REG_US

    qlib.init(provider_uri=str(provider_uri), region=REG_US, kernels=1)


def expression_frames(expressions: dict[str, str], market: str, start: str, end: str,
                      dates: pd.DatetimeIndex | None = None) -> dict[str, pd.DataFrame]:
    """用 qlib 表达式引擎计算一组因子，返回 {名字: 日期 × 代码 宽表}（只含 market 区间内的股票日）。"""
    from qlib.data import D

    inst = D.instruments(market)
    df = D.features(inst, list(expressions.values()), start_time=start, end_time=end, freq="day")
    df.columns = list(expressions)
    out = {}
    for name in expressions:
        w = df[name].unstack(level=0)
        w.index = pd.DatetimeIndex(w.index)
        out[name] = w.reindex(dates) if dates is not None else w
    return out


def alpha158_expressions() -> dict[str, str]:
    """qlib Alpha158 的 158 个价量因子表达式。"""
    from qlib.contrib.data.loader import Alpha158DL

    exprs, names = Alpha158DL.get_feature_config()
    return dict(zip(names, exprs))
