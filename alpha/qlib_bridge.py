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


def _allow_mlflow_file_store() -> None:
    # qlib 的实验记录默认写本地 ./mlruns，新版 mlflow 需要显式允许文件存储
    import os

    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")


def init_qlib(provider_uri: str | Path) -> None:
    import qlib
    from qlib.constant import REG_US

    _allow_mlflow_file_store()
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


def alpha158_frames(market: str, start: str, end: str, dates: pd.DatetimeIndex, batch: int = 20,
                    prefix: str = "a158_") -> dict[str, pd.DataFrame]:
    """分批计算 Alpha158，只保留 dates（如月末调仓日），控制内存。

    Alpha158 在 t 日只用到 t 日收盘及以前的数据，配合 T+1 执行没有前视。
    """
    exprs = alpha158_expressions()
    names = list(exprs)
    out: dict[str, pd.DataFrame] = {}
    for i in range(0, len(names), batch):
        part = {n: exprs[n] for n in names[i : i + batch]}
        for n, w in expression_frames(part, market, start, end).items():
            out[prefix + n] = w.reindex(dates).astype(float)
    return out


def _long(frames: dict[str, pd.DataFrame], group: str) -> pd.DataFrame:
    cols = {}
    for name, w in frames.items():
        try:
            s = w.stack(future_stack=True)
        except TypeError:  # pandas < 2.1
            s = w.stack(dropna=False)
        cols[(group, name)] = s
    df = pd.DataFrame(cols)
    df.index = df.index.set_names(["datetime", "instrument"])
    return df


def train_lgb_score(
    features: dict[str, pd.DataFrame],
    label: pd.DataFrame,
    train: pd.DatetimeIndex,
    valid: pd.DatetimeIndex,
    test: pd.DatetimeIndex,
    **lgb_params,
) -> pd.DataFrame:
    """用 qlib 的 DatasetH + LGBModel 训练截面打分模型，返回验证期和测试期的打分宽表。

    features  {名字: 日期 × 代码}（建议已做截面标准化，股票池外为 NaN）
    label     与调仓匹配的未来收益（日期 × 代码），训练时逐日转成截面排序分位，减少极端值影响
    train / valid / test  调仓日；调用方负责在各段之间留出 horizon 长度的隔离
    """
    import tempfile

    import qlib
    from qlib.config import C
    from qlib.contrib.model.gbdt import LGBModel
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.data.dataset.loader import StaticDataLoader

    _allow_mlflow_file_store()
    if not C.registered:  # 模型训练会写 qlib 的实验记录，需要先 init（不读行情数据，空目录即可）
        tmp = tempfile.mkdtemp(prefix="qlib_", dir=tempfile.gettempdir())
        qlib.init(provider_uri=tmp, exp_manager={"class": "MLflowExpManager", "module_path": "qlib.workflow.expm",
                                                 "kwargs": {"uri": (Path(tmp) / "mlruns").as_uri(),
                                                            "default_exp_name": "alpha"}})
    dates = train.union(valid).union(test)
    feats = {k: v.reindex(dates) for k, v in features.items()}
    lab = label.reindex(dates).rank(axis=1, pct=True) - 0.5
    # 只保留至少有一个特征的股票日
    any_feat = sum(v.notna().astype(int) for v in feats.values()) > 0
    lab = lab.where(any_feat)
    df = pd.concat([_long(feats, "feature"), _long({"LABEL0": lab}, "label")], axis=1)
    df = df[df["feature"].notna().any(axis=1)].sort_index()

    handler = DataHandlerLP(data_loader=StaticDataLoader(config=df), learn_processors=[{"class": "DropnaLabel"}])
    seg = {"train": (train[0], train[-1]), "valid": (valid[0], valid[-1]), "test": (test[0], test[-1])}
    ds = DatasetH(handler, segments=seg)
    params = {"learning_rate": 0.05, "num_leaves": 31, "max_depth": 6, "subsample": 0.8, "colsample_bytree": 0.8,
              "lambda_l2": 10.0, "num_threads": 4, "early_stopping_rounds": 50, "num_boost_round": 500}
    params.update(lgb_params)
    model = LGBModel(loss="mse", **params)
    from qlib.workflow import R

    with R.start(experiment_name="alpha_lgb"):   # 训练曲线记到 qlib 的实验记录里，结束后释放
        model.fit(ds)
        pred = pd.concat([model.predict(ds, "valid"), model.predict(ds, "test")])
    pred = pred[~pred.index.duplicated()]
    return pred.unstack(level="instrument").reindex(columns=label.columns)
