"""带大盘择时的动量选股（迁移自本地 main_momentum_11_1_live.py）。

规则：
- 基准（QQQ）收盘 > N 日均线 → risk-on，持有动量 Top n_on（默认 3）
- 否则 risk-off，持有动量 Top n_off（默认 10）
- 每月末用当月历史成分做股票池刷新排名；risk-on 时可每日重排（实盘逻辑）

信号只用 t 日收盘及之前的数据；成交时点由 run_backtest 的 lag 决定。
"""

from __future__ import annotations

import pandas as pd


def trend_regime(bench_close: pd.Series, ma_days: int = 200) -> pd.Series:
    """收盘价 > ma_days 日均线为 True。均线未满窗口时为 False。"""
    ma = bench_close.rolling(ma_days, min_periods=ma_days).mean()
    return (bench_close > ma) & ma.notna()


def _top(row: pd.Series, n: int) -> list[str]:
    return row.dropna().sort_values(ascending=False, kind="mergesort").head(n).index.to_list()


def regime_switch_targets(
    score: pd.DataFrame,
    universe: pd.DataFrame,
    regime: pd.Series,
    rebalance_dates: pd.DatetimeIndex,
    n_on: int = 3,
    n_off: int = 10,
    on_rerank: str = "daily",
    off_rerank: str = "monthly",
    reweight_daily: bool = False,
) -> pd.DataFrame:
    """逐日生成目标权重（只在持仓需要变化的日子输出一行）。

    on_rerank / off_rerank  'daily'：每天按最新得分重排；'monthly'：沿用最近一次月末的排名
                            （本地回测脚本两种状态都是 monthly；实盘脚本 risk-on 为 daily）
    reweight_daily          True 时每天都恢复等权（本地回测的做法，会产生额外换手）
    """
    for v in (on_rerank, off_rerank):
        if v not in ("daily", "monthly"):
            raise ValueError(f"rerank 只能是 daily/monthly，得到 {v!r}")
    s = score.where(universe.reindex_like(score).fillna(False).astype(bool))
    reg = regime.reindex(s.index, method="ffill").fillna(False).astype(bool)
    reb = set(pd.DatetimeIndex(rebalance_dates))

    rows: dict[pd.Timestamp, pd.Series] = {}
    month_on: list[str] = []
    month_off: list[str] = []
    started = False
    prev: list[str] | None = None
    for d in s.index:
        if d in reb:
            month_on, month_off = _top(s.loc[d], n_on), _top(s.loc[d], n_off)
            started = True
        if not started:
            continue
        if reg.loc[d]:
            picks = _top(s.loc[d], n_on) if on_rerank == "daily" else month_on
        else:
            picks = _top(s.loc[d], n_off) if off_rerank == "daily" else month_off
        if reweight_daily or d in reb or prev is None or set(picks) != set(prev):
            w = pd.Series(0.0, index=s.columns)
            if picks:
                w[picks] = 1.0 / len(picks)
            rows[d] = w
        prev = picks
    if not rows:
        return pd.DataFrame(columns=s.columns, dtype=float)
    return pd.DataFrame(rows).T
