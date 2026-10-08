"""日频组合回测：带执行延迟、交易成本、持仓漂移。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CostModel:
    """单边交易成本（基点）。

    commission_bps  佣金 + 交易所费用
    spread_bps      半个买卖价差
    impact_bps      冲击成本的固定近似
    borrow_bps_annual 空头借券费（年化）
    """

    commission_bps: float = 1.0
    spread_bps: float = 5.0
    impact_bps: float = 4.0
    borrow_bps_annual: float = 50.0

    @property
    def one_way(self) -> float:
        return (self.commission_bps + self.spread_bps + self.impact_bps) / 1e4


@dataclass
class BacktestResult:
    returns: pd.Series          # 扣成本后的日收益
    gross_returns: pd.Series    # 扣成本前
    costs: pd.Series            # 每日成本（占净值比例）
    turnover: pd.Series         # 每次调仓的单边换手（占净值比例）
    weights: pd.DataFrame       # 每日收盘后的实际权重
    nav: pd.Series

    def summary(self, periods_per_year: int = 252) -> dict[str, float]:
        from alpha.metrics import performance_summary

        s = performance_summary(self.returns, periods_per_year)
        g = performance_summary(self.gross_returns, periods_per_year)
        years = len(self.returns) / periods_per_year
        s["gross_ann_return"] = g["ann_return"]
        s["gross_sharpe"] = g["sharpe"]
        s["annual_turnover"] = self.turnover.sum() / years if years > 0 else np.nan
        s["annual_cost"] = self.costs.sum() / years if years > 0 else np.nan
        return s


def run_backtest(
    target_weights: pd.DataFrame,
    close: pd.DataFrame,
    lag: int = 1,
    costs: CostModel | None = None,
    tradable: pd.DataFrame | None = None,
    delisting_return: float = 0.0,
) -> BacktestResult:
    """在 close 价上模拟。

    target_weights  调仓信号日的目标权重（index 为信号日）
    lag             信号日之后第 lag 个交易日收盘成交（默认 1，避免用信号当天收盘价成交）
    tradable        可交易掩码；不可交易的股票保持原有仓位
    delisting_return 持仓股价格中断（退市）时计入的收益，之后清仓。
                    默认 0 偏乐观，有退市收益数据时应替换。
    """
    costs = costs or CostModel()
    close = close.sort_index()
    dates = close.index
    tickers = close.columns
    rets = close.pct_change(fill_method=None).to_numpy()
    px_ok = close.notna().to_numpy()
    trd = tradable.reindex(index=dates, columns=tickers).fillna(False).to_numpy() if tradable is not None else px_ok

    tw = target_weights.reindex(columns=tickers).fillna(0.0)
    exec_pos = dates.get_indexer(tw.index)
    if (exec_pos < 0).any():
        raise ValueError("target_weights 的日期必须是交易日")
    exec_pos = exec_pos + lag
    schedule = {int(p): tw.iloc[i].to_numpy() for i, p in enumerate(exec_pos) if p < len(dates)}

    n, m = close.shape
    pos = np.zeros(m)          # 持仓市值（以初始净值 1 为单位）
    nav = 1.0
    out_ret = np.zeros(n)
    out_gross = np.zeros(n)
    out_cost = np.zeros(n)
    out_w = np.zeros((n, m))
    turnover = {}
    borrow_daily = costs.borrow_bps_annual / 1e4 / 252

    for t in range(n):
        # 1) 当日收益作用于昨日收盘持仓
        if t > 0 and pos.any():
            r = rets[t].copy()
            dead = (pos != 0) & ~px_ok[t]
            r[dead] = delisting_return
            r = np.nan_to_num(r, nan=0.0)
            pnl = float(pos @ r)
            borrow = float(-pos[pos < 0].sum()) * borrow_daily
            pos = pos * (1 + r)
            pos[dead] = 0.0
            gross = pnl / nav
            nav_new = nav + pnl - borrow
            out_gross[t] = gross
            out_ret[t] = (nav_new - nav) / nav
            out_cost[t] = borrow / nav
            nav = nav_new
        # 2) 收盘调仓
        if t in schedule:
            can = trd[t]
            # 目标仓位按扣除成本后的净值计算（成本依赖于仓位，迭代几次即收敛）
            nav_post = nav
            for _ in range(5):
                target = np.where(can, schedule[t] * nav_post, pos)  # 不可交易的保持原仓位
                trade = np.abs(target - pos).sum()
                cost = trade * costs.one_way
                nav_post = nav - cost
            turnover[dates[t]] = trade / nav / 2 if nav > 0 else np.nan
            pos = target
            nav -= cost
            # 成本记在调仓当日
            out_ret[t] = (1 + out_ret[t]) * (1 - cost / (nav + cost)) - 1
            out_cost[t] += cost / (nav + cost)
        out_w[t] = pos / nav if nav > 0 else 0.0

    idx = dates
    ret = pd.Series(out_ret, index=idx)
    first = min(schedule) if schedule else n
    sl = slice(first, None)
    return BacktestResult(
        returns=ret.iloc[sl],
        gross_returns=pd.Series(out_gross, index=idx).iloc[sl],
        costs=pd.Series(out_cost, index=idx).iloc[sl],
        turnover=pd.Series(turnover, dtype=float),
        weights=pd.DataFrame(out_w, index=idx, columns=tickers).iloc[sl],
        nav=(1 + ret.iloc[sl]).cumprod(),
    )
