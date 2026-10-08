from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class MarketData:
    """横截面研究所需的全部数据，全部为对齐好的宽表。

    close     复权收盘价（必须）
    open      复权开盘价（可选）
    volume    成交量（股）（可选，流动性因子/过滤需要）
    universe  布尔宽表，point-in-time 的股票池成员（如历史纳指100成分）。
              缺省为“有价格即在池中”，这会引入幸存者偏差，正式研究必须提供。
    sector    股票 -> 行业 的映射（用于行业中性化），可随时间变化时传宽表
    fields    其它任意宽表，例如 market_cap、book_value（必须已按披露日对齐）
    """

    close: pd.DataFrame
    open: pd.DataFrame | None = None
    volume: pd.DataFrame | None = None
    universe: pd.DataFrame | None = None
    sector: pd.Series | pd.DataFrame | None = None
    fields: dict[str, pd.DataFrame] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.close = self.close.sort_index()
        idx, cols = self.close.index, self.close.columns
        for name in ("open", "volume", "universe"):
            df = getattr(self, name)
            if df is not None:
                setattr(self, name, df.reindex(index=idx, columns=cols))
        self.fields = {k: v.reindex(index=idx, columns=cols) for k, v in self.fields.items()}
        if self.universe is None:
            self.universe = self.close.notna()
        else:
            self.universe = self.universe.fillna(False).astype(bool) & self.close.notna()

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.close.index

    @property
    def tickers(self) -> pd.Index:
        return self.close.columns

    @property
    def returns(self) -> pd.DataFrame:
        """日收益（close-to-close）。"""
        return self.close.pct_change(fill_method=None)

    @property
    def dollar_volume(self) -> pd.DataFrame | None:
        if self.volume is None:
            return None
        return self.close * self.volume

    def field(self, name: str) -> pd.DataFrame:
        if name in self.fields:
            return self.fields[name]
        if name in ("close", "open", "volume"):
            df = getattr(self, name)
            if df is not None:
                return df
        raise KeyError(f"MarketData 中没有字段 {name!r}")

    def forward_returns(self, horizon: int, lag: int = 1, delisting_return: float = 0.0) -> pd.DataFrame:
        """t 日信号对应的未来收益：从 t+lag 收盘买入，持有 horizon 天。

        lag=1 表示 t 日收盘算出信号、t+1 日收盘成交，避免用当天收盘价成交的偏差。
        持有期内退市（价格中断）的股票不会被丢弃：收益算到最后一个价格，再叠加
        delisting_return。直接丢弃等于提前知道“谁会退市”，会让评估结果偏乐观。
        仅用于评估，绝不能作为因子输入。
        """
        entry = self.close.shift(-lag)
        exit_raw = self.close.shift(-(lag + horizon))
        # 数据末尾的行 shift 之后整行为 NaN，ffill 版本同样为 NaN，因此不会把“数据结束”误当成退市
        exit_last = self.close.ffill().shift(-(lag + horizon))
        fwd = exit_last / entry - 1.0
        delisted = exit_raw.isna() & exit_last.notna() & entry.notna()
        fwd = fwd.mask(delisted, (1 + fwd) * (1 + delisting_return) - 1)
        return fwd.where(entry.notna())

    def slice(self, start=None, end=None) -> "MarketData":
        sl = slice(start, end)

        def cut(df):
            return None if df is None else df.loc[sl]

        sector = self.sector
        if isinstance(sector, pd.DataFrame):
            sector = sector.loc[sl]
        return MarketData(
            close=self.close.loc[sl],
            open=cut(self.open),
            volume=cut(self.volume),
            universe=cut(self.universe),
            sector=sector,
            fields={k: v.loc[sl] for k, v in self.fields.items()},
        )

    def sector_frame(self) -> pd.DataFrame | None:
        """把行业信息统一成宽表。"""
        if self.sector is None:
            return None
        if isinstance(self.sector, pd.DataFrame):
            return self.sector.reindex(index=self.dates, columns=self.tickers)
        s = self.sector.reindex(self.tickers)
        return pd.DataFrame(
            np.tile(s.to_numpy(dtype=object), (len(self.dates), 1)),
            index=self.dates,
            columns=self.tickers,
        )
