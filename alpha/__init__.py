"""alpha: 美股横截面多因子研究框架。

数据约定：所有面板数据都是"宽表" DataFrame，index 为交易日 (DatetimeIndex)，
columns 为股票代码。因子值 f.loc[t] 只能使用 t 日收盘及之前的信息。
"""

from alpha.data.market import MarketData

__all__ = ["MarketData"]
__version__ = "0.1.0"
