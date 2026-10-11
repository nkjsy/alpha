"""另类数据因子：内部人交易（Form 4）与机构持仓（13F）。字段由 alpha.data.insider / alpha.data.thirteenf 生成。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha.data.market import MarketData
from alpha.factors.base import Factor, register


@register
class InsiderNetRatio(Factor):
    """过去半年内部人公开市场 (买入额 - 卖出额) / (买入额 + 卖出额)，越高越好（Lakonishok & Lee 2001）。"""

    name = "insider_net_ratio"

    def compute(self, data: MarketData) -> pd.DataFrame:
        return data.field("insider_net_ratio")


@register
class InsiderBuyers(Factor):
    """过去半年有公开市场买入的内部人数，取 log(1+x)。内部人买入比卖出信息量大得多（Cohen 等 2012）。"""

    name = "insider_buyers"

    def compute(self, data: MarketData) -> pd.DataFrame:
        return np.log1p(data.field("insider_buyers"))


@register
class InstBreadthChange(Factor):
    """持有机构比例的季度变化（Chen, Hong & Stein 2002），越高越好。"""

    name = "inst_breadth_chg"

    def compute(self, data: MarketData) -> pd.DataFrame:
        return data.field("inst_breadth_chg")
