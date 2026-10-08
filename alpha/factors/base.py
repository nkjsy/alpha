from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from alpha.data.market import MarketData

FACTOR_REGISTRY: dict[str, type["Factor"]] = {}


def register(cls: type["Factor"]) -> type["Factor"]:
    """类装饰器：把因子注册进 FACTOR_REGISTRY，配置文件里用 name 引用。"""
    FACTOR_REGISTRY[cls.name] = cls
    return cls


class Factor(ABC):
    """因子基类。

    约定：
    - compute 返回与 data.close 同形状的宽表；
    - 值越大代表预期收益越高（反向因子在内部取负号）；
    - t 行只能用 t 日收盘及以前的数据（tests/test_factors.py 会做未来数据扰动检查）。
    """

    name: str = "base"

    def __init__(self, **params: Any) -> None:
        self.params = params

    @abstractmethod
    def compute(self, data: MarketData) -> pd.DataFrame: ...

    @property
    def label(self) -> str:
        if not self.params:
            return self.name
        p = ",".join(f"{k}={v}" for k, v in sorted(self.params.items()))
        return f"{self.name}({p})"

    def __repr__(self) -> str:
        return self.label


def build_factor(spec: str | dict) -> Factor:
    """由配置构造因子：'momentum' 或 {'name': 'momentum', 'lookback': 252}。"""
    if isinstance(spec, str):
        spec = {"name": spec}
    spec = dict(spec)
    name = spec.pop("name")
    if name not in FACTOR_REGISTRY:
        raise KeyError(f"未知因子 {name!r}，可选：{sorted(FACTOR_REGISTRY)}")
    return FACTOR_REGISTRY[name](**spec)
