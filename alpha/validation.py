"""样本外验证与试验记录，防止“挑最优变体”式的过拟合。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


@dataclass
class Split:
    train: tuple[pd.Timestamp, pd.Timestamp]
    test: tuple[pd.Timestamp, pd.Timestamp]


def holdout_split(dates: pd.DatetimeIndex, holdout_start: str) -> Split:
    """固定留出样本：holdout_start 之后的数据在研究阶段不许看。"""
    h = pd.Timestamp(holdout_start)
    train = dates[dates < h]
    test = dates[dates >= h]
    return Split((train[0], train[-1]), (test[0], test[-1]))


def walk_forward_splits(
    dates: pd.DatetimeIndex, train_years: float = 5, test_years: float = 1, expanding: bool = True,
    embargo_days: int = 21,
) -> list[Split]:
    """滚动/扩展窗口切分。embargo_days 在训练集末尾留空，避免未来收益标签跨界泄漏。"""
    splits = []
    start = dates[0]
    train_end = start + pd.DateOffset(years=train_years)
    while True:
        test_end = train_end + pd.DateOffset(years=test_years)
        tr = dates[(dates >= start) & (dates < train_end)]
        te = dates[(dates >= train_end) & (dates < test_end)]
        if len(te) == 0:
            break
        if embargo_days and len(tr) > embargo_days:
            tr = tr[:-embargo_days]
        splits.append(Split((tr[0], tr[-1]), (te[0], te[-1])))
        train_end = test_end
        if not expanding:
            start = start + pd.DateOffset(years=test_years)
    return splits


class ExperimentLog:
    """把每一次回测尝试追加记录到 JSONL。

    统计 n_trials 用于 deflated_sharpe_ratio：只报告最好的结果而不计入失败的尝试，
    是回测过拟合最常见的来源。
    """

    def __init__(self, path: str | Path = "output/experiments.jsonl") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, name: str, config: dict, metrics: dict) -> None:
        rec = {
            "time": datetime.now(timezone.utc).isoformat(),
            "name": name,
            "config": config,
            "metrics": {k: (None if pd.isna(v) else float(v)) for k, v in metrics.items()},
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    def load(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame()
        rows = [json.loads(l) for l in self.path.read_text(encoding="utf-8").splitlines() if l.strip()]
        return pd.json_normalize(rows)

    def n_trials(self) -> int:
        return max(len(self.load()), 1)

    def trial_sharpes(self, periods_per_year: int = 252) -> list[float]:
        df = self.load()
        col = "metrics.sharpe"
        if df.empty or col not in df:
            return []
        # 记录的是年化夏普，DSR 需要日度
        return (df[col].dropna() / periods_per_year**0.5).tolist()
