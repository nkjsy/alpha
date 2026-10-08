# alpha：美股横截面多因子研究框架

一个轻量、可测试的 Python 框架，覆盖横截面多因子研究的完整链路：

```
数据 + 历史股票池 → 因子计算 → 截面预处理 → 单因子评估 → 多因子合成 → 组合构建 → 带成本回测 → 过拟合检验
```

设计重点是**避免回测里最常见的自欺**：前视偏差、幸存者偏差、忽略交易成本、以及在很多变体里挑最好的那个。

## 快速开始

```bash
pip install -e ".[dev]"
pytest -q                                       # 39 个测试，含端到端未来函数检查
python -m alpha factors                         # 列出内置因子
python -m alpha run configs/example_synthetic.yaml   # 用模拟数据跑完整流程
```

模拟数据里植入了已知的动量 alpha，可用来验证流程本身是否正确。输出写到 `output/<name>/`：
`report.txt`、`factor_summary.csv`、`factor_corr.csv`、`backtest_daily.csv`、`turnover.csv`、`experiments.jsonl`。

接真实数据：按 [data/README.md](data/README.md) 准备 CSV，然后 `python -m alpha run configs/example_csv.yaml`。

## 目录结构

| 模块 | 作用 |
|---|---|
| `alpha/data/` | `MarketData` 宽表容器；CSV / yfinance 加载；历史成分股掩码；流动性过滤；模拟数据 |
| `alpha/factors/` | 因子基类与注册表；内置价量因子（动量、反转、低波、彩票、规模、Amihud、低贝塔）与基本面因子模板 |
| `alpha/preprocess.py` | 股票池掩码、MAD/分位数去极值、zscore/排序标准化、行业与市值中性化 |
| `alpha/evaluate.py` | Rank IC、ICIR、Newey-West t 值、分组收益与单调性、IC 衰减、排名自相关、因子相关性 |
| `alpha/combine.py` | 等权、IC 加权、ICIR 加权（权重严格只用已实现的 IC） |
| `alpha/portfolio.py` | 调仓日历；按得分选股（只数或比例）；等权/得分/逆波动加权；单票上限；多头/多空 |
| `alpha/backtest.py` | 日频净值模拟：执行延迟、持仓漂移、佣金+价差+冲击成本、借券费、退市处理 |
| `alpha/metrics.py` | 年化收益/波动/夏普/回撤；PSR；**Deflated Sharpe**；块自助法夏普置信区间 |
| `alpha/validation.py` | 留出样本、带 embargo 的滚动切分、**试验记录 ExperimentLog** |
| `alpha/pipeline.py` | YAML 配置驱动的端到端流程与报告 |
| `alpha/strategies/` | 具体策略，目前有纳指100动量 + QQQ 均线择时（Top3/Top10 切换） |
| `scripts/nasdaq100_momentum.py` | 本地动量策略的迁移：原版复现与逐项修正对比 |

## 防过拟合的约定

这些规则来自对已有动量策略的复盘（挑选了最优变体、没计交易成本）：

1. **执行延迟**：t 日收盘算信号，默认 t+1 日收盘成交（`execution.lag: 1`）。评估用的未来收益也按同样的延迟计算。
2. **历史股票池**：`universe` 必须是 point-in-time 的（如历史纳指100成分），所有截面处理前先掩码。缺省“有价格即在池”会引入幸存者偏差。
3. **交易成本默认开启**：`CostModel` 默认单边 10bp（佣金 1 + 半价差 5 + 冲击 4）和 50bp/年借券费，报告同时给出扣成本前后的结果和年化换手。
4. **合成权重无前视**：t 日信号的 IC 要到 t+lag+horizon 才实现，`ic_weights` 自动滞后，`tests/test_evaluate_combine.py` 会打乱未来收益来验证。
5. **每次尝试都记账**：每次 `run` 都追加到 `experiments.jsonl`，`n_trials` 随之增长，Deflated Sharpe 会因此下降。换参数、换因子、换调仓频率都算一次试验。不要删这个文件来“重置”。
6. **留出样本只看一次**：`holdout_start` 之后的数据不参与单因子评估和选择；报告最后单独列出留出期结果。看了之后再改参数，留出期就不再是样本外了。
7. **因子前视检查**：`tests/test_factors.py` 对每个内置因子篡改 t 之后的数据，断言 t 及之前的因子值不变。新增因子请加进 `PRICE_FACTORS` 列表。
8. **端到端前视检查**：`tests/test_lookahead.py` 篡改 t 之后的全部价格、成交量和市值，断言整条流程在 t 及之前输出的合成得分和目标仓位逐位不变；另外篡改留出期，断言样本内绩效和单因子评估不变。
9. **退市股不丢弃**：评估用的未来收益在持有期内退市时，算到最后一个价格（可叠加退市收益），而不是变成 NaN 被剔除。
10. **成分快照不提前生效**：`snapshot_mask` 让月末成分快照从该日（或之前最后一个交易日）才开始生效。

### 数据层面仍需注意的前视

框架代码管不到数据本身，以下几点要靠数据源解决：

- **行业分类**：`sectors.csv` 是静态映射，用今天的行业去中性化过去，有轻微前视。有历史行业数据时传入宽表即可。
- **复权价**：yfinance 的复权价按未来分红向前调整，收益率没问题，但价格水平（以及“价格 × 成交量”的成交额）会受未来信息影响。用成交额绝对阈值过滤时要留意。
- **成分股重建**：历史成分表若是从当前成分反推，准确性取决于调整记录是否完整。

## 新增一个因子

```python
# alpha/factors/my_factors.py
from alpha.factors.base import Factor, register

@register
class High52w(Factor):
    """距离 52 周高点：越接近高点越好（George & Hwang 2004）。"""
    name = "high52w"

    def __init__(self, window: int = 252):
        super().__init__(window=window)
        self.window = window

    def compute(self, data):
        c = data.close
        return c / c.rolling(self.window, min_periods=self.window // 2).max()
```

在 `alpha/factors/__init__.py` 中 import 该模块使其注册，然后在配置里写 `- {name: high52w, window: 252}`。

约定：返回与 `data.close` 同形状的宽表；值越大预期收益越高；只用 t 日及以前的数据。

## 用 Python 直接调用

```python
from alpha.data import make_synthetic_market
from alpha.factors import build_factor
from alpha.preprocess import standard_pipeline
from alpha.evaluate import evaluate_factor, ic_decay

data = make_synthetic_market()
mom = standard_pipeline(build_factor("momentum").compute(data), data.universe,
                        groups=data.sector_frame())
ev = evaluate_factor(mom, data.forward_returns(horizon=21), horizon=21)
print(ev["summary"])
print(ic_decay(mom, data.close))
```

## 纳指100动量策略迁移

`scripts/nasdaq100_momentum.py` 把本地的 11-1 动量 + QQQ 200 日均线择时策略搬了过来，逐项打开修正（T+1 成交、交易成本、不 ffill 价格、只在持仓变化时调仓），并和无择时的 Top10 基准、QQQ 买入持有对比。需要能访问 Yahoo Finance：

```bash
pip install -e ".[yfinance]"
python scripts/nasdaq100_momentum.py --membership C:/money/fin/nasdaq/nasdaq100_monthly_constituents_backtest_2010_2026.csv --n-trials 12
```

脚本会先打印历史成分的价格覆盖率：yfinance 拿不到退市股，覆盖率低于 100% 的部分就是幸存者偏差。

## 已知局限与后续方向

- yfinance 没有退市股，只适合快速原型；正式回测建议用含退市股的数据（CRSP、Norgate、Sharadar 等）。
- 回测在收盘价成交，冲击成本为固定基点近似，未按成交额占比建模。
- 组合构建是排序选股，尚未实现带风险模型的均值-方差优化（可在 `portfolio.py` 扩展）。
- 基本面因子需要自备按披露日对齐的数据。
