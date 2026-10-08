# 数据文件格式

`configs/example_csv.yaml` 读取以下文件（本目录不提交真实数据，`data/cache/` 已被忽略）。

**prices.csv**（长表，必须）

```
date,ticker,open,close,volume,market_cap
2020-01-02,AAPL,74.06,75.09,135480400,1.30e12
```

- `close`、`open` 必须是复权价（含拆股和分红）。
- 其它数值列（如 `market_cap`）会进入 `MarketData.fields`，可被因子引用。
- 应包含已退市、已被剔除出指数的股票，否则会有幸存者偏差。

**membership.csv**（历史成分区间，强烈建议）

```
ticker,start,end
AAPL,1985-01-01,
WBA,2009-01-01,2024-12-20
```

`end` 留空表示至今仍在指数中；同一股票可有多行。

**sectors.csv**（可选，行业中性化用）

```
ticker,sector
AAPL,Information Technology
```

基本面数据请先用 `alpha.factors.fundamental.align_fundamental` 按披露日对齐后再放入 `fields`。
