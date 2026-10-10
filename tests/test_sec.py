import numpy as np
import pandas as pd
import pytest

from alpha.data.sec import build_fields, extract_records, point_in_time_series, quarterly, sue, ttm


def _fact(start, end, val, filed, form="10-Q"):
    d = {"end": end, "val": val, "filed": filed, "form": form}
    if start:
        d["start"] = start
    return d


def _facts():
    ni = [
        _fact("2020-01-01", "2020-03-31", 10, "2020-05-05"),
        _fact("2020-04-01", "2020-06-30", 20, "2020-08-04"),
        _fact("2020-01-01", "2020-06-30", 30, "2020-08-04"),          # 半年累计，应被忽略
        _fact("2020-07-01", "2020-09-30", 30, "2020-11-03"),
        _fact("2020-01-01", "2020-12-31", 100, "2021-02-25", "10-K"),  # 推出 Q4 = 40
        _fact("2020-01-01", "2020-03-31", 999, "2021-05-04"),          # 后续重述 Q1，应被忽略
        _fact("2021-01-01", "2021-03-31", 15, "2021-05-04"),
    ]
    assets = [_fact(None, "2020-12-31", 1000, "2021-02-25", "10-K"), _fact(None, "2021-03-31", 1100, "2021-05-04")]
    shares = [_fact(None, "2020-04-28", 100, "2020-05-05"), _fact(None, "2020-10-28", 400, "2020-11-03")]
    return {"facts": {
        "us-gaap": {"NetIncomeLoss": {"units": {"USD": ni}}, "Assets": {"units": {"USD": assets}}},
        "dei": {"EntityCommonStockSharesOutstanding": {"units": {"shares": shares}}},
    }}


def test_quarterly_derives_q4_and_ignores_restatement():
    rec = extract_records(_facts())
    q = quarterly(rec[rec.field == "net_income"])
    assert q["val"].tolist() == [10, 20, 30, 40, 15]
    t = ttm(rec[rec.field == "net_income"])
    row = t[t["end"] == pd.Timestamp("2020-12-31")].iloc[0]
    assert row["val"] == 100 and row["filed"] == pd.Timestamp("2021-02-25")
    # 2021Q1 的 TTM = 20 + 30 + 40 + 15，可获得日为 2021-05-04
    row = t[t["end"] == pd.Timestamp("2021-03-31")].iloc[0]
    assert row["val"] == 105 and row["filed"] == pd.Timestamp("2021-05-04")


def test_build_fields_point_in_time_and_splits():
    dates = pd.bdate_range("2020-01-01", "2021-12-31")
    tickers = pd.Index(["AAA"])
    close = pd.DataFrame(10.0, index=dates, columns=tickers)  # 已做拆股调整的价格
    splits = pd.DataFrame(0.0, index=dates, columns=tickers)
    splits.loc["2020-08-31", "AAA"] = 4.0
    f = build_fields({"AAA": extract_records(_facts())}, dates, tickers, close, splits)
    ni = f["net_income_ttm"]["AAA"]
    # 2021-02-25 提交，滞后 1 个交易日才可用
    assert np.isnan(ni.loc["2021-02-25"])
    assert ni.loc["2021-02-26"] == 100
    assert ni.loc["2021-05-05"] == 105
    # 拆股前披露的 100 股，换算成拆股后单位为 400 股
    assert f["shares"]["AAA"].loc["2020-06-01"] == 400
    assert f["shares"]["AAA"].loc["2020-11-05"] == 400
    assert f["market_cap"]["AAA"].loc["2020-06-01"] == 4000
    assert np.isnan(f["assets"]["AAA"].loc["2021-02-25"])
    assert f["assets"]["AAA"].loc["2021-03-01"] == 1000


def test_sue_standardizes_yoy_change():
    ends = pd.date_range("2015-03-31", periods=20, freq="QE")
    eps = pd.Series(np.arange(20, dtype=float) * 0.1 + np.tile([0, 0.05, 0.1, 0.0], 5), index=ends)
    eps.iloc[-1] += 1.0  # 最后一季大幅超预期
    rec = pd.DataFrame({
        "field": "eps", "prio": 0, "tag": "EarningsPerShareDiluted",
        "start": ends - pd.offsets.QuarterBegin(startingMonth=1) + pd.Timedelta(days=0),
        "end": ends, "val": eps.values, "filed": ends + pd.Timedelta(days=35), "form": "10-Q",
    })
    rec["start"] = [e - pd.Timedelta(days=89) for e in ends]
    s = sue(rec)
    assert len(s) > 5
    assert s["val"].iloc[-1] > 3


def test_point_in_time_series_prefers_primary_tag():
    facts = _facts()
    facts["facts"]["us-gaap"]["ProfitLoss"] = {"units": {"USD": [
        _fact("2020-01-01", "2020-12-31", 555, "2021-02-25", "10-K")]}}
    series = point_in_time_series({"AAA": extract_records(facts)})
    ni = series["net_income_ttm"]
    assert ni.loc[ni["end"] == pd.Timestamp("2020-12-31"), "val"].tolist() == [100]
