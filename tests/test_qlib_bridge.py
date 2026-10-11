import numpy as np
import pandas as pd
import pytest

from alpha.qlib_bridge import dump_qlib, membership_spans, qlib_price_fields


def test_membership_spans_split_on_gaps():
    d = pd.bdate_range("2020-01-01", periods=6)
    u = pd.DataFrame({"A": [1, 1, 0, 0, 1, 1], "B": [0, 0, 0, 0, 0, 0]}, index=d).astype(bool)
    s = membership_spans(u)
    assert s.values.tolist() == [["A", d[0], d[1]], ["A", d[4], d[5]]]


def test_price_fields_follow_qlib_convention():
    d = pd.bdate_range("2020-01-01", periods=2)
    raw = {k: pd.DataFrame({"A": v}, index=d) for k, v in {
        "close": [10.0, 10.0], "adj_close": [5.0, 10.0], "open": [8.0, 9.0], "high": [12.0, 11.0],
        "low": [8.0, 9.0], "volume": [100.0, 100.0]}.items()}
    f = qlib_price_fields(raw)
    assert f["factor"]["A"].tolist() == [0.5, 1.0]
    assert f["open"]["A"].iloc[0] == 4.0 and f["volume"]["A"].iloc[0] == 200.0


def test_dump_and_read_back_with_qlib(tmp_path):
    pytest.importorskip("qlib")
    from alpha.qlib_bridge import expression_frames, init_qlib

    d = pd.bdate_range("2015-01-01", periods=120)
    rng = np.random.default_rng(0)
    close = pd.DataFrame(np.exp(np.cumsum(rng.normal(0, 0.01, (120, 3)), 0)) * 50, index=d, columns=["AAA", "CON", "BBB"])
    u = pd.DataFrame(True, index=d, columns=close.columns)
    u.iloc[40:60, 0] = False
    dump_qlib({"close": close, "roe": close * 0 + 0.1}, tmp_path, {"univ": u})
    init_qlib(tmp_path)
    out = expression_frames({"r5": "$close/Ref($close,5)-1", "roe": "$roe"}, "univ", "2015-01-01", "2015-06-30")
    exp = (close / close.shift(5) - 1).reindex_like(out["r5"])
    assert np.nanmax(np.abs(out["r5"] - exp)) < 1e-5
    assert out["r5"].loc[d[45], "AAA"] != out["r5"].loc[d[45], "AAA"]   # 池外为 NaN
    assert set(out["roe"].columns) == {"AAA", "CON", "BBB"}


def test_alpha158_runs_on_dumped_data(tmp_path):
    pytest.importorskip("qlib")
    from alpha.qlib_bridge import alpha158_frames, init_qlib

    d = pd.bdate_range("2015-01-01", periods=150)
    rng = np.random.default_rng(1)
    close = pd.DataFrame(np.exp(np.cumsum(rng.normal(0, 0.01, (150, 3)), 0)) * 50, index=d, columns=["A", "B", "C"])
    f = {"close": close, "open": close * 0.995, "high": close * 1.01, "low": close * 0.99, "vwap": close,
         "volume": close * 0 + 1e6 * rng.uniform(0.5, 1.5, (150, 3)), "factor": close * 0 + 1}
    dump_qlib(f, tmp_path, {"univ": close.notna()})
    init_qlib(tmp_path)
    me = d[d.is_month_end | (d == d[-1])]
    out = alpha158_frames("univ", str(d[0].date()), str(d[-1].date()), me, batch=60)
    assert len(out) == 158
    assert out["a158_ROC5"].iloc[-1].notna().all()
