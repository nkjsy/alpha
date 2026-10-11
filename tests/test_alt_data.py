import io
import zipfile

import numpy as np
import pandas as pd

from alpha.data.insider import insider_fields, parse_form345
from alpha.data.sec import industry_frame, sic_to_ff12
from alpha.data.thirteenf import breadth_fields, cusip_map, parse_13f, parse_ftd
from alpha.data.universe import cap_rank_membership, liquid_pool
from alpha.preprocess import group_demean, winsorize_mad


def _zip(files: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, text in files.items():
            z.writestr(name, text)
    return buf.getvalue()


def _tsv(rows: list[list]) -> str:
    return "\n".join("\t".join(str(c) for c in r) for r in rows) + "\n"


def test_parse_form345_keeps_open_market_form4_only():
    sub = _tsv([["ACCESSION_NUMBER", "FILING_DATE", "PERIOD_OF_REPORT", "DOCUMENT_TYPE", "ISSUERCIK", "ISSUERTRADINGSYMBOL"],
                ["a1", "05-JAN-2015", "02-JAN-2015", "4", "320193", "aapl"],
                ["a2", "06-JAN-2015", "02-JAN-2015", "4/A", "320193", "AAPL"],
                ["a3", "07-JAN-2015", "06-JAN-2015", "4", "789019", "MSFT"]])
    own = _tsv([["ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNERNAME"], ["a1", "111", "X"], ["a2", "111", "X"], ["a3", "222", "Y"]])
    tr = _tsv([["ACCESSION_NUMBER", "NONDERIV_TRANS_SK", "TRANS_CODE", "TRANS_SHARES", "TRANS_PRICEPERSHARE"],
               ["a1", 1, "P", 100, 10], ["a1", 2, "M", 50, 0], ["a2", 3, "P", 999, 10], ["a3", 4, "S", 10, 20]])
    ev = parse_form345(_zip({"2015q1/SUBMISSION.tsv": sub, "2015q1/REPORTINGOWNER.tsv": own, "2015q1/NONDERIV_TRANS.tsv": tr}))
    assert sorted(ev["code"]) == ["P", "S"]                      # 行权 M 和 4/A 被剔除
    assert ev.loc[ev["code"] == "P", "value"].item() == 1000
    assert ev.loc[ev["code"] == "P", "filed"].item() == pd.Timestamp("2015-01-05")


def test_insider_fields_point_in_time():
    dates = pd.bdate_range("2015-01-01", "2015-12-31")
    ev = pd.DataFrame({"cik": [1, 1, 2], "symbol": ["OLD", "OLD", "BBB"],
                       "filed": pd.to_datetime(["2015-03-02", "2015-03-02", "2015-03-03"]),
                       "owner": ["o1", "o2", "o3"], "code": ["P", "P", "S"], "shares": [10, 10, 10],
                       "value": [100.0, 100.0, 50.0]})
    f = insider_fields(ev, dates, pd.Index(["AAA", "BBB", "CCC"]), cik_to_ticker={1: "AAA"}, window=21)
    nb, ratio = f["insider_buyers"], f["insider_net_ratio"]
    assert nb.loc["2015-03-02", "AAA"] == 0                      # 提交当天不可用
    assert nb.loc["2015-03-03", "AAA"] == 2
    assert ratio.loc["2015-03-03", "AAA"] == 1.0
    assert ratio.loc["2015-03-04", "BBB"] == -1.0
    assert ratio.loc["2015-06-01", "AAA"] == 0.0                 # 窗口过后归零
    assert ratio.loc["2015-03-02", "CCC"] == 0.0
    assert np.isnan(ratio.loc["2015-02-27", "AAA"])              # 数据集开始之前为空


def test_ftd_cusip_map_uses_latest_symbol():
    a = "SETTLEMENT DATE|CUSIP|SYMBOL|QUANTITY (FAILS)|DESCRIPTION|PRICE\n20200115|30303M102|FB|100|FACEBOOK|200\n"
    b = "SETTLEMENT DATE|CUSIP|SYMBOL|QUANTITY (FAILS)|DESCRIPTION|PRICE\n20230115|30303M102|META|100|META|200\n" \
        "20230115|084670702|BRK.B|5|BERKSHIRE|300\n"
    ftd = pd.concat([parse_ftd(_zip({"a.txt": a})), parse_ftd(_zip({"b.txt": b}))])
    m = cusip_map(ftd)
    assert m["30303M10"] == "META"
    assert m["08467070"] == "BRK-B"


def test_parse_13f_counts_on_time_original_filings():
    sub = _tsv([["ACCESSION_NUMBER", "FILING_DATE", "SUBMISSIONTYPE", "CIK", "PERIODOFREPORT"],
                ["f1", "10-FEB-2020", "13F-HR", "1", "31-DEC-2019"],
                ["f2", "14-FEB-2020", "13F-HR", "2", "31-DEC-2019"],
                ["f3", "20-MAR-2020", "13F-HR", "3", "31-DEC-2019"],     # 迟交，不计
                ["f4", "15-FEB-2020", "13F-HR/A", "1", "31-DEC-2019"]])  # 修正，不计
    info = _tsv([["ACCESSION_NUMBER", "INFOTABLE_SK", "NAMEOFISSUER", "CUSIP", "VALUE", "SSHPRNAMT", "SSHPRNAMTTYPE", "PUTCALL"],
                 ["f1", 1, "A", "037833100", 1, 1, "SH", ""],
                 ["f1", 2, "A", "037833100", 1, 1, "SH", ""],     # 同一机构重复行只算一次
                 ["f1", 3, "A", "037833100", 1, 1, "SH", "Put"],  # 期权不计
                 ["f2", 4, "A", "37833100", 1, 1, "SH", ""],      # 丢了前导零
                 ["f2", 5, "B", "594918104", 1, 1, "PRN", ""],    # 债券不计
                 ["f3", 6, "A", "037833100", 1, 1, "SH", ""],
                 ["f4", 7, "B", "594918104", 1, 1, "SH", ""]])
    holders, filers = parse_13f(_zip({"SUBMISSION.tsv": sub, "INFOTABLE.tsv": info}))
    assert holders.set_index("cusip8")["n_holders"].to_dict() == {"03783310": 2}
    assert filers["n_filers"].item() == 2


def test_breadth_change_available_after_deadline():
    holders = pd.DataFrame({"period": pd.to_datetime(["2019-09-30", "2019-12-31", "2019-12-31"]),
                            "cusip8": ["AAAAAAAA", "AAAAAAAA", "BBBBBBBB"], "n_holders": [10, 20, 5]})
    filers = pd.DataFrame({"period": pd.to_datetime(["2019-09-30", "2019-12-31"]), "n_filers": [100, 100]})
    dates = pd.bdate_range("2019-10-01", "2020-06-30")
    f = breadth_fields(holders, filers, {"AAAAAAAA": "AAA", "BBBBBBBB": "BBB"}, dates, pd.Index(["AAA", "BBB"]))["inst_breadth_chg"]
    deadline = pd.Timestamp("2019-12-31") + pd.Timedelta(days=47)   # 2020-02-16（周日）
    assert f.loc[:deadline, "AAA"].isna().all()
    first = f["AAA"].first_valid_index()
    assert first == pd.Timestamp("2020-02-17")                       # 只统计 2-16 前提交的，2-17 才使用
    assert abs(f.loc[first, "AAA"] - 0.10) < 1e-12
    assert abs(f.loc[first, "BBB"] - 0.05) < 1e-12


def test_cap_rank_membership_uses_prior_may_ranking():
    dates = pd.bdate_range("2010-01-01", "2012-12-31")
    mcap = pd.DataFrame(1.0, index=dates, columns=["a", "b", "c"])
    mcap.loc["2011-05-31":, "c"] = 10.0     # 5 月末排名日起 c 变大
    mem = cap_rank_membership(mcap, n=1)
    by = mem.set_index("month_end_date")["ticker"]
    assert by[pd.Timestamp("2011-05-31")] != "c"
    assert by[pd.Timestamp("2011-06-30")] == "c"
    assert mem["month_end_date"].min() == pd.Timestamp("2010-06-30")


def test_liquid_pool():
    dates = pd.bdate_range("2015-01-01", "2015-12-31")
    dv = pd.DataFrame({"a": 1.0, "b": 2.0, "c": 3.0}, index=dates)
    dv.loc["2015-06-01":"2015-07-15", "a"] = 100.0
    assert liquid_pool(dv, 1) == ["a", "c"]


def test_industry_helpers():
    assert sic_to_ff12(3571) == "BusEq" and sic_to_ff12(6022) == "Money" and sic_to_ff12(np.nan) is None
    dates = pd.bdate_range("2020-01-01", periods=3)
    g = industry_frame(pd.Series({"AAPL": 3571, "JPM": 6022, "X": np.nan}), dates)
    assert list(g.iloc[0, :2]) == ["BusEq", "Money"] and pd.isna(g.iloc[0, 2])
    x = pd.DataFrame([[1.0, 5.0, 2.0]] * 3, index=dates, columns=g.columns)
    gd = group_demean(x, g)
    assert gd.iloc[0].tolist()[:2] == [0.0, 0.0] and np.isnan(gd.iloc[0, 2])


def test_winsorize_keeps_signal_when_mad_is_zero():
    x = pd.DataFrame([[0.0] * 8 + [1.0, 3.0]])
    assert winsorize_mad(x).iloc[0].tolist() == x.iloc[0].tolist()


def test_yahoo_download_resumes_from_chunk_cache(tmp_path, monkeypatch):
    import sys
    import types

    from alpha.data.loaders import download_yahoo_ohlcv

    calls = []

    def fake_download(sub, **kw):
        calls.append(list(sub))
        if len(calls) == 2 and not hasattr(fake_download, "ok"):
            raise ConnectionError("断线")
        idx = pd.bdate_range("2020-01-01", periods=3)
        cols = pd.MultiIndex.from_product([["Open", "Close", "Adj Close", "Volume", "Stock Splits"], sub])
        return pd.DataFrame(1.0, index=idx, columns=cols)

    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(download=fake_download))
    cache = tmp_path / "raw.pkl"
    tickers = ["A", "B", "C", "D"]
    try:
        download_yahoo_ohlcv(tickers, cache, chunk=2)
    except ConnectionError:
        pass
    assert not cache.exists()
    fake_download.ok = True
    out = download_yahoo_ohlcv(tickers, cache, chunk=2)
    assert calls == [["A", "B"], ["C", "D"], ["C", "D"]]          # 第一批没有重新下载
    assert list(out["close"].columns) == tickers and cache.exists()
    assert not (tmp_path / "raw_parts").exists()


def _rec(rows):
    df = pd.DataFrame(rows, columns=["field", "prio", "tag", "start", "end", "val", "filed", "form"])
    for c in ("start", "end", "filed"):
        df[c] = pd.to_datetime(df[c])
    return df


def test_shares_fall_back_to_weighted_average_without_interleaving():
    from alpha.data.sec import point_in_time_series

    rec = _rec([
        ("shares", 0, "dei", None, "2015-01-31", 100.0, "2015-02-20", "10-K"),
        ("shares", 0, "dei", None, "2015-04-30", 101.0, "2015-05-10", "10-Q"),
        ("shares", 2, "wavg", "2015-01-01", "2015-03-31", 150.0, "2015-05-10", "10-Q"),   # 紧挨着 dei，不用
        ("shares", 2, "wavg", "2016-07-01", "2016-09-30", 160.0, "2016-11-01", "10-Q"),   # dei 断档后才用
    ])
    s = point_in_time_series({"X": rec})["shares"].set_index("end")["val"]
    assert 150.0 not in s.values and s.loc[pd.Timestamp("2016-09-30")] == 160.0


def test_historical_ciks_from_form4():
    from alpha.data.sec import historical_ciks

    ev = pd.DataFrame({"cik": [34088, 2115436, 111, 222], "symbol": ["XOM", "XOM", "ABC", "ABC"],
                       "filed": pd.to_datetime(["2020-01-01", "2025-01-01", "2012-01-01", "2020-01-01"])})
    cmap = {"XOM": 2115436, "ABC": 222, "DEF": 111}   # 111 现在是另一家公司的 CIK，不能并进来
    assert historical_ciks(ev, cmap) == {"XOM": [34088]}


def test_cap_rank_fallback_and_forced_members():
    dates = pd.bdate_range("2010-01-01", "2011-12-31")
    mcap = pd.DataFrame(np.nan, index=dates, columns=["a", "b", "c", "d"])
    mcap.loc["2011-01-01":, ["a", "b"]] = [2.0, 1.0]
    adv = pd.DataFrame({"a": 1.0, "b": 1.0, "c": 5.0, "d": 3.0}, index=dates)
    always = pd.DataFrame({"month_end_date": ["2010-07-30"], "ticker": ["d"]})
    mem = cap_rank_membership(mcap, n=2, fallback=adv, always=always)
    jun10 = set(mem.loc[mem["month_end_date"] == pd.Timestamp("2010-06-30"), "ticker"])
    jul10 = set(mem.loc[mem["month_end_date"] == pd.Timestamp("2010-07-30"), "ticker"])
    jun11 = set(mem.loc[mem["month_end_date"] == pd.Timestamp("2011-06-30"), "ticker"])
    assert jun10 == {"c", "d"}          # 2010-05 没有市值，按成交额补足
    assert jul10 == {"c", "d"}
    assert jun11 == {"a", "b"}


def test_same_company_names():
    from alpha.data.sec import same_company

    assert same_company(["EXXON MOBIL CORP"], ["Exxon Mobil Corporation"])
    assert same_company(["MEDTRONIC INC"], ["Medtronic plc", "Medtronic Holdings Ltd"])
    assert not same_company(["BARNES GROUP INC"], ["Barrick Mining Corp", "BARRICK GOLD CORP"])
    assert not same_company(["GREAT ATLANTIC & PACIFIC TEA CO INC"], ["GAP INC"])
    assert not same_company([""], ["GAP INC"])


def test_merge_old_ciks_guards():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location("b", Path(__file__).parents[1] / "scripts" / "build_r1000_universe.py")
    b = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(b)
    names = {"XOM": ["Exxon Mobil Corp"], 34088: ["EXXON MOBIL CORP"], "B": ["Barrick Mining Corp"],
             9984: ["BARNES GROUP INC"], "NEW": ["New Co"], 1: ["New Co"], "GOOG": ["Alphabet Inc."],
             1288776: ["Google Inc."]}
    ts = pd.Timestamp
    cand = {"XOM": [34088], "B": [9984], "NEW": [1], "GOOG": [1288776]}
    first_px = pd.Series({"XOM": ts("2008-01-02"), "B": ts("2008-01-02"), "NEW": ts("2020-01-02"),
                          "GOOG": ts("2008-01-02")})
    first_filed = {"XOM": ts("2025-06-01"), "B": ts("2015-01-01"), "NEW": ts("2020-06-01"),
                   "GOOG": ts("2015-10-01")}
    out = b.merge_old_ciks(cand, first_px, first_filed, lambda x: names.get(x, []), {"GOOG": [1288776]})
    assert out == {"XOM": [34088], "GOOG": [1288776]}
