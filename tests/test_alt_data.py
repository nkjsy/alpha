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
