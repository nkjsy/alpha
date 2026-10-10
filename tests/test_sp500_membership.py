import importlib.util
from pathlib import Path

import pandas as pd


def _load():
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_sp500_membership.py"
    spec = importlib.util.spec_from_file_location("sp500", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_reconstruct_reverses_changes():
    mod = _load()
    raw = pd.DataFrame({
        ("Effective Date", "Effective Date"): ["March 15, 2021", "June 1, 2020"],
        ("Added", "Ticker"): ["NEW", "BRK.B"],
        ("Added", "Security"): ["New Co", "Berkshire"],
        ("Removed", "Ticker"): ["OLD", None],
        ("Removed", "Security"): ["Old Co", None],
    })
    raw.columns = pd.MultiIndex.from_tuples(raw.columns)
    ch = mod.parse_changes(raw)
    assert ch["added"].tolist() == ["NEW", "BRK-B"]
    df = mod.reconstruct(["AAA", "NEW", "BRK.B"], ch, "2020-01-01", "2021-06-30")
    snap = lambda d: set(df.loc[df["month_end_date"] == pd.Timestamp(d).date(), "ticker"])
    assert snap("2021-04-30") == {"AAA", "NEW", "BRK-B"}
    assert snap("2021-02-28") == {"AAA", "OLD", "BRK-B"}
    assert snap("2020-05-31") == {"AAA", "OLD"}
