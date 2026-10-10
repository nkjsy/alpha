import importlib.util
from pathlib import Path

from alpha.data import make_synthetic_market


def test_mine_script_runs_on_synthetic(tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts" / "mine_factors.py"
    spec = importlib.util.spec_from_file_location("mine_factors", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    m = make_synthetic_market(n_stocks=100, n_days=252 * 7, seed=5, momentum_strength=0.3, start="2014-01-02")
    res, hold, bts, n_trials = mod.run(m, "2015-01-01", "2019-06-01", min_t=2.0, log_path=tmp_path / "e.jsonl")
    assert len(res.table) >= 20
    assert n_trials == len(res.table)
    assert "基准 11-1 动量" in bts
    assert set(hold.index) == set(res.selected)
