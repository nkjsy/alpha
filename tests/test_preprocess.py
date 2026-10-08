import numpy as np
import pandas as pd

from alpha.preprocess import apply_universe, neutralize, rank_normalize, winsorize_mad, zscore


def _panel(seed=0, n=50, t=20):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.standard_t(3, (t, n)), index=pd.bdate_range("2020-01-01", periods=t),
                        columns=[f"S{i}" for i in range(n)])


def test_zscore_moments():
    z = zscore(_panel())
    assert np.allclose(z.mean(axis=1), 0, atol=1e-12)
    assert np.allclose(z.std(axis=1), 1)


def test_rank_normalize_centered():
    r = rank_normalize(_panel())
    assert np.allclose(r.mean(axis=1), 0, atol=1e-12)
    assert r.abs().max().max() <= 0.5


def test_winsorize_bounds_outlier():
    x = _panel()
    x.iloc[0, 0] = 1e6
    w = winsorize_mad(x)
    assert w.iloc[0, 0] < 100


def test_universe_mask():
    x = _panel()
    u = pd.DataFrame(True, index=x.index, columns=x.columns)
    u.iloc[:, :10] = False
    m = apply_universe(x, u)
    assert m.iloc[:, :10].isna().all().all()
    assert m.iloc[:, 10:].notna().all().all()


def test_neutralize_orthogonal_to_groups_and_exposure():
    x = _panel(n=200)
    rng = np.random.default_rng(1)
    g = pd.DataFrame(np.tile(rng.choice(list("ABCD"), 200), (len(x), 1)), index=x.index, columns=x.columns)
    e = _panel(seed=2, n=200)
    y = x + 2 * e + (g == "A") * 3.0
    res = neutralize(y, groups=g, exposures={"e": e})
    for i in range(len(res)):
        r = res.iloc[i]
        assert abs(np.corrcoef(r, e.iloc[i])[0, 1]) < 1e-8
        for k in "ABCD":
            assert abs(r[g.iloc[i] == k].mean()) < 1e-8
