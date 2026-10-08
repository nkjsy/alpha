import numpy as np
import pandas as pd

from alpha.combine import combine, ic_weights, realized_ic_frame
from alpha.evaluate import ic_summary, information_coefficient, quantile_returns
from alpha.factors import build_factor
from alpha.preprocess import standard_pipeline


def test_ic_perfect_and_inverse(market):
    fwd = market.forward_returns(21)
    assert np.allclose(information_coefficient(fwd, fwd), 1.0)
    assert np.allclose(information_coefficient(-fwd, fwd), -1.0)


def test_planted_momentum_detected(market):
    fwd = market.forward_returns(21)
    mom = standard_pipeline(build_factor("momentum").compute(market), market.universe)
    ic = information_coefficient(mom, fwd)
    s = ic_summary(ic, horizon=21)
    assert s["ic_mean"] > 0.03 and s["t_stat"] > 2
    q = quantile_returns(mom, fwd).mean()
    assert q[5] > q[1]


def test_ic_weights_have_no_lookahead(market):
    """修改某日之后的未来收益，不应影响该日及之前的因子权重。"""
    h, lag = 21, 1
    facs = {
        "mom": standard_pipeline(build_factor("momentum").compute(market), market.universe),
        "rev": standard_pipeline(build_factor("reversal").compute(market), market.universe),
    }
    fwd = market.forward_returns(h, lag)
    cal = market.dates
    w1 = ic_weights(realized_ic_frame(facs, fwd), cal, h, lag, window=126, min_periods=60)

    t = cal[600]
    # t 日可用的 IC 只来自信号日 <= t - lag - h；把更晚信号日的未来收益全部打乱
    cutoff = cal[cal.get_loc(t) - lag - h]
    fwd2 = fwd.copy()
    late = fwd2.index > cutoff
    fwd2.loc[late] = np.random.default_rng(0).normal(0, 0.1, fwd2.loc[late].shape)
    w2 = ic_weights(realized_ic_frame(facs, fwd2), cal, h, lag, window=126, min_periods=60)
    pd.testing.assert_frame_equal(w1.loc[:t], w2.loc[:t])


def test_combine_methods_run(market):
    facs = {
        "mom": standard_pipeline(build_factor("momentum").compute(market), market.universe),
        "vol": standard_pipeline(build_factor("low_vol").compute(market), market.universe),
    }
    fwd = market.forward_returns(21)
    for m in ("equal", "ic", "icir"):
        score, w = combine(facs, method=m, fwd_returns=fwd, window=126, min_periods=60)
        assert score.shape == market.close.shape
        valid = score.dropna(how="all")
        assert np.allclose(valid.mean(axis=1), 0, atol=1e-9)
        if w is not None:
            assert np.allclose(w.abs().sum(axis=1), 1)
