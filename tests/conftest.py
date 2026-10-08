import pytest

from alpha.data import make_synthetic_market


@pytest.fixture(scope="session")
def market():
    return make_synthetic_market(n_stocks=120, n_days=252 * 4, seed=1, momentum_strength=0.3)
