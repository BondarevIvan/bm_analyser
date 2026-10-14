"""Случайные величины: uniform, normal, exponential (+ bernoulli, poisson)."""
import math
import statistics

import pytest

from magnum import MagnumRuntimeError, MagnumValidationError, load_model, run_model
from conftest import wrap

N_SAMPLES = 20000


def sample(expr, n=N_SAMPLES, seed=0):
    """Значения выражения expr в n независимых срабатываниях события."""
    r = run_model(wrap(f"""
        controls {{ default {{ mu = 10  sigma = 2 }} }}
        event _init {{ do {{ xs = create_array({n}) i = 0 }} trigger {{ on start }} }}
        event draw {{ do {{ xs[i] = {expr}  i = i + 1 }} trigger {{ every 1s }} }}
    """), horizon=(n + 0.5) / 86400, sampling="1w", seed=seed)
    xs = r["final"]["xs"]
    assert r["final"]["i"] == n
    return xs


def test_uniform():
    xs = sample("uniform(2, 5)")
    assert min(xs) >= 2 and max(xs) <= 5
    assert statistics.mean(xs) == pytest.approx(3.5, abs=0.03)
    assert statistics.pvariance(xs) == pytest.approx(9 / 12, rel=0.05)


def test_uniform_default_is_0_1():
    xs = sample("uniform()")
    assert 0 <= min(xs) and max(xs) <= 1
    assert statistics.mean(xs) == pytest.approx(0.5, abs=0.01)


def test_normal():
    xs = sample("normal(mu, sigma)")
    assert statistics.mean(xs) == pytest.approx(10, abs=0.05)
    assert statistics.pstdev(xs) == pytest.approx(2, rel=0.03)
    assert statistics.mean(sample("normal()")) == pytest.approx(0, abs=0.03)


def test_normal_keyword_args():
    xs = sample("normal(mu=-3, sigma=0.5)")
    assert statistics.mean(xs) == pytest.approx(-3, abs=0.02)


@pytest.mark.parametrize("expr", ["exponential(4)", "exponential(mean=4)"])
def test_exponential(expr):
    xs = sample(expr)
    assert min(xs) >= 0
    assert statistics.mean(xs) == pytest.approx(4, rel=0.03)
    assert statistics.pstdev(xs) == pytest.approx(4, rel=0.05)
    # P(X > mean) = e^-1
    assert sum(x > 4 for x in xs) / len(xs) == pytest.approx(math.exp(-1), abs=0.015)


def test_exponential_with_duration():
    # длительности — это дни, так что exponential(2h) — среднее 2 часа в днях
    xs = sample("exponential(2h)")
    assert statistics.mean(xs) == pytest.approx(2 / 24, rel=0.03)


def test_bernoulli_and_poisson():
    assert statistics.mean(sample("bernoulli(0.3)")) == pytest.approx(0.3, abs=0.015)
    xs = sample("poisson(lambda=3)")
    assert statistics.mean(xs) == pytest.approx(3, abs=0.05)
    assert statistics.pvariance(xs) == pytest.approx(3, rel=0.05)


def test_random_delay_and_period():
    r = run_model(wrap("""
        event _init { do { served = 0 } trigger { on start } }
        event done { do { served = served + 1 } }
        event visit { emit done after exponential(mean=30m)  trigger { every exponential(1h) } }
    """), horizon=100, seed=1)
    # в среднем 24 визита в день
    assert r["event_counts"]["visit"] / 100 == pytest.approx(24, rel=0.05)
    assert r["final"]["served"] >= r["event_counts"]["visit"] - 5


def test_seed_reproducibility():
    assert sample("normal(0, 1)", n=100, seed=5) == sample("normal(0, 1)", n=100, seed=5)
    assert sample("normal(0, 1)", n=100, seed=5) != sample("normal(0, 1)", n=100, seed=6)


@pytest.mark.parametrize("expr,match", [
    ("uniform(5, 2)", "a <= b"),
    ("normal(0, -1)", "sigma"),
    ("exponential(0)", "mean"),
    ("exponential(-1h)", "mean"),
])
def test_bad_parameters(expr, match):
    with pytest.raises(MagnumRuntimeError, match=match):
        run_model(wrap(f"event e {{ do {{ x = {expr} }} trigger {{ on start }} }}"), horizon=0)


@pytest.mark.parametrize("expr", ["uniform(1, 2, 3)", "normal(scale=1)", "exponential()", "exponential(lambda=1)"])
def test_bad_arguments_at_load(expr):
    with pytest.raises(MagnumValidationError, match="неверные аргументы"):
        load_model(wrap(f"event e {{ do {{ x = {expr} }} }}"))
