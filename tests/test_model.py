"""Scorecard, validation and policy tests."""
import numpy as np
import pandas as pd
import pytest

from pdse.binning import BinningSet
from pdse.capital import irb_capital_requirement, irb_correlation, portfolio_capital
from pdse.config import BASE_ODDS, BASE_SCORE, PDO
from pdse.policy import cutoff_curve, optimal_cutoff, swap_set
from pdse.scorecard import fit_scorecard
from pdse.validation import decile_table, gini, ks_statistic, psi, psi_verdict, rank_ordering_breaks

RNG = np.random.default_rng(1)


def _frame(n: int = 30_000) -> pd.DataFrame:
    x1 = RNG.uniform(0, 1, n)
    x2 = RNG.normal(50, 15, n)
    logit = -3.0 + 3.0 * x1 - 0.03 * (x2 - 50)
    y = (RNG.uniform(0, 1, n) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


@pytest.fixture(scope="module")
def fitted():
    frame = _frame()
    binning = BinningSet().fit(frame, "y", ["x1", "x2"])
    return frame, fit_scorecard(frame, "y", ["x1", "x2"], binning)


def test_higher_score_means_lower_probability_of_default(fitted):
    """Rank correlation, not linear: score is a log-odds transform of the probability,
    so the relationship is exactly monotone but deliberately not linear."""
    from scipy.stats import spearmanr

    frame, card = fitted
    assert spearmanr(card.score(frame), card.probability(frame)).statistic == pytest.approx(-1.0)


def test_the_score_scaling_doubles_the_odds_every_pdo_points(fitted):
    frame, card = fitted
    probabilities = np.clip(card.probability(frame), 1e-9, 1 - 1e-9)
    odds = (1 - probabilities) / probabilities
    scores = card.score(frame).to_numpy()
    lo, hi = np.argmin(scores), np.argmax(scores)
    expected_gap = PDO * np.log2(odds[hi] / odds[lo])
    assert scores[hi] - scores[lo] == pytest.approx(expected_gap, rel=1e-9)


def test_the_reference_point_of_the_scale_is_where_it_was_set(fitted):
    _, card = fitted
    # By construction, odds of BASE_ODDS:1 must score exactly BASE_SCORE.
    assert card.offset + card.factor * np.log(BASE_ODDS) == pytest.approx(BASE_SCORE, abs=1e-9)


def test_coefficients_on_woe_are_negative_because_high_woe_is_low_risk(fitted):
    _, card = fitted
    assert np.all(card.model.coef_[0] < 0)


def test_explain_adds_up_to_the_total_score(fitted):
    frame, card = fitted
    sample = frame.head(200)
    assert np.allclose(card.explain(sample).sum(axis=1), card.score(sample), atol=1e-6)


def test_gini_and_ks_agree_with_a_perfect_and_a_useless_model():
    y = np.array([0, 0, 0, 1, 1, 1])
    assert gini(y, np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])) == pytest.approx(1.0)
    assert ks_statistic(y, np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])) == pytest.approx(1.0)


def test_a_well_ranking_model_has_no_decile_inversions(fitted):
    frame, card = fitted
    assert rank_ordering_breaks(decile_table(frame["y"], card.score(frame))) == 0


def test_psi_is_zero_against_an_identical_population_and_large_against_a_shifted_one():
    base = pd.Series(RNG.normal(600, 40, 20_000))
    same, _ = psi(base, pd.Series(RNG.normal(600, 40, 20_000)))
    shifted, _ = psi(base, pd.Series(RNG.normal(540, 40, 20_000)))
    assert same < 0.02 and psi_verdict(same) == "stable"
    assert shifted > 0.25 and psi_verdict(shifted).startswith("materially")


def test_irb_correlation_falls_as_pd_rises_and_stays_inside_its_bounds():
    pds = np.array([0.001, 0.01, 0.05, 0.2, 0.5])
    r = irb_correlation(pds)
    assert np.all(np.diff(r) < 0)
    assert np.all((r >= 0.03 - 1e-9) & (r <= 0.16 + 1e-9))


def test_capital_requirement_is_positive_and_rises_with_pd():
    k = irb_capital_requirement(np.array([0.005, 0.02, 0.10]))
    assert np.all(k > 0) and np.all(np.diff(k) > 0)


def test_a_riskier_book_needs_more_capital():
    safe = portfolio_capital(np.full(1000, 0.01))
    risky = portfolio_capital(np.full(1000, 0.08))
    assert risky["rwa_cr"] > safe["rwa_cr"]
    assert risky["expected_loss_cr"] > safe["expected_loss_cr"]


def test_tightening_the_cutoff_lowers_the_approved_bad_rate(fitted):
    frame, card = fitted
    curve = cutoff_curve(card.score(frame), card.probability(frame), frame["y"], grid=25)
    ordered = curve.sort_values("approval_rate")
    assert ordered["observed_bad_rate"].is_monotonic_increasing
    assert 0.0 < optimal_cutoff(curve)["approval_rate"] <= 1.0


def test_swap_set_partitions_the_population_exactly():
    n = 10_000
    a = pd.Series(RNG.normal(600, 30, n))
    b = a + RNG.normal(0, 15, n)
    y = pd.Series((RNG.uniform(0, 1, n) < 0.07).astype(int))
    table = swap_set(a, b, y, approval_rate=0.7)
    assert table["accounts"].sum() == n
    assert table["share"].sum() == pytest.approx(1.0, abs=1e-9)
    swap_in = table.loc[table.segment.str.startswith("swap-in"), "accounts"].item()
    swap_out = table.loc[table.segment.str.startswith("swap-out"), "accounts"].item()
    assert abs(swap_in - swap_out) <= 1  # equal approval rates means equal swaps
