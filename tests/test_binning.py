"""Binning tests: WOE has a closed form, so most of these check it exactly."""
import numpy as np
import pandas as pd
import pytest

from pdse.binning import BinningSet, fit_binning, iv_strength

RNG = np.random.default_rng(0)


def _monotone_frame(n: int = 20_000) -> pd.DataFrame:
    x = RNG.uniform(0, 1, n)
    y = (RNG.uniform(0, 1, n) < 0.02 + 0.30 * x).astype(int)
    return pd.DataFrame({"x": x, "y": y})


def test_woe_is_the_log_odds_ratio_of_the_bin():
    frame = _monotone_frame()
    binning = fit_binning(frame["x"], frame["y"])
    row = binning.table.iloc[0]
    total_good, total_bad = (frame["y"] == 0).sum(), (frame["y"] == 1).sum()
    good, bad = row["n"] - row["bads"], row["bads"]
    k = len(binning.table)
    expected = np.log(((good + 0.5) / (total_good + 0.5 * k)) / ((bad + 0.5) / (total_bad + 0.5 * k)))
    assert row["woe"] == pytest.approx(expected, abs=1e-10)


def test_iv_is_the_sum_of_its_bin_contributions_and_never_negative():
    frame = _monotone_frame()
    binning = fit_binning(frame["x"], frame["y"])
    assert binning.iv == pytest.approx(binning.table["iv_contribution"].sum(), abs=1e-12)
    assert binning.iv > 0


def test_event_rates_come_out_monotone():
    frame = _monotone_frame()
    binning = fit_binning(frame["x"], frame["y"])
    rates = binning.table["event_rate"].to_numpy()
    assert np.all(np.diff(rates) >= -1e-12) or np.all(np.diff(rates) <= 1e-12)
    assert binning.monotonic


def test_a_pure_noise_feature_has_almost_no_information_value():
    n = 20_000
    frame = pd.DataFrame({"x": RNG.uniform(0, 1, n), "y": (RNG.uniform(0, 1, n) < 0.07).astype(int)})
    assert fit_binning(frame["x"], frame["y"]).iv < 0.02


def test_missing_values_get_their_own_woe_rather_than_an_imputed_one():
    frame = _monotone_frame()
    frame.loc[frame.sample(3000, random_state=1).index, "x"] = np.nan
    frame.loc[frame["x"].isna(), "y"] = 1  # make the missingness maximally informative
    binning = fit_binning(frame["x"], frame["y"])
    assert binning.missing_rate == pytest.approx(0.15, abs=0.01)
    assert binning.missing_woe < -1.0  # missing is bad, and the WOE says so


def test_transform_maps_a_value_to_its_own_bin_woe():
    frame = _monotone_frame()
    binning = fit_binning(frame["x"], frame["y"])
    woe = binning.transform(frame["x"])
    assert set(np.round(woe.unique(), 10)) <= set(np.round(binning.table["woe"], 10))


def test_a_counter_that_is_95_percent_zeros_still_gets_split():
    """The tree path returns one bin here; the discrete path must not."""
    n = 40_000
    x = pd.Series(np.where(RNG.uniform(0, 1, n) < 0.95, 0, RNG.integers(1, 4, n)))
    y = pd.Series((RNG.uniform(0, 1, n) < np.where(x > 0, 0.35, 0.03)).astype(int))
    binning = fit_binning(x, y)
    assert len(binning.table) >= 2
    assert binning.iv > 0.2


def test_iv_strength_labels():
    assert iv_strength(0.01) == "unpredictive"
    assert iv_strength(0.2) == "medium"
    assert iv_strength(0.9).startswith("very strong")


def test_binning_set_produces_one_woe_column_per_feature():
    frame = _monotone_frame()
    frame["z"] = RNG.uniform(0, 1, len(frame))
    result = BinningSet().fit(frame, "y", ["x", "z"]).transform(frame)
    assert list(result.columns) == ["woe_x", "woe_z"]
    assert len(result) == len(frame)
