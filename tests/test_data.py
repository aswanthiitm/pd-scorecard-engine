"""Data-cleaning tests — these encode the decisions the audit log claims were made."""
import numpy as np
import pandas as pd

from pdse.data import DELINQUENCY, SENTINELS, clean


def _raw() -> pd.DataFrame:
    return pd.DataFrame({
        "target": ["No", "Yes", "No", "No", "Yes"],
        "revolving_utilisation": [0.2, 0.9, 50_000.0, 0.4, 0.1],
        "age": [35, 0, 52, 44, 61],
        "dpd_30_59": [0, 1, 96, 0, 98],
        "debt_ratio": [0.3, 0.5, 2_500.0, 0.2, 0.9],
        "monthly_income": [40_000.0, np.nan, np.nan, 60_000.0, 25_000.0],
        "open_lines": [5, 3, 8, 11, 6],
        "dpd_90_plus": [0, 2, 96, 0, 98],
        "real_estate_lines": [1, 0, 2, 1, 0],
        "dpd_60_89": [0, 0, 96, 0, 98],
        "dependents": [2, np.nan, 1, 0, 3],
    })


def test_target_becomes_a_zero_one_integer():
    out, _ = clean(_raw())
    assert out["target"].tolist() == [0, 0, 0, 1]  # the age-0 row (a "Yes") is dropped
    assert out["target"].dtype.kind in "iu"


def test_sentinel_codes_become_missing_not_large_counts():
    out, _ = clean(_raw())
    for column in DELINQUENCY:
        assert not out[column].isin(SENTINELS).any()
        assert out[column].isna().any()


def test_a_sentinel_row_is_flagged():
    out, _ = clean(_raw())
    assert out["has_sentinel_history"].sum() == 2


def test_under_age_rows_are_dropped():
    out, _ = clean(_raw())
    assert (out["age"] >= 18).all()
    assert len(out) == 4


def test_extreme_values_are_flagged_rather_than_capped():
    out, _ = clean(_raw())
    assert out["revolving_utilisation"].max() == 50_000.0  # rank-based models need no cap
    assert out["debt_ratio_is_amount"].sum() == 1


def test_the_audit_log_records_every_repair():
    _, audit = clean(_raw())
    assert set(audit.columns) == {"issue", "rows", "action"}
    assert len(audit) >= 7
