"""Loading and repairing the Give Me Some Credit portfolio.

The raw file is a real retail credit bureau extract and it arrives with the damage a
real extract has. Four problems have to be dealt with before any of it is modellable,
and each one is a decision that changes the answer:

1. **Sentinel codes.** The three delinquency counters take the values 96 and 98 for a
   few hundred accounts. They are not "someone was 96 times 30-days late" — they are
   the bureau's codes for *no history* and *refer to narrative*. Left in, they become
   the single most predictive thing in the file and the model learns to detect a
   missing-data flag instead of credit risk.
2. **A field that is not what it says.** ``DebtRatio`` exceeds 10 for 19% of accounts,
   and **93% of those rows have no monthly income**. On an income-missing row the field
   is not a ratio at all — it holds a monthly rupee expense. The two populations are
   flagged apart rather than merged.
3. **Missing income.** ~20% of ``MonthlyIncome`` is absent, and the missingness is
   itself predictive. It is kept as its own bin rather than imputed away.
4. **age = 0.** One account. Dropped.

Note what is deliberately *not* done: no winsorising of ``RevolvingUtilizationOfUnsecured
Lines`` despite values above 50,000. Both the scorecard's WOE binning and the gradient-
boosting challenger are rank-based, so an outlier lands in the top bin either way and
capping only destroys the ordering inside it. Capping is a fix for models that read the
magnitude — this pipeline has none.
"""
from __future__ import annotations

import io
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import RANDOM_STATE, RAW, TARGET, TEST_SIZE

ARFF = RAW / "GiveMeSomeCredit.arff"
SOURCE = "https://openml.org/data/v1/download/22125240/GiveMeSomeCredit.arff"

RENAME = {
    "FinancialDistressNextTwoYears": TARGET,
    "RevolvingUtilizationOfUnsecuredLines": "revolving_utilisation",
    "age": "age",
    "NumberOfTime30-59DaysPastDueNotWorse": "dpd_30_59",
    "DebtRatio": "debt_ratio",
    "MonthlyIncome": "monthly_income",
    "NumberOfOpenCreditLinesAndLoans": "open_lines",
    "NumberOfTimes90DaysLate": "dpd_90_plus",
    "NumberRealEstateLoansOrLines": "real_estate_lines",
    "NumberOfTime60-89DaysPastDueNotWorse": "dpd_60_89",
    "NumberOfDependents": "dependents",
}
DELINQUENCY = ["dpd_30_59", "dpd_60_89", "dpd_90_plus"]
SENTINELS = (96, 98)


def read_arff(path: Path = ARFF) -> pd.DataFrame:
    """Parse the ARFF file without a dependency — the body is plain CSV."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    names = re.findall(r"^@ATTRIBUTE\s+(\S+)", text, flags=re.M | re.I)
    body = text.split("@DATA", 1)[1].lstrip("\r\n")
    frame = pd.read_csv(io.StringIO(body), header=None, names=names, na_values=["?"])
    return frame.rename(columns=RENAME)


def clean(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the repairs above. Returns the cleaned frame and an audit log."""
    out = frame.copy()
    audit: list[dict] = []

    out[TARGET] = (out[TARGET].astype(str).str.strip().str.lower() == "yes").astype(int)

    sentinel_mask = pd.Series(False, index=out.index)
    for column in DELINQUENCY:
        hit = out[column].isin(SENTINELS)
        audit.append({"issue": f"{column} sentinel code (96/98)", "rows": int(hit.sum()),
                      "action": "set to missing, later given its own WOE bin"})
        out.loc[hit, column] = np.nan
        sentinel_mask |= hit
    out["has_sentinel_history"] = sentinel_mask.astype(int)

    extreme_util = out["revolving_utilisation"] > 10
    audit.append({"issue": "revolving utilisation above 10x the limit", "rows": int(extreme_util.sum()),
                  "action": "left as is — WOE binning and the GBM are both rank-based"})

    ratio_is_amount = (out["debt_ratio"] > 10) & out["monthly_income"].isna()
    share = ratio_is_amount.sum() / max((out["debt_ratio"] > 10).sum(), 1)
    audit.append({"issue": "debt ratio above 10 (field holds a rupee amount, not a ratio)",
                  "rows": int((out["debt_ratio"] > 10).sum()),
                  "action": f"flagged, not capped — {share:.0%} of them have no monthly income"})
    out["debt_ratio_is_amount"] = ratio_is_amount.astype(int)

    bad_age = out["age"] < 18
    audit.append({"issue": "age below 18", "rows": int(bad_age.sum()), "action": "dropped"})
    out = out[~bad_age].copy()

    audit.append({"issue": "monthly income missing", "rows": int(out["monthly_income"].isna().sum()),
                  "action": "kept missing — the missingness itself carries signal"})
    audit.append({"issue": "dependents missing", "rows": int(out["dependents"].isna().sum()),
                  "action": "kept missing"})

    return out.reset_index(drop=True), pd.DataFrame(audit)


def load(path: Path = ARFF) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download it first:\n  curl -L {SOURCE} -o {path}"
        )
    return clean(read_arff(path))


def split(frame: pd.DataFrame, test_size: float = TEST_SIZE,
          random_state: int = RANDOM_STATE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split.

    Stratification matters here: the bad rate is under 7%, so an unstratified split can
    move the test-set base rate by enough to shift every calibration statistic.
    """
    return train_test_split(frame, test_size=test_size, random_state=random_state,
                            stratify=frame[TARGET])
