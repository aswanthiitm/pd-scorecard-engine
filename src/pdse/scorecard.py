"""Logistic regression on WOE, turned into an auditable points table.

The scaling is the standard one. Pick a reference: score ``BASE_SCORE`` means odds of
``BASE_ODDS`` to one, and every ``PDO`` points doubles the odds. Then

    factor = PDO / ln(2)
    offset = BASE_SCORE - factor * ln(BASE_ODDS)
    points(bin) = -(beta_feature * WOE(bin) + intercept / n_features) * factor + offset / n_features

so a customer's score is the plain sum of one number per characteristic. That additivity
is the whole point: it is what lets a branch explain a decline, what lets an auditor
re-derive a score by hand, and what makes an adverse-action notice possible.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .binning import BinningSet
from .config import BASE_ODDS, BASE_SCORE, PDO


@dataclass
class Scorecard:
    """A fitted scorecard: the bins, the logistic model, and the points table."""

    binning: BinningSet
    model: LogisticRegression
    features: list[str]
    points: pd.DataFrame
    factor: float
    offset: float

    # --------------------------------------------------------------- scoring
    def probability(self, frame: pd.DataFrame) -> np.ndarray:
        """Probability of default over the two-year outcome window."""
        woe = self.binning.transform(frame)[[f"woe_{f}" for f in self.features]]
        return self.model.predict_proba(woe.to_numpy())[:, 1]

    def score(self, frame: pd.DataFrame) -> pd.Series:
        """Scorecard points. Higher is better — the convention everywhere but Kaggle."""
        odds = (1 - self.probability(frame)) / np.clip(self.probability(frame), 1e-12, 1)
        return pd.Series(self.offset + self.factor * np.log(odds), index=frame.index, name="score")

    def explain(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Per-characteristic points for each row — the adverse-action reason codes."""
        parts = {}
        for feature in self.features:
            binning = self.binning.binnings[feature]
            woe = binning.transform(frame[feature])
            beta = float(self.model.coef_[0][self.features.index(feature)])
            base = -(self.model.intercept_[0] / len(self.features)) * self.factor \
                + self.offset / len(self.features)
            parts[feature] = -beta * woe * self.factor + base
        return pd.DataFrame(parts, index=frame.index)


def fit_scorecard(train: pd.DataFrame, target: str, features: list[str],
                  binning: BinningSet, C: float = 1.0) -> Scorecard:
    """Fit the logistic model on WOE and build the points table.

    The coefficients are *expected* to come out negative on WOE, because high WOE means
    low risk. A positive coefficient means that feature is fighting the others — usually
    correlation with a stronger characteristic — and in a production scorecard it is a
    reason to drop the feature rather than to ship it.
    """
    woe_train = binning.transform(train)[[f"woe_{f}" for f in features]]
    model = LogisticRegression(C=C, max_iter=2000, solver="lbfgs")
    model.fit(woe_train.to_numpy(), train[target].to_numpy())

    factor = PDO / np.log(2)
    offset = BASE_SCORE - factor * np.log(BASE_ODDS)
    intercept_share = model.intercept_[0] / len(features)

    rows = []
    for i, feature in enumerate(features):
        beta = float(model.coef_[0][i])
        b = binning.binnings[feature]
        for _, r in b.table.iterrows():
            rows.append({
                "feature": feature, "bin": r["range"], "share": r["share"],
                "event_rate": r["event_rate"], "woe": r["woe"], "beta": beta,
                "points": -(beta * r["woe"] + intercept_share) * factor + offset / len(features),
            })
        if b.missing_rate > 0:
            rows.append({
                "feature": feature, "bin": "missing", "share": b.missing_rate,
                "event_rate": np.nan, "woe": b.missing_woe, "beta": beta,
                "points": -(beta * b.missing_woe + intercept_share) * factor + offset / len(features),
            })
    points = pd.DataFrame(rows)
    points["points"] = points["points"].round(1)

    return Scorecard(binning=binning, model=model, features=features, points=points,
                     factor=factor, offset=offset)


def fit_challenger(train: pd.DataFrame, target: str, features: list[str]):
    """A gradient-boosting challenger on raw values, to price the cost of interpretability.

    This is the model a scorecard is always compared against in a model-risk paper. It
    sees the raw values, handles missingness natively, and finds interactions the additive
    scorecard cannot. Whatever AUC it wins is the premium the bank pays for a model it can
    explain — and that premium is usually small enough that the scorecard still wins the
    argument.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier

    model = HistGradientBoostingClassifier(
        max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
        l2_regularization=1.0, early_stopping=True, validation_fraction=0.15,
        random_state=0,
    )
    model.fit(train[features].to_numpy(float), train[target].to_numpy())
    return model
