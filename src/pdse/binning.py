"""Monotonic weight-of-evidence binning and information value.

A scorecard is a logistic regression on **weight of evidence**, not on raw values, and
that choice is what makes it survive a model-risk review:

* WOE(bin) = ln( share of goods in the bin / share of bads in the bin ). It is a
  log-odds contribution, so a logistic regression on WOE is linear in exactly the
  quantity it is trying to predict.
* Binning absorbs outliers, missing values and non-linearity without a single
  imputation or transformation choice that has to be defended later.
* And the result is readable: "utilisation above 60% costs you 84 points" is a sentence
  a credit committee can approve and an ombudsman can audit. A gradient booster's SHAP
  value is not.

The cost is real and worth stating: binning throws away within-bin ordering, and the
challenger model in ``scorecard.py`` measures exactly how much AUC that costs.

**Monotonicity** is enforced because a scorecard where risk rises from bin 3 to bin 4
and falls again at bin 5 cannot be explained to a customer or a regulator, and is
usually noise. Candidate cuts come from a shallow decision tree, then adjacent bins are
merged until the event rate is monotone.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

EPS = 0.5  # Haldane-Anscombe correction: keeps WOE finite in a bin with zero bads


@dataclass
class Binning:
    """The fitted bins of one feature, plus its WOE table and information value."""

    feature: str
    edges: np.ndarray
    table: pd.DataFrame
    iv: float
    monotonic: bool
    missing_woe: float = 0.0
    missing_rate: float = 0.0

    def transform(self, values: pd.Series) -> pd.Series:
        """Map raw values onto their bin's WOE."""
        idx = np.digitize(values.to_numpy(float), self.edges[1:-1], right=True)
        woe_by_bin = self.table.set_index("bin")["woe"]
        out = pd.Series(
            [woe_by_bin.get(int(i), 0.0) for i in idx], index=values.index, dtype=float
        )
        return out.mask(values.isna(), self.missing_woe)

    def label(self, values: pd.Series) -> pd.Series:
        """Human-readable bin label, for the scorecard document."""
        idx = np.digitize(values.to_numpy(float), self.edges[1:-1], right=True)
        names = self.table.set_index("bin")["range"]
        out = pd.Series([names.get(int(i), "?") for i in idx], index=values.index)
        return out.mask(values.isna(), "missing")


def _woe_table(x: np.ndarray, y: np.ndarray, edges: np.ndarray) -> pd.DataFrame:
    bins = np.digitize(x, edges[1:-1], right=True)
    total_good = float((y == 0).sum())
    total_bad = float((y == 1).sum())
    rows = []
    for b in range(len(edges) - 1):
        mask = bins == b
        n = int(mask.sum())
        bad = int(y[mask].sum())
        good = n - bad
        share_good = (good + EPS) / (total_good + EPS * (len(edges) - 1))
        share_bad = (bad + EPS) / (total_bad + EPS * (len(edges) - 1))
        woe = float(np.log(share_good / share_bad))
        rows.append({
            "bin": b,
            "range": f"({edges[b]:,.4g}, {edges[b + 1]:,.4g}]",
            "n": n,
            "share": n / len(x) if len(x) else 0.0,
            "bads": bad,
            "event_rate": bad / n if n else np.nan,
            "woe": woe,
            "iv_contribution": (share_good - share_bad) * woe,
        })
    return pd.DataFrame(rows)


def _is_monotonic(rates: pd.Series) -> bool:
    clean = rates.dropna().to_numpy()
    if clean.size < 3:
        return True
    diffs = np.diff(clean)
    return bool(np.all(diffs >= -1e-12) or np.all(diffs <= 1e-12))


def _discrete_edges(xv: np.ndarray, min_count: int) -> np.ndarray:
    """Cut points for a low-cardinality counter, merging rare values upwards.

    A decision tree with a 5%-of-population leaf constraint cannot split a variable
    where 95% of accounts sit on the value zero — it returns a single bin and an
    information value of exactly zero, which reads as "no signal" when the truth is
    "the split you need is at 0 versus 1 or more". Counters are therefore binned on
    their own distinct values.
    """
    values, counts = np.unique(xv, return_counts=True)
    edges, running = [-np.inf], 0
    for value, count in zip(values[:-1], counts[:-1]):
        running += count
        if running >= min_count and (xv > value).sum() >= min_count:
            edges.append(float(value) + 0.5)
            running = 0
    return np.array(edges + [np.inf])


def fit_binning(x: pd.Series, y: pd.Series, *, max_bins: int = 6,
                min_share: float = 0.03, force_monotonic: bool = True,
                discrete_max_levels: int = 15) -> Binning:
    """Bin one numeric feature against a binary target.

    `min_share` sets the smallest bin as a fraction of the population. Below roughly 5%
    a bin's event rate is too noisy to survive into next quarter's data, which is how
    scorecards that look excellent in development fall over in production.
    """
    present = x.notna()
    xv = x[present].to_numpy(float)
    yv = y[present].to_numpy(int)

    missing_rate = float((~present).mean())
    missing_woe = 0.0
    if (~present).any():
        good_missing = int((y[~present] == 0).sum())
        bad_missing = int((y[~present] == 1).sum())
        total_good, total_bad = float((y == 0).sum()), float((y == 1).sum())
        missing_woe = float(np.log(
            ((good_missing + EPS) / (total_good + EPS)) / ((bad_missing + EPS) / (total_bad + EPS))
        ))

    min_count = max(int(min_share * len(xv)), 50)
    if np.unique(xv).size <= discrete_max_levels:
        edges = _discrete_edges(xv, min_count)
    else:
        tree = DecisionTreeClassifier(
            max_leaf_nodes=max_bins, min_samples_leaf=min_count, random_state=0,
        ).fit(xv.reshape(-1, 1), yv)
        cuts = np.sort(tree.tree_.threshold[tree.tree_.feature >= 0])
        edges = np.unique(np.concatenate([[-np.inf], cuts, [np.inf]]))

    table = _woe_table(xv, yv, edges)

    if force_monotonic:
        while len(edges) > 3 and not _is_monotonic(table["event_rate"]):
            rates = table["event_rate"].to_numpy(float)
            diffs = np.diff(rates)
            # Direction is whichever way the feature mostly runs; then merge the single
            # adjacent pair that violates it worst. Merging the *smallest* gap instead
            # would keep collapsing perfectly good splits and can walk a strong feature
            # down to two bins and a near-zero information value.
            direction = 1.0 if np.nansum(diffs) >= 0 else -1.0
            violation = np.where(direction * diffs < 0, np.abs(diffs), 0.0)
            if not np.any(violation > 0):
                break
            edges = np.delete(edges, int(np.nanargmax(violation)) + 1)
            table = _woe_table(xv, yv, edges)

    return Binning(
        feature=str(x.name), edges=edges, table=table,
        iv=float(table["iv_contribution"].sum()),
        monotonic=_is_monotonic(table["event_rate"]),
        missing_woe=missing_woe, missing_rate=missing_rate,
    )


def iv_strength(iv: float) -> str:
    """Siddiqi's conventional reading of an information value."""
    if iv < 0.02:
        return "unpredictive"
    if iv < 0.10:
        return "weak"
    if iv < 0.30:
        return "medium"
    if iv < 0.50:
        return "strong"
    return "very strong (verify it is not leakage)"


@dataclass
class BinningSet:
    """All fitted binnings for a model, and the transform they define together."""

    binnings: dict[str, Binning] = field(default_factory=dict)

    def fit(self, frame: pd.DataFrame, target: str, features: list[str], **kwargs) -> "BinningSet":
        for feature in features:
            self.binnings[feature] = fit_binning(frame[feature], frame[target], **kwargs)
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {f"woe_{name}": b.transform(frame[name]) for name, b in self.binnings.items()},
            index=frame.index,
        )

    def iv_summary(self) -> pd.DataFrame:
        rows = [{"feature": name, "iv": b.iv, "strength": iv_strength(b.iv),
                 "bins": len(b.table), "monotonic": b.monotonic,
                 "missing_rate": b.missing_rate}
                for name, b in self.binnings.items()]
        return pd.DataFrame(rows).sort_values("iv", ascending=False).reset_index(drop=True)
