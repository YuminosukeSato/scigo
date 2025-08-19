#!/usr/bin/env python3
"""
Generate golden data with scikit-learn's LogisticRegression.
The Go parity test reads this JSON and performs numerical comparisons.

Example:
  uv run --with scikit-learn --with numpy --with scipy python scripts/golden/gen_logreg.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def main() -> None:
    out_dir = Path("tests/golden")
    out_dir.mkdir(parents=True, exist_ok=True)

    X, y = make_classification(
        n_samples=600,
        n_features=12,
        n_informative=8,
        n_redundant=0,
        random_state=0,
    )
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)

    clf = LogisticRegression(
        penalty="l2",
        C=0.7,
        solver="lbfgs",
        max_iter=1000,
        tol=1e-6,
        random_state=0,
    )
    clf.fit(Xtr, ytr)

    out = {
        "meta": {
            "sklearn_version_hint": "1.5.x",
            "solver": "lbfgs",
            "penalty": "l2",
            "C": 0.7,
            "fit_intercept": bool(clf.fit_intercept),
            "random_state": 0,
        },
        "coef_": clf.coef_.tolist(),
        "intercept_": clf.intercept_.tolist(),
        "classes_": clf.classes_.astype(int).tolist(),
        "n_iter_": getattr(clf, "n_iter_", np.array([0])).astype(int).tolist(),
        "X_test": Xte.tolist(),
        "y_test": yte.astype(int).tolist(),
        "pred": clf.predict(Xte).astype(int).tolist(),
        "proba": clf.predict_proba(Xte).tolist(),
    }

    with (out_dir / "logreg_case1.json").open("w") as f:
        json.dump(out, f)

    print("Wrote:", out_dir / "logreg_case1.json")


if __name__ == "__main__":
    main()
