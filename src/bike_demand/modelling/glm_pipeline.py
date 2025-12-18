from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd
from glum import GeneralizedLinearRegressor, TweedieDistribution
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from bike_demand.feature_engineering.transformers import CyclicalEncoder


# ---------------------------------------------------------------------
# Feature specification
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class BikeFeatureSpec:
    """
    Single source of truth for feature groups used in modelling.

    Notes
    -----
    - Pipelines are designed to be robust to missing columns
      (e.g. older parquet versions).
    - Only columns present at fit-time are used.
    """

    target: str = "rented_bike_count"

    # cyclical variables (encoded into sin/cos; originals dropped)
    cyclical: Tuple[str, ...] = ("hour", "month", "day_of_week")

    # continuous numeric predictors
    numeric: Tuple[str, ...] = (
        "dew_point_temp",
        "temperature",
        "humidity",
        "wind_speed",
        "visibility",
        "solar_radiation",
    )

    # categorical predictors
    categorical: Tuple[str, ...] = ()

    # binary flags (treated as categorical for GLM)
    binary: Tuple[str, ...] = ("rain_binary", "snow_binary", "holiday")

    # columns always dropped from X
    drop: Tuple[str, ...] = (
        "date",
        "sample",
        "functioning_day",
        "seasons",
        "rainfall",
        "snowfall",
    )


def split_xy(
    df: pd.DataFrame,
    spec: BikeFeatureSpec | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split a dataframe into feature matrix X and target vector y.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing features and target.
    spec : BikeFeatureSpec or None, optional
        Feature specification. If None, default specification is used.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix with target and dropped columns removed.
    y : pd.Series
        Target variable.
    """
    spec = spec or BikeFeatureSpec()
    X = df.drop(columns=[spec.target, *spec.drop], errors="ignore")
    y = df[spec.target]
    return X, y


# ---------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------
class _ColumnAwarePreprocess(BaseEstimator, TransformerMixin):
    """
    Build a ColumnTransformer dynamically based on columns present in X.

    This avoids pipeline failures when some features are unavailable
    (e.g. older cleaned datasets).
    """

    def __init__(self, spec: BikeFeatureSpec):
        self.spec = spec

    def fit(self, X: pd.DataFrame, y=None):
        cols = set(X.columns)

        numeric_base = [c for c in self.spec.numeric if c in cols]
        categorical_base = [c for c in self.spec.categorical if c in cols]
        binary_base = [c for c in self.spec.binary if c in cols]

        # cyclical features produced by CyclicalEncoder
        cyc_num: list[str] = []
        for c in self.spec.cyclical:
            sin_col, cos_col = f"{c}_sin", f"{c}_cos"
            if sin_col in cols and cos_col in cols:
                cyc_num.extend([sin_col, cos_col])

        numeric_features = numeric_base + cyc_num
        categorical_features = categorical_base + binary_base

        num_pipe = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]
        )

        cat_pipe = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        drop="first",
                    ),
                ),
            ]
        )

        self.pre_ = ColumnTransformer(
            transformers=[
                ("num", num_pipe, numeric_features),
                ("cat", cat_pipe, categorical_features),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )

        self.pre_.fit(X, y)
        return self

    def transform(self, X: pd.DataFrame):
        return self.pre_.transform(X)


# ---------------------------------------------------------------------
# GLM pipeline
# ---------------------------------------------------------------------
def build_glm_pipeline(
    spec: Optional[BikeFeatureSpec] = None,
    *,
    tweedie_power: float = 1.5,
) -> Pipeline:
    """
    Build a GLM pipeline with Tweedie distribution.

    Pipeline steps
    --------------
    1. Cyclical encoding of hour / month / day_of_week (sin & cos).
    2. Imputation and scaling of numeric features.
    3. Imputation and one-hot encoding of categorical and binary features.
    4. Generalized linear model with Tweedie loss.

    Notes
    -----
    - Regularisation parameters can be tuned via GridSearchCV:
        * model__alpha
        * model__l1_ratio
    """
    spec = spec or BikeFeatureSpec()

    cyc = CyclicalEncoder(
        columns=list(spec.cyclical),
        periods={"hour": 24, "month": 12, "day_of_week": 7},
        drop_original=True,
    )

    model = GeneralizedLinearRegressor(
        family=TweedieDistribution(tweedie_power),
        fit_intercept=True,
        alpha=0.0,  # tuned via CV
        l1_ratio=0.0,  # tuned via CV
    )

    return Pipeline(
        steps=[
            ("cyclical", cyc),
            ("preprocess", _ColumnAwarePreprocess(spec)),
            ("model", model),
        ]
    )
