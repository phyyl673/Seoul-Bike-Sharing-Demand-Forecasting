from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


@dataclass
class CyclicalEncoder(BaseEstimator, TransformerMixin):
    """
    Encode cyclical (periodic) features into sine/cosine components.

    This transformer is useful for periodic variables where the endpoints
    are adjacent on a circle (e.g. hour 23 is close to hour 0).

    Parameters
    ----------
    columns : Sequence[str]
        Column names to encode (e.g. ["hour", "month"]).
    periods : dict[str, int | float]
        Period per column (e.g. {"hour": 24, "month": 12}).
    drop_original : bool, optional
        If True, drop original columns after encoding. Defaults to False.

    Attributes
    ----------
    feature_names_in_ : np.ndarray
        Names of features seen during `fit`, stored for scikit-learn
        compatibility and `get_feature_names_out`.

    Raises
    ------
    KeyError
        If requested columns are missing from the input, or if any period is
        not provided for a requested column.
    TypeError
        If a period is not numeric, or if X is not a pandas DataFrame.
    ValueError
        If any provided period is not strictly positive.
    """

    columns: Sequence[str]
    periods: dict[str, int | float]
    drop_original: bool = False

    # Fitted attributes (created only after fit)
    feature_names_in_: np.ndarray = field(init=False, repr=False)

    def fit(self, X: pd.DataFrame, y: Any = None) -> CyclicalEncoder:
        """
        Validate configuration and record input feature names.

        Parameters
        ----------
        X : pd.DataFrame
            Input features.
        y : Any, optional
            Ignored. Present for sklearn API compatibility.

        Returns
        -------
        CyclicalEncoder
            Fitted transformer.
        """
        X_df = _ensure_dataframe(X)
        cols = list(self.columns)

        missing_cols = [c for c in cols if c not in X_df.columns]
        if missing_cols:
            raise KeyError(f"CyclicalEncoder: missing column(s): {missing_cols}")

        for c in cols:
            if c not in self.periods:
                raise KeyError(f"CyclicalEncoder: missing period for '{c}'")

            p = self.periods[c]
            if not isinstance(p, (int, float)):
                raise TypeError(
                    f"CyclicalEncoder: period for '{c}' must be numeric, got {type(p).__name__}"
                )
            if p <= 0:
                raise ValueError(f"CyclicalEncoder: period for '{c}' must be > 0, got {p}")

        self.feature_names_in_ = np.array(list(X_df.columns), dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform cyclical columns into sine/cosine features.

        Parameters
        ----------
        X : pd.DataFrame
            Input features.

        Returns
        -------
        pd.DataFrame
            Transformed dataframe containing new `<col>_sin` and `<col>_cos`
            columns (and optionally dropping originals).

        Raises
        ------
        KeyError
            If requested columns are missing from the input.
        ValueError
            If a requested column cannot be converted to numeric.
        """
        check_is_fitted(self, attributes=["feature_names_in_"])

        X_df = _ensure_dataframe(X)
        cols = list(self.columns)

        missing_cols = [c for c in cols if c not in X_df.columns]
        if missing_cols:
            raise KeyError(f"CyclicalEncoder: missing column(s): {missing_cols}")

        out = X_df.copy()

        for c in cols:
            period = float(self.periods[c])
            vals = pd.to_numeric(out[c], errors="raise").astype(float).to_numpy()
            angle = 2.0 * np.pi * vals / period

            out[f"{c}_sin"] = np.sin(angle)
            out[f"{c}_cos"] = np.cos(angle)

            if self.drop_original:
                out = out.drop(columns=[c])

        return out

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> np.ndarray:
        """
        Get output feature names for scikit-learn pipelines.

        Parameters
        ----------
        input_features : Sequence[str] or None, optional
            If provided, use these as the input feature names; otherwise use
            names recorded during `fit`.

        Returns
        -------
        np.ndarray
            Output feature names after transformation.
        """
        check_is_fitted(self, attributes=["feature_names_in_"])

        in_feats = list(self.feature_names_in_) if input_features is None else list(input_features)
        out_feats = list(in_feats)

        for c in list(self.columns):
            out_feats.append(f"{c}_sin")
            out_feats.append(f"{c}_cos")
            if self.drop_original and c in out_feats:
                out_feats.remove(c)

        return np.array(out_feats, dtype=object)


def _ensure_dataframe(X: Any) -> pd.DataFrame:
    """
    Ensure the input is a pandas DataFrame.

    Parameters
    ----------
    X : Any
        Input object.

    Returns
    -------
    pd.DataFrame
        The same object if it is a DataFrame.

    Raises
    ------
    TypeError
        If X is not a pandas DataFrame.
    """
    if isinstance(X, pd.DataFrame):
        return X
    raise TypeError(f"CyclicalEncoder expects a pandas DataFrame input (got {type(X).__name__}).")
