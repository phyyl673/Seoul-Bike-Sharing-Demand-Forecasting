from __future__ import annotations

from pathlib import Path

import pandas as pd


def _project_root() -> Path:
    """
    Return the project root directory.

    Notes
    -----
    This project assumes the repository root is three levels above this file.
    If you move this module to a different depth, update `parents[3]`
    accordingly.
    """
    return Path(__file__).resolve().parents[3]


def load_data(csv_path: str = "raw/SeoulBikeData.csv") -> pd.DataFrame:
    """
    Load the raw Seoul Bike data from the project's `data/` directory.

    Parameters
    ----------
    csv_path : str, optional
        Path to the CSV file *relative to* `<project_root>/data/`.
        Defaults to ``"raw/SeoulBikeData.csv"``.

    Returns
    -------
    pd.DataFrame
        Raw bike data loaded from CSV.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist.
    """
    project_root = _project_root()
    file_path = project_root / "data" / csv_path

    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    return pd.read_csv(file_path, encoding="latin1")


def load_cleaned_data(
    parquet_path: str = "processed/seoul_bike_cleaned.parquet",
    *,
    ensure_calendar_features: bool = True,
) -> pd.DataFrame:
    """
    Load cleaned (processed) Seoul Bike data from a parquet file.

    This loader is defensive: if the stored parquet was generated before
    calendar features were added, it can recreate them from the `date` column.

    Parameters
    ----------
    parquet_path : str, optional
        Path to the parquet file *relative to* `<project_root>/data/`.
        Defaults to ``"processed/seoul_bike_cleaned.parquet"``.
    ensure_calendar_features : bool, optional
        If True, ensure `month`, `day_of_week` exists
        (derived from `date`) when missing. Defaults to True.

    Returns
    -------
    pd.DataFrame
        Cleaned dataset loaded from parquet.

    Raises
    ------
    FileNotFoundError
        If the parquet file does not exist.
    KeyError
        If `date` is missing from the cleaned data.
    ValueError
        If `date` cannot be converted to datetime.
    """
    root = _project_root()
    file_path = root / "data" / parquet_path

    if not file_path.exists():
        raise FileNotFoundError(f"Processed parquet not found: {file_path}")

    df = pd.read_parquet(file_path)

    if "date" not in df.columns:
        raise KeyError("Expected column 'date' not found in cleaned data.")

    # Ensure datetime (raise if conversion fails to keep bugs visible)
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"], errors="raise")

    if ensure_calendar_features:
        if "month" not in df.columns:
            df["month"] = df["date"].dt.month
        if "day_of_week" not in df.columns:
            df["day_of_week"] = df["date"].dt.dayofweek  # Monday=0, Sunday=6

    return df
