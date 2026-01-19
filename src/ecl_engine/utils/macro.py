from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_Z_COLS = ["unemployment_z", "gdp_yoy_z", "policy_rate_z"]


def _as_month_end(x) -> pd.Timestamp:
    return (pd.to_datetime(x) + pd.offsets.MonthEnd(0)).normalize()


def prepare_macro_for_ecl(macro: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a MultiIndex DataFrame indexed by (scenario, date)
    with z-scored macro columns:
      unemployment_z, gdp_yoy_z, policy_rate_z

    Z-scoring is anchored to the Base scenario distribution.
    """
    df = macro.copy()

    # standardize column names
    if "date" not in df.columns:
        raise KeyError(f"macro missing 'date' column. cols={list(df.columns)}")
    if "scenario" not in df.columns:
        raise KeyError(f"macro missing 'scenario' column. cols={list(df.columns)}")

    df["date"] = df["date"].apply(_as_month_end)
    df["scenario"] = df["scenario"].astype(str)

    need_raw = ["unemployment", "gdp_yoy", "policy_rate"]
    missing = [c for c in need_raw if c not in df.columns]
    if missing:
        raise KeyError(f"macro missing raw columns {missing}. cols={list(df.columns)}")

    base = df[df["scenario"] == "Base"].sort_values("date")
    if base.empty:
        raise ValueError("macro has no Base scenario rows; cannot anchor z-scores")

    mu = base[need_raw].mean(numeric_only=True)
    sd = base[need_raw].std(numeric_only=True).replace(0.0, 1.0)

    for c in need_raw:
        df[f"{c}_z"] = (df[c] - mu[c]) / sd[c]

    mz = df.set_index(["scenario", "date"])[REQUIRED_Z_COLS].sort_index()
    return mz


def prepare_macro_z_all_scenarios(macro: pd.DataFrame) -> pd.DataFrame:
    """
    Backwards-compatible helper used by older modules.

    Returns a *flat* DataFrame with columns:
      scenario, date, unemployment_z, gdp_yoy_z, policy_rate_z
    """
    mz = prepare_macro_for_ecl(macro)
    return mz.reset_index()
