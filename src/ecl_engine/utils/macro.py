from __future__ import annotations

import numpy as np
import pandas as pd


CANON_Z = ["unemployment_z", "gdp_yoy_z", "policy_rate_z"]
RAW_ALIASES = {
    "unemployment_z": ["unemployment_z", "unemp_z", "unemployment"],
    "gdp_yoy_z": ["gdp_yoy_z", "gdp_z", "gdp_yoy"],
    "policy_rate_z": ["policy_rate_z", "rate_z", "policy_rate"],
}


def _month_end(d: pd.Series) -> pd.Series:
    dt = pd.to_datetime(d)
    return dt + pd.offsets.MonthEnd(0)


def _ensure_scenario_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "scenario" not in df.columns:
        raise KeyError(f"Macro is missing 'scenario'. Available: {list(df.columns)}")
    if "date" not in df.columns:
        raise KeyError(f"Macro is missing 'date'. Available: {list(df.columns)}")
    df["date"] = _month_end(df["date"])
    return df


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _compute_z_by_base(df: pd.DataFrame, raw_col: str, z_col: str) -> pd.Series:
    """
    Compute z-scores using Base scenario mean/std (common IFRS9 approach for normalization).
    If Base not available, fallback to all scenarios pooled.
    """
    base = df[df["scenario"].astype(str).str.lower() == "base"]
    ref = base if len(base) > 0 else df

    mu = float(ref[raw_col].mean())
    sd = float(ref[raw_col].std(ddof=0))
    sd = sd if sd > 1e-12 else 1.0  # avoid divide-by-zero

    return (df[raw_col] - mu) / sd


def prepare_macro_for_ecl(macro: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a macro DataFrame indexed by (scenario, date) with canonical z columns:
      unemployment_z, gdp_yoy_z, policy_rate_z

    Accepts:
      A) already-z macro (has *_z columns), OR
      B) raw macro (unemployment, gdp_yoy, policy_rate) and will create z columns.

    Output format is what ecl.py expects for:
      mz.loc[(scenario, future_dates), ["unemployment_z","gdp_yoy_z","policy_rate_z"]]
    """
    df = macro.copy()

    # If macro came with index levels, reset to columns safely
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index()

    df = _ensure_scenario_date_columns(df)

    # Build z columns if needed
    for z_name in CANON_Z:
        # if z already exists, keep it
        if z_name in df.columns:
            continue

        raw_candidate = _pick_col(df, RAW_ALIASES[z_name])
        if raw_candidate is None:
            raise KeyError(
                f"Macro missing required inputs for {z_name}. "
                f"Need one of {RAW_ALIASES[z_name]}. Available: {list(df.columns)}"
            )

        # If raw_candidate itself is the z col name, we'd have continued above.
        # Here raw_candidate may be raw level; create z via Base reference.
        df[z_name] = _compute_z_by_base(df, raw_candidate, z_name)

    # Final sanity
    missing = [c for c in CANON_Z if c not in df.columns]
    if missing:
        raise KeyError(f"Macro z columns missing after prep: {missing}. Available: {list(df.columns)}")

    # Return MultiIndex (scenario, date)
    out = (
        df[["scenario", "date"] + CANON_Z]
        .copy()
        .sort_values(["scenario", "date"])
        .set_index(["scenario", "date"])
    )
    return out
