from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def load_policy(path: str | Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def prepare_macro_base(macro: pd.DataFrame) -> pd.DataFrame:
    """
    Keep Base scenario and compute z-scores for macro variables over the history window.
    """
    base = macro.loc[
        macro["scenario"] == "Base",
        ["date", "unemployment", "gdp_yoy", "policy_rate"],
    ].copy()
    base["date"] = pd.to_datetime(base["date"])

    for col in ["unemployment", "gdp_yoy", "policy_rate"]:
        mu = base[col].mean()
        sd = base[col].std(ddof=0)
        sd = sd if sd > 0 else 1.0
        base[f"{col}_z"] = (base[col] - mu) / sd

    return base[["date", "unemployment_z", "gdp_yoy_z", "policy_rate_z"]]


def align_to_macro_date(d: pd.Series, macro_dates: pd.DatetimeIndex) -> pd.Series:
    """
    Align arbitrary dates to closest available macro date (month-end).
    If date is before macro starts -> clamp to first.
    If after macro ends -> clamp to last.
    """
    d = pd.to_datetime(d)
    first = macro_dates.min()
    last = macro_dates.max()
    d = d.clip(lower=first, upper=last)

    # Convert to month-end to match our macro index
    d_me = (d + pd.offsets.MonthEnd(0)).dt.normalize()
    # If some month-ends are missing, map to nearest existing macro date
    # (usually not needed since both are month-end, but safe)
    lookup = pd.Index(macro_dates)
    pos = lookup.get_indexer(d_me, method="nearest")
    return lookup.take(pos)


def compute_pit_pd_panel(
    accounts: pd.DataFrame,
    perf: pd.DataFrame,
    macro_base_z: pd.DataFrame,
    policy: dict,
) -> pd.DataFrame:
    """
    Returns a panel keyed by (account_id, snapshot_date) with:
      - pd_pit
      - pd_pit_orig
      - pd_ratio
    """
    betas = policy["default"]["pd_logit"]
    pd_floor = float(policy["default"]["pd_floor"])
    pd_cap = float(policy["default"]["pd_cap"])

    # Macro base z keyed by date
    mz = macro_base_z.copy()
    mz = mz.set_index("date")

    macro_dates = mz.index

    # Snapshot macro join
    perf_dates = pd.to_datetime(perf["snapshot_date"])
    aligned_snap = align_to_macro_date(perf_dates, macro_dates)
    snap_macro = mz.loc[aligned_snap].reset_index(drop=True)
    snap_macro["snapshot_date"] = pd.to_datetime(perf["snapshot_date"]).values

    # One row per snapshot_date (fast merge)
    snap_macro = snap_macro.groupby("snapshot_date", as_index=False).first()

    # Origination macro join (one row per account)
    orig_me = (
        pd.to_datetime(accounts["origination_date"]) + pd.offsets.MonthEnd(0)
    ).dt.normalize()
    aligned_orig = align_to_macro_date(orig_me, macro_dates)
    orig_macro = mz.loc[aligned_orig].reset_index(drop=True)
    orig_macro["account_id"] = accounts["account_id"].values

    # Precompute orig PD PIT (one per account)
    ttc = accounts["ttc_pd_annual"].astype(float).values
    logit_ttc = _logit(ttc)

    x_orig = (
        betas["intercept"]
        + betas["unemployment_z"] * orig_macro["unemployment_z"].to_numpy()
        + betas["gdp_yoy_z"] * orig_macro["gdp_yoy_z"].to_numpy()
        + betas["policy_rate_z"] * orig_macro["policy_rate_z"].to_numpy()
    )
    logit_pit_orig = logit_ttc + x_orig
    pd_pit_orig = np.clip(_sigmoid(logit_pit_orig), pd_floor, pd_cap)

    orig_pd_df = pd.DataFrame(
        {"account_id": accounts["account_id"].values, "pd_pit_orig": pd_pit_orig}
    )

    # Merge snapshot macro to perf (on snapshot_date)
    perf2 = perf[["snapshot_date", "account_id"]].copy()
    perf2["snapshot_date"] = pd.to_datetime(perf2["snapshot_date"])

    perf2 = (
        perf2.merge(snap_macro, on="snapshot_date", how="left")
        .merge(orig_pd_df, on="account_id", how="left")
    )

    # Compute PD PIT at snapshot using same TTC logit + snapshot macro shift
    # (ttc depends on account -> merge ttc)
    ttc_df = accounts[["account_id", "ttc_pd_annual"]].copy()
    perf2 = perf2.merge(ttc_df, on="account_id", how="left")

    logit_ttc_panel = _logit(perf2["ttc_pd_annual"].to_numpy())

    x_snap = (
        betas["intercept"]
        + betas["unemployment_z"] * perf2["unemployment_z"].to_numpy()
        + betas["gdp_yoy_z"] * perf2["gdp_yoy_z"].to_numpy()
        + betas["policy_rate_z"] * perf2["policy_rate_z"].to_numpy()
    )
    logit_pit = logit_ttc_panel + x_snap
    pd_pit = np.clip(_sigmoid(logit_pit), pd_floor, pd_cap)

    perf2["pd_pit"] = pd_pit
    perf2["pd_ratio"] = perf2["pd_pit"] / np.clip(
        perf2["pd_pit_orig"], 1e-12, None
    )

    return perf2[["snapshot_date", "account_id", "pd_pit", "pd_pit_orig", "pd_ratio"]]


def assign_stage(
    accounts: pd.DataFrame,
    perf: pd.DataFrame,
    pd_panel: pd.DataFrame,
    policy: dict,
) -> pd.DataFrame:
    default_dpd = int(policy["default"]["default_dpd"])
    sicr_dpd = int(policy["default"]["sicr_dpd_backstop"])
    use_default_flag = bool(policy["default"]["stage3_if_default_flag"])

    thr_map = policy["sicr_pd_ratio_threshold"]

    acc = accounts[["account_id", "segment"]].copy()
    df = perf.copy()
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])

    df = df.merge(acc, on="account_id", how="left").merge(
        pd_panel, on=["snapshot_date", "account_id"], how="left"
    )

    # Segment-specific thresholds (default fallback)
    df["sicr_pd_ratio_threshold"] = (
        df["segment"].map(thr_map).fillna(2.0).astype(float)
    )

    # Flags
    df["stage3_flag"] = df["dpd"].astype(int) >= default_dpd
    if use_default_flag and "default_flag" in df.columns:
        df["stage3_flag"] = df["stage3_flag"] | (
            df["default_flag"].astype(int) == 1
        )

    df["sicr_flag_dpd"] = (df["dpd"].astype(int) >= sicr_dpd) & (
        ~df["stage3_flag"]
    )
    df["sicr_flag_pd"] = (
        df["pd_ratio"].astype(float) >= df["sicr_pd_ratio_threshold"]
    ) & (~df["stage3_flag"])

    # Stage assignment
    df["stage"] = 1
    df.loc[df["sicr_flag_dpd"] | df["sicr_flag_pd"], "stage"] = 2
    df.loc[df["stage3_flag"], "stage"] = 3

    # Reason string (auditable)
    reason = np.where(
        df["stage"] == 3,
        "Stage3: Default/90+ DPD",
        np.where(
            df["stage"] == 2,
            np.where(
                df["sicr_flag_dpd"] & df["sicr_flag_pd"],
                "Stage2: SICR (DPD>=30 + PD deterioration)",
                np.where(
                    df["sicr_flag_dpd"],
                    "Stage2: SICR (DPD>=30 backstop)",
                    "Stage2: SICR (PD deterioration)",
                ),
            ),
            "Stage1",
        ),
    )
    df["stage_reason"] = reason

    return df


def stage_qc_summary(staged: pd.DataFrame) -> pd.DataFrame:
    """
    Counts by snapshot_date, segment, stage (for quick sanity checks).
    """
    out = (
        staged.groupby(["snapshot_date", "segment", "stage"], as_index=False)
        .size()
        .rename(columns={"size": "n_accounts"})
        .sort_values(["snapshot_date", "segment", "stage"])
    )
    return out


def run(
    accounts_path: str | Path = "data/curated/accounts.parquet",
    perf_path: str | Path = "data/curated/performance_monthly.parquet",
    macro_path: str | Path = "data/curated/macro_scenarios_monthly.parquet",
    policy_path: str | Path = "configs/policy.yml",
    outdir: str | Path = "data/curated",
) -> None:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    accounts = pd.read_parquet(accounts_path)
    perf = pd.read_parquet(perf_path)
    macro = pd.read_parquet(macro_path)
    policy = load_policy(policy_path)

    macro_base_z = prepare_macro_base(macro)
    pd_panel = compute_pit_pd_panel(accounts, perf, macro_base_z, policy)
    staged = assign_stage(accounts, perf, pd_panel, policy)
    qc = stage_qc_summary(staged)

    staged_out = outdir / "staging_output.parquet"
    qc_out = outdir / "staging_qc_summary.parquet"

    staged.to_parquet(staged_out, index=False)
    qc.to_parquet(qc_out, index=False)

    # Console sanity
    latest = staged["snapshot_date"].max()
    print("Wrote:")
    print(f" - {staged_out}")
    print(f" - {qc_out}")
    print("\nLatest snapshot stage mix:")
    print(
        staged.loc[staged["snapshot_date"] == latest, "stage"]
        .value_counts(dropna=False)
        .sort_index()
    )


if __name__ == "__main__":
    run()
