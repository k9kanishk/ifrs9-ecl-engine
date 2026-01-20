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

    thr_ratio_map = policy.get("sicr_pd_ratio_threshold", {})
    thr_abs_map = policy.get("sicr_pd_abs_threshold", {})

    use_lcr = bool(policy["default"].get("use_lcr_exemption", True))
    lcr_default = float(policy["default"].get("lcr_pd_threshold_default", 0.01))
    lcr_map = policy.get("lcr_pd_threshold", {})

    use_watchlist = bool(policy["default"].get("use_watchlist_flag", False))
    watchlist_col = str(policy["default"].get("watchlist_column", "watchlist_flag"))

    cure_cfg = policy.get("cure", {})
    cure_s2_to_s1 = int(cure_cfg.get("stage2_to_1_months", 3))
    cure_s3_to_s2 = int(cure_cfg.get("stage3_to_2_months", 6))
    cure_dpd_below = int(cure_cfg.get("dpd_below_for_cure", 30))
    cure_watch_clear = bool(cure_cfg.get("require_watchlist_clear", True))
    cure_s3_no_default = bool(
        cure_cfg.get("require_no_default_flag_for_stage3_cure", True)
    )

    acc = accounts[["account_id", "segment"]].copy()
    df = perf.copy()
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])

    df = (
        df.merge(acc, on="account_id", how="left")
        .merge(pd_panel, on=["snapshot_date", "account_id"], how="left")
    )

    # thresholds
    df["sicr_pd_ratio_threshold"] = (
        df["segment"]
        .map(thr_ratio_map)
        .fillna(thr_ratio_map.get("default", 2.0))
        .astype(float)
    )
    df["sicr_pd_abs_threshold"] = (
        df["segment"]
        .map(thr_abs_map)
        .fillna(thr_abs_map.get("default", 0.01))
        .astype(float)
    )
    df["lcr_pd_threshold"] = (
        df["segment"].map(lcr_map).fillna(lcr_default).astype(float)
    )

    # ensure core fields exist
    df["dpd"] = pd.to_numeric(df.get("dpd", 0), errors="coerce").fillna(0).astype(int)
    if "default_flag" in df.columns:
        df["default_flag"] = (
            pd.to_numeric(df["default_flag"], errors="coerce").fillna(0).astype(int)
        )
    else:
        df["default_flag"] = 0

    # SICR PD measures
    df["pd_pit"] = pd.to_numeric(df["pd_pit"], errors="coerce")
    df["pd_pit_orig"] = pd.to_numeric(df["pd_pit_orig"], errors="coerce")

    # fallback if missing (should be rare)
    df["pd_pit"] = df["pd_pit"].fillna(0.02)
    df["pd_pit_orig"] = df["pd_pit_orig"].fillna(0.02)

    df["pd_delta"] = df["pd_pit"] - df["pd_pit_orig"]
    df["pd_ratio"] = df["pd_pit"] / np.clip(df["pd_pit_orig"], 1e-12, None)

    # Stage 3 flag
    df["stage3_flag"] = df["dpd"] >= default_dpd
    if use_default_flag:
        df["stage3_flag"] = df["stage3_flag"] | (df["default_flag"] == 1)

    # SICR triggers (Stage 2) — only if not Stage 3
    df["sicr_flag_dpd"] = (df["dpd"] >= sicr_dpd) & (~df["stage3_flag"])
    df["sicr_flag_pd_ratio"] = (
        df["pd_ratio"] >= df["sicr_pd_ratio_threshold"]
    ) & (~df["stage3_flag"])
    df["sicr_flag_pd_abs"] = (df["pd_delta"] >= df["sicr_pd_abs_threshold"]) & (
        ~df["stage3_flag"]
    )

    # Optional qualitative trigger
    if use_watchlist and watchlist_col in df.columns:
        df["watchlist_flag"] = (
            pd.to_numeric(df[watchlist_col], errors="coerce").fillna(0).astype(int)
        )
    else:
        df["watchlist_flag"] = 0
    df["sicr_flag_watchlist"] = (df["watchlist_flag"] == 1) & (~df["stage3_flag"])

    # Combine SICR
    df["sicr_flag_pd"] = df["sicr_flag_pd_ratio"] | df["sicr_flag_pd_abs"]
    df["sicr_flag"] = (
        df["sicr_flag_dpd"] | df["sicr_flag_pd"] | df["sicr_flag_watchlist"]
    )

    # Low credit risk exemption (only applies to potential Stage 2, never Stage 3)
    df["lcr_flag"] = False
    if use_lcr:
        df["lcr_flag"] = (
            (df["pd_pit"] <= df["lcr_pd_threshold"])
            & (df["dpd"] < sicr_dpd)
            & (~df["stage3_flag"])
        )

    # Raw stage
    df["stage_raw"] = 1
    df.loc[df["sicr_flag"] & (~df["lcr_flag"]), "stage_raw"] = 2
    df.loc[df["stage3_flag"], "stage_raw"] = 3

    # Apply cure / probation per account
    df = df.sort_values(["account_id", "snapshot_date"]).reset_index(drop=True)

    stages = []
    reasons = []

    for acc_id, g in df.groupby("account_id", sort=False):
        cur_stage = 1
        good_s2 = 0
        good_s3 = 0

        for _, r in g.iterrows():
            raw = int(r["stage_raw"])

            # determine “good” conditions for cure
            watch_ok = (int(r["watchlist_flag"]) == 0) if cure_watch_clear else True
            dpd_ok = int(r["dpd"]) < cure_dpd_below

            s3_default_ok = True
            if cure_s3_no_default:
                s3_default_ok = (int(r["default_flag"]) == 0)

            # State machine
            if cur_stage == 3:
                if raw < 3 and dpd_ok and watch_ok and s3_default_ok:
                    good_s3 += 1
                    if good_s3 >= cure_s3_to_s2:
                        cur_stage = 2 if raw == 2 else 1
                        good_s3 = 0
                else:
                    good_s3 = 0
                    cur_stage = 3  # stays 3

            elif cur_stage == 2:
                if raw == 1 and dpd_ok and watch_ok:
                    good_s2 += 1
                    if good_s2 >= cure_s2_to_s1:
                        cur_stage = 1
                        good_s2 = 0
                else:
                    good_s2 = 0
                    cur_stage = 3 if raw == 3 else 2

            else:  # cur_stage == 1
                cur_stage = 3 if raw == 3 else (2 if raw == 2 else 1)
                good_s2 = 0
                good_s3 = 0

            stages.append(cur_stage)

            # Reason string (auditable)
            if cur_stage == 3:
                reasons.append("Stage3: Default/90+ DPD (credit-impaired)")
            elif cur_stage == 2:
                if raw == 1:
                    reasons.append("Stage2: SICR (cure probation)")
                elif r["sicr_flag_watchlist"]:
                    reasons.append("Stage2: SICR (watchlist/qualitative)")
                elif r["sicr_flag_dpd"] and r["sicr_flag_pd"]:
                    reasons.append("Stage2: SICR (DPD>=30 + PD deterioration)")
                elif r["sicr_flag_dpd"]:
                    reasons.append("Stage2: SICR (DPD>=30 backstop)")
                elif r["sicr_flag_pd_abs"] and r["sicr_flag_pd_ratio"]:
                    reasons.append("Stage2: SICR (PD abs + ratio)")
                elif r["sicr_flag_pd_abs"]:
                    reasons.append("Stage2: SICR (PD absolute deterioration)")
                else:
                    reasons.append("Stage2: SICR (PD ratio deterioration)")
            else:
                if r["lcr_flag"] and raw == 2:
                    reasons.append("Stage1: Low credit risk exemption (LCR)")
                else:
                    reasons.append("Stage1")

        # end for rows

    df["stage"] = stages
    df["stage_reason"] = reasons

    # Keep key audit flags for dashboard/validation
    keep = df.columns.tolist()

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
