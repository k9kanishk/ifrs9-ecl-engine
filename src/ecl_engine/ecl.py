from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from ecl_engine.models.lgd_workout import stage3_workout_table_scenarios
from ecl_engine.utils.io import load_yml
from ecl_engine.utils.macro import prepare_macro_for_ecl


def as_month_end(x) -> pd.Timestamp:
    return (pd.to_datetime(x) + pd.offsets.MonthEnd(0)).normalize()


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def month_ends_forward(asof_dt: pd.Timestamp, periods: int) -> pd.DatetimeIndex:
    # freq="ME" avoids pandas 'M' deprecation warning
    start_me = as_month_end(asof_dt)
    return pd.date_range(start=start_me + pd.offsets.MonthEnd(1), periods=periods, freq="ME")


def _macro_matrix(
    mz: pd.DataFrame,
    scenario: str,
    dates: pd.DatetimeIndex,
    cols: list[str],
) -> np.ndarray:
    """
    Safe macro extraction even if dates are missing.

    Strategy:
      - take scenario slice (date-indexed)
      - build union index = existing dates U requested dates
      - forward-fill then back-fill (so future dates beyond max get last known value)
      - return values on requested dates
    """
    # scenario slice: date-indexed frame
    try:
        s = mz.xs(scenario, level=0)
    except KeyError as e:
        raise KeyError(
            f"Scenario '{scenario}' not found in macro index. Available: {mz.index.get_level_values(0).unique()}"
        ) from e

    s = s.sort_index()

    # ensure month-end alignment
    dates = (pd.to_datetime(dates) + pd.offsets.MonthEnd(0)).to_period("M").to_timestamp("M")

    # build full grid then fill
    full_idx = s.index.union(dates)
    grid = s.reindex(full_idx).sort_index()
    grid[cols] = grid[cols].ffill().bfill()

    # if still NaN (e.g., macro totally empty), fail loudly
    if grid[cols].isna().any().any():
        raise ValueError(
            f"Macro still has NaNs after ffill/bfill for scenario={scenario}. "
            f"Check macro_scenarios_monthly.parquet generation."
        )

    return grid.loc[dates, cols].to_numpy(dtype=np.float64)


def compute_stress_scalar(
    mz: pd.DataFrame,
    scenario: str,
    future_dates: pd.DatetimeIndex,
    stress_weights: dict,
    horizon_months: int,
) -> float:
    cols = ["unemployment_z", "gdp_yoy_z", "policy_rate_z"]
    m = _macro_matrix(mz, scenario, future_dates, cols)[:horizon_months, :]
    w_u = float(stress_weights.get("unemployment_z", 1.0))
    w_g = float(stress_weights.get("gdp_yoy_z", -0.5))
    w_r = float(stress_weights.get("policy_rate_z", 0.25))
    stress = w_u * m[:, 0] + w_g * m[:, 1] + w_r * m[:, 2]
    return float(np.mean(stress))


def months_to_maturity(accounts_asof: pd.DataFrame, asof_dt: pd.Timestamp, params: dict) -> np.ndarray:
    default_map = params.get("default_maturity_months_by_segment", {})
    if "maturity_date" in accounts_asof.columns:
        md = pd.to_datetime(accounts_asof["maturity_date"], errors="coerce")
        m = ((md.dt.to_period("M") - asof_dt.to_period("M")).apply(lambda x: x.n)).astype("float64")
        m = m.fillna(0.0).to_numpy()
        # if missing/<=0 fall back to segment defaults
        seg = accounts_asof["segment"].astype(str).to_numpy()
        fallback = np.array([float(default_map.get(s, 36)) for s in seg], dtype=np.float64)
        m = np.where(m > 0, m, fallback)
    else:
        seg = accounts_asof["segment"].astype(str).to_numpy()
        m = np.array([float(default_map.get(s, 36)) for s in seg], dtype=np.float64)
    return np.clip(m, 1.0, 360.0)


def compute_ead(balance: np.ndarray, limit_amount: np.ndarray, segment: np.ndarray, params: dict) -> np.ndarray:
    ccf_map = params.get("ccf_base", {})
    ccf_rev = float(ccf_map.get("Revolving", 0.75))
    is_rev = segment == "Revolving"
    undrawn = np.maximum(limit_amount - balance, 0.0)
    ead = balance.copy()
    ead[is_rev] = balance[is_rev] + ccf_rev * undrawn[is_rev]
    return np.maximum(ead, 0.0)


def compute_ecl_from_frames(
    staged: pd.DataFrame,
    accounts: pd.DataFrame,
    macro: pd.DataFrame,
    params: dict,
    asof_dt: pd.Timestamp,
    policy: dict | None = None,
    scenario_weights: dict | None = None,
) -> pd.DataFrame:
    """
    Core ECL computation using provided dataframes.
    Used by driver_decomp / explain / tests.
    """
    staged = staged.copy()
    accounts = accounts.copy()
    macro = macro.copy()

    staged["snapshot_date"] = staged["snapshot_date"].apply(as_month_end)
    # macro must be multi-index (scenario, date) with *_z columns
    mz = prepare_macro_for_ecl(macro)

    asof_dt = as_month_end(asof_dt)
    staging_asof = staged[staged["snapshot_date"] == asof_dt][["account_id", "stage"]].copy()

    base = accounts.copy()
    if "balance" not in base.columns and "balance" in staged.columns:
        bal_asof = staged[staged["snapshot_date"] == asof_dt][["account_id", "balance"]].copy()
        base = base.merge(bal_asof, on="account_id", how="left")

    base = base.merge(staging_asof, on="account_id", how="left").copy()
    base["stage"] = base["stage"].fillna(1).astype(int)
    base["balance"] = base.get("balance", pd.Series(0.0, index=base.index)).fillna(0.0)

    # PD inputs
    scores_path = Path(f"data/curated/pd_scores_asof_{asof_dt.date().isoformat()}.parquet")
    anchor_path = Path(f"data/curated/pd_anchor_shift_asof_{asof_dt.date().isoformat()}.parquet")

    if scores_path.exists():
        sc = pd.read_parquet(scores_path)[["account_id", "pd_12m_hat"]]
        base = base.merge(sc, on="account_id", how="left")
    else:
        base["pd_12m_hat"] = np.nan

    if anchor_path.exists():
        sh = pd.read_parquet(anchor_path)[["account_id", "pd_anchor_shift"]]
        base = base.merge(sh, on="account_id", how="left")
    else:
        base["pd_anchor_shift"] = 0.0

    # defaults for missing PDs
    base["pd_12m_hat"] = base["pd_12m_hat"].fillna(0.02).clip(1e-5, 0.5)
    base["pd_anchor_shift"] = base["pd_anchor_shift"].fillna(0.0)

    # Anchor-adjusted baseline PD (logit shift)
    pd_anchor = sigmoid(logit(base["pd_12m_hat"].to_numpy(dtype=np.float64)) + base["pd_anchor_shift"].to_numpy(dtype=np.float64))
    base["pd_12m_anchor"] = pd_anchor

    # Macro setup
    scen_list = ["Base", "Upside", "Downside"]
    future_dates_36 = month_ends_forward(asof_dt, periods=36)
    future_dates_12 = future_dates_36[:12]

    stress_weights = params.get("stress_weights", {})
    macro_scale = float(params.get("macro_scale", 0.35))
    pd_beta = float(params.get("pd_macro_beta", 0.8))
    h_pd = int(params.get("stress_horizon_months_pd", 12))
    h_lgd = int(params.get("stress_horizon_months_lgd", 12))

    stress_pd = {s: macro_scale * compute_stress_scalar(mz, s, future_dates_12, stress_weights, h_pd) for s in scen_list}
    stress_lgd = {s: macro_scale * compute_stress_scalar(mz, s, future_dates_12, stress_weights, h_lgd) for s in scen_list}
    stress_s3 = {s: macro_scale * compute_stress_scalar(mz, s, future_dates_36, stress_weights, 36) for s in scen_list}

    # Save scenario severity (used in your diagnostics)
    sev = []
    for s in scen_list:
        cols = ["unemployment_z", "gdp_yoy_z", "policy_rate_z"]
        z = _macro_matrix(mz, s, future_dates_12, cols)
        sev.append(
            {
                "asof_date": asof_dt.date().isoformat(),
                "scenario": s,
                "macro_scale": macro_scale,
                "unemp_z_mean": float(z[:, 0].mean()),
                "unemp_z_min": float(z[:, 0].min()),
                "unemp_z_max": float(z[:, 0].max()),
                "gdp_z_mean": float(z[:, 1].mean()),
                "gdp_z_min": float(z[:, 1].min()),
                "gdp_z_max": float(z[:, 1].max()),
                "rate_z_mean": float(z[:, 2].mean()),
                "rate_z_min": float(z[:, 2].min()),
                "rate_z_max": float(z[:, 2].max()),
                "stress_pd": float(stress_pd[s]),
                "stress_lgd": float(stress_lgd[s]),
                "stress_stage3": float(stress_s3[s]),
            }
        )
    Path("data/curated").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(sev).to_csv(f"data/curated/scenario_severity_asof_{asof_dt.date().isoformat()}.csv", index=False)

    # --- Build scenario PDs ---
    # logit(PD_scen) = logit(PD_anchor) + pd_beta * stress_pd[scenario]
    pd_anchor_logit = logit(pd_anchor)
    pd_12m = {}
    for s in scen_list:
        pd_12m[s] = sigmoid(pd_anchor_logit + pd_beta * float(stress_pd[s]))

    # Convert 12m PD to constant monthly hazard (simple but stable)
    pd_m1 = {s: 1.0 - np.power(1.0 - pd_12m[s], 1.0 / 12.0) for s in scen_list}
    pd_cum12 = {s: 1.0 - np.power(1.0 - pd_m1[s], 12.0) for s in scen_list}  # ~ pd_12m[s]

    # Lifetime PD via maturity months
    seg = base["segment"].astype(str).to_numpy()
    bal = base["balance"].astype(np.float64).to_numpy()
    lim = base.get("limit_amount", pd.Series(0.0, index=base.index)).fillna(0.0).astype(np.float64).to_numpy()
    ead = compute_ead(bal, lim, seg, params)
    mat_m = months_to_maturity(base, asof_dt, params)
    pd_cumlt = {s: 1.0 - np.power(1.0 - pd_m1[s], mat_m) for s in scen_list}
    for s in scen_list:
        pd_cumlt[s] = np.clip(pd_cumlt[s], 0.0, 1.0)

    # --- Scenario LGD for Stage 1/2 ---
    lgd_base_map = params.get("lgd_base", {})
    lgd_beta_map = params.get("lgd_scenario_beta", {})
    lgd_floor = float(params.get("lgd_mult_floor", 0.8))
    lgd_cap = float(params.get("lgd_mult_cap", 1.6))

    lgd_base = np.array([float(lgd_base_map.get(x, 0.45)) for x in seg], dtype=np.float64)
    beta = np.array([float(lgd_beta_map.get(x, 0.20)) for x in seg], dtype=np.float64)

    lgd_scen = {}
    for s in scen_list:
        mult = np.exp(beta * float(stress_lgd[s]))
        mult = np.clip(mult, lgd_floor, lgd_cap)
        lgd_scen[s] = np.clip(lgd_base * mult, 0.01, 0.99)

    # --- Scenario ECL for Stage 1/2 (pre Stage3 override) ---
    ecl12 = {s: ead * lgd_scen[s] * pd_cum12[s] for s in scen_list}
    ecllt = {s: ead * lgd_scen[s] * pd_cumlt[s] for s in scen_list}

    # --- Stage 3 override via scenario-linked workout ---
    if scenario_weights is None:
        scenario_weights = params.get("scenario_weights", {"Base": 0.6, "Upside": 0.2, "Downside": 0.2})
    weight_map = scenario_weights.get("weights", scenario_weights)
    stage3_tbl = stage3_workout_table_scenarios(
        asof_dt=asof_dt,
        accounts=accounts,
        perf_asof=base[["account_id", "balance"]].copy(),
        staging_asof=staging_asof,
        portfolio_params=params,
        stress_by_scenario=stress_s3,
        scenario_weights=weight_map,
        workout_cfg_path="configs/workout_lgd.yml",
    )

    out = base.copy()
    out["ead"] = ead
    # IMPORTANT: asof_date must exist for ALL rows (not just Stage 3)
    out["asof_date"] = asof_dt

    # add scenario PD columns + scenario LGD columns (for explainability)
    for s in scen_list:
        out[f"pd_12m_{s.lower()}"] = pd_cum12[s]
        out[f"pd_lt_{s.lower()}"] = pd_cumlt[s]
        out[f"lgd_{s.lower()}"] = lgd_scen[s]
        out[f"ecl12_{s.lower()}"] = ecl12[s]
        out[f"ecllt_{s.lower()}"] = ecllt[s]

    # merge stage 3 workout scenario outputs
    if not stage3_tbl.empty:
        # avoid partially populated asof_date from stage3_tbl
        stage3_tbl = stage3_tbl.drop(columns=["asof_date"], errors="ignore")
        out = out.merge(stage3_tbl, on="account_id", how="left")
        stage3 = out["stage"] == 3

        # override stage3 scenario ECLs with workout ECLs
        out.loc[stage3, "ecl12_base"] = out.loc[stage3, "ecl_stage3_base"]
        out.loc[stage3, "ecl12_upside"] = out.loc[stage3, "ecl_stage3_upside"]
        out.loc[stage3, "ecl12_downside"] = out.loc[stage3, "ecl_stage3_downside"]

        out.loc[stage3, "ecllt_base"] = out.loc[stage3, "ecl_stage3_base"]
        out.loc[stage3, "ecllt_upside"] = out.loc[stage3, "ecl_stage3_upside"]
        out.loc[stage3, "ecllt_downside"] = out.loc[stage3, "ecl_stage3_downside"]

        # also override LGD columns for stage3 to workout lgd by scenario
        out.loc[stage3, "lgd_base"] = out.loc[stage3, "workout_lgd_base"]
        out.loc[stage3, "lgd_upside"] = out.loc[stage3, "workout_lgd_upside"]
        out.loc[stage3, "lgd_downside"] = out.loc[stage3, "workout_lgd_downside"]

    # IMPORTANT: overwrite again to guarantee no NaT survives any merge logic
    out["asof_date"] = asof_dt

    # Ensure the scenario columns exist with your naming convention
    # (your repo previously used ecl12_base, etc.)
    if "ecl12_base" not in out.columns:
        out["ecl12_base"] = out["ecl12_base"]  # no-op

    # scenario weights
    w_base = float(weight_map.get("Base", 0.6))
    w_up = float(weight_map.get("Upside", 0.2))
    w_dn = float(weight_map.get("Downside", 0.2))
    w_sum = w_base + w_up + w_dn
    w_base, w_up, w_dn = w_base / w_sum, w_up / w_sum, w_dn / w_sum

    # weighted horizon ECL
    out["ecl12_weighted"] = w_base * out["ecl12_base"] + w_up * out["ecl12_upside"] + w_dn * out["ecl12_downside"]
    out["ecllt_weighted"] = w_base * out["ecllt_base"] + w_up * out["ecllt_upside"] + w_dn * out["ecllt_downside"]

    # scenario-selected ECL (apply horizon rule per stage)
    stage = out["stage"].astype(int)
    out["ecl_selected_base"] = np.where(stage == 1, out["ecl12_base"], out["ecllt_base"])
    out["ecl_selected_upside"] = np.where(stage == 1, out["ecl12_upside"], out["ecllt_upside"])
    out["ecl_selected_downside"] = np.where(stage == 1, out["ecl12_downside"], out["ecllt_downside"])
    out["ecl_selected_weighted"] = w_base * out["ecl_selected_base"] + w_up * out["ecl_selected_upside"] + w_dn * out["ecl_selected_downside"]

    # final (pre-overlay) selected
    out["ecl_selected"] = out["ecl_selected_weighted"].astype("float64")

    # PD summary output (your diagnostics file)
    pd_sum = []
    for s in scen_list:
        col = f"pd_12m_{s.lower()}"
        pd_sum.append(
            {
                "asof_date": asof_dt.date().isoformat(),
                "scenario": s,
                "n_accounts": int(len(out)),
                "pd_m1_mean": float(pd_m1[s].mean()),
                "pd_cum12_mean": float(out[col].mean()),
                "pd_cum12_p50": float(np.quantile(out[col], 0.50)),
                "pd_cum12_p90": float(np.quantile(out[col], 0.90)),
                "pd_cum12_p99": float(np.quantile(out[col], 0.99)),
                "pd_cum12_max": float(out[col].max()),
            }
        )
    pd.DataFrame(pd_sum).to_csv(f"data/curated/scenario_pd_summary_asof_{asof_dt.date().isoformat()}.csv", index=False)

    return out


def compute_ecl_asof(asof_dt: pd.Timestamp, params: dict) -> pd.DataFrame:
    # --- Load curated inputs ---
    asof_dt = as_month_end(asof_dt)
    accounts = pd.read_parquet("data/curated/accounts.parquet")
    perf = pd.read_parquet("data/curated/performance_monthly.parquet")
    staging = pd.read_parquet("data/curated/staging_output.parquet")
    macro = pd.read_parquet("data/curated/macro_scenarios_monthly.parquet")

    perf["snapshot_date"] = perf["snapshot_date"].apply(as_month_end)
    perf_asof = perf[perf["snapshot_date"] == asof_dt][["account_id", "balance"]].copy()

    accounts = accounts.merge(perf_asof, on="account_id", how="left")

    return compute_ecl_from_frames(
        staged=staging,
        accounts=accounts,
        macro=macro,
        params=params,
        asof_dt=asof_dt,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", default=None, help="ASOF date (YYYY-MM-DD). Default = max snapshot in staging_output.")
    args = ap.parse_args()

    params = load_yml("configs/portfolio_params.yml")

    staging = pd.read_parquet("data/curated/staging_output.parquet")
    staging["snapshot_date"] = staging["snapshot_date"].apply(as_month_end)

    if args.asof is None:
        asof_dt = staging["snapshot_date"].max()
    else:
        asof_dt = as_month_end(args.asof)

    out = compute_ecl_asof(asof_dt=asof_dt, params=params)

    out_path = Path(f"data/curated/ecl_output_asof_{asof_dt.date().isoformat()}.parquet")
    out.to_parquet(out_path, index=False)
    print(f"Wrote: {out_path}")

    # Console summary
    print("\nECL selected totals by stage:")
    print(out.groupby("stage")["ecl_selected"].sum())

    print("\nECL selected totals by segment (top 10):")
    print(out.groupby("segment")["ecl_selected"].sum().sort_values(ascending=False).head(10))


if __name__ == "__main__":
    main()
