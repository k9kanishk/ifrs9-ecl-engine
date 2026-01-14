from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def load_yml(path: str | Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def month_end_index(start_me: pd.Timestamp, periods: int) -> pd.DatetimeIndex:
    # future month-ends starting next month-end
    return pd.date_range(start=start_me + pd.offsets.MonthEnd(1), periods=periods, freq="ME")


def macro_block_ffill(mz: pd.DataFrame, scen: str, dates: pd.DatetimeIndex) -> np.ndarray:
    """
    mz: MultiIndex (scenario, date) -> columns unemployment_z, gdp_yoy_z, policy_rate_z
    dates: desired future dates (month-ends)
    Returns (H,3) macro z block, forward-filled beyond last known macro date.
    """
    dates = pd.to_datetime(dates)

    # Extract one scenario with date index
    df = mz.xs(scen, level=0).sort_index()

    # Extend index to include required dates, then forward fill
    idx = df.index.union(dates)
    df2 = df.reindex(idx).sort_index().ffill()

    # If dates are earlier than the first macro date (unlikely), backfill them
    df2 = df2.bfill()

    return df2.loc[dates, ["unemployment_z", "gdp_yoy_z", "policy_rate_z"]].to_numpy(dtype=np.float32)


def prepare_macro_z_all_scenarios(macro: pd.DataFrame) -> pd.DataFrame:
    """
    Compute z-scores using Base scenario mean/std, apply to all scenarios.
    """
    macro = macro.copy()
    macro["date"] = pd.to_datetime(macro["date"])

    base = macro.loc[macro["scenario"] == "Base", ["unemployment", "gdp_yoy", "policy_rate"]]
    stats = {}
    for col in ["unemployment", "gdp_yoy", "policy_rate"]:
        mu = base[col].mean()
        sd = base[col].std(ddof=0)
        stats[col] = (mu, sd if sd > 0 else 1.0)

    for col in ["unemployment", "gdp_yoy", "policy_rate"]:
        mu, sd = stats[col]
        macro[f"{col}_z"] = (macro[col] - mu) / sd

    return macro[["date", "scenario", "unemployment_z", "gdp_yoy_z", "policy_rate_z"]]


def months_to_maturity(asof: pd.Timestamp, maturity: pd.Series) -> np.ndarray:
    """
    Integer number of months between asof month-end and maturity month-end (>=1).
    """
    asof_me = (asof + pd.offsets.MonthEnd(0)).normalize()
    mat_me = (pd.to_datetime(maturity) + pd.offsets.MonthEnd(0)).dt.normalize()
    m = (mat_me.dt.year - asof_me.year) * 12 + (mat_me.dt.month - asof_me.month)
    m = m.astype(int).to_numpy()
    return np.maximum(m, 1)


def compute_ecl_asof(
    staged: pd.DataFrame,
    accounts: pd.DataFrame,
    macro: pd.DataFrame,
    policy: dict,
    scenario_weights: dict,
    portfolio_params: dict,
    asof: pd.Timestamp,
    max_horizon_months: int = 360,
) -> pd.DataFrame:
    """
    Returns per-account ECL at asof date:
      - scenario ECLs (12m, lifetime)
      - scenario-weighted ECL
      - stage-selected ECL (Stage1=12m, Stage2/3=lifetime)
    """
    asof = pd.to_datetime(asof)
    df = staged.loc[staged["snapshot_date"] == asof].copy()
    if df.empty:
        raise ValueError(f"No rows found for asof={asof.date()} in staging_output.")

    # Join account fields we need
    acc_cols = ["account_id", "segment", "eir", "maturity_date", "limit_amount", "ttc_pd_annual"]
    df = df.merge(accounts[acc_cols], on=["account_id", "segment"], how="left", validate="many_to_one")

    n = len(df)
    seg = df["segment"].astype(str).to_numpy()

    # Horizon per account (months)
    h_m = months_to_maturity(asof, df["maturity_date"])
    H = int(min(max_horizon_months, int(np.max(h_m))))
    t = np.arange(1, H + 1, dtype=np.float32)[None, :]  # (1,H)
    mask = (t <= h_m[:, None]).astype(np.float32)  # (n,H)

    # Discount factors
    eir = df["eir"].astype(float).to_numpy()[:, None]  # (n,1)
    df_disc = np.power(1.0 + eir, -(t / 12.0), dtype=np.float32)  # (n,H)

    # EAD projection (simple)
    bal0 = df["balance"].astype(float).to_numpy()[:, None]  # (n,1)

    amort_map = portfolio_params["amort_rate_monthly"]
    amort = np.vectorize(lambda s: float(amort_map.get(s, 0.01)))(seg).astype(np.float32)[:, None]
    bal_path = bal0 * np.power((1.0 - amort), t, dtype=np.float32)  # (n,H)

    # Revolving EAD uses CCF on undrawn
    limit = df["limit_amount"].fillna(0.0).astype(float).to_numpy()[:, None]
    is_rev = (seg == "Revolving")[:, None]

    ccf = float(portfolio_params.get("ccf_base", {}).get("Revolving", 0.75))
    ead_rev = bal0 + ccf * np.maximum(limit - bal0, 0.0)

    ead = np.where(is_rev, ead_rev, bal_path).astype(np.float32)  # (n,H)
    ead *= mask

    # LGD by segment + scenario multipliers
    lgd_base_map = portfolio_params["lgd_base"]
    lgd_base = np.vectorize(lambda s: float(lgd_base_map.get(s, 0.5)))(seg).astype(np.float32)[:, None]
    lgd_base = np.clip(lgd_base, 0.0, 1.0)

    # PD PIT term structure driven by macro scenarios using same policy betas
    betas = policy["default"]["pd_logit"]
    pd_floor = float(policy["default"]["pd_floor"])
    pd_cap = float(policy["default"]["pd_cap"])

    logit_ttc = _logit(np.clip(df["ttc_pd_annual"].astype(float).to_numpy(), 1e-12, 1 - 1e-12)).astype(
        "float64"
    )[:, None]  # (n,1)

    # Phase 3: account-level anchor shift to match fitted 12M PD under Base scenario
    if "pd_anchor_shift" in df.columns:
        logit_ttc = logit_ttc + df["pd_anchor_shift"].fillna(0.0).to_numpy(dtype=np.float64)[:, None]

    # Macro z table for all scenarios
    mz = prepare_macro_z_all_scenarios(macro).copy()
    mz = mz.set_index(["scenario", "date"]).sort_index()

    future_dates = month_end_index((asof + pd.offsets.MonthEnd(0)).normalize(), H)

    results = {
        "account_id": df["account_id"].values,
        "segment": df["segment"].values,
        "stage": df["stage"].astype(int).values,
        "balance": df["balance"].values,
    }

    for scen in ["Base", "Upside", "Downside"]:
        m = macro_block_ffill(mz, scen, future_dates)  # (H,3)

        x = (
            float(betas["intercept"])
            + float(betas["unemployment_z"]) * m[:, 0]
            + float(betas["gdp_yoy_z"]) * m[:, 1]
            + float(betas["policy_rate_z"]) * m[:, 2]
        ).astype(np.float32)[None, :]  # (1,H)

        logit_pit = logit_ttc + x
        pd_annual = np.clip(_sigmoid(logit_pit), pd_floor, pd_cap).astype(np.float32)  # (n,H)

        # monthly hazard from annual PD
        h = (1.0 - np.power((1.0 - pd_annual), (1.0 / 12.0))).astype(np.float32)  # (n,H)
        h *= mask

        # survival and marginal PD
        surv = np.cumprod(1.0 - h, axis=1, dtype=np.float32)
        surv_prev = np.concatenate([np.ones((n, 1), dtype=np.float32), surv[:, :-1]], axis=1)
        mpd = (surv_prev * h).astype(np.float32)  # (n,H)

        # scenario LGD
        mult = float(portfolio_params["lgd_scenario_multiplier"].get(scen, 1.0))
        lgd = np.clip(lgd_base * mult, 0.0, 1.0).astype(np.float32)

        el = df_disc * ead * lgd * mpd  # (n,H)

        ecl_12m = el[:, : min(12, H)].sum(axis=1)
        ecl_lt = el.sum(axis=1)

        results[f"ecl12_{scen.lower()}"] = ecl_12m
        results[f"ecllt_{scen.lower()}"] = ecl_lt

    # Scenario weights
    w = scenario_weights["weights"]
    w_base = float(w["Base"])
    w_up = float(w["Upside"])
    w_dn = float(w["Downside"])

    results["ecl12_weighted"] = (
        w_base * results["ecl12_base"] + w_up * results["ecl12_upside"] + w_dn * results["ecl12_downside"]
    )
    results["ecllt_weighted"] = (
        w_base * results["ecllt_base"] + w_up * results["ecllt_upside"] + w_dn * results["ecllt_downside"]
    )

    out = pd.DataFrame(results)
    # Ensure float64 for stable downstream assignments / reports
    for c in out.columns:
        if c.startswith("ecl"):
            out[c] = out[c].astype("float64")

    # Stage-selected ECL
    # Stage1 -> 12m, Stage2/3 -> lifetime
    out["ecl_selected"] = np.where(
        out["stage"].to_numpy() == 1,
        out["ecl12_weighted"].to_numpy(),
        out["ecllt_weighted"].to_numpy(),
    ).astype("float64")

    # Simple Stage 3 override (optional): defaulted exposure -> immediate loss proxy
    # Keeps things sensible for a skeleton engine.
    stage3 = out["stage"].to_numpy() == 3
    out.loc[stage3, "ecl_selected"] = (out.loc[stage3, "balance"].astype("float64") * 0.9).to_numpy()  # conservative proxy

    out["asof_date"] = asof
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", type=str, default=None, help="YYYY-MM-DD month-end; default uses latest in staging_output")
    ap.add_argument("--max_horizon_months", type=int, default=360)
    ap.add_argument("--staging_path", type=str, default="data/curated/staging_output.parquet")
    ap.add_argument("--accounts_path", type=str, default="data/curated/accounts.parquet")
    ap.add_argument("--macro_path", type=str, default="data/curated/macro_scenarios_monthly.parquet")
    ap.add_argument("--policy_path", type=str, default="configs/policy.yml")
    ap.add_argument("--weights_path", type=str, default="configs/scenario_weights.yml")
    ap.add_argument("--params_path", type=str, default="configs/portfolio_params.yml")
    ap.add_argument("--outdir", type=str, default="data/curated")
    args = ap.parse_args()

    staged = pd.read_parquet(args.staging_path)
    accounts = pd.read_parquet(args.accounts_path)
    macro = pd.read_parquet(args.macro_path)

    policy = load_yml(args.policy_path)
    weights = load_yml(args.weights_path)
    params = load_yml(args.params_path)

    staged["snapshot_date"] = pd.to_datetime(staged["snapshot_date"])
    asof = pd.to_datetime(args.asof) if args.asof else staged["snapshot_date"].max()
    asof_dt = asof

    # --- Phase 2/3: merge fitted PD + optional anchor shift ---
    pd_scores_path = Path(f"data/curated/pd_scores_asof_{asof_dt.date().isoformat()}.parquet")
    if pd_scores_path.exists():
        pd_scores = pd.read_parquet(pd_scores_path)
        accounts = accounts.merge(pd_scores[["account_id", "pd_12m_hat"]], on="account_id", how="left")
        print(f"Loaded fitted PDs from: {pd_scores_path}")

    shift_path = Path(f"data/curated/pd_anchor_shift_asof_{asof_dt.date().isoformat()}.parquet")
    if shift_path.exists():
        sh = pd.read_parquet(shift_path)
        accounts = accounts.merge(sh[["account_id", "pd_anchor_shift"]], on="account_id", how="left")
        print(f"Loaded PD anchor shifts from: {shift_path}")
    else:
        accounts["pd_anchor_shift"] = 0.0

    out = compute_ecl_asof(
        staged=staged,
        accounts=accounts,
        macro=macro,
        policy=policy,
        scenario_weights=weights,
        portfolio_params=params,
        asof=asof,
        max_horizon_months=args.max_horizon_months,
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / f"ecl_output_asof_{asof.date().isoformat()}.parquet"
    out.to_parquet(out_path, index=False)

    # QC prints
    print(f"Wrote: {out_path}")
    print("\nECL selected totals by stage:")
    print(out.groupby("stage")["ecl_selected"].sum().round(2))
    print("\nECL selected totals by segment (top 10):")
    print(out.groupby("segment")["ecl_selected"].sum().sort_values(ascending=False).head(10).round(2))


if __name__ == "__main__":
    main()
