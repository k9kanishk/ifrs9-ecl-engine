from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd


def _latest_ecl_with_overlays() -> Path:
    p = Path("data/curated/ecl_with_overlays.parquet")
    if not p.exists():
        raise FileNotFoundError("Missing data/curated/ecl_with_overlays.parquet. Run: python -m ecl_engine.run_ecl_with_overlays")
    return p


def _pick_asof(df: pd.DataFrame) -> pd.Timestamp:
    if "asof_date" in df.columns:
        x = pd.to_datetime(df["asof_date"], errors="coerce")
        if x.notna().any():
            return pd.Timestamp(x.max()).normalize()
    # fallback: try from filename patterns
    ecl_paths = sorted(glob.glob("data/curated/ecl_output_asof_*.parquet"))
    if ecl_paths:
        asof = Path(ecl_paths[-1]).stem.replace("ecl_output_asof_", "")
        return pd.Timestamp(asof).normalize()
    raise ValueError("Could not determine ASOF date.")


def _as_month_end(x) -> pd.Timestamp:
    return (pd.to_datetime(x) + pd.offsets.MonthEnd(0)).normalize()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", default=None, help="ASOF date YYYY-MM-DD. Default = inferred from ecl_with_overlays.")
    ap.add_argument("--n", type=int, default=0, help="If >0, write only top N accounts by reported ECL.")
    ap.add_argument("--stage", type=int, default=None, help="Optional stage filter (1/2/3).")
    args = ap.parse_args()

    ecl_ov = pd.read_parquet(_latest_ecl_with_overlays())
    ecl_ov["asof_date"] = pd.to_datetime(ecl_ov.get("asof_date"), errors="coerce")

    inferred_asof = _pick_asof(ecl_ov)
    asof_dt = _as_month_end(args.asof) if args.asof else _as_month_end(inferred_asof)

    # filter to asof
    if "asof_date" in ecl_ov.columns and ecl_ov["asof_date"].notna().any():
        ecl_ov = ecl_ov[ecl_ov["asof_date"].apply(_as_month_end) == asof_dt].copy()

    # staging (for dpd + stage_reason + SICR metadata)
    staging = pd.read_parquet("data/curated/staging_output.parquet").copy()
    staging["snapshot_date"] = pd.to_datetime(staging["snapshot_date"], errors="coerce").apply(_as_month_end)
    stg = staging[staging["snapshot_date"] == asof_dt].copy()

    # accounts (for maturity/eir/limits)
    accounts = pd.read_parquet("data/curated/accounts.parquet").copy()

    # merge
    out = ecl_ov.merge(
        stg[["account_id", "dpd", "stage_reason"]].drop_duplicates("account_id"),
        on="account_id",
        how="left",
    ).merge(
        accounts[["account_id", "maturity_date", "eir", "limit_amount", "ttc_pd_annual"]].copy(),
        on="account_id",
        how="left",
    )

    # derived fields
    out["maturity_date"] = pd.to_datetime(out.get("maturity_date"), errors="coerce")
    out["months_to_maturity"] = (
        (out["maturity_date"].dt.to_period("M") - asof_dt.to_period("M"))
        .apply(lambda x: x.n if pd.notna(x) else np.nan)
        .astype("float64")
    )

    out["utilization"] = np.where(
        (out.get("limit_amount").fillna(0.0) > 0),
        out.get("balance").fillna(0.0) / out.get("limit_amount").replace(0.0, np.nan),
        np.nan,
    )

    out["ead_rule"] = np.where(out.get("segment").astype(str) == "Revolving", "Balance + CCF * Undrawn", "Balance")
    if "ead" in out.columns:
        out["ead0"] = out["ead"]
    else:
        out["ead0"] = out.get("balance", 0.0)

    out["ead12_avg"] = out["ead0"]

    # optional stage filter
    if args.stage is not None:
        out = out[out["stage"].astype(int) == int(args.stage)].copy()

    # ranking (reported ECL)
    rank_col = "ecl_post_overlay" if "ecl_post_overlay" in out.columns else "ecl_selected"
    out = out.sort_values(rank_col, ascending=False)

    if args.n and args.n > 0:
        out = out.head(args.n).copy()

    # keep a clean, governance-friendly column order
    preferred = [
        "account_id",
        "asof_date",
        "segment",
        "stage",
        "stage_reason",
        "dpd",
        "balance",
        "limit_amount",
        "utilization",
        "maturity_date",
        "months_to_maturity",
        "eir",
        "ead_rule",
        "ead0",
        "ead12_avg",
        "ead_default",
        "pv_recoveries",
        "workout_lgd",
        "ecl_stage3_workout",
        "pd_12m_hat",
        "pd_12m_anchor",
        "pd_12m_base",
        "pd_12m_upside",
        "pd_12m_downside",
        "lgd_base",
        "lgd_upside",
        "lgd_downside",
        "ecl_pre_overlay",
        "overlay_amount",
        "ecl_post_overlay",
        "overlay_audit",
    ]
    cols = [c for c in preferred if c in out.columns] + [c for c in out.columns if c not in preferred]
    out = out[cols]

    Path("data/curated").mkdir(parents=True, exist_ok=True)
    out_path = Path(f"data/curated/account_explain_asof_{asof_dt.date().isoformat()}.parquet")
    out.to_parquet(out_path, index=False)
    print(f"Read: data/curated/ecl_with_overlays.parquet")
    print(f"Wrote: {out_path} ({out.shape[0]:,} rows)")


if __name__ == "__main__":
    main()
