from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def _month_end(x) -> pd.Timestamp:
    return (pd.to_datetime(x) + pd.offsets.MonthEnd(0)).normalize()


def main(asof: str = "2024-12-31", n: int = 0) -> None:
    asof_dt = _month_end(asof).date().isoformat()
    asof_ts = pd.to_datetime(asof_dt)

    p_post = Path("data/curated/ecl_with_overlays.parquet")
    p_pre = Path(f"data/curated/ecl_output_asof_{asof_dt}.parquet")

    if p_post.exists():
        df = pd.read_parquet(p_post)
        print(f"Read: {p_post}")
    elif p_pre.exists():
        df = pd.read_parquet(p_pre)
        print(f"Read: {p_pre}")
    else:
        raise FileNotFoundError("Run ECL first: python src/ecl_engine/ecl.py and overlays if desired")

    # Ensure consistent columns
    df = df.copy()
    if "asof_date" not in df.columns:
        df["asof_date"] = asof_ts
    else:
        df["asof_date"] = pd.to_datetime(df["asof_date"], errors="coerce")
        if df["asof_date"].isna().any():
            df["asof_date"] = df["asof_date"].fillna(asof_ts)
    df = df[df["asof_date"] == asof_ts].copy()

    # Prefer post-overlay column if available
    if "ecl_post_overlay" in df.columns:
        df["ecl_final"] = df["ecl_post_overlay"]
    else:
        df["ecl_final"] = df["ecl_selected"]

    # Build a compact explain table for dashboard drilldown
    keep = ["account_id", "segment", "stage", "balance", "ecl_final"]
    if "ead" in df.columns:
        keep.append("ead")
    if "ead_default" in df.columns:
        keep.append("ead_default")

    # Scenario columns (Phase 5)
    for c in [
        "ecl_selected_base", "ecl_selected_upside", "ecl_selected_downside", "ecl_selected_weighted",
        "pd_12m_base", "pd_12m_upside", "pd_12m_downside",
        "pd_lt_base", "pd_lt_upside", "pd_lt_downside",
        "lgd_base", "lgd_upside", "lgd_downside",
        "pv_recoveries_base", "pv_recoveries_upside", "pv_recoveries_downside",
        "ecl_stage3_base", "ecl_stage3_upside", "ecl_stage3_downside", "ecl_stage3_workout",
        "workout_lgd_base", "workout_lgd_upside", "workout_lgd_downside",
        "workout_lgd", "pv_recoveries",
    ]:
        if c in df.columns:
            keep.append(c)

    keep = [c for c in keep if c in df.columns]
    out = df[keep].copy()

    # Add handy derived columns
    if "ead" in out.columns:
        out["ecl_over_ead"] = np.where(out["ead"] > 0, out["ecl_final"] / out["ead"], np.nan)
    elif "ead_default" in out.columns:
        out["ecl_over_ead"] = np.where(out["ead_default"] > 0, out["ecl_final"] / out["ead_default"], np.nan)
    else:
        out["ecl_over_ead"] = np.nan

    # Identify biggest accounts by ECL for drilldown
    out = out.sort_values("ecl_final", ascending=False).reset_index(drop=True)

    # cap size for dashboard performance
    if int(n) > 0:
        out = out.head(int(n)).reset_index(drop=True)

    out_path = Path(f"data/curated/account_explain_asof_{asof_dt}.parquet")
    out.to_parquet(out_path, index=False)
    print(f"Wrote: {out_path} ({len(out):,} rows)")


if __name__ == "__main__":
    # Optional CLI usage:
    # python -m ecl_engine.explain
    # python -m ecl_engine.explain --asof 2024-12-31 --n 12000
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", default="2024-12-31")
    ap.add_argument("--n", type=int, default=0)
    args = ap.parse_args()
    main(asof=args.asof, n=args.n)
