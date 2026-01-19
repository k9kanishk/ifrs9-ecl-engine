from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np


def main(asof: str = "2024-12-31") -> None:
    asof_dt = (pd.to_datetime(asof) + pd.offsets.MonthEnd(0)).date().isoformat()
    in_path = Path(f"data/curated/ecl_output_asof_{asof_dt}.parquet")
    if not in_path.exists():
        raise FileNotFoundError(f"Missing {in_path}. Run: python src/ecl_engine/ecl.py")

    df = pd.read_parquet(in_path)
    s3 = df[df["stage"] == 3].copy()
    if s3.empty:
        print("No Stage 3 rows.")
        return

    # Weighted summary (backward compatible)
    g = s3.groupby("segment").agg(
        ead_default_sum=("ead_default", "sum"),
        pv_recoveries_sum=("pv_recoveries", "sum"),
        ecl_sum=("ecl_selected", "sum"),
        n=("account_id", "count"),
    )
    g["implied_lgd"] = 1.0 - g["pv_recoveries_sum"] / g["ead_default_sum"]
    g = g.reset_index().sort_values("ecl_sum", ascending=False)
    out1 = Path(f"data/curated/stage3_workout_summary_asof_{asof_dt}.csv")
    g.to_csv(out1, index=False)
    print(f"Wrote: {out1}")

    # Scenario summary (Base/Upside/Downside + weighted)
    cols_needed = [
        "pv_recoveries_base", "pv_recoveries_upside", "pv_recoveries_downside",
        "ecl_stage3_base", "ecl_stage3_upside", "ecl_stage3_downside",
        "ecl_stage3_workout",
        "ead_default",
    ]
    missing = [c for c in cols_needed if c not in s3.columns]
    if missing:
        print("Scenario Stage3 columns missing, skipping scenario summary:", missing)
        return

    s = s3.groupby("segment").agg(
        ead_default_sum=("ead_default", "sum"),

        pv_base=("pv_recoveries_base", "sum"),
        pv_up=("pv_recoveries_upside", "sum"),
        pv_dn=("pv_recoveries_downside", "sum"),
        pv_w=("pv_recoveries", "sum"),

        ecl_base=("ecl_stage3_base", "sum"),
        ecl_up=("ecl_stage3_upside", "sum"),
        ecl_dn=("ecl_stage3_downside", "sum"),
        ecl_w=("ecl_stage3_workout", "sum"),

        n=("account_id", "count"),
    ).reset_index()

    s["lgd_base"] = 1.0 - s["pv_base"] / s["ead_default_sum"]
    s["lgd_up"] = 1.0 - s["pv_up"] / s["ead_default_sum"]
    s["lgd_dn"] = 1.0 - s["pv_dn"] / s["ead_default_sum"]
    s["lgd_w"] = 1.0 - s["pv_w"] / s["ead_default_sum"]

    s = s.sort_values("ecl_w", ascending=False)

    out2 = Path(f"data/curated/stage3_workout_summary_scenarios_asof_{asof_dt}.csv")
    s.to_csv(out2, index=False)
    print(f"Wrote: {out2}")


if __name__ == "__main__":
    main()
