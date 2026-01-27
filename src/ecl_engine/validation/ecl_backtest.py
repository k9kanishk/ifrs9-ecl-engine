from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", default=None, help="ASOF date (YYYY-MM-DD). Default: infer from latest ecl output.")
    ap.add_argument("--h", type=int, default=12, help="Forward horizon in months for realized default proxy.")
    args = ap.parse_args()

    if args.asof is None:
        paths = sorted(Path("data/curated").glob("ecl_output_asof_*.parquet"))
        if not paths:
            raise FileNotFoundError("No ecl_output_asof_*.parquet found. Run: python -m ecl_engine.ecl")
        p = paths[-1]
        asof = p.stem.replace("ecl_output_asof_", "")
    else:
        asof = args.asof
        p = Path(f"data/curated/ecl_output_asof_{asof}.parquet")

    out = pd.read_parquet(p)

    # crude "realized" proxy using performance + default_flag/dpd if available (toy governance check)
    perf = pd.read_parquet("data/curated/performance_monthly.parquet")
    perf["snapshot_date"] = pd.to_datetime(perf["snapshot_date"])
    asof_dt = pd.Timestamp(asof)

    # next 12 months window
    end_dt = (asof_dt + pd.offsets.MonthEnd(args.h)).normalize()

    fwd = perf[(perf["snapshot_date"] > asof_dt) & (perf["snapshot_date"] <= end_dt)].copy()
    if "default_flag" in fwd.columns:
        realized = (
            fwd.groupby("account_id")["default_flag"]
            .max()
            .reset_index()
            .rename(columns={"default_flag": "realized_default_h"})
        )
    else:
        # fallback: dpd >= 90 means default
        realized = (
            fwd.groupby("account_id")["dpd"]
            .max()
            .reset_index()
            .assign(realized_default_h=lambda x: (x["dpd"].astype(float) >= 90).astype(int))
            .drop(columns=["dpd"])
        )

    bt = out[["account_id", "segment", "stage", "ecl_selected"]].merge(realized, on="account_id", how="left")
    bt["realized_default_h"] = bt["realized_default_h"].fillna(0).astype(int)

    # simple calibration: average ECL per account vs realized default rate by segment/stage
    seg = (
        bt.groupby(["segment", "stage"], as_index=False)
        .agg(
            n=("account_id", "count"),
            ecl_mean=("ecl_selected", "mean"),
            default_rate=("realized_default_h", "mean"),
        )
        .sort_values(["segment", "stage"])
    )

    Path("reports").mkdir(parents=True, exist_ok=True)
    out_path = Path("reports") / f"ecl_backtest_{asof}.csv"
    seg.to_csv(out_path, index=False)

    print(f"Read: {p}")
    print(f"Wrote: {out_path}")
    print(seg.head(10))


if __name__ == "__main__":
    main()
