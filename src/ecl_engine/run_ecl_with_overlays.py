from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd

from ecl_engine.overlay import apply_overlays


def main() -> None:
    # pick latest ecl output
    p = sorted(glob.glob("data/curated/ecl_output_asof_*.parquet"))[-1]
    asof = Path(p).stem.replace("ecl_output_asof_", "")
    asof_ts = pd.Timestamp(asof)
    ecl = pd.read_parquet(p)

    out = apply_overlays(ecl, "data/curated/overlays.csv")
    if "asof_date_y" in out.columns:
        out = out.rename(columns={"asof_date_y": "overlay_asof_date"})
    if "asof_date_overlay" in out.columns:
        out = out.rename(columns={"asof_date_overlay": "overlay_asof_date"})
    asof_ts = pd.to_datetime(asof_ts)
    if "asof_date" not in out.columns:
        out["asof_date"] = asof_ts
    else:
        out["asof_date"] = pd.to_datetime(out["asof_date"], errors="coerce")
        out["asof_date"] = out["asof_date"].fillna(asof_ts)
        out["asof_date"] = asof_ts

    out_path = Path("data/curated") / "ecl_with_overlays.parquet"
    out.to_parquet(out_path, index=False)

    print(f"Read: {p}")
    print(f"Wrote: {out_path}")

    # Governance summary
    seg_sum = (
        out.groupby("segment")[["ecl_pre_overlay", "overlay_amount", "ecl_post_overlay"]]
        .sum()
        .sort_values("ecl_post_overlay", ascending=False)
    )
    stage_sum = out.groupby("stage")[["ecl_pre_overlay", "overlay_amount", "ecl_post_overlay"]].sum()

    print("\nSegment overlay summary (top 10):")
    print(seg_sum.head(10).round(2))

    print("\nStage overlay summary:")
    print(stage_sum.round(2))


if __name__ == "__main__":
    main()
