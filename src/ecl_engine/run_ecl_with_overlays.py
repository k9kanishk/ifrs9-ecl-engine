from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd

from ecl_engine.overlay import apply_overlays


def main() -> None:
    # pick latest ecl output
    p = sorted(glob.glob("data/curated/ecl_output_asof_*.parquet"))[-1]
    ecl = pd.read_parquet(p)

    out = apply_overlays(ecl, "data/curated/overlays.csv")

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
