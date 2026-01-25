from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pandas as pd

from ecl_engine.overlay import apply_overlays


def pick_ecl_output(asof: str | None) -> tuple[str, pd.Timestamp]:
    if asof:
        asof_ts = pd.Timestamp(asof)
        p = f"data/curated/ecl_output_asof_{asof_ts.date().isoformat()}.parquet"
        if not Path(p).exists():
            raise FileNotFoundError(f"Missing: {p}. Run: python -m ecl_engine.ecl --asof {asof}")
        return p, asof_ts

    paths = sorted(glob.glob("data/curated/ecl_output_asof_*.parquet"))
    if not paths:
        raise FileNotFoundError("No ecl_output_asof_*.parquet found. Run: python -m ecl_engine.ecl")
    p = paths[-1]
    asof_str = Path(p).stem.replace("ecl_output_asof_", "")
    return p, pd.Timestamp(asof_str)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", default=None, help="YYYY-MM-DD. If omitted, uses latest available output.")
    args = ap.parse_args()

    p, asof_ts = pick_ecl_output(args.asof)
    ecl = pd.read_parquet(p)

    out = apply_overlays(ecl, "data/curated/overlays.csv")

    # normalize any merge artifacts
    if "asof_date_y" in out.columns:
        out = out.rename(columns={"asof_date_y": "overlay_asof_date"})
    if "asof_date_overlay" in out.columns:
        out = out.rename(columns={"asof_date_overlay": "overlay_asof_date"})

    # hard-set asof_date to the file ASOF (this avoids stale/NaT)
    out["asof_date"] = asof_ts

    out_path = Path("data/curated") / "ecl_with_overlays.parquet"
    out.to_parquet(out_path, index=False)

    print(f"Read: {p}")
    print(f"Wrote: {out_path}")

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
