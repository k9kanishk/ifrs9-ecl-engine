from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd


def _latest_ecl_path(outdir: str) -> Path:
    paths = sorted(glob.glob(str(Path(outdir) / "ecl_output_asof_*.parquet")))
    if not paths:
        raise FileNotFoundError("No ecl_output_asof_*.parquet found. Run: python src/ecl_engine/ecl.py")
    return Path(paths[-1])


def build_stage3_summary(
    asof: str | None = None,
    ecl_path: str | None = None,
    outdir: str = "data/curated",
) -> Path:
    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)

    if ecl_path:
        ecl_file = Path(ecl_path)
    elif asof:
        ecl_file = outdir_path / f"ecl_output_asof_{asof}.parquet"
    else:
        ecl_file = _latest_ecl_path(outdir)

    ecl = pd.read_parquet(ecl_file)
    if "asof_date" in ecl.columns:
        asof_dt = pd.to_datetime(asof) if asof else pd.to_datetime(ecl["asof_date"]).max()
    else:
        if not asof:
            raise ValueError("Missing asof_date in ECL output; provide --asof or --ecl_path.")
        asof_dt = pd.to_datetime(asof)

    s3 = ecl[ecl["stage"] == 3].copy()
    if s3.empty:
        summary = pd.DataFrame(
            columns=["segment", "ead_default_sum", "pv_recoveries_sum", "ecl_sum", "implied_lgd"]
        )
    else:
        summary = (
            s3.groupby("segment")
            .agg(
                ead_default_sum=("ead_default", "sum"),
                pv_recoveries_sum=("pv_recoveries", "sum"),
                ecl_sum=("ecl_selected", "sum"),
            )
            .reset_index()
        )
        summary["implied_lgd"] = 1.0 - np.where(
            summary["ead_default_sum"] > 0,
            summary["pv_recoveries_sum"] / summary["ead_default_sum"],
            0.0,
        )

    out_path = outdir_path / f"stage3_workout_summary_asof_{asof_dt.date().isoformat()}.csv"
    summary.to_csv(out_path, index=False)
    print(f"Wrote: {out_path}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", type=str, default=None, help="YYYY-MM-DD month-end; defaults to latest in ECL output")
    ap.add_argument("--ecl_path", type=str, default=None, help="Optional explicit ECL parquet path")
    ap.add_argument("--outdir", type=str, default="data/curated")
    args = ap.parse_args()
    build_stage3_summary(asof=args.asof, ecl_path=args.ecl_path, outdir=args.outdir)


if __name__ == "__main__":
    main()
