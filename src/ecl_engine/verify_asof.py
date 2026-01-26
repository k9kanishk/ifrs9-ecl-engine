"""Verify ASOF consistency across all pipeline outputs."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def verify_asof_consistency(asof: str) -> bool:
    """Check that all output files have matching ASOF dates."""
    asof_dt = pd.Timestamp(asof).normalize()

    files_to_check = [
        (f"data/curated/ecl_output_asof_{asof}.parquet", "asof_date"),
        (f"data/curated/ecl_dcf_asof_{asof}.parquet", "asof_date"),
        (f"data/curated/account_explain_asof_{asof}.parquet", "asof_date"),
        ("data/curated/ecl_with_overlays.parquet", "asof_date"),
    ]

    print(f"🔍 Verifying ASOF consistency for {asof}...\n")

    all_good = True
    for fpath, col in files_to_check:
        p = Path(fpath)
        if not p.exists():
            print(f"⚠️  Missing: {fpath}")
            continue  # Not a hard failure - file might not be generated yet

        df = pd.read_parquet(p)
        if col not in df.columns:
            print(f"❌ {fpath}: missing column '{col}'")
            all_good = False
            continue

        file_asofs = pd.to_datetime(df[col]).dt.normalize().unique()
        if len(file_asofs) != 1:
            print(f"❌ {fpath}: multiple ASOF dates found: {file_asofs}")
            all_good = False
        elif file_asofs[0] != asof_dt:
            print(
                f"❌ {fpath}: ASOF mismatch. Expected {asof_dt.date()}, "
                f"got {file_asofs[0].date()}"
            )
            all_good = False
        else:
            print(f"✅ {fpath}: ASOF = {file_asofs[0].date()}")

    if all_good:
        print(f"\n✅ All files consistent with ASOF = {asof}")
    else:
        print("\n❌ ASOF consistency check FAILED")

    return all_good


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("asof", help="ASOF date to verify (YYYY-MM-DD)")
    args = ap.parse_args()

    success = verify_asof_consistency(args.asof)
    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()
