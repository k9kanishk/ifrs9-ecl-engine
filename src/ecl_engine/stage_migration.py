from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_stage_migration(
    staging_path: str = "data/curated/staging_output.parquet",
    out_path: str = "data/curated/stage_migration.parquet",
) -> None:
    stg = pd.read_parquet(staging_path, columns=["snapshot_date", "account_id", "segment", "stage"])
    stg["snapshot_date"] = pd.to_datetime(stg["snapshot_date"])
    stg["stage"] = stg["stage"].astype(int)

    # sort and shift to get previous month’s stage per account
    stg = stg.sort_values(["account_id", "snapshot_date"])
    stg["prev_stage"] = stg.groupby("account_id")["stage"].shift(1)
    stg["prev_date"] = stg.groupby("account_id")["snapshot_date"].shift(1)

    # keep only consecutive month-end transitions
    stg["expected_prev"] = stg["snapshot_date"] - pd.offsets.MonthEnd(1)
    stg = stg[stg["prev_date"] == stg["expected_prev"]].dropna(subset=["prev_stage"])

    stg["prev_stage"] = stg["prev_stage"].astype(int)

    mig = (
        stg.groupby(["prev_date", "snapshot_date", "segment", "prev_stage", "stage"], as_index=False)
        .size()
        .rename(
            columns={
                "size": "n_accounts",
                "prev_date": "from_date",
                "snapshot_date": "to_date",
                "prev_stage": "stage_from",
                "stage": "stage_to",
            }
        )
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mig.to_parquet(out_path, index=False)
    print(f"Wrote: {out_path} ({mig.shape[0]:,} rows)")


if __name__ == "__main__":
    build_stage_migration()
