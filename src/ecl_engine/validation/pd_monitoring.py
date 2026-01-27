from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", default=None, help="ASOF date (YYYY-MM-DD). Default: infer from latest pd_scores file.")
    args = ap.parse_args()

    if args.asof is None:
        paths = sorted(Path("data/curated").glob("pd_scores_asof_*.parquet"))
        if not paths:
            raise FileNotFoundError("No pd_scores_asof_*.parquet found. Run: python -m ecl_engine.models.pd_score")
        p = paths[-1]
        asof = p.stem.replace("pd_scores_asof_", "")
    else:
        asof = args.asof
        p = Path(f"data/curated/pd_scores_asof_{asof}.parquet")

    df = pd.read_parquet(p)
    if "segment" not in df.columns:
        # try join from accounts
        acc = pd.read_parquet("data/curated/accounts.parquet")[["account_id", "segment"]]
        df = df.merge(acc, on="account_id", how="left")

    # simple monitoring stats by segment
    out = (
        df.groupby("segment")["pd_12m_hat"]
        .agg(["count", "mean", "std", "min", "median", "max"])
        .reset_index()
        .rename(columns={"count": "n"})
        .sort_values("mean", ascending=False)
    )

    Path("reports").mkdir(parents=True, exist_ok=True)
    out_path = Path("reports") / f"pd_monitoring_{asof}.csv"
    out.to_csv(out_path, index=False)

    # stability: compare to anchor shifts if present
    anchor_path = Path(f"data/curated/pd_anchor_shift_asof_{asof}.parquet")
    if anchor_path.exists():
        sh = pd.read_parquet(anchor_path)
        if "segment" not in sh.columns:
            acc = pd.read_parquet("data/curated/accounts.parquet")[["account_id", "segment"]]
            sh = sh.merge(acc, on="account_id", how="left")
        sh_out = (
            sh.groupby("segment")["pd_anchor_shift"]
            .agg(["count", "mean", "std", "min", "median", "max"])
            .reset_index()
            .rename(columns={"count": "n"})
            .sort_values("mean", ascending=False)
        )
        sh_path = Path("reports") / f"pd_anchor_monitoring_{asof}.csv"
        sh_out.to_csv(sh_path, index=False)
        print(f"Wrote: {sh_path}")

    print(f"Read: {p}")
    print(f"Wrote: {out_path}")
    print("Top 5 segments by mean PD:")
    print(out.head(5))


if __name__ == "__main__":
    main()
