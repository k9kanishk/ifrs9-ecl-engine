from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from ecl_engine.utils.math import logit, sigmoid


def _latest_scores_path() -> Path:
    paths = sorted(glob.glob("data/curated/pd_scores_asof_*.parquet"))
    if not paths:
        raise FileNotFoundError("No pd_scores_asof_*.parquet found. Run: python -m ecl_engine.models.pd_score")
    return Path(paths[-1])


def _solve_logit_shift(p_hat: np.ndarray, target_mean: float) -> float:
    """
    Find shift 's' such that mean(sigmoid(logit(p_hat) + s)) ~= target_mean.
    Monotone => binary search.
    """
    p_hat = np.clip(p_hat.astype(np.float64), 1e-6, 1.0 - 1e-6)
    target_mean = float(np.clip(target_mean, 1e-6, 1.0 - 1e-6))

    lo, hi = -8.0, 8.0
    x = logit(p_hat)

    for _ in range(80):
        mid = 0.5 * (lo + hi)
        m = float(sigmoid(x + mid).mean())
        if m < target_mean:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--level",
        choices=["portfolio", "segment"],
        default="segment",
        help="Calibration level for anchor shift (default: segment).",
    )
    ap.add_argument(
        "--target",
        default=None,
        help="Optional explicit target PD mean (0-1). If omitted, uses mean(ttc_pd_annual) per group.",
    )
    ap.add_argument(
        "--asof",
        default=None,
        help="ASOF date YYYY-MM-DD (optional). If omitted, uses latest pd_scores file.",
    )
    args = ap.parse_args()

    # Determine ASOF and scores path
    if args.asof:
        asof = args.asof
        asof_ts = pd.Timestamp(asof)
        scores_path = Path(f"data/curated/pd_scores_asof_{asof}.parquet")
        if not scores_path.exists():
            raise FileNotFoundError(
                f"Missing {scores_path}. Run: python -m ecl_engine.models.pd_score --asof {asof}"
            )
    else:
        scores_path = _latest_scores_path()
        asof = scores_path.stem.replace("pd_scores_asof_", "")
        asof_ts = pd.Timestamp(asof)

    scores = pd.read_parquet(scores_path)[["account_id", "pd_12m_hat"]].copy()
    accounts = pd.read_parquet("data/curated/accounts.parquet")[
        ["account_id", "segment", "ttc_pd_annual"]
    ].copy()

    df = scores.merge(accounts, on="account_id", how="left")
    df["segment"] = df["segment"].astype(str).fillna("UNKNOWN")
    df["ttc_pd_annual"] = df["ttc_pd_annual"].fillna(df["ttc_pd_annual"].median())
    df["pd_12m_hat"] = df["pd_12m_hat"].fillna(0.02).clip(1e-6, 0.5)

    if args.target is not None:
        explicit_target = float(args.target)
        explicit_target = float(np.clip(explicit_target, 1e-6, 1.0 - 1e-6))
    else:
        explicit_target = None

    if args.level == "portfolio":
        tgt = explicit_target if explicit_target is not None else float(df["ttc_pd_annual"].mean())
        shift = _solve_logit_shift(df["pd_12m_hat"].to_numpy(), tgt)
        df["pd_anchor_shift"] = shift
        report = pd.DataFrame([{"level": "portfolio", "target_mean": tgt, "shift": shift}])
    else:
        shifts = []
        for seg, g in df.groupby("segment", sort=True):
            tgt = explicit_target if explicit_target is not None else float(g["ttc_pd_annual"].mean())
            s = _solve_logit_shift(g["pd_12m_hat"].to_numpy(), tgt)
            shifts.append({"segment": seg, "target_mean": tgt, "shift": s})

        rep = pd.DataFrame(shifts)
        df = df.merge(rep[["segment", "shift"]], on="segment", how="left")
        df = df.rename(columns={"shift": "pd_anchor_shift"})
        report = rep.rename(columns={"shift": "pd_anchor_shift"})

    out = df[["account_id", "pd_anchor_shift"]].copy()
    out["asof_date"] = asof_ts

    Path("data/curated").mkdir(parents=True, exist_ok=True)
    out_path = Path(f"data/curated/pd_anchor_shift_asof_{asof_ts.date().isoformat()}.parquet")
    out.to_parquet(out_path, index=False)

    rep_path = Path(f"reports/pd_anchor_shift_{asof_ts.date().isoformat()}.csv")
    Path("reports").mkdir(parents=True, exist_ok=True)
    report.to_csv(rep_path, index=False)

    print(f"Wrote: {out_path}")
    print(f"Wrote: {rep_path}")
    print(report.head(20))


if __name__ == "__main__":
    main()
