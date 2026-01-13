from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def apply_overlays(ecl: pd.DataFrame, overlays_path: str | Path) -> pd.DataFrame:
    """
    Apply overlays by asof_date + segment.
    Produces:
      - ecl_pre_overlay
      - overlay_amount
      - ecl_post_overlay
      - overlay_audit (text)
    """
    out = ecl.copy()
    out["asof_date"] = pd.to_datetime(out["asof_date"])
    out["ecl_pre_overlay"] = out["ecl_selected"].astype("float64")
    out["overlay_amount"] = 0.0
    out["overlay_audit"] = ""

    overlays = pd.read_csv(overlays_path)
    overlays["asof_date"] = pd.to_datetime(overlays["asof_date"])

    # Apply overlays segment-by-segment
    for _, ov in overlays.iterrows():
        asof = ov["asof_date"]
        seg = ov["segment"]
        method = ov["method"]
        val = float(ov["value"])

        m = (out["asof_date"] == asof) & (out["segment"] == seg)
        if not m.any():
            continue

        if method == "multiplicative":
            # overlay = (factor - 1) * base
            add = (val - 1.0) * out.loc[m, "ecl_pre_overlay"]
        elif method == "additive":
            # allocate additive overlay pro-rata to ECL (or balance if ECL=0)
            base = out.loc[m, "ecl_pre_overlay"].to_numpy()
            w = base / base.sum() if base.sum() > 0 else np.ones_like(base) / len(base)
            add = pd.Series(val * w, index=out.loc[m].index)
        else:
            raise ValueError(f"Unknown overlay method: {method}")

        out.loc[m, "overlay_amount"] += add.astype("float64")
        out.loc[m, "overlay_audit"] = (
            out.loc[m, "overlay_audit"].astype(str)
            + f"[{ov['overlay_id']}:{method}:{val}:{ov['reason_code']}]"
        )

    out["ecl_post_overlay"] = out["ecl_pre_overlay"] + out["overlay_amount"]
    return out
