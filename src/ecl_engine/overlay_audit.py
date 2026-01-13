from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_overlay_audit(
    ecl_with_overlays_path: str = "data/curated/ecl_with_overlays.parquet",
    overlays_path: str = "data/curated/overlays.csv",
    out_path: str = "data/curated/overlay_audit.parquet",
    top_n_accounts: int = 20,
) -> None:
    e = pd.read_parquet(ecl_with_overlays_path)
    e["asof_date"] = pd.to_datetime(e["asof_date"])

    ov = pd.read_csv(overlays_path)
    ov["asof_date"] = pd.to_datetime(ov["asof_date"])

    asof = e["asof_date"].max()
    e = e[e["asof_date"] == asof].copy()
    ov = ov[ov["asof_date"] == asof].copy()

    # Summary by overlay_id (allocated impact)
    # We use overlay_audit string tags created in apply_overlays
    rows = []
    for _, r in ov.iterrows():
        oid = r["overlay_id"]
        seg = r["segment"]

        m = (e["segment"] == seg) & (
            e["overlay_audit"].astype(str).str.contains(f"[{oid}:", regex=False)
        )
        impact = float(e.loc[m, "overlay_amount"].sum())

        # top accounts impacted
        top = (
            e.loc[
                m,
                [
                    "account_id",
                    "segment",
                    "stage",
                    "balance",
                    "ecl_pre_overlay",
                    "overlay_amount",
                    "ecl_post_overlay",
                ],
            ]
            .sort_values("overlay_amount", ascending=False)
            .head(top_n_accounts)
        )
        top["overlay_id"] = oid

        rows.append(
            {
                "overlay_id": oid,
                "segment": seg,
                "method": r["method"],
                "value": r["value"],
                "reason_code": r["reason_code"],
                "owner": r["owner"],
                "approved_by": r["approved_by"],
                "asof_date": asof,
                "allocated_overlay_amount": impact,
            }
        )

        # save top accounts per overlay as separate parquet (optional but useful)
        top_out = Path(out_path).with_name(f"overlay_top_accounts_{oid}.parquet")
        top.to_parquet(top_out, index=False)

    audit = pd.DataFrame(rows).sort_values("allocated_overlay_amount", ascending=False)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    audit.to_parquet(out_path, index=False)

    print(f"Wrote: {out_path}")
    print("Wrote per-overlay top accounts: overlay_top_accounts_<overlay_id>.parquet")


if __name__ == "__main__":
    build_overlay_audit()
