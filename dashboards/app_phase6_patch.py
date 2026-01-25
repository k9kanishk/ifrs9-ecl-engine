"""
Patch snippet to integrate Phase 6+ (DCF ECL + validation artifacts) into dashboards/app.py.

How to use:
- Open dashboards/app.py
- Add a new section after your Phase 5 blocks (or near PD model validation).
- Paste the relevant blocks from this file.

This patch assumes the following outputs exist:
- data/curated/ecl_dcf_asof_<asof>.parquet   (from: python -m ecl_engine.dcf_ecl)
- reports/pd_monitoring_<asof>.csv
- reports/ecl_backtest_<asof>.csv
"""

import os
import pandas as pd
import streamlit as st


def phase6_blocks(asof: str):
    st.markdown("## Phase 6 — DCF ECL (Optional Enhancements)")

    dcf_path = f"data/curated/ecl_dcf_asof_{asof}.parquet"
    if os.path.exists(dcf_path):
        dcf = pd.read_parquet(dcf_path)
        st.metric("DCF Selected ECL (total)", f"{dcf['dcf_ecl_selected'].sum():,.2f}")
        if "ecl_selected" in dcf.columns:
            st.metric("DCF vs Model Delta (total)", f"{(dcf['dcf_ecl_selected'] - dcf['ecl_selected']).sum():,.2f}")
        st.dataframe(
            dcf.groupby("segment")[["dcf_ecl_selected"]].sum().sort_values("dcf_ecl_selected", ascending=False).reset_index(),
            width="stretch",
        )
    else:
        st.info("Run: python -m ecl_engine.dcf_ecl")

    st.markdown("## Phase 6 — Monitoring & Backtest (Toy Governance)")

    pdm = f"reports/pd_monitoring_{asof}.csv"
    if os.path.exists(pdm):
        st.markdown("### PD monitoring by segment")
        st.dataframe(pd.read_csv(pdm), width="stretch")
    else:
        st.info("Run: python -m ecl_engine.validation.pd_monitoring")

    bt = f"reports/ecl_backtest_{asof}.csv"
    if os.path.exists(bt):
        st.markdown("### ECL backtest proxy (segment x stage)")
        st.dataframe(pd.read_csv(bt), width="stretch")
    else:
        st.info("Run: python -m ecl_engine.validation.ecl_backtest")
