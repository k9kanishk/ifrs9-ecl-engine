from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd

from ecl_engine.utils.macro import prepare_macro_z_all_scenarios


def main() -> None:
    ecl_paths = sorted(glob.glob("data/curated/ecl_output_asof_*.parquet"))
    if not ecl_paths:
        raise FileNotFoundError("No ecl_output_asof_*.parquet found. Run: python src/ecl_engine/ecl.py")

    p = Path(ecl_paths[-1])
    asof = p.stem.replace("ecl_output_asof_", "")

    macro = pd.read_parquet("data/curated/macro_scenarios_monthly.parquet")
    mz = prepare_macro_z_all_scenarios(macro)

    # simple QC summary: mean macro z by scenario over next 12 months isn't done here
    # (your ecl.py already writes a better 'scenario_severity_asof_*.csv').
    # This is just a sanity output to prove z columns exist and look finite.
    out = (
        mz.groupby("scenario")[["unemployment_z", "gdp_yoy_z", "policy_rate_z"]]
        .mean()
        .reset_index()
    )
    out["asof_date"] = asof

    Path("data/curated").mkdir(parents=True, exist_ok=True)
    out_path = Path(f"data/curated/scenario_macro_z_means_asof_{asof}.csv")
    out.to_csv(out_path, index=False)
    print(f"Wrote: {out_path}")
    print(out)


if __name__ == "__main__":
    main()
