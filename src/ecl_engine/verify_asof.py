from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


def _asof_str(x) -> str:
    x = pd.to_datetime(x)
    return str(x.date())


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python -m ecl_engine.verify_asof YYYY-MM-DD")

    asof = sys.argv[1]
    asof_s = _asof_str(asof)

    checks = [
        ("ECL output", Path(f"data/curated/ecl_output_asof_{asof_s}.parquet"), "asof_date"),
        ("Overlays", Path("data/curated/ecl_with_overlays.parquet"), "asof_date"),
        ("Explain", Path(f"data/curated/account_explain_asof_{asof_s}.parquet"), "asof_date"),
        ("DCF", Path(f"data/curated/ecl_dcf_asof_{asof_s}.parquet"), "asof_date"),
    ]

    for name, path, col in checks:
        if not path.exists():
            raise SystemExit(f"[FAIL] Missing: {name}: {path}")

        df = pd.read_parquet(path, columns=[col])
        if df[col].isna().any():
            raise SystemExit(f"[FAIL] {name}: {path} has NaT in {col}")

        u = df[col].dropna().unique()
        if len(u) != 1:
            raise SystemExit(f"[FAIL] {name}: {path} has multiple {col} values: {u[:5]}")

        found = _asof_str(u[0])
        if found != asof_s:
            raise SystemExit(f"[FAIL] {name}: {path} {col}={found} != requested {asof_s}")

        print(f"[OK] {name}: {path} asof={found}")

    print("\nAll ASOF checks passed.")


if __name__ == "__main__":
    main()
