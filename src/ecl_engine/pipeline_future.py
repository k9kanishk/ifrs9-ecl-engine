from __future__ import annotations

import argparse
import subprocess
import sys


def _run(cmd: list[str]) -> None:
    print("\n>>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", default=None, help="ASOF date YYYY-MM-DD (optional)")
    args = ap.parse_args()

    py = sys.executable

    # DCF ECL (CRITICAL: pass ASOF)
    if args.asof:
        _run([py, "-m", "ecl_engine.dcf_ecl", "--asof", args.asof])
        _run([py, "-m", "ecl_engine.validation.pd_monitoring", "--asof", args.asof])
        _run([py, "-m", "ecl_engine.validation.ecl_backtest", "--asof", args.asof])
    else:
        _run([py, "-m", "ecl_engine.dcf_ecl"])
        _run([py, "-m", "ecl_engine.validation.pd_monitoring"])
        _run([py, "-m", "ecl_engine.validation.ecl_backtest"])

    print("\n✅ Future/validation pipeline complete.")
    print("All outputs ready for dashboard.")


if __name__ == "__main__":
    main()
