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

    def with_asof(cmd: list[str]) -> list[str]:
        return cmd + (["--asof", args.asof] if args.asof else [])

    # Future phases
    _run(with_asof([py, "-m", "ecl_engine.dcf_ecl"]))
    _run(with_asof([py, "-m", "ecl_engine.validation.pd_monitoring"]))
    _run(with_asof([py, "-m", "ecl_engine.validation.ecl_backtest"]))

    print("\nFuture pipeline done.")
    print("Next: integrate the dashboard patch in dashboards/app.py (see dashboards/app_phase6_patch.py).")


if __name__ == "__main__":
    main()
