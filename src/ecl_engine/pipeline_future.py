from __future__ import annotations

import subprocess
import sys


def _run(cmd: list[str]) -> None:
    print("\n>>", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    py = sys.executable

    # Assumes you already run your core pipeline first (pd_train/score/anchor, ecl, overlays, driver_decomp, stage3_summary, explain)
    _run([py, "-m", "ecl_engine.dcf_ecl"])
    _run([py, "-m", "ecl_engine.validation.pd_monitoring"])
    _run([py, "-m", "ecl_engine.validation.ecl_backtest"])

    print("\nFuture pipeline done.")
    print("Next: integrate the dashboard patch in dashboards/app.py (see dashboards/app_phase6_patch.py).")


if __name__ == "__main__":
    main()
