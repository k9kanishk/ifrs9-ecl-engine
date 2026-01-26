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
    ap.add_argument("--skip-pd-train", action="store_true")
    ap.add_argument("--skip-pd-anchor", action="store_true")
    args = ap.parse_args()

    py = sys.executable

    # Helper: append --asof consistently
    def with_asof(cmd: list[str]) -> list[str]:
        return cmd + (["--asof", args.asof] if args.asof else [])

    # Core pipeline
    if not args.skip_pd_train:
        _run([py, "-m", "ecl_engine.models.pd_train"])

    # If your pd_score / pd_anchor accept --asof, this is ideal.
    # If they DON'T accept --asof, remove with_asof(...) here.
    _run(with_asof([py, "-m", "ecl_engine.models.pd_score"]))

    if not args.skip_pd_anchor:
        _run(with_asof([py, "-m", "ecl_engine.models.pd_anchor", "--level", "segment"]))

    _run(with_asof([py, "-m", "ecl_engine.ecl"]))

    _run(with_asof([py, "-m", "ecl_engine.run_ecl_with_overlays"]))
    _run(with_asof([py, "-m", "ecl_engine.driver_decomp"]))
    _run(with_asof([py, "-m", "ecl_engine.stage3_summary"]))

    # explain already supports --asof in your usage
    _run(with_asof([py, "-m", "ecl_engine.explain", "--n", "0"]))

    print("\nPipeline done.")
    print("Next: streamlit run dashboards/app.py")


if __name__ == "__main__":
    main()
