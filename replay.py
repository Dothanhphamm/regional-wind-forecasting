"""Recreate the validation, calibration and final scores from saved Tufts fits."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys

from verify_release import ROOT, ARCHIVED, verify_files, check_replay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path, help="New directory to hold replay outputs")
    destination = parser.parse_args().destination.resolve()
    verify_files()
    destination.mkdir(parents=True, exist_ok=False)
    for index in range(99):
        shutil.copyfile(ARCHIVED / f"trial-{index:02d}.pt", destination / f"trial-{index:02d}.pt")
    env = os.environ.copy()
    env["TORCH_HOME"] = str(ROOT / "torch-cache")
    script = ROOT / "wind_histories_2021/followup.py"
    for stage in ("freeze", "calibrate", "score"):
        print(f"Running {stage}", flush=True)
        subprocess.run([sys.executable, "-u", str(script), stage, str(destination)], env=env, check=True)
    check_replay(destination)


if __name__ == "__main__":
    main()
