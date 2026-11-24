"""Verify the release payload; optionally compare a replay with the Tufts run."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARCHIVED = ROOT / "wind-6afo1x"


def verify_files():
    entries = (ROOT / "SHA256SUMS").read_text().splitlines()
    for line in entries:
        digest, name = line.split("  ", 1)
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"SHA-256 mismatch for {name}")
    print(f"Verified all {len(entries)} release file hashes.")


def check_csv(original, replay):
    with original.open(newline="") as a, replay.open(newline="") as b:
        observed, expected = csv.DictReader(b), csv.DictReader(a)
        if observed.fieldnames != expected.fieldnames:
            raise ValueError(f"Column mismatch in {original.name}")
        ref_rows, new_rows = list(expected), list(observed)
    if len(ref_rows) != len(new_rows):
        raise ValueError(f"Row count mismatch in {original.name}")
    for i, (ref, new) in enumerate(zip(ref_rows, new_rows), 2):
        for key, ref_value in ref.items():
            if new[key] == ref_value:
                continue
            try:
                x, y = float(ref_value), float(new[key])
            except ValueError:
                raise ValueError(f"Different text in {original.name}, row {i}, {key}") from None
            # Allow CPU floating-point variation. Mass diagnostics have much smaller units.
            atol = 1e-10 if key == "mass_residual_rmse_per_s" else 1e-5
            if not math.isclose(x, y, rel_tol=1e-5, abs_tol=atol):
                raise ValueError(f"{original.name}, row {i}, {key}: {y} differs from {x}")


def check_replay(replay):
    import torch

    ref = json.loads((ARCHIVED / "frozen.json").read_text())
    new = json.loads((replay / "frozen.json").read_text())
    if ref != new:
        raise ValueError("Frozen finalist configurations differ.")
    for source in sorted(ARCHIVED.glob("*.csv")):
        check_csv(source, replay / source.name)
    saved, regenerated = [torch.load(p / "intervals.pt", map_location="cpu", weights_only=True)
                          for p in (ARCHIVED, replay)]
    for key in ("trial_indices", "calibration_year"):
        if saved[key] != regenerated[key]:
            raise ValueError(f"Calibration metadata differs in {key}")
    if saved["radii"].keys() != regenerated["radii"].keys():
        raise ValueError("Calibration finalists differ.")
    for key, radii in saved["radii"].items():
        torch.testing.assert_close(radii, regenerated["radii"][key], atol=1e-4, rtol=1e-5)
    print("Replay agrees with the frozen selections, calibration radii and all ten CSVs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", type=Path)
    args = parser.parse_args()
    verify_files()
    if args.replay_dir:
        check_replay(args.replay_dir.resolve())
