"""Slurm stages for a validation-selected regional WeatherBench experiment."""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xarray as xr

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT.parent))
from experiment import Diagnostics, FIELDS, LEADS, GridLSTM, fit, forecast, windows
from paper_models import PaperPredRNN, WindMapMetrics

DEV = ROOT / "regional_2008_2020.nc"
HOLDOUT = ROOT / "holdout_2021.nc"
HISTORIES = (2, 5, 10)
FINAL_HISTORY = 10
ARMS = ("lstm", "plain", "both")
LABELS = {"lstm": "Conventional LSTM", "plain": "PredRNN without physical constraints",
          "both": "PredRNN with mass and dry-static-energy constraints"}


def trials(smoke):
    result = []
    for arm in ARMS:
        positive = [.01] if smoke else [.003, .01, .03]
        pairs = [(m, e) for m in positive for e in positive] if arm == "both" else [(0., 0.)]
        for mass, energy in pairs:
            for seed in ([42] if smoke else [42, 43, 44]):
                result.append(dict(arm=arm, mass=mass, energy=energy, lr=.0001, seed=seed))
    return [dict(p, training_years=n) for n in HISTORIES for p in result]


def build(arm, shape):
    return GridLSTM(shape) if arm == "lstm" else PaperPredRNN()


def read_maps(path, years):
    with xr.open_dataset(path, engine="scipy") as ds:
        region = ds.sel(time=slice(*years))
        first, last = map(int, years)
        times = np.arange(np.datetime64(f"{first}-01-01T00"), np.datetime64(f"{last+1}-01-01T00"),
                          np.timedelta64(6, "h"))
        np.testing.assert_array_equal(region.time.values, times)
        lat, lon = region.latitude.values.copy(), region.longitude.values.copy()
        np.testing.assert_allclose(lat, np.arange(5)*5.625 + 30.9375)
        np.testing.assert_allclose(lon, np.arange(7)*5.625 + 264.375)
        maps = region[FIELDS].to_array().transpose("time", "variable", "latitude", "longitude")
        return torch.tensor(maps.values.copy()), lat, lon


def cases(data, mean, std, smoke):
    x, y = windows((data - mean) / std)
    return (x[:32], y[:32]) if smoke else (x, y)


def train(run, index, smoke):
    grid = trials(smoke)
    if not 0 <= index < len(grid):
        raise ValueError(f"Trial index must be in [0, {len(grid)-1}]")
    params = grid[index]
    target = run / f"trial-{index:02d}.pt"
    if target.exists():
        saved = torch.load(target, weights_only=True)
        if saved["params"] != params or saved["smoke"] != smoke:
            raise ValueError(f"Incompatible existing trial {target}")
        return
    data, lat, lon = read_maps(DEV, (str(2018-params["training_years"]), "2017"))
    mean = data.mean((0, 2, 3))[:, None, None]
    std = data.std((0, 2, 3))[:, None, None]
    x, y = cases(data, mean, std, smoke)
    val = cases(read_maps(DEV, ("2018", "2018"))[0], mean, std, smoke)
    diag = Diagnostics(mean, std, lat, lon)
    budgets = diag.budgets(x, y)
    scale = budgets.std((0, 1, 3, 4), keepdim=True).clamp_min(1e-12)
    weights = (params["mass"], params["energy"])
    torch.manual_seed(params["seed"])
    history = []
    best, epoch, state = fit(build(params["arm"], x.shape[2:]), x, y, budgets, val, diag, scale,
                             params["lr"], weights, 2 if smoke else 60, params["seed"],
                             patience=10, history=history, paper_loss=True)
    tmp = target.with_suffix(f".{os.getpid()}.part")
    torch.save(dict(params=params, smoke=smoke, mean=mean, std=std, best=best, epoch=epoch,
                    n_fit=len(x), budget_scale=scale, curve=history, state=state), tmp)
    tmp.replace(target)
    print(f"Trial {index}: {params}, best epoch {epoch}/{len(history)}, validation {best:.3f}")


def load_trial(run, index):
    return torch.load(run / f"trial-{index:02d}.pt", map_location="cpu", weights_only=True)


def freeze(run, smoke):
    grid = trials(smoke)
    rows = []
    curves = []
    for i, p in enumerate(grid):
        saved = load_trial(run, i)
        if p != saved["params"] or smoke != saved["smoke"]:
            raise ValueError(f"Trial mismatch for index {i}")
        rows.append(dict(index=i, **p, n_fit=saved["n_fit"], score=saved["best"], epoch=saved["epoch"]))
        curves.extend(dict(trial=i, epoch=j, validation_rmse=s)
                      for j, s in enumerate(saved["curve"], 1))
    df = pd.DataFrame(rows)
    df.to_csv(run / "validation_trials.csv", index=False)
    pd.DataFrame(curves).to_csv(run / "learning_curves.csv", index=False)
    scores = df.groupby(["training_years", "arm", "mass", "energy"], as_index=False).score.mean()
    # A common physical speed scale keeps validation image scores comparable. All
    # training periods contain 2016–17; this scale never uses validation weather.
    speed_max = read_maps(DEV, ("2016", "2017"))[0][:, :2, 1:-1, 1:-1].norm(dim=1).max().item()
    images = WindMapMetrics(speed_max)
    val, lat, lon = read_maps(DEV, ("2018", "2018"))
    validation_rows = []
    selected_by_size = {}
    for years in HISTORIES:
        selection = {}
        for arm in ARMS:
            group = scores[(scores.training_years == years) & (scores.arm == arm)]
            best = group.sort_values(["score", "mass", "energy"]).iloc[0]
            matching = df[(df.training_years == years) & (df.arm == arm)
                          & (df.mass == best.mass) & (df.energy == best.energy)]
            indices = matching.sort_values("seed")["index"].tolist()
            selection[arm] = indices
            for i in indices:
                saved = load_trial(run, i)
                mean, std = saved["mean"], saved["std"]
                x, y = cases(val, mean, std, smoke)
                diag = Diagnostics(mean, std, lat, lon)
                prediction = predictions(saved, x)
                rmse = diag.wind_mse(y, prediction).mean(0).sqrt()
                ssim, distance = images.score(prediction, y, mean, std)
                for j, lead in enumerate(LEADS):
                    validation_rows.append(dict(training_years=years, train_start=2018-years, train_end=2017,
                        train_examples=saved["n_fit"], validation_year=2018, validation_days=len(y),
                        Model=LABELS[arm], lead_h=lead, seed=grid[i]["seed"], trial=i,
                        mass_weight=best.mass, energy_weight=best.energy, epoch=saved["epoch"],
                        RMSE_m_s=rmse[j].item(), SSIM=ssim[j].item(), LPIPS=distance[j].item()))
        selected_by_size[years] = selection
    detail = pd.DataFrame(validation_rows)
    detail.to_csv(run / "validation_by_history_per_seed.csv", index=False)
    keys = ["training_years", "train_start", "train_end", "train_examples", "validation_year",
            "validation_days", "Model", "lead_h", "mass_weight", "energy_weight"]
    summary = detail.groupby(keys, sort=False, as_index=False)[["RMSE_m_s", "SSIM", "LPIPS"]].mean()
    summary.to_csv(run / "validation_by_history.csv", index=False)
    wide = summary.pivot(index=["training_years", "lead_h"], columns="Model", values="RMSE_m_s")
    (wide[LABELS["plain"]] - wide[LABELS["both"]]).rename("wind_rmse_gain_ms").to_csv(
        run / "validation_physics_gain.csv")
    selections = selected_by_size[FINAL_HISTORY]
    choices = {label: [{**grid[i], "trial": i, "epoch": rows[i]["epoch"]} for i in indices]
               for label, indices in selections.items()}
    (run / "frozen.json").write_text(json.dumps(dict(smoke=smoke, test_year=2020 if smoke else 2021,
                         training_years=FINAL_HISTORY, image_speed_max_ms=speed_max,
                         primary=["plain", "both"], trial_indices=selections, settings=choices), indent=2))
    print("Frozen trial indices", selections)


def predictions(saved, x):
    model = build(saved["params"]["arm"], x.shape[2:])
    model.load_state_dict(saved["state"])
    return forecast(model.eval(), x)


def evaluate(run, smoke, calibrating):
    design = json.loads((run / "frozen.json").read_text())
    if (design["smoke"] != smoke or design["test_year"] != (2020 if smoke else 2021)
            or design["training_years"] != FINAL_HISTORY or set(design["trial_indices"]) != set(ARMS)):
        raise ValueError("Incompatible final design or run directory")
    selections = design["trial_indices"]
    grid = trials(smoke)
    # Check all finalists and calibration provenance before opening the evaluation year.
    checkpoints = {}
    for arm, indices in selections.items():
        seeds = []
        for i in indices:
            saved = load_trial(run, i)
            p = saved["params"]
            if (p != grid[i] or p["training_years"] != FINAL_HISTORY or p["arm"] != arm
                    or saved["smoke"] != smoke or saved["n_fit"] != (32 if smoke else 3650)):
                raise ValueError("Only matching ten-year checkpoints may reach calibration or testing")
            checkpoints[i] = saved
            seeds.append(p["seed"])
        if sorted(seeds) != ([42] if smoke else [42, 43, 44]):
            raise ValueError("Missing or duplicated final seed")
    year = "2019" if calibrating else str(design["test_year"])
    path = DEV if calibrating or smoke else HOLDOUT
    # Require a complete calibration artifact before loading any final test cases.
    fitted = {}
    if not calibrating:
        calibrated = torch.load(run / "intervals.pt", weights_only=True)
        if calibrated["trial_indices"] != selections or calibrated["calibration_year"] != 2019:
            raise ValueError("Intervals must belong to the selected ten-year checkpoints")
        fitted = calibrated["radii"]
    raw, lat, lon = read_maps(path, (year, year))
    first = checkpoints[selections["plain"][0]]
    mean, std = first["mean"], first["std"]
    x, y = cases(raw, mean, std, smoke)
    diag = Diagnostics(mean, std, lat, lon)
    truth_budgets = diag.budgets(x, y)
    rows, paired = [], {}
    image_scores = None if calibrating else WindMapMetrics(design["image_speed_max_ms"])
    for arm, indices in selections.items():
        for i in indices:
            saved = checkpoints[i]
            pred = predictions(saved, x)
            key = f"{arm}/{i}"
            wind_error = ((pred-y)*std)[:, :, :2, 1:-1, 1:-1]
            if calibrating:
                rank = min(len(y), math.ceil(.90*(len(y)+1)))
                fitted[key] = wind_error.abs().kthvalue(rank, dim=0).values
                continue
            radius = fitted[key]
            cov = ((wind_error.abs() <= radius).float() * diag.weight).mean((0, 3, 4))
            widths = (2*radius*diag.weight).mean((2, 3))
            error = diag.wind_mse(y, pred)
            diag_rmse = ((diag.budgets(x, pred)-truth_budgets).square()*diag.weight).mean((0, 3, 4)).sqrt()
            ssim, lpips = image_scores.score(pred, y, mean, std)
            seed = saved["params"]["seed"]
            if arm in design["primary"]:
                paired[(arm, seed)] = error.numpy()
            for j, lead in enumerate(LEADS):
                rows.append(dict(arm=arm, seed=seed, training_years=FINAL_HISTORY, test_year=int(year), n_days=len(y), lead_h=lead,
                             wind_rmse_ms=error[:, j].mean().sqrt().item(), SSIM=ssim[j].item(), LPIPS=lpips[j].item(),
                             mass_residual_rmse_per_s=diag_rmse[j, 0].item(),
                             dse_tendency_rmse_W_kg=diag_rmse[j, 1].item(),
                             u_coverage=cov[j, 0].item(), v_coverage=cov[j, 1].item(),
                             u_width_ms=widths[j, 0].item(), v_width_ms=widths[j, 1].item()))
    if calibrating:
        tmp = run / "intervals.part"
        torch.save(dict(trial_indices=selections, calibration_year=2019, radii=fitted), tmp)
        tmp.replace(run / "intervals.pt")
        return
    metrics = pd.DataFrame(rows)
    metrics.to_csv(run / "per_seed.csv", index=False)
    summary = metrics.drop(columns="seed").groupby(["arm", "lead_h"]).mean()
    summary.to_csv(run / "summary.csv")
    for lead in [24, 48]:
        table = summary.xs(lead, level="lead_h").loc[list(ARMS), ["wind_rmse_ms", "SSIM", "LPIPS"]]
        table.rename(index=LABELS, columns={"wind_rmse_ms":"RMSE_m_s"}).rename_axis("Model").to_csv(
            run / f"table1_{lead}h.csv")
    # Resample the same consecutive weather cases for all seeds and both models.
    seeds = sorted(s for a, s in paired if a == "plain")
    n = len(y)
    rng = np.random.default_rng(2021)
    starts = rng.integers(n, size=(1000, math.ceil(n/7)))
    ids = ((starts[:, :, None]+np.arange(7)) % n).reshape(1000, -1)[:, :n]
    boot, gain = [], []
    for s in seeds:
        p, m = paired["plain", s], paired["both", s]
        boot.append(np.sqrt(p[ids].mean(1))-np.sqrt(m[ids].mean(1)))
        gain.append(np.sqrt(p.mean(0))-np.sqrt(m.mean(0)))
    lo, hi = np.quantile(np.mean(boot, axis=0), [.025, .975], axis=0)
    pd.DataFrame(dict(lead_h=LEADS, wind_rmse_gain_ms=np.mean(gain, axis=0),
                      lower95=lo, upper95=hi)).to_csv(run / "primary_gain.csv", index=False)
    print("Scored", year, "with", n, "daily forecast cases")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["train", "freeze", "calibrate", "score", "smoke"])
    parser.add_argument("run", type=Path)
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "2")))
    args.run.mkdir(parents=True, exist_ok=True)
    if args.stage == "smoke":
        if len(trials(False)) != 99:
            raise RuntimeError("Update the Slurm array size to match the trial grid")
        for i in range(len(trials(True))):
            train(args.run, i, True)
        freeze(args.run, True)
        evaluate(args.run, True, True)
        evaluate(args.run, True, False)
    elif args.stage == "train":
        train(args.run, args.index, False)
    elif args.stage == "freeze":
        freeze(args.run, False)
    else:
        evaluate(args.run, False, args.stage == "calibrate")


if __name__ == "__main__":
    main()
