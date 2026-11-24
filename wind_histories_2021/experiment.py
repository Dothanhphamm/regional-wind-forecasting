"""Train, tune, calibrate and evaluate the regional WeatherBench pilot."""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import xarray as xr

ROOT = Path(__file__).resolve().parent
FIELDS = ["u500", "v500", "omega400", "omega500", "omega600",
          "T400", "T500", "T600", "phi400", "phi500", "phi600"]
LEADS = [12, 24, 36, 48]


class MemoryCell(nn.Module):
    """PredRNN-style temporal and zigzag spatial memories, without layer norm."""
    def __init__(self, inputs, hidden):
        super().__init__()
        self.temporal = nn.Conv2d(inputs + hidden, 3 * hidden, 3, padding=1)
        self.spatial = nn.Conv2d(inputs + hidden, 3 * hidden, 3, padding=1)
        self.output_gate = nn.Conv2d(inputs + 3 * hidden, hidden, 1)
        self.read = nn.Conv2d(2 * hidden, hidden, 1)

    @staticmethod
    def update(gates, memory):
        forget, write, value = gates.chunk(3, dim=1)
        return (forget + 1).sigmoid() * memory + write.sigmoid() * value.tanh()

    def forward(self, x, h, c, m):
        c = self.update(self.temporal(torch.cat((x, h), 1)), c)
        m = self.update(self.spatial(torch.cat((x, m), 1)), m)
        gate = self.output_gate(torch.cat((x, h, c, m), 1)).sigmoid()
        h = gate * self.read(torch.cat((c, m), 1)).tanh()
        return h, c, m


class PredRNN(nn.Module):
    def __init__(self, channels, hidden=8):
        super().__init__()
        self.hidden = hidden
        self.cells = nn.ModuleList([MemoryCell(channels, hidden), MemoryCell(hidden, hidden)])
        self.readout = nn.Conv2d(hidden, channels, 1)

    def forward(self, history):
        batch, length, _, height, width = history.shape
        memory = history.new_zeros((batch, self.hidden, height, width))
        h = [memory, memory]
        c = [memory, memory]
        outputs = []
        for step in range(length + len(LEADS) - 1):
            x = history[:, step] if step < length else outputs[-1]
            frame = x
            for i, cell in enumerate(self.cells):
                h[i], c[i], memory = cell(x, h[i], c[i], memory)
                x = h[i]
            if step >= length - 1:
                outputs.append(frame + self.readout(x))
        return torch.stack(outputs, dim=1)


class GridLSTM(nn.Module):
    """Conventional LSTM that sees a flattened regional map."""
    def __init__(self, shape):
        super().__init__()
        n = math.prod(shape)
        self.cell = nn.LSTMCell(n, 24)
        self.readout = nn.Linear(24, n)

    def forward(self, history):
        state = None
        outputs = []
        length = history.shape[1]
        for step in range(length + len(LEADS) - 1):
            x = history[:, step] if step < length else outputs[-1]
            state = self.cell(x.flatten(1), state)
            if step >= length - 1:
                outputs.append(x + self.readout(state[0]).reshape_as(x))
        return torch.stack(outputs, dim=1)


def windows(frames):
    # Source is six-hourly. Issue daily at 00 UTC using three maps 12 h apart.
    issue = np.arange(4, len(frames) - 8, 4)
    return frames[issue[:, None] + np.array([-4, -2, 0])], frames[issue[:, None] + np.array([2, 4, 6, 8])]


class Diagnostics:
    def __init__(self, mean, std, lat, lon):
        self.mean, self.std = mean, std
        self.cos = torch.tensor(np.cos(np.deg2rad(lat)), dtype=torch.float32).view(-1, 1)
        self.dx = 6371000 * np.deg2rad(lon[1] - lon[0]) * self.cos[1:-1]
        self.dy = 6371000 * np.deg2rad(lat[1] - lat[0])
        self.weight = self.cos[1:-1] / self.cos[1:-1].mean()

    def dx_of(self, a):
        return (a[..., 1:-1, 2:] - a[..., 1:-1, :-2]) / (2 * self.dx)

    def dy_of(self, a):
        return (a[..., 2:, 1:-1] - a[..., :-2, 1:-1]) / (2 * self.dy)

    def budgets(self, x, y):
        physical = y * self.std + self.mean
        u, v, wa, w, wb, ta, t, tb, za, z, zb = physical.unbind(dim=2)
        center = (..., slice(1, -1), slice(1, -1))
        mass = (self.dx_of(u) + self.dy_of(v * self.cos) / self.cos[1:-1]
                + (wb - wa)[center] / 20000)
        # Dry static energy in J/kg. The apparent source includes unresolved
        # heating and work as well as interpolation and discretization effects.
        s = 1004.0 * t + z
        initial = x[:, -1] * self.std + self.mean
        previous = torch.cat(((1004.0 * initial[:, 6] + initial[:, 9]).unsqueeze(1), s[:, :-1]), 1)
        energy = ((s - previous)[center] / 43200 + u[center] * self.dx_of(s)
                  + v[center] * self.dy_of(s)
                  + w[center] * (1004.0 * (tb - ta) + zb - za)[center] / 20000)
        return torch.stack((mass, energy), dim=2)

    def wind_mse(self, y, pred):
        err = ((pred - y) * self.std)[:, :, :2, 1:-1, 1:-1]
        return (err.square().sum(2) * self.weight).mean((-2, -1))


@torch.no_grad()
def forecast(model, x):
    return torch.cat([model(batch) for batch in x.split(32)])


def fit(model, x, y, true_budget, val, diag, scale, lr, weights, epochs, seed,
        *, patience=None, history=None, paper_loss=False):
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    generator = torch.Generator().manual_seed(seed)
    best = float("inf")
    for epoch in range(1, epochs + 1):
        for idx in torch.randperm(len(x), generator=generator).split(32):
            aux = 0.
            if paper_loss and hasattr(model, "sequence"):
                prediction, decouple = model.sequence(x[idx], y[idx], (epoch-1)/max(epochs-1, 1))
                aux = .1 * decouple
            else:
                prediction = model(x[idx])
            loss = (prediction - y[idx]).square().mean()
            if paper_loss:
                loss = (loss + 1e-12).sqrt() + aux
            if any(weights):
                diff = (diag.budgets(x[idx], prediction) - true_budget[idx]) / scale
                penalties = (diff.square() * diag.weight).mean((0, 1, 3, 4))
                loss = loss + penalties @ torch.tensor(weights, dtype=penalties.dtype)
            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
        vp = forecast(model, val[0])
        score = diag.wind_mse(val[1], vp).mean(0).sqrt()[[1, 3]].mean().item()
        if not math.isfinite(score):
            raise ValueError(f"Nonfinite validation RMSE at epoch {epoch}")
        if history is not None:
            history.append(score)
        if score < best:
            best = score
            chosen_epoch = epoch
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if patience is not None and epoch - chosen_epoch >= patience:
            break
    return best, chosen_epoch, state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--out", type=Path, default=ROOT / "run")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("Use at least one epoch")
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    start = time.perf_counter()
    data = xr.load_dataset(ROOT / "regional_era5.nc", engine="scipy")
    maps = data[FIELDS].to_array("channel").transpose("time", "channel", "latitude", "longitude")
    dates = [("2016", "2017"), ("2018", "2018"), ("2019", "2019"), ("2020", "2020")]
    periods = [torch.from_numpy(maps.sel(time=slice(a, b)).values.copy()) for a, b in dates]
    mean = periods[0].mean((0, 2, 3))[:, None, None]
    std = periods[0].std((0, 2, 3))[:, None, None]
    diag = Diagnostics(mean, std, data.latitude.values, data.longitude.values)
    train, val, cal, test = [windows((p - mean) / std) for p in periods]
    budgets = diag.budgets(*train)
    scale = budgets.std((0, 1, 3, 4), keepdim=True).clamp_min(1e-12)
    # Each physics arm shares the same architecture, initial seed, ordering,
    # two learning rates and training budget. Weight magnitudes are prespecified.
    arms = {"Grid LSTM": (0., 0.), "PredRNN": (0., 0.), "Mass": (.1, 0.),
            "Energy": (0., .1), "Both": (.1, .1)}
    def build(name):
        return GridLSTM(train[0].shape[2:]) if name == "Grid LSTM" else PredRNN(len(FIELDS))
    trials, selected = [], {}
    for name, weights in arms.items():
        best = float("inf")
        for lr in [.001, .003]:
            torch.manual_seed(args.seed)
            model = build(name)
            score, epoch, state = fit(model, *train, budgets, val, diag, scale, lr, weights,
                                      args.epochs, args.seed)
            row = dict(arm=name, lr=lr, epoch=epoch, validation_wind_rmse=score)
            trials.append(row)
            print(row, flush=True)
            if score < best:
                best = score
                selected[name] = (row, state)
    # Freeze all choices before looking at either the calibration or test year.
    physics_name = min(("Mass", "Energy", "Both"), key=lambda n: selected[n][0]["validation_wind_rmse"])
    metrics = {}
    wind_case_mse = {}
    bundle = dict(fields=FIELDS, leads_hours=LEADS, mean=mean, std=std, budget_scale=scale,
                  selected_physics=physics_name, seed=args.seed, models={})
    for name in ["Persistence", *arms]:
        if name == "Persistence":
            cp, tp = [x[:, -1:].expand_as(y) for x, y in [cal, test]]
        else:
            model = build(name)
            model.load_state_dict(selected[name][1])
            cp, tp = forecast(model, cal[0]), forecast(model, test[0])
        # Calibration quantiles by lead, wind component and grid cell.
        ce = ((cp - cal[1]) * std)[:, :, :2, 1:-1, 1:-1]
        quantile_rank = min(len(ce), math.ceil(.90 * (len(ce) + 1)))
        radius = ce.abs().kthvalue(quantile_rank, dim=0).values
        te = ((tp - test[1]) * std)[:, :, :2, 1:-1, 1:-1]
        covered = (te.abs() <= radius).float()
        wm = diag.wind_mse(test[1], tp)
        wind_case_mse[name] = wm.numpy()
        resid = diag.budgets(test[0], tp) - diag.budgets(*test)
        budget_rmse = (resid.square() * diag.weight).mean((0, 3, 4)).sqrt()
        field_errors = ((tp - test[1]) * std)[:, :, :, 1:-1, 1:-1]
        field_rmse = (field_errors.square() * diag.weight).mean((0, 3, 4)).sqrt()
        metrics[name] = {}
        for j, lead in enumerate(LEADS):
            metrics[name][lead] = {
                "wind_vector_rmse_ms": wm[:, j].mean().sqrt().item(),
                "mass_residual_error_per_s": budget_rmse[j, 0].item(),
                "dse_tendency_error_W_per_kg": budget_rmse[j, 1].item(),
                "rmse_by_field_native_units": dict(zip(FIELDS, field_rmse[j].tolist())),
                "wind_interval_coverage": (covered[:, j] * diag.weight).mean((0, 2, 3)).tolist(),
                "wind_interval_width_ms": (2 * radius[j] * diag.weight).mean((1, 2)).tolist(),
            }
        bundle["models"][name] = {"interval_radius": radius}
        if name != "Persistence":
            bundle["models"][name].update(weights=selected[name][1], tuning=selected[name][0])
    rng = np.random.default_rng(args.seed)
    n = len(test[0])
    blocks = rng.integers(0, n, (1000, math.ceil(n / 7)))
    idx = ((blocks[:, :, None] + np.arange(7)) % n).reshape(1000, -1)[:, :n]
    gain = np.sqrt(wind_case_mse["PredRNN"][idx].mean(1)) - np.sqrt(wind_case_mse[physics_name][idx].mean(1))
    results = {"years": dates, "cases": [len(x) for x, _ in [train, val, cal, test]],
               "epochs_per_trial": args.epochs, "seed": args.seed, "trials": trials,
               "selected": {name: row for name, (row, _) in selected.items()},
               "physics_weights_mass_energy": arms, "selected_physics": physics_name,
               "test_scores": metrics, "paired_wind_gain_interval_95_ms_by_lead":
                   dict(zip(LEADS, np.quantile(gain, [.025, .975], axis=0).T.tolist())),
               "elapsed_seconds": time.perf_counter() - start}
    torch.save(bundle, args.out / "models.pt")
    (args.out / "results.json").write_text(json.dumps(results, indent=2))
    print("Finished", physics_name, results["elapsed_seconds"], flush=True)


if __name__ == "__main__":
    main()
