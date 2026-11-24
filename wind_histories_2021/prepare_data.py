"""Extract the regional ERA5 fields used by the PredRNN pilot."""
from pathlib import Path

import fsspec
import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parent
ARCHIVE = ("https://storage.googleapis.com/weatherbench2/datasets/era5/"
           "1959-2023_01_10-6h-64x32_equiangular_conservative.zarr")
FIELDS = {
    "u_component_of_wind": ("u", [500]),
    "v_component_of_wind": ("v", [500]),
    "vertical_velocity": ("omega", [400, 500, 600]),
    "temperature": ("T", [400, 500, 600]),
    "geopotential": ("phi", [400, 500, 600]),
}


def main():
    path = ROOT / "regional_era5.nc"
    if path.exists():
        raise SystemExit(f"Data are already present at {path}")
    with xr.open_zarr(fsspec.get_mapper(ARCHIVE), consolidated=True, chunks={}) as raw:
        region = raw.sel(time=slice("2016", "2020"), latitude=slice(29.375, 55.625),
                         longitude=slice(264.375, 300.625))
        fields = {}
        for variable, (prefix, levels) in FIELDS.items():
            for level in levels:
                fields[f"{prefix}{level}"] = region[variable].sel(level=level, drop=True)
        fields["surface_pressure"] = region.surface_pressure
        data = xr.Dataset(fields).transpose("time", "latitude", "longitude").compute(num_workers=4)
    assert data.sizes == {"time": 7308, "latitude": 5, "longitude": 7}
    assert np.all(np.diff(data.time.values) == np.timedelta64(6, "h"))
    assert np.isfinite(data.to_array().values).all()
    assert float(data.surface_pressure.min()) > 60000, "Pressure levels intersect terrain"
    data.attrs = {"source": ARCHIVE, "pressure_levels_hPa": "400, 500, 600",
                  "purpose": "Great Lakes and Northeast regional PredRNN pilot"}
    data.to_netcdf(path, engine="scipy")
    print(path, data.sizes, flush=True)


if __name__ == "__main__":
    main()
