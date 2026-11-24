"""Stage one new ERA5 year. This step only checks the data schema and coverage."""
from pathlib import Path
import sys
import fsspec
import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT.parent))
from prepare_data import ARCHIVE, FIELDS


def main():
    path = ROOT / "holdout_2021.nc"
    if path.exists():
        raise SystemExit(f"Already staged: {path}")
    with xr.open_zarr(fsspec.get_mapper(ARCHIVE), consolidated=True, chunks={}) as archive:
        region = archive.sel(time="2021", latitude=slice(29.375, 55.625),
                             longitude=slice(264.375, 300.625))
        fields = {f"{name}{p}": region[variable].sel(level=p, drop=True)
                  for variable, (name, levels) in FIELDS.items() for p in levels}
        fields["surface_pressure"] = region.surface_pressure
        year = xr.Dataset(fields).transpose("time", "latitude", "longitude").compute(num_workers=4)
    expected = np.arange(np.datetime64("2021-01-01T00"), np.datetime64("2022-01-01T00"),
                         np.timedelta64(6, "h"))
    np.testing.assert_array_equal(year.time.values, expected)
    assert year.sizes["latitude"] == 5 and year.sizes["longitude"] == 7
    assert np.isfinite(year.to_array().values).all() and year.surface_pressure.min().item() > 60000
    year.attrs = {"source": ARCHIVE, "role": "2021 held-out test year"}
    tmp = path.with_suffix(".partial.nc")
    year.to_netcdf(tmp, engine="scipy")
    tmp.replace(path)
    print(path, year.sizes)


if __name__ == "__main__":
    main()
