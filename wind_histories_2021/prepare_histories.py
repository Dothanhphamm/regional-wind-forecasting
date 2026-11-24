"""Build the continuous 2008–2020 regional development archive."""
import argparse
import sys
from pathlib import Path

import fsspec
import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT.parent))
from prepare_data import ARCHIVE, FIELDS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, help="Reuse the complete regional 2016–2020 pilot file")
    args = parser.parse_args()
    path = ROOT / "regional_2008_2020.nc"
    if path.exists():
        raise SystemExit(f"Already prepared: {path}")
    years = []
    with xr.open_zarr(fsspec.get_mapper(ARCHIVE), consolidated=True, chunks={}) as archive:
        for year in range(2008, 2016 if args.pilot else 2021):
            region = archive.sel(time=str(year), latitude=slice(29.375, 55.625),
                                 longitude=slice(264.375, 300.625))
            fields = {f"{prefix}{p}": region[var].sel(level=p, drop=True)
                      for var, (prefix, levels) in FIELDS.items() for p in levels}
            fields["surface_pressure"] = region.surface_pressure
            years.append(xr.Dataset(fields).transpose("time", "latitude", "longitude").compute(num_workers=4))
            print(f"Read {year}", flush=True)
    if args.pilot:
        years.append(xr.load_dataset(args.pilot, engine="scipy"))
    full = xr.concat(years, dim="time", join="exact", data_vars="all")
    expected = np.arange(np.datetime64("2008-01-01T00"), np.datetime64("2021-01-01T00"),
                         np.timedelta64(6, "h"))
    np.testing.assert_array_equal(full.time.values, expected)
    np.testing.assert_allclose(full.latitude.values, np.arange(5)*5.625 + 30.9375)
    np.testing.assert_allclose(full.longitude.values, np.arange(7)*5.625 + 264.375)
    assert np.isfinite(full.to_array().values).all() and full.surface_pressure.min().item() > 60000
    full.attrs = {"source": ARCHIVE, "role": "2008–2020 development and historical smoke-test weather"}
    tmp = path.with_suffix(".partial.nc")
    full.to_netcdf(tmp, engine="scipy")
    tmp.replace(path)
    print(path, dict(full.sizes))


if __name__ == "__main__":
    main()
