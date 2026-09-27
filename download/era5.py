"""Download the ERA5 monthly-mean fields preprocess_monthly.py turns into forcing.

The 5 raw fields (u10, v10, d2m, t2m, sp) over 40-90N; humidity, rotated winds and
degree-days are derived later. Include the year BEFORE the one you need: the yearly
degree-day features are a trailing 12-month mean and need that spin-up.

  Training forcing (original):   python download/era5.py --years 1994-2018
  Out-of-sample extension:       python download/era5.py --years 2018-2025
  Extra channels (sst, ssrd, strd for add_forcing_channels.py):
                                 python download/era5.py --extra --years 1995-2025

Writes to data/train_data/era5_<y0>_<y1>/ by default, so an extension never
overwrites the training file. Needs ~/.cdsapirc (see cdsapirc.example) and the
ERA5 licence accepted on the CDS website.
"""
import argparse
import os
import zipfile

import cdsapi

DATASET = "reanalysis-era5-single-levels-monthly-means"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--years", default="1994-2018", help="YYYY-YYYY, inclusive")
    ap.add_argument("--out-dir", default=None, help="default: data/train_data/era5[_extra]_<y0>_<y1>")
    ap.add_argument("--extra", action="store_true",
                    help="download sea-surface temperature + downward solar and thermal "
                         "radiation instead of the 5 core fields (no degree-day spin-up needed)")
    args = ap.parse_args()
    y0, y1 = (int(s) for s in args.years.split("-"))
    kind = "era5_extra" if args.extra else "era5"
    out_dir = args.out_dir or f"data/train_data/{kind}_{y0}_{y1}"
    os.makedirs(out_dir, exist_ok=True)

    request = {
        "product_type": ["monthly_averaged_reanalysis"],
        "variable": (["sea_surface_temperature", "surface_solar_radiation_downwards",
                      "surface_thermal_radiation_downwards"] if args.extra else
                     ["10m_u_component_of_wind", "10m_v_component_of_wind",
                      "2m_dewpoint_temperature", "2m_temperature", "surface_pressure"]),
        "year": [str(y) for y in range(y0, y1 + 1)],
        "month": [f"{m:02d}" for m in range(1, 13)],
        "time": ["00:00"],
        "data_format": "netcdf",
        "download_format": "zip",
        "area": [90, -180, 40, 180],
    }
    zpath = os.path.join(out_dir, f"{kind}_monthly_{y0}_{y1}.zip")
    cdsapi.Client().retrieve(DATASET, request).download(zpath)
    with zipfile.ZipFile(zpath) as z:
        z.extractall(out_dir)
        print(f"[era5] extracted {z.namelist()} -> {out_dir}/")


if __name__ == "__main__":
    main()
