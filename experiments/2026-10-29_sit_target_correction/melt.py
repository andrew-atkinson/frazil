# Apr->Sep thickness loss on the shelves and pan-Arctic: targets (teacher vs sitra) and both hindcasts (2019-22)
import glob, os, numpy as np, pandas as pd, xarray as xr
from frazil import paths
aux = xr.open_dataset(paths.AUX); oc = aux.mask.values.astype(bool); area = aux.cell_area.values
lat, lon = aux.latitude.values, (aux.longitude.values + 360) % 360
REG = {"ESS+Chukchi": oc & (lat >= 64) & (lon >= 140) & (lon < 205), "Beaufort": oc & (lat >= 64) & (lon >= 205) & (lon < 240), "all": oc}
def mean(f, w): return float((np.nan_to_num(f) * area * w).sum() / (area * w).sum())
def row(name, get, years):
    r = {"source": name}
    for g, w in REG.items():
        for m in (4, 9):
            r[f"{g} {m}"] = np.mean([mean(get(y, m), w) for y in years])
        r[f"{g} loss"] = r[f"{g} 4"] - r[f"{g} 9"]
    return r
rows = []
for dc in ("monthly_datacube_sicqmw9", "monthly_datacube_sicqmw9_sitra"):
    c = xr.open_mfdataset(sorted(glob.glob(os.path.join(paths.TRAIN_DATA, dc, "*_datacube_*.nc"))), combine="by_coords")["datacube"].sel(var_names="sit")
    t = pd.DatetimeIndex(c.time.values)
    rows.append(row(f"targets {dc[17:] or 'teacher'} 1995-2014", lambda y, m: c.isel(time=int(np.where((t.year == y) & (t.month == m))[0][0])).values, range(1995, 2015)))
    rows.append(row(f"targets {dc[17:] or 'teacher'} 2015-18", lambda y, m: c.isel(time=int(np.where((t.year == y) & (t.month == m))[0][0])).values, range(2015, 2019)))
for mdl in ("monthly_pf_qm_ema_qmw", "monthly_pf_qm_ema_qmw_sit"):
    d = os.path.join(paths.SNAPSHOTS, f"{mdl}_era5_2018")
    rows.append(row(f"{mdl} 2019-22", lambda y, m: xr.open_dataset(f"{d}/state_{y}{m:02d}.nc").state.sel(var_names="sit").values.squeeze(), range(2019, 2023)))
pd.set_option("display.width", 250)
print(pd.DataFrame(rows).set_index("source").round(2).to_string())
