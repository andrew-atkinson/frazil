"""Are ice drift (siu, siv) and wind (uas, vas) in the same frame? Free drift: ice moves at ~1-3% of
wind speed, turned ~20-40 deg to the right of the wind in the Arctic. A missed or wrong rotation of
either shows up as an angle that varies with longitude."""
import glob, os
import numpy as np, xarray as xr, pandas as pd
from frazil import paths
aux = xr.open_dataset(paths.AUX); ocean = aux.mask.values.astype(bool); lon = aux.longitude.values; lat = aux.latitude.values
fs = sorted(f for f in glob.glob(os.path.join(paths.DATACUBE, "*.nc")) if f[-7:-3].isdigit() and 2005 <= int(f[-7:-3]) <= 2014)
da = xr.open_mfdataset(fs, combine="by_coords")["datacube"]
print(list(da.var_names.values))
win = da.time.dt.month.isin([11, 12, 1, 2, 3])
g = lambda v: np.nan_to_num(da.sel(var_names=v).sel(time=win).values)
u, v, U, V, c = g("siu"), g("siv"), g("uas"), g("vas"), g("sic")
ice = (c > 0.9) & ocean[None]
ang = np.degrees(np.angle((u + 1j * v) / (U + 1j * V + 1e-9)))   # drift direction relative to wind, + = left
ratio = np.hypot(u, v) / (np.hypot(U, V) + 1e-9)
strong = ice & (np.hypot(U, V) > 3)
rows = []
for lo in range(0, 360, 45):
    k = strong & (((lon + 360) % 360 >= lo) & ((lon + 360) % 360 < lo + 45))[None] & (lat > 70)[None]
    if k.sum() < 100: continue
    a = ang[k]; mean_dir = np.degrees(np.angle(np.exp(1j * np.radians(a)).mean()))
    rows.append({"lon": f"{lo}-{lo+45}E", "n": int(k.sum()), "drift turned vs wind (deg, + left)": round(mean_dir, 1),
                 "spread R": round(float(abs(np.exp(1j*np.radians(a)).mean())), 2), "speed ratio %": round(100*np.median(ratio[k]), 2)})
print(pd.DataFrame(rows).to_string(index=False))
