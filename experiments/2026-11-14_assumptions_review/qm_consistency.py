"""Do the QM targets stay physically consistent? The QM correction changes sic only: how often is sic 0 with
thickness left, and how does the implied per-ice thickness move; and the 0-15% band in teacher, QM and NSIDC-0051.
July-September 1995-2014, Pacific shelves and Barents/Kara. Run from the repo root."""
import glob, os, importlib.util
import numpy as np, pandas as pd, xarray as xr
import frazil.ledger as RL
from frazil import paths
spec = importlib.util.spec_from_file_location("sc", os.path.join(RL.ROOT, "scripts/evaluate/scorecard.py"))
sc = importlib.util.module_from_spec(spec); spec.loader.exec_module(sc)
aux = xr.open_dataset(paths.AUX); ocean = aux.mask.values.astype(bool); area = aux.cell_area.values/1e6
sec = sc.sector_masks(aux.longitude.values, ocean)
R = {"Pacific": sec["East Siberian"] | sec["Chukchi"] | sec["Beaufort"], "Barents/Kara": sec["Barents/Kara"]}
def cube(d, var):
    fs = sorted(f for f in glob.glob(os.path.join(d, "*.nc")) if f[-7:-3].isdigit() and 1995 <= int(f[-7:-3]) <= 2014)
    da = xr.open_mfdataset(fs, combine="by_coords"); n = "datacube" if "datacube" in da else "state"
    return da[n].sel(var_names=var)
raw_c, qm_c = cube(paths.DATACUBE, "sic"), cube(os.path.join(paths.TRAIN_DATA, "monthly_datacube_sicqmw9"), "sic")
raw_t, qm_t = cube(paths.DATACUBE, "sit"), cube(os.path.join(paths.TRAIN_DATA, "monthly_datacube_sicqmw9"), "sit")
ob = cube(paths.NSIDC0051_GRID, "sic")
print("sit identical in raw and QM cubes:", bool(np.allclose(np.nan_to_num(raw_t.isel(time=200).values), np.nan_to_num(qm_t.isel(time=200).values))))
rows = []
for m in (7, 8, 9):
    sel = lambda da: np.nan_to_num(da.sel(time=da.time.dt.month == m).values)
    c_raw, c_qm, t, o = sel(raw_c), sel(qm_c), sel(raw_t), sel(ob)
    for r, mk in R.items():
        w = np.broadcast_to(area * mk, c_raw.shape)
        def share(x, lo, hi): return (w * ((x > lo) & (x <= hi))).sum() / w.sum()
        rows.append({"m": m, "region": r,
            "raw 0<c<15": share(c_raw, 1e-4, .15), "QM 0<c<15": share(c_qm, 1e-4, .15), "NSIDC 0<c<15": share(o, 1e-4, .15),
            "raw c=0": share(c_raw, -1, 1e-4), "QM c=0": share(c_qm, -1, 1e-4), "NSIDC c=0": share(o, -1, 1e-4),
            # thickness per unit ice: sit/sic where sic>0.15, raw vs after QM changed sic
            "QM c=0 but sit>0.1m": (w * ((c_qm <= 1e-4) & (t > 0.1))).sum() / w.sum(),
            "QM c<15 & raw c>50": (w * ((c_qm < .15) & (c_raw > .5))).sum() / w.sum(),
            "med sit/sic raw (c>15)": np.median((t / np.maximum(c_raw, 1e-3))[(c_raw > .15) & mk[None]]),
            "med sit/sic QM (c>15)": np.median((t / np.maximum(c_qm, 1e-3))[(c_qm > .15) & mk[None]])})
pd.set_option("display.width", 250)
print(pd.DataFrame(rows).set_index(["region", "m"]).sort_index().round(3).to_string())
# sit grid-cell-mean or per-ice? sit for raw cells with sic in 5-15% vs 90-100%
c, t = np.nan_to_num(raw_c.sel(time=raw_c.time.dt.month == 8).values), np.nan_to_num(raw_t.sel(time=raw_t.time.dt.month == 8).values)
for lo, hi in ((.05, .15), (.3, .5), (.9, 1.01)):
    k = (c >= lo) & (c < hi) & ocean[None]; print(f"raw Aug sic {lo}-{hi}: median sit {np.median(t[k]):.2f} m, median sit/sic {np.median(t[k]/c[k]):.2f}")
