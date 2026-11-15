"""v0.4.0 members (from the QM target's previous month) vs the QM target and the uncorrected teacher, same cells,
by the model's starting class. Which one do the members' partly-open shares look like?"""
import glob, os, importlib.util
import numpy as np, pandas as pd, xarray as xr
import frazil.ledger as RL
from frazil import paths
spec = importlib.util.spec_from_file_location("sc", os.path.join(RL.ROOT, "scripts/evaluate/scorecard.py"))
sc = importlib.util.module_from_spec(spec); spec.loader.exec_module(sc)
aux = xr.open_dataset(paths.AUX); ocean = aux.mask.values.astype(bool); area = aux.cell_area.values / 1e6
sec = sc.sector_masks(aux.longitude.values, ocean)
R = {"Pacific": sec["East Siberian"] | sec["Chukchi"] | sec["Beaufort"], "Barents/Kara": sec["Barents/Kara"]}
def cube(d):
    fs = sorted(f for f in glob.glob(os.path.join(d, "*.nc")) if f[-7:-3].isdigit())
    return xr.open_mfdataset(fs, combine="by_coords")["datacube"].sel(var_names="sic")
qm, raw = cube(os.path.join(paths.TRAIN_DATA, "monthly_datacube_sicqmw9")), cube(paths.DATACUBE)
ct = pd.DatetimeIndex(qm.time.values); Z = 1e-4
rows = []
for run, fdir in (("2015-18", "one_month_fields"), ("2011-14", "one_month_fields_2011")):
    for f in sorted(glob.glob(os.path.join(paths.MODELS, "monthly_pf_qm_ema_qmw", fdir, "pred_??????.nc"))):
        y, m = int(f[-9:-5]), int(f[-5:-3])
        if m not in (8, 9): continue
        i = int(np.where((ct.year == y) & (ct.month == m))[0][0])
        s = np.nan_to_num(qm.isel(time=i - 1).values)
        ends = {"QM target": np.nan_to_num(qm.isel(time=i).values)[None], "raw teacher": np.nan_to_num(raw.isel(time=i).values)[None],
                "members": np.nan_to_num(xr.open_dataset(f).sic_members.values)}
        for r, mk in R.items():
            for cls, lo, hi in (("15-50%", .15, .5), ("50-80%", .5, .8)):
                c = mk & (s >= lo) & (s < hi)
                for k, e in ends.items():
                    e = e[:, c]
                    rows.append({"years": run, "region": r, "m": m, "cls": cls, "who": k, "w": area[c].sum(),
                                 "partly": np.average(((e > Z) & (e < .15)).mean(0), weights=area[c]),
                                 "zero": np.average((e <= Z).mean(0), weights=area[c]), "mean": np.average(e.mean(0), weights=area[c])})
D = pd.DataFrame(rows)
T = D.groupby(["years", "region", "m", "cls", "who"]).apply(lambda g: pd.Series({k: np.average(g[k], weights=g.w) for k in ("partly", "zero", "mean")})).unstack("who")
pd.set_option("display.width", 250)
print(T.round(3).to_string())
