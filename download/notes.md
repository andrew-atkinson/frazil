
## Sources

The 6 sea-ice state variables it predicts — sit (thickness), sic (concentration), sid (damage), siu/siv (drift), snt (snow-on-ice) — come from neXtSIM-OPA, a sea-ice model (the Moorings_YYYYmMM.nc files). These are the training targets/truth. ERA5 supplies only the atmospheric forcing (temperature, humidity, winds).

### ERA5 

What ERA5 parameters you actually need

The pipeline downloads 5 raw ERA5 fields and derives everything else:

Download from ERA5	Used to produce 2 m temperature	tus + all 4 degree-day features (pdd_month, fdd_month, pdd_year, fdd_year) 2 m dewpoint temperature	huss (specific humidity), rhus (relative humidity) surface pressure	needed for the specific-humidity calc 10 m u-component of wind	uas (after rotation to the grid) 10 m v-component of wind	vas (after rotation to the grid)

So you do not download degree-days or humidity — they're computed. Note the model at inference only consumes tus, huss, uas, vas + the 4 degree-days; the raw list above is what the preprocessing needs to build those.


## The real source: SASIP THREDDS server

The neXtSIM-OPA simulation GenSIM was trained on is **`OPA-neXtSIM_CREG025-ILBOXE140`**, hosted on the IGE-MEOM THREDDS server in Grenoble. It's served as **per-file NetCDF over OPeNDAP/HTTP**, *not* a Zarr store — so `xr.open_zarr` will never work here regardless of the URL.

**Catalog (browse it in a browser):** `https://ige-meom-opendap.univ-grenoble-alpes.fr/thredds/catalog/meomopendap/extract/SASIP/model-outputs/OPA-neXtSIM_CREG025/catalog.html`

Under it, the time-averaged tree has exactly what you want: `OPA-neXtSIM_CREG025-ILBOXE140-MEAN/` → **`1m/`** (monthly), `1d/` (daily), `6h/` (6-hourly), organized `1m/<year>/`.

**The monthly sea-ice files** are `*_icemod.nc` (thickness, concentration, **damage**, u/v drift) and `*_simba.nc` (snow/ice thermodynamics). A verified real filename:

```
OPA-neXtSIM_CREG025-ILBOXE140_y2015m12.1m_icemod.nc
```

Which gives these two working URL forms:

- **OPeNDAP** (open remotely, subset server-side): replace `catalog/…/catalog.html` with `dodsC/…/<file>` → `.../thredds/dodsC/meomopendap/extract/SASIP/model-outputs/OPA-neXtSIM_CREG025/OPA-neXtSIM_CREG025-ILBOXE140-MEAN/1m/2015/OPA-neXtSIM_CREG025-ILBOXE140_y2015m12.1m_icemod.nc` → `xr.open_dataset(url)`
- **Direct download**: same but `fileServer/…` instead of `dodsC/…`.

There's also a `dl_data.sh` helper and a `README` in the root folder. Contact for the dataset: **Guillaume Boutin (NERSC)**; citation DOI **[10.5281/zenodo.7277523](https://doi.org/10.5281/zenodo.7277523)** (Boutin et al., 2023, *The Cryosphere*).

## Three corrections this forces on your plan

1. **Time coverage is 1995–2018 — full stop.** The simulation is only ~24 years. Your `slice('1970-01-01','2025-12-31')` (and the 1940–2025 idea) can't be met from neXtSIM-OPA; it doesn't exist outside 1995–2018. If you truly need 1940–2025, that has to come from a *different* product (ERA5 sea-ice cover, reanalyses) — but then you lose damage/thickness/drift, as we discussed.

2. **It's `open_dataset` over OPeNDAP, not `open_zarr`.** Your line 8 must change. And because it's one file per month per type, you either loop over years/months and `open_mfdataset`, or download the `_icemod.nc`/`_simba.nc` files.

3. **The grid is curvilinear CREG025 (~¼° ORCA-type), not lat/lon.** Coordinates are 2-D `nav_lat`/`nav_lon` over `x`/`y`, so `.sel(lat=…, lon=…)` fails (your line-16 comment was right to worry). CREG025 is already an Arctic regional domain, so you may not need to subset spatially at all — and note this is the model's **native** grid, *different* from GenSIM's 512×512 polar-stereographic grid, so the regrid+wind-rotation step from `data_02` still applies.

## Want me to rewrite `download.py`?

I can turn it into a working monthly fetcher: loop 1995–2018, open the `_icemod.nc` (and `_simba.nc`) files over OPeNDAP, select the variables that map to `sit/sic/sid/siu/siv/snt`, and save yearly NetCDFs — with the time range and access method corrected. Before I do, one useful check: I can open one `_icemod.nc` remotely and print its actual variable names so we map them correctly (neXtSIM uses names like `sit`, `sic`, `damage`, `siu`, `siv`). Say the word and I'll verify the variables and rewrite the script.

**Sources:**
- [arXiv:2508.14984 — Generative AI models capture realistic sea-ice evolution](https://arxiv.org/abs/2508.14984)
- [SASIP catalog entry: OPA-neXtSIM_CREG025-ILBOXE140](https://github.com/sasip-climate/catalog-shared-data-SASIP/blob/main/outputs/OPA-neXtSIM_CREG025-ILBOXE140.md)
- [IGE-MEOM THREDDS catalog (OPA-neXtSIM_CREG025)](https://ige-meom-opendap.univ-grenoble-alpes.fr/thredds/catalog/meomopendap/extract/SASIP/model-outputs/OPA-neXtSIM_CREG025/catalog.html)
- [SASIP project (CMCC)](https://www.cmcc.it/projects/sasip-the-scale-aware-sea-ice-project) · [sasip-climate on GitHub](https://github.com/sasip-climate)
- [neXtSIM-OPA dataset DOI (Zenodo 7277523)](https://doi.org/10.5281/zenodo.7277523)


### variables

All six GenSIM targets are present in a **single** file type — `_icemod.nc` — so we don't even need `simba`:

| GenSIM target | Variable in file | Long name              | Units |
|---------------|------------------|------------------------|-------|
| `sit`         | `sit`            | Sea Ice Thickness      | m     |
| `sic`         | `sic`            | Sea Ice Concentration  | 1     |
| `sid`         | **`damage`**     | Sea Ice Damage         | 1     |
| `siu`         | `siu`            | Sea Ice X Velocity     | m/s   |
| `siv`         | `siv`            | Sea Ice Y Velocity     | m/s   |
| `snt`         | `snt`            | Surface Snow Thickness | m     |

Only rename needed: **`damage` → `sid`**. Grid is native curvilinear CREG025 (603×528, 2-D `latitude`/`longitude`), one timestep per monthly file. OPeNDAP works and selecting the 6 vars keeps the transfer small.



### Grid Dimensions

The grid is clear: 512×512 North Polar Stereographic, ~12.5 km, spanning 41.6°N–90°N, full −180…+180 longitude, ~48% ocean. Now let me confirm the data provenance and how forcings/degree-days are built, from the preprocessing notebooks.


## Config file

`.cdsapirc` is located in the home directory.

