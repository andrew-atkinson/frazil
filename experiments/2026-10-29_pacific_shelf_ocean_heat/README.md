# The Pacific-shelf excess as an ocean-heat problem

**Date:** 2026-10-29 · **Branch:** `exp/pacific-shelf-ocean-heat` · **Status:** step 1 done · **Issues:** ISS-002, ISS-028

## Question
[regional_model_vs_target](../2026-10-25_regional_model_vs_target/) showed the September excess in the East Siberian, Chukchi and Beaufort seas is Frazil's own error, not its training target's: in the East Siberian and Beaufort seas the ice thickens beyond its target over 4–5 years, in the Chukchi there is too much ice from the first year. Treated first as an oceanography question: what does the literature say drives sea ice in these seas that Frazil can't see?

## What the literature says
Local copies are in the gitignored `literature/` folder (manifest: `literature/index.md`).

1. **Pacific Water through the Bering Strait melts Chukchi and Beaufort ice, and has increased.** Mooring records 1991–2021: warming of about 0.03 °C per year and a flow increase of about 0.01 Sv per year, together adding about 6×10¹⁸ J per year to the heat entering the Arctic ([Woodgate & Peralta-Ferriz](https://par.nsf.gov/search/author:%22Woodgate,%20Rebecca%20A%22); [Frontiers 2021](https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2021.650909/full)). In 2007 the extra inflow could melt about 0.5 m of ice over the Chukchi Sea, and the heat stayed against the ice in the Chukchi–Beaufort region from July to September ([Arctic Portal](https://library.arcticportal.org/2352/)). One climate-model evaluation attributes about two-thirds of Beaufort and Chukchi ice loss to ocean heat fluxes ([US CLIVAR poster](https://usclivar.org/sites/default/files/2022/posters/Alberty_Marion_Poster.pdf)). Warm Pacific water delays freeze-up by weeks independently of the air ([U Tokyo](https://www.u-tokyo.ac.jp/focus/en/press/z0508_00154.html); [AGU abstract](https://agu.confex.com/agu/osm20/preliminaryview.cgi/Paper658200.html)); storm mixing of it raises basal melt by 0.1–1.7 cm per day ([Wang et al.](https://par.nsf.gov/search/author:%22Wang,%20Zhaomin%22)).
2. **Heat stored below the surface slows winter ice growth from year to year, especially in the Beaufort.** Summer sunlight absorbed by open water is stored in the near-surface temperature maximum below the mixed layer and mixed up in autumn and winter; after 2007 it cut end-of-winter thickness by about 25% ([Timmermans 2015](https://earth.yale.edu/sites/default/files/2025-04/Timmerman%2043.pdf); [Smith et al. 2018](https://data-ww3.ifremer.fr/BIB/Smith_etal_JGR2018.pdf)). The heat in the Beaufort Gyre's Pacific layer has nearly doubled in 30 years ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8065036); [WHOI](https://www2.whoi.edu/site/beaufortgyre/results/beaufort-gyre-heat-content-increase)).
3. **River heat on the Siberian shelves.** Up to 10% of shelf ice loss 1980–2015, most on the Siberian shelves, through earlier break-up whose effect roughly doubles itself ([Park et al. 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7673719)). The East Siberian Sea accounts for the largest share of September loss, 22% ([The Cryosphere 2024](https://tc.copernicus.org/articles/18/3559/2024/)).
4. **The Beaufort depends on what wind and currents bring in.** Thick old ice is imported, but less of it survives the summer; annual loss there quadrupled since 1997 ([Babb et al. 2022](https://essopenarchive.org/doi/full/10.1002/essoar.10509833.1); [2020–21 transport](https://par.nsf.gov/biblio/10403817)). The Beaufort was low in September 2024 and 2025; in 2025 from warm air and winds melting thin ice in August ([NSIDC](https://nsidc.org/sea-ice-today/analyses/taking-bite-out-beaufort); [NOAA Arctic Report Card 2025](https://arctic.noaa.gov/report-card/report-card-2025/sea-ice-2025/)).

## How it matches what we measured
| what we found | oceanographic explanation |
| --- | --- |
| The Pacific-shelf excess is the model's own; the target is right | the target comes from an ice–ocean model (neXtSIM coupled to the OPA ocean) that includes this heat; Frazil sees only the ice and the air |
| East Siberian and Beaufort: ice **thickens** beyond the target over 4–5 years | subsurface and Pacific-layer heat slows winter growth and melts the ice base: a few millimetres a month, **accumulating in thickness** |
| Chukchi: excess **from the first year**, no thickening | Bering Strait heat arrives fresh every summer and melts directly; seasonal, so nothing carries over |
| Appears in **August–September** | when stored and imported ocean heat is in contact with the ice |
| Worse in **later years** (2023–25) | the ocean heat has been **increasing**; a model trained on 1995–2014 learned that period's average |

Frazil knows where each cell is, so it can learn an average regional correction, but not ocean heat that varies and trends over time. The earlier [ocean_memory](../2026-10-21_ocean_memory/) test probably missed this: it tested next-month concentration with this summer's local sunlight, the wrong quantity (the heat is subsurface, built up over seasons, partly advected) and the wrong variable (it shows in thickness), with too little power for a bias of millimetres a month.

## Step 1: test it against the training simulation's own ocean (no training)
neXtSIM-OPA's monthly ocean output (SASIP server, 1995–2018) includes water temperature on 75 levels, 24 of them in the top 100 m.

- **Data:** `scripts/download/nextsim_ocean_heat.py` streams that temperature for the Pacific and Siberian shelf seas (north of 64° N, 90° E to 110° W) and keeps only the ocean heat content above freezing in two layers, 0–20 m (the mixed layer) and 20–100 m (where the stored summer heat and Pacific water sit): `data/train_data/nextsim_ocean_heat/` (~0.1 GB kept; ~2–3 GB streamed).
- **Analysis** (`analysis.py`, this folder), per sector (East Siberian, Chukchi, Beaufort; Laptev as a contrast):
  1. the training simulation's ocean heat by month and its trend 1995–2018;
  2. whether, year by year in 2009–18, Frazil's own error in a sector (model − target September extent and June thickness, from the drift runs with fields) is larger when the simulation's ocean heat there is larger.

**Reading it:** the hypothesis is supported if the Pacific-shelf seas carry substantial and rising subsurface heat (more than the Laptev), and Frazil's own regional error tracks it. If not, ocean heat isn't the missing piece, and the next candidates are ice import by winds (item 4) and river heat (item 3).

## Commands
```bash
python scripts/download/nextsim_ocean_heat.py --dry-run          # one month: box, levels, sizes
python scripts/download/nextsim_ocean_heat.py                    # 1995-2018, resumable by year
python experiments/2026-10-29_pacific_shelf_ocean_heat/analysis.py
```

## Results
`nextsim_ocean_heat.py` streamed all 288 months (84 MB kept); `analysis.py` against `monthly_pf_qm_ema_qmw`'s drift runs with fields.

**A. The training simulation's subsurface ocean heat (20–100 m) is largest and rising fastest exactly where Frazil is too icy.**

| sector | 20–100 m heat, MJ m⁻² (mean) | trend per decade | 1995–2014 (training) → 2009–18 (runs) | Frazil's own error, Sept extent / June thickness |
| --- | --- | --- | --- | --- |
| Beaufort | 184 | **+38 ± 5** | 177 → 215 (+21%) | +0.11 M km² / +0.09 m |
| Chukchi | 177 | **+25 ± 6** | 173 → 191 (+11%) | +0.12 M km² / +0.01 m |
| East Siberian | 97 | +13 ± 4 | 95 → 100 (+5%) | +0.09 M km² / +0.07 m |
| Laptev (contrast) | 108 | −2 ± 4 | 110 → 103 (−6%) | −0.04 M km² / +0.03 m |

The mixed layer (0–20 m) warms by a similar +6 to +13 per decade in all four, so it doesn't separate them; the subsurface does. The subsurface heat peaks in September–October (Chukchi 355, Beaufort 264 MJ m⁻²), i.e. late in the melt season, as stored summer and Pacific heat.

**Order of magnitude.** Melting ice takes about 306 MJ per m³ (density × latent heat), so 31 MJ m⁻² melts 0.1 m. The Beaufort's subsurface heat in 2009–18 is about 37 MJ m⁻² above the training years: about 0.12 m of ice if delivered to it over a year, the same order as Frazil's Beaufort thickness drift (+0.09 m on average, +0.15–0.2 m after 5–8 years). For the East Siberian Sea (+5 MJ m⁻² ≈ 0.02 m) it accounts for much less than the +0.07 m drift.

**B. Year to year, no clear link.** With the growth over a run removed, the correlation between a sector's yearly error and that year's ocean heat is weak (|r| ≤ 0.44 in the Pacific sectors, pooled +0.18 to +0.28, not significant); the Laptev shows −0.66 to −0.68 (fewer than 10 independent years; opposite in sign to the hypothesis). Frazil's error is accumulated thickness, which integrates several years, so a one-year link was always going to be weak.

## Conclusion
Supported where it matters most, not proven. Across regions, Frazil's own error follows the training simulation's subsurface ocean heat: too icy where that heat is high and rising (Beaufort, Chukchi), not where it isn't (Laptev). In the Beaufort the size matches too. Frazil learned the ocean's effect as it was in 1995–2014 and has no way to see it grow. The East Siberian Sea fits only partly: its subsurface heat changed little, so something else probably adds to it (river heat or ice import, items 3–4 of the literature). Year-to-year variability doesn't show the link.

**Next:** give Frazil ocean information, starting with subsurface heat (20–100 m), either as an extra state it predicts (it carries the ocean's memory itself; works for hindcasts and projections without outside data) or as an input from an ocean reanalysis and climate models (simpler to learn, but needs a leak test). Either needs the heat content over the whole model domain from the training simulation (~9 GB to stream, ~1 GB kept). For the East Siberian Sea, check river heat and ice import separately.
