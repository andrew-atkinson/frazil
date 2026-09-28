# Credits

Frazil is an adaptation of other people's work. This file lists what it is built on, how each source asks to be cited, and the part of each dataset this project uses. None of the data is included in the repository; the scripts in `scripts/download/` fetch it from the providers. Several providers make citation **a condition of use**, and that applies to anything made with this project, including published animations and trained models.

## Software

**GenSIM**, the generative sea-ice model this project adapts (the `gensim/` package, MIT licence, © 2025 Tobias Finn). Code: <https://github.com/cerea-daml/gensim>.

> Finn, T. S., Bocquet, M., Rampal, P., Durand, C., Porro, F., Farchi, A. & Carrassi, A. (2025). *Generative AI models capture realistic sea-ice evolution from days to decades*. arXiv:2508.14984. https://doi.org/10.48550/arXiv.2508.14984

## Data

### neXtSIM-OPA sea-ice simulation — the training targets

The model learns all of its sea-ice behaviour from this coupled ice–ocean simulation.

> Boutin, G., Ólason, E. Ö., Rampal, P., Regan, H., Lique, C., Talandier, C., Brodeau, L. & Ricker, R. (2023). Arctic sea ice mass balance in a new coupled ice–ocean model using a brittle rheology framework. *The Cryosphere*, 17, 617–638. https://doi.org/10.5194/tc-17-617-2023

Used: run `OPA-neXtSIM_CREG025-ILBOXE140`, monthly means, 1995–2018, from the SASIP THREDDS server (IGE-MEOM, Grenoble). The data accompanying the paper is also on Zenodo under CC BY 4.0 (https://doi.org/10.5281/zenodo.7277523). *Check the SASIP server's terms before redistributing any of its files.*

### ERA5 reanalysis — atmospheric forcing

> Copernicus Climate Change Service (C3S) (2019). ERA5 monthly averaged data on single levels from 1940 to present. Copernicus Climate Change Service (C3S) Climate Data Store (CDS). https://doi.org/10.24381/cds.f17050d7

> Hersbach, H., Bell, B., Berrisford, P., et al. (2020). The ERA5 global reanalysis. *Quarterly Journal of the Royal Meteorological Society*, 146, 1999–2049. https://doi.org/10.1002/qj.3803

Licence: CC BY. Required acknowledgement: **"Generated using Copernicus Climate Change Service information [year]."**

Used: monthly averaged reanalysis, 40–90° N. 2 m temperature, 2 m dewpoint, surface pressure and 10 m winds for 1994–2025; sea-surface temperature and surface downward solar and thermal radiation for 1995–2025.

### NSIDC-0051 sea-ice concentration — observations

> DiGirolamo, N., Parkinson, C. L., Cavalieri, D. J., Gloersen, P. & Zwally, H. J. (2022). *Sea Ice Concentrations from Nimbus-7 SMMR and DMSP SSM/I-SSMIS Passive Microwave Data*. (NSIDC-0051, Version 2). [Data Set]. Boulder, Colorado USA. NASA National Snow and Ice Data Center Distributed Active Archive Center. https://doi.org/10.5067/MPYG15WAA4WX

Used: Northern Hemisphere, monthly, 1995–2025, regridded to the model grid. Fits the quantile-mapped targets (1995–2018) and scores the 2019–2025 hindcast.

### NSIDC Sea Ice Index — observations

> Fetterer, F., Knowles, K., Meier, W. N., Savoie, M., Windnagel, A. K. & Stafford, T. (2025). *Sea Ice Index*. (G02135, Version 4). [Data Set]. Boulder, Colorado USA. National Snow and Ice Data Center. https://doi.org/10.7265/a98x-0f50

Used: Northern Hemisphere monthly extent and area files, all 12 months, 1979–2025. Evaluation only.

### CMIP6 MPI-ESM1-2-LR — future forcing

> Wieners, K.-H., Giorgetta, M., Jungclaus, J., Reick, C., Esch, M., Bittner, M., et al. (2019). *MPI-M MPI-ESM1.2-LR model output prepared for CMIP6 CMIP historical*. Earth System Grid Federation. https://doi.org/10.22033/ESGF/CMIP6.6595

> Wieners, K.-H., Giorgetta, M., Jungclaus, J., Reick, C., Esch, M., Bittner, M., et al. (2019). *MPI-M MPI-ESM1.2-LR model output prepared for CMIP6 ScenarioMIP ssp245*. Earth System Grid Federation. https://doi.org/10.22033/ESGF/CMIP6.6693

Licence: CC BY 4.0 (full author lists at the DOI pages). Used: member r1i1p1f1, monthly atmosphere (Amon), native grid, variables `tas huss uas vas`. Historical is used up to 2014 (for the bias correction against ERA5), ssp245 from 2015 to 2100.

## Suggested credit line (exhibitions, videos, stills)

> Sea-ice model: Frazil by Andrew Atkinson, adapted from GenSIM (Finn et al., 2025). Trained on the neXtSIM-OPA simulation (Boutin et al., 2023) with ERA5 forcing; generated using Copernicus Climate Change Service information [year]. Observations: NSIDC. Future climate: CMIP6 MPI-ESM1-2-LR, SSP2-4.5.
