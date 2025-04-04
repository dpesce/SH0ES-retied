# SH0ES retied

The code here repackages the SH0ES analysis published in [Riess et al. (2022)](https://ui.adsabs.harvard.edu/abs/2022ApJ...934L...7R/abstract), starting from the public data provided in the [associated GitHub repository](https://github.com/PantheonPlusSH0ES/DataRelease).

The datasets that have been incorporated into the code thus far include:

* Cepheid variable measurements (including full measurement covariance) from [Riess et al. (2022)](https://ui.adsabs.harvard.edu/abs/2022ApJ...934L...7R/abstract)
* SNe Ia measurements (including full measurement covariance) from [Riess et al. (2022)](https://ui.adsabs.harvard.edu/abs/2022ApJ...934L...7R/abstract)
* Megamaser measurements from [Pesce et al. (2020)](https://ui.adsabs.harvard.edu/abs/2020ApJ...891L...1P/abstract)
* Mira variable measurements from [Huang et al. (2018)](https://ui.adsabs.harvard.edu/abs/2018ApJ...857...67H/abstract), [Huang et al. (2020)](https://ui.adsabs.harvard.edu/abs/2020ApJ...889....5H/abstract), and [Huang et al. (2024)](https://ui.adsabs.harvard.edu/abs/2024ApJ...963...83H/abstract)
* TRGB measurements from [Anand et al. (2022)](https://ui.adsabs.harvard.edu/abs/2022ApJ...932...15A/abstract)

# Using the scripts

The main script to use is `model_SH0ES_plus.py`, which can just be run from the command line.  There are some flags in the script to control whether individual external datasets are included or not; at the moment, the SH0ES data are always included.

The script `model_SH0ES.py` reproduces the SH0ES analysis identically.

The script `model_SH0ES_reprior.py` repackages the SH0ES analysis to use explicit priors on certain parameters (e.g., NGC 4258 and LMC distance moduli, MW constraints on Cepheid PLR) rather than including the sorts of dummy parameters that were originally used in [Riess et al. (2022)](https://ui.adsabs.harvard.edu/abs/2022ApJ...934L...7R/abstract).  The resulting posterior should be near-identical to the exact SH0ES analysis.

