Overview
============

What is pyTENAX?
------------------

**PyTENAX** contains a set of methods to apply The TEmperature-dependent Non-Asymptotic statistical model for eXtreme return levels (TENAX).

pyTENAX is essentially the Pythonized version of the TENAX MATLAB code for this model.
The link to original repository in `Cross-Language Implementations`_.

The model is based on a parsimonious non-stationary and non-asymptotic theoretical framework that 
incorporates temperature as a covariate to estimate changes in precipitation return levels.

| The model is presented in: 
| Marra, F., Koukoula, M., Canale, A., & Peleg, N. (2023).
| Predicting extreme sub-hourly precipitation intensification based on temperature shifts. 
| Hydrology and Earth System Sciences Discussions, 2023, 1-23.
| https://doi.org/10.5194/hess-28-375-2024

.. _cross-language-implementations:

Cross-Language Implementations
--------------------------------

| Original TENAX model has been developed in MATLAB:
| TEmperature-dependent Non-Asymptotic statistical model for eXtreme return levels (TENAX)
| Source code: https://doi.org/10.5281/zenodo.8332232

+-----------------------------------+------------------------+------------------------+
| **Feature**                       | **PYTHON**             | **MATLAB**             |
+===================================+========================+========================+
| **Ordinary Events**               | ✅                     | ✅                     |
+-----------------------------------+------------------------+------------------------+
| **Magnitude Model**               |                        |                        |
+-----------------------------------+------------------------+------------------------+
| • Estimate 4 parameters           | ✅                     | ✅                     |
+-----------------------------------+------------------------+------------------------+
| • Alpha value test on scale param.| ✅                     | ✅                     |
+-----------------------------------+------------------------+------------------------+
| • Fixed-b parameter estimation    | ✅                     | ❌                     |
+-----------------------------------+------------------------+------------------------+
| **Temperature Model**             |                        |                        |
+-----------------------------------+------------------------+------------------------+
| • Generalized Gaussian dist.      | ✅                     | ✅                     |
+-----------------------------------+------------------------+------------------------+
| • Skewed Normal (β fitted)        | ✅ (beta testing)      | ❌                     |
+-----------------------------------+------------------------+------------------------+

Developer community
--------------------------------
Current pyTENAX developers:
- Petr Vohnicky (PhD student at the University of Padova; petr.vohnicky@unipd.it)
- Ella Thomas (Research Assistant at the University of Padova)
- Jannis Hoch (Senior Hydrologist at Fathom)
- Rashid Akbary (PhD student at the University of Padova)

We would like to express our gratitude to Riccardo Ciceri (riccardo.ciceri@studenti.unipd.it) for his contribution to the initial development phase of pyTENAX.


Important notes
--------------------------------
pyTENAX also includes SMEV class (Simplified Metastatistical Extreme Value)

| For more information about SMEV, please see manuscripts:
| Francesco Marra, Davide Zoccatelli, Moshe Armon, Efrat Morin.
| A simplified MEV formulation to model extremes emerging from multiple nonstationary underlying processes.
| Advances in Water Resources, 127, 280-290, 2019
| https://doi.org/10.1016/j.advwatres.2019.04.002
  
| Francesco Marra, Marco Borga, Efrat Morin.
| A unified framework for extreme sub-daily precipitation frequency analyses based on ordinary events. 
| Geophys. Res. Lett., 47, 18, e2020GL090209. 2020.
| https://doi.org/10.1029/2020GL090209 

| We have used pythonized version of SMEV code from:
| https://github.com/luigicesarini/pysmev 
| The original code of SMEV written in Matlab is available from:
| https://zenodo.org/records/11934843

