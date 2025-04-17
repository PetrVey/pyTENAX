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
| Source code: https://zenodo.org/records/8345905




Important notes
--------------------------------
pyTENAX also includes SMEV class (Simplified Metastatistical Extreme Value)

| For more information about SMEV, please see manuscript:  
| Marra F, M Borga, E Morin, (2020). 
| A unified framework for extreme sub-daily precipitation frequency analyses based on ordinary events. 
| Geophys. Res. Lett., 47, 18, e2020GL090209. 
| https://doi.org/10.1029/2020GL090209 

| We have used pythonized version of SMEV code from:
| https://github.com/luigicesarini/pysmev 
| The original code of SMEV written in Matlab is available from:
| https://zenodo.org/records/11934843

