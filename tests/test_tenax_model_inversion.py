# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 14:58:31 2025

@author: Petr
"""
import pandas as pd
import numpy as np
from scipy.stats import gennorm
from src.pyTENAX import smev, tenax, plotting
from scipy.optimize import root_scalar, minimize
from typing import List, Tuple, Union, Optional

S = tenax.TENAX(
    return_period=[2, 5, 10, 20, 50, 100, 200],
    durations=[10,],
    time_resolution=5,
    left_censoring=[0, 0.90],
    alpha=0.05,
)


np.random.seed(42)
n_samples = 2000
# Generate T first
temperature_values = gennorm.rvs(beta=4, loc=10, scale=11.5, size=n_samples)
temperature_values = np.clip(temperature_values, -10, 50)
# Generate P depending on T
# For example: higher T → slightly higher scale of Weibull
shape = 0.7
base_scale = 2
# Scale P proportionally to T (e.g., scale = base_scale + k*T)
k = 0.2
scales = base_scale + k * temperature_values
pr_values = scales * np.random.weibull(shape, size=n_samples)
pr_values = np.clip(pr_values, 0.2, 300)
# Random years
years = np.random.randint(2000, 2026, size=n_samples)
# Create DataFrame
df_example = pd.DataFrame({
    'year': years,
    'P': pr_values,
    'T': temperature_values
})
n_ordinary_per_year = df_example['year'].value_counts().sort_index()

# Your data (P, T arrays)
P = df_example["P"].to_numpy()  # Replace with your actual data
T = df_example["T"].to_numpy()  # Replace with your actual data
blocks_id = df_example["year"].to_numpy()  # Replace with your actual data

# Number of threshold
thr = df_example["P"].quantile(S.left_censoring[1])

# Sampling intervals for the Montecarlo
Ts = np.arange(
    np.min(T) - S.temp_delta, np.max(T) + S.temp_delta, S.temp_res_monte_carlo
)

# TENAX MODEL HERE
# magnitude model
F_phat, loglik, _, _ = S.magnitude_model(P, T, thr)
# temperature model
g_phat = S.temperature_model(T)
# M is mean n of ordinary events
n = n_ordinary_per_year.sum() / len(n_ordinary_per_year)
# estimates return levels using MC samples


#this is ripoff of tenex class model inversion 
method_root_scalar="brentq"
P_mc = []
ret_lev = []
pdf_values = tenax.gen_norm_pdf(Ts, g_phat[0], g_phat[1], 4)
df = np.vstack([pdf_values, Ts])
# Generates random T values according to the temperature model
T_mc = tenax.randdf(S.n_monte_carlo, df, "pdf").T
# Generates random P according to the magnitude model
wbl_phat = np.column_stack((
                            F_phat[2] * np.exp(F_phat[3] * T_mc),
                            F_phat[0] + F_phat[1] * T_mc
                            )) #linear model for b
vguess = 10 ** np.arange(np.log10(0.05), np.log10(5e2), 0.05)

def MC_tSMEV_cdf_old(
    y: Union[float, np.ndarray], wbl_phat: np.ndarray, n: int
) -> Tuple[float, np.ndarray]:
    """
    Calculate the cumulative distribution function (CDF) based on the given Weibull parameters.

    Parameters
    ----------
        y (Union[float, np.ndarray]): Value(s) at which to evaluate the CDF.
        wbl_phat (np.ndarray): Array of Weibull parameters, where each row contains [shape, scale].
        n (int): Power to raise the final probability to.

    Returns
    -------
        Tuple[float, np.ndarray]: Calculated CDF value(s).
    """
    p = 0
    for i in range(wbl_phat.shape[0]):
        p += 1 - np.exp(-((y / wbl_phat[i, 0]) ** wbl_phat[i, 1]))
    p = (p / wbl_phat.shape[0]) ** n
    return p

def SMEV_Mc_inversion_old(
    wbl_phat: np.ndarray,
    n: Union[int, float, pd.Series],
    target_return_periods: Union[list, np.ndarray],
    vguess: np.ndarray,
    method_root_scalar: Union[str, None],
) -> np.ndarray:
    """
    Invert to find quantiles corresponding to the target return periods.

    Parameters
    ----------
        wbl_phat (numpy.ndarray): Array of Weibull parameters, where each row contains [shape, scale].
        n (int): Power to raise the final probability to.
        target_return_periods (list or array-like): Desired target return periods.
        vguess (numpy.ndarray): Initial guesses for inversion.

    Returns
    -------
        np.ndarray: Quantiles corresponding to the target return periods.
    """
    if isinstance(n, pd.Series):
        n = float(n.values[0])

    pr = 1 - 1 / np.array(
        target_return_periods
    )  # Probabilities associated with target_return_periods
    pv = MC_tSMEV_cdf_old(
        vguess, wbl_phat, n
    )  # Probabilities associated with vguess values
    qnt = np.full(
        len(target_return_periods), np.nan
    )  # Initialize output array with NaNs

    for t in range(len(target_return_periods)):
        # Find the first guess where pv exceeds pr
        first_guess_idx = np.where(pv > pr[t])[0]
        if len(first_guess_idx) > 0:
            first_guess = vguess[first_guess_idx[0]]
        else:
            # Use the last valid guess if none exceeds pr
            last_valid_idx = np.where(pv < 1)[0]
            first_guess = (
                vguess[last_valid_idx[-1]] if len(last_valid_idx) > 0 else vguess[-1]
            )

        # Define the function for root finding
        def func(y):
            return MC_tSMEV_cdf_old(y, wbl_phat, n) - pr[t]

        # Use root_scalar as an alternative to MATLAB's fzero
        result = root_scalar(
            func,
            bracket=[vguess[0], vguess[-1]],
            x0=first_guess,
            method=method_root_scalar,
        )

        if result.converged:
            qnt[t] = result.root

    return qnt

ret_lev_old = SMEV_Mc_inversion_old(
            wbl_phat,
            n,
            S.return_period,
            vguess,
            method_root_scalar=method_root_scalar,
        )

ret_lev_new = tenax.SMEV_Mc_inversion(wbl_phat,
                                  n,
                                  S.return_period,
                                  vguess,
                                  method_root_scalar=method_root_scalar,)