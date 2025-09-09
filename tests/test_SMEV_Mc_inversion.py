# -*- coding: utf-8 -*-
"""
Created on Mon Sep  8 14:58:31 2025

@author: Petr
"""
import unittest
import pandas as pd
import numpy as np
from pyTENAX import tenax
from scipy.stats import gennorm
from scipy.optimize import root_scalar
from typing import List, Tuple, Union, Optional
import time

# Old function for the reference.
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


class TestMc_inversion(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # The time resolution and durations are just placeholders, 
        # but they must be defined in the class; otherwise, some functions will not work.
        cls.S = tenax.TENAX(
           return_period=[2, 5, 10, 20, 50, 100, 200],
           durations=[10],
           time_resolution=5,
           left_censoring=[0, 0.90],
           alpha=0.05,
       )
        
        # Define random seed for reproduductyibily 
        np.random.seed(42)
        # totaal samples to build
        n_samples = 2000

        # Temperature samples
        temperature_values = gennorm.rvs(beta=4, loc=10, scale=11.5, size=n_samples)
        temperature_values = np.clip(temperature_values, -10, 50)

        # Precipitation samples depending on T based on weibull distribution
        shape = 0.7
        base_scale = 2
        k = 0.2
        scales = base_scale + k * temperature_values
        pr_values = scales * np.random.weibull(shape, size=n_samples)
        pr_values = np.clip(pr_values, 0.2, 300)

        # Random years
        years = np.random.randint(2000, 2026, size=n_samples)

        # Build dataframe
        df_example = pd.DataFrame({
            'year': years,
            'P': pr_values,
            'T': temperature_values
        })
        # For clarity purpose, sort by years, as they were random before.
        df_example = df_example.sort_values("year").reset_index(drop=True)
        
        cls.n_ordinary_per_year = df_example['year'].value_counts().sort_index()
        P = df_example["P"].to_numpy()
        T = df_example["T"].to_numpy()

        # Threshold
        thr = df_example["P"].quantile(cls.S.left_censoring[1])

        # Fit magnitude + temperature models
        cls.F_phat, _, _, _ = cls.S.magnitude_model(P, T, thr)
        cls.g_phat = cls.S.temperature_model(T)
        cls.n = cls.n_ordinary_per_year.sum() / len(cls.n_ordinary_per_year)

        # Monte Carlo setup
        Ts = np.arange(np.min(T) - cls.S.temp_delta,
                       np.max(T) + cls.S.temp_delta,
                       cls.S.temp_res_monte_carlo)
        pdf_values = tenax.gen_norm_pdf(Ts, cls.g_phat[0], cls.g_phat[1], 4)
        df = np.vstack([pdf_values, Ts])
        T_mc = tenax.randdf(cls.S.n_monte_carlo, df, "pdf").T

        # Weibull parameters for MC
        cls.wbl_phat = np.column_stack((
            cls.F_phat[2] * np.exp(cls.F_phat[3] * T_mc),
            cls.F_phat[0] + cls.F_phat[1] * T_mc
        ))

        # Value guesses for inversion
        cls.vguess = 10 ** np.arange(np.log10(0.05), np.log10(500), 0.05)

    def test_inversion_consistency(cls):
        """
        Compare SMEV_Mc_inversion_old vs library SMEV_Mc_inversion.
        Runs both 100 times, checks equality on each run, and reports average timings.
        """
        n_runs = 1
        times_old = []
        times_new = []

        for _ in range(n_runs):
            # Old version timing
            start = time.perf_counter()
            ret_old = SMEV_Mc_inversion_old(
                cls.wbl_phat,
                cls.n,
                cls.S.return_period,
                cls.vguess,
                method_root_scalar="brentq",
            )
            times_old.append(time.perf_counter() - start)

            # New version timing
            start = time.perf_counter()
            ret_new = tenax.SMEV_Mc_inversion(
                cls.wbl_phat,
                cls.n,
                cls.S.return_period,
                cls.vguess,
                method_root_scalar="brentq",
            )
            times_new.append(time.perf_counter() - start)

            # Validate per-run equality
            np.testing.assert_allclose(ret_new, ret_old, rtol=1e-5, atol=1e-8)

        avg_old = np.mean(times_old)
        avg_new = np.mean(times_new)

        print(f"\nAverage runtime over {n_runs} runs:")
        print(f"  Old inversion: {avg_old:.6f} s")
        print(f"  New inversion: {avg_new:.6f} s")


if __name__ == "__main__":
    # Load all tests from TestMc_inversion
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestMc_inversion)
    # Create a runner with verbosity=2
    runner = unittest.TextTestRunner(verbosity=2)
    # Run the suite
    runner.run(suite)
