import pandas as pd
import numpy as np
from scipy.special import gamma
from scipy.stats import weibull_min, norm, skewnorm, chi2
from scipy.optimize import root_scalar, minimize
import time
from scipy.optimize import curve_fit
from typing import Union, Tuple, Dict
from multiprocessing.dummy import Pool as ThreadPool
from pyTENAX.smev import (
    _smev_inner_loop_numba_seq,
    _smev_inner_loop_numba,
    _NUMBA_AVAILABLE,
)


class TENAX:
    def __init__(
        self,
        return_period: list[Union[int, float]],
        durations: list[int],
        time_resolution: int,
        beta: Union[float, int] = 4,
        temp_time_hour: int = 24,
        alpha=0.05,
        n_monte_carlo=int(2e4),
        tolerance=0.1,
        min_event_duration=30,
        storm_separation_time=24,
        left_censoring: list = [0, 1],
        niter_smev=100,  # why is this here?
        niter_tenax=100,
        temp_res_monte_carlo=0.001,
        temp_delta=10,
        init_param_guess=[0.7, 0, 2, 0],
        min_rain: Union[float, int] = 0,
    ) -> None:
        """Initialize the TENAX model with the specified parameters.

        The TEmperaturedependent Non-Asymptotic statistical model for eXtreme
        return levels (TENAX), is based on a parsimonious nonstationary and
        non-asymptotic theoretical framework that incorporates temperature
        as a covariate in a physically consistent manner.

        Parameters
        ----------
        return_period : list[Union[int, float]]
            Return periods [years].
        durations : list[int]
            Durations of interest [min].
        time_resolution : int
            Temporal resolution of the precipitation data [min].
        beta : Union[float, int], optional
            Shape parameter of the Generalized Normal for g(T). Defaults to 4.
        temp_time_hour : int, optional
            Time window to compute T [h]. Will be converted to negative if
            needed. Defaults to 24.
        alpha : float, optional
            Unitless significance level for the dependence of the shape on T.
            Defaults to 0.05.
            0 → dependence always allowed; 1 → never allowed;
            (0, 1) → depends on statistical significance at alpha-level.
        n_monte_carlo : int, optional
            Number of elements in the MC samples. Defaults to int(2e4).
        tolerance : float, optional
            Maximum allowed fraction of missing data in one year.
            If exceeded, year will be disregarded. Defaults to 0.1.
        min_event_duration : int, optional
            Minimum event duration [min]. Defaults to 30.
        storm_separation_time : int, optional
            Separation time between independent storms [hours]. Defaults to 24.
        left_censoring : list, optional
            2-element list with lower and upper probability limits for
            parameter estimation. Defaults to [0, 1].
        niter_smev : int, optional
            Number of bootstrap iterations for SMEV uncertainty. Defaults to
            100.
        niter_tenax : int, optional
            Number of bootstrap iterations for TENAX uncertainty. Defaults to
            100.
        temp_res_monte_carlo : float, optional
            Resolution in T for the MC samples. Defaults to 0.001.
        temp_delta : int, optional
            Range in T of MC samples — explores temperatures up to this many
            degrees above and below observed values. Defaults to 10.
        init_param_guess : list, optional
            Initial values of Weibull parameters for optimisation.
            Defaults to [0.7, 0, 2, 0].
        min_rain : Union[float, int], optional
            Minimum rainfall value. Defaults to 0.
        """
        self.return_period = return_period
        self.durations = durations
        self.time_resolution = time_resolution
        self.beta = beta
        self.temp_time_hour = -abs(temp_time_hour)
        self.alpha = alpha
        self.n_monte_carlo = n_monte_carlo
        self.tolerance = tolerance
        self.min_event_duration = min_event_duration
        self.storm_separation_time = storm_separation_time
        self.left_censoring = left_censoring
        self.niter_smev = niter_smev
        self.niter_tenax = niter_tenax
        self.temp_res_monte_carlo = temp_res_monte_carlo
        self.temp_delta = temp_delta
        self.init_param_guess = init_param_guess
        self.min_rain = min_rain

        self.__incomplete_years_removed__ = False

    def remove_incomplete_years(
        self,
        data_pr: pd.DataFrame,
        name_col="value",
        nan_to_zero=True,
    ) -> pd.DataFrame:
        """Delete incomplete years in precipitation data.

        An incomplete year is defined as a year where observations are
        missing above a given threshold.

        Parameters
        ----------
        data_pr : pd.DataFrame
            Dataframe containing (hourly) precipitation values.
        name_col : str, optional
            Column name in `data_pr` with precipitation values.
            Defaults to "value".
        nan_to_zero : bool, optional
            Set `nan` to zero. Defaults to True.

        Returns
        -------
        pd.DataFrame
            Dataframe containing (hourly) precipitation values
            with incomplete years removed.
        """
        # Step 1: get resolution of dataset (MUST BE SAME in whole dataset!!!)
        time_res = (
            (data_pr.index[-1] - data_pr.index[-2]).total_seconds() / 60
        )
        # Validate: if user provided time_resolution, it must match the data
        if (
            self.time_resolution is not None
            and self.time_resolution != time_res
        ):
            raise ValueError(
                "time_resolution provided "
                f"({self.time_resolution} min) does not match "
                f"the resolution detected from data ({time_res} min)."
            )
        # Step 2: Resample by year and count total and NaN values
        yearly_valid = data_pr.resample("YE").apply(
            lambda x: x.notna().sum()
        )  # Count not NaNs per year
        # Step 3: Estimate expected length of yearly timeseries
        expected = pd.DataFrame(index=yearly_valid.index)
        # 1440 = number of minutes in a day
        expected["Total"] = 1440 / time_res * 365
        # Step 4: Calculate percentage of missing data per year
        valid_percentage = yearly_valid[name_col] / expected["Total"]
        # Step 5: Filter out years where more than tolerance% of values are NaN
        years_to_remove = valid_percentage[
            valid_percentage < (1 - self.tolerance)
        ].index
        # Step 6: Remove data for those years from the original DataFrame
        data_cleanded = data_pr[
            ~data_pr.index.year.isin(years_to_remove.year)
        ]
        # Replace NaN values with 0 in the specific column
        if nan_to_zero:
            data_cleanded.loc[:, name_col] = (
                data_cleanded[name_col].fillna(0)
            )

        self.time_resolution = time_res

        self.__incomplete_years_removed__ = True

        return data_cleanded

    def get_ordinary_events(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        dates: np.ndarray,
        name_col: str = "value",
        check_gaps=True,
    ) -> list:
        """Extract ordinary precipitation events from a time series.

        Groups timesteps at or above ``self.min_rain`` into independent storm
        events separated by at least ``self.storm_separation_time`` hours.
        Optionally removes events too close to dataset boundaries or data gaps.

        Parameters
        ----------
        data : Union[pd.DataFrame, np.ndarray]
            Precipitation values.
        dates : np.ndarray
            Timestamps of the precipitation data.
        name_col : str, optional
            Column name to use when ``data`` is a DataFrame. Defaults to
            "value".
        check_gaps : bool, optional
            Remove events that fall within ``storm_separation_time`` of the
            dataset boundaries or internal data gaps. Defaults to True.

        Returns
        -------
        list
            List of np.ndarray, each containing the timestamps of one ordinary
            event (values >= ``self.min_rain`` separated by more than
            ``self.storm_separation_time`` hours).
        """
        if not self.__incomplete_years_removed__:
            raise ValueError(
                "You must run 'remove_incomplete_years' "
                "before running this function. "
                "If you are sure your data is complete, set "
                "self.__incomplete_years_removed__ = True "
                "to bypass this check."
            )

        if isinstance(data, pd.DataFrame):
            data = np.array(data[name_col])

        dates = dates.astype("datetime64[ns]")

        if self.min_rain == 0:
            above_threshold_indices = np.where(data > self.min_rain)[0]
        else:
            above_threshold_indices = np.where(data >= self.min_rain)[0]

        if len(above_threshold_indices) == 0:
            return []

        # Get dates at above-threshold positions
        above_dates = dates[above_threshold_indices]

        # Compute time differences between consecutive above-threshold
        # timesteps (in nanoseconds)
        time_diffs_above = np.diff(above_dates).astype(np.int64)

        # Find where gaps exceed separation time (hours to nanoseconds)
        separation_ns = int(self.storm_separation_time * 3.6e12)
        gap_mask = time_diffs_above > separation_ns

        # Split indices at gap locations
        split_points = np.where(gap_mask)[0] + 1

        # Split into groups of indices, then map back to dates
        index_groups = np.split(above_threshold_indices, split_points)

        # Convert to list of date arrays (same format as original)
        consecutive_values = [dates[group] for group in index_groups]

        if check_gaps:
            # Remove event too close to the start of the dataset
            if (consecutive_values[0][0] - dates[0]).item() < (
                self.storm_separation_time * 3.6e12
            ):  # numpy dt is in nanoseconds
                consecutive_values.pop(0)
            else:
                pass

            # Remove event too close to the end of the dataset
            if (dates[-1] - consecutive_values[-1][-1]).item() < (
                self.storm_separation_time * 3.6e12
            ):  # numpy dt is in nanoseconds
                consecutive_values.pop()
            else:
                pass

            # Locate OE that ends before gaps in data starts.
            # Calculate the differences between consecutive elements
            time_diffs = np.diff(dates)
            # difference of first element is time resolution
            time_res = time_diffs[0]
            # Identify gaps larger than separation time
            sep_td = np.timedelta64(
                int(self.storm_separation_time * 3.6e12), "ns"
            )
            gap_indices_end = np.where(time_diffs > sep_td)[0]
            # Extend by one index to also check OE near gap start
            gap_indices_start = gap_indices_end + 1

            match_info = []
            for gap_idx in gap_indices_end:
                end_date = dates[gap_idx]
                start_date = end_date - np.timedelta64(
                    int(self.storm_separation_time * 3.6e12), "ns"
                )
                temp_date_array = np.arange(start_date, end_date, time_res)

                for i, sub_array in enumerate(consecutive_values):
                    match_indices = np.where(
                        np.isin(sub_array, temp_date_array)
                    )[0]
                    if match_indices.size > 0:
                        match_info.append(i)

            for gap_idx in gap_indices_start:
                start_date = dates[gap_idx]
                end_date = start_date + np.timedelta64(
                    int(self.storm_separation_time * 3.6e12), "ns"
                )
                temp_date_array = np.arange(start_date, end_date, time_res)

                for i, sub_array in enumerate(consecutive_values):
                    match_indices = np.where(
                        np.isin(sub_array, temp_date_array)
                    )[0]
                    if match_indices.size > 0:
                        match_info.append(i)

            for del_index in sorted(match_info, reverse=True):
                del consecutive_values[del_index]

        return consecutive_values

    def remove_short(
        self,
        list_ordinary: list,
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """Remove ordinary events that are too short.

        Parameters
        ----------
        list_ordinary : list
            List of ordinary events as returned by `get_ordinary_events()`.
            Each event may contain pd.Timestamp or np.datetime64 values.

        Returns
        -------
        arr_vals : np.ndarray
            Boolean array (all True) of length equal to the number of kept
            events, one entry per event that passed the duration filter.
        arr_dates : np.ndarray
            Array of (end, start) date tuples for each kept event.
        n_ordinary_per_year : pd.DataFrame
            DataFrame with the count of ordinary events per year.
        """
        if not self.__incomplete_years_removed__:
            raise ValueError(
                "You must run 'remove_incomplete_years' "
                "before running this function. "
                "If you are sure your data is complete, set "
                "self.__incomplete_years_removed__ = True "
                "to bypass this check."
            )

        # Convert pd.Timestamp events to np.datetime64 if needed
        if isinstance(list_ordinary[0][0], pd.Timestamp):
            list_ordinary = [
                np.array([t.to_datetime64() for t in ev])
                for ev in list_ordinary
            ]

        min_duration = np.timedelta64(int(self.min_event_duration), "m")
        time_res = np.timedelta64(int(self.time_resolution), "m")

        ll_short = [
            (ev[-1] - ev[0]).astype("timedelta64[m]") + time_res
            >= min_duration
            for ev in list_ordinary
        ]
        ll_dates = [
            (ev[-1], ev[0]) if keep else (np.nan, np.nan)
            for ev, keep in zip(list_ordinary, ll_short)
        ]

        arr_vals = np.array(ll_short)[ll_short]
        arr_dates = np.array(ll_dates)[ll_short]

        filtered_list = [
            ev for ev, keep in zip(list_ordinary, ll_short) if keep
        ]
        list_year = pd.DataFrame(
            [ev[0].astype("datetime64[Y]").item().year
             for ev in filtered_list],
            columns=["year"],
        )
        n_ordinary_per_year = list_year.reset_index().groupby(["year"]).count()

        return arr_vals, arr_dates, n_ordinary_per_year

    def get_ordinary_events_values(
        self,
        data: np.ndarray,
        dates: np.ndarray,
        arr_dates_oe: np.ndarray,
        method: str = "vectorized",
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        """Extract ordinary events and annual maxima from precipitation data.

        Parameters
        ----------
        data : np.ndarray
            Full precipitation time series.
        dates : np.ndarray
            Timestamps of the full precipitation dataset.
        arr_dates_oe : np.ndarray
            End and start times of ordinary events as returned by
            `remove_short`.
        method : str, optional
            Backend used for the sliding-window maximum search. Defaults to
            ``"vectorized"``. One of:

            - ``"vectorized"``    — pure numpy, ``np.convolve`` per event.
            - ``"njit"``          — numba JIT-compiled loop, single-threaded.
            - ``"njit_parallel"`` — numba JIT-compiled loop, parallelised
              over events. Requires ``numba`` to be installed.

        Returns
        -------
        dict_ordinary : dict
            Key is duration (str), value is a ``pd.DataFrame`` with columns
            ``year``, ``oe_time``, ``ordinary`` (event depth/intensity).
        dict_AMS : dict
            Key is duration (str), value is a ``pd.DataFrame`` with columns
            ``year`` and ``AMS`` (annual maximum value).
        """
        if method in ("njit", "njit_parallel") and not _NUMBA_AVAILABLE:
            raise ImportError(
                "numba is required for method='njit'/'njit_parallel'. "
                "Install with: pip install numba"
            )

        dict_ordinary = {}
        dict_AMS = {}

        time_index = dates.reshape(-1)
        n_events = arr_dates_oe.shape[0]

        oe_end = arr_dates_oe[:, 0].astype("datetime64[ns]")
        oe_start = arr_dates_oe[:, 1].astype("datetime64[ns]")
        start_indices = np.searchsorted(time_index, oe_start).astype(np.int64)
        end_indices = np.searchsorted(time_index, oe_end).astype(np.int64)

        ll_yrs = np.array([
            oe_end[i].astype("datetime64[Y]").item().year
            for i in range(n_events)
        ], dtype=np.int64)

        unique_years = np.unique(ll_yrs)
        year_masks = {yr: ll_yrs == yr for yr in unique_years}

        data_int = np.round(data * 10000).astype(np.int64)

        for d in range(len(self.durations)):
            window_size = int(self.durations[d] / self.time_resolution)

            if method == "vectorized":
                ones_kernel = np.ones(window_size, dtype=np.int64)
                max_vals = np.empty(n_events, dtype=np.float64)
                max_global_idx = np.empty(n_events, dtype=np.int64)
                for i in range(n_events):
                    si = start_indices[i]
                    ei = end_indices[i]
                    if si == ei:
                        max_vals[i] = data_int[si] / 10000.0
                        max_global_idx[i] = si
                    else:
                        arr_conv = np.convolve(
                            data_int[si:ei + 1], ones_kernel, "same"
                        )
                        ll_idx = np.nanargmax(arr_conv)
                        max_vals[i] = arr_conv[ll_idx] / 10000.0
                        max_global_idx[i] = si + ll_idx
            elif method == "njit":
                max_vals_int, max_global_idx = _smev_inner_loop_numba_seq(
                    data_int, start_indices, end_indices, window_size, n_events
                )
                max_vals = max_vals_int / 10000.0
            else:  # njit_parallel
                max_vals_int, max_global_idx = _smev_inner_loop_numba(
                    data_int, start_indices, end_indices, window_size, n_events
                )
                max_vals = max_vals_int / 10000.0

            ll_dates_arr = time_index[max_global_idx]
            ams_vals = np.array([
                np.max(max_vals[mask]) for _, mask in year_masks.items()
            ])

            df_ams = pd.DataFrame({"year": unique_years, "AMS": ams_vals})
            df_oe = pd.DataFrame({
                "year": ll_yrs,
                "oe_time": ll_dates_arr,
                "ordinary": max_vals,
            })
            dict_AMS[f"{self.durations[d]}"] = df_ams
            dict_ordinary[f"{self.durations[d]}"] = df_oe

        return dict_ordinary, dict_AMS

    def associate_vars(
        self,
        dict_ordinary,
        data_temperature,
        dates_temperature
    ):
        """Associate temperature with each ordinary event.

        The associated temperature is the mean over the past ``temp_time_hour``
        hours before the event. Events for which no temperature can be found
        are dropped.

        Parameters
        ----------
        dict_ordinary : dict
            Dictionary of ordinary events as returned by
            `get_ordinary_events_values`.
        data_temperature : np.ndarray
            Full temperature time series.
        dates_temperature : np.ndarray
            Timestamps of the full temperature dataset.

        Returns
        -------
        dict_ordinary : dict
            Ordinary events with an added ``T`` column, keyed by duration.
            Example: ``{"10": pd.DataFrame(
                columns=['year', 'oe_time', 'ordinary', 'T'])}``.
        dict_dropped_oe : dict
            Ordinary events dropped because no temperature was found,
            keyed by duration.
        n_ordinary_per_year_new : pd.DataFrame
            DataFrame with the count of ordinary events per year after
            dropping events without temperature.
        """
        # start here
        dict_dropped_oe = {}
        time_index = dates_temperature.reshape(-1)

        delta_time = np.timedelta64(int(self.temp_time_hour), "h")
        for d in self.durations:
            df_oe = dict_ordinary[f"{d}"]
            arr_dates_oe = np.array(df_oe["oe_time"])
            ll_vals = []

            # Prepare indices for using pandas `merge_asof` function
            df_time_index = pd.DataFrame({"time_index": time_index})
            df_arr_dates_oe = pd.DataFrame({"oe_time": arr_dates_oe})

            # Use pandas to perform an "as-of" merge that efficiently finds
            # the closest index, 30 are handled to nearest lower
            merged = pd.merge_asof(
                df_arr_dates_oe,
                df_time_index,
                left_on="oe_time",
                right_on="time_index",
                direction="nearest",
            )

            for _, row in merged.iterrows():
                end_time = row["time_index"]

                # Find the index of the closest time
                # directly using `np.searchsorted`
                if end_time is None:
                    continue  # Skip this iteration if no match was found

                # Use `np.searchsorted` to find the index in `time_index`
                end_dt = np.datetime64(end_time)
                closest_idx = np.searchsorted(time_index, end_dt)

                # Calculate end time with delta
                end_time_minus_delta = time_index[closest_idx] + delta_time

                # Find start index efficiently
                start_time_idx = np.searchsorted(time_index,
                                                 end_time_minus_delta
                                                 )

                # Slice array more efficiently
                ll_idx_in_slice_vals = data_temperature[
                    start_time_idx: closest_idx + 1
                ]

                # Compute mean using vectorized method
                if np.all(np.isnan(ll_idx_in_slice_vals)):
                    ll_val = np.nan
                else:
                    ll_val = np.nanmean(ll_idx_in_slice_vals).round(decimals=3)
                ll_vals.append(ll_val)

            # Assign computed list back to DataFrame
            dict_ordinary[f"{d}"]["T"] = ll_vals

            # Locate rows with NaN and saved them
            dict_dropped_oe[f"{d}"] = dict_ordinary[f"{d}"][
                dict_ordinary[f"{d}"]["T"].isna()
            ]

            # Drop rows with NaN in the "T" column from the original DataFrame
            dict_ordinary[f"{d}"] = (
                dict_ordinary[f"{d}"]
                .dropna(subset=["T"])
                .reset_index(drop=True)
            )

        # Recalculate number of OE
        n_ordinary_per_year_new = (
            dict_ordinary[f"{d}"]
            .groupby(["year"])["ordinary"]
            .count()
            .to_frame(name="N_oe")
        )

        return dict_ordinary, dict_dropped_oe, n_ordinary_per_year_new

    def magnitude_model(
        self,
        data_oe_prec,
        data_oe_temp,
        thr,
        b_set=None,
        b_exp=False
    ):
        """
        Fits the data to the magnitude model of TENAX.

        Parameters
        ----------
        data_oe_prec : numpy.ndarray
            Array of precipitation ordinary events data.
        data_oe_temp : numpy.ndarray
            Array of temperature ordinary events data.
        thr : numpy.float64
            Magnitude of precipitation threshold.
        b_set : NoneType or float
            Set value of b. fits magnitude model with a specified value for b.
        b_exp : bool
            If True, uses the exponential rather than linear fit for b.


        Returns
        -------
        phat : np.ndarray
            Fitted parameters ``[kappa_0, b, lambda_0, a]``.
        loglik : float
            Log-likelihood of the selected model.
        loglik_H1 : float or None
            Log-likelihood of the alternative hypothesis (H1). ``None`` when
            ``b_set`` is provided.
        loglik_H0shape : float or None
            Log-likelihood of the null hypothesis (constant shape). ``None``
            when ``b_set`` is provided.
        """
        # alpha=0 --> dependence of shape on T is always allowed
        # alpha=1 --> dependence of shape on T is never allowed
        # else    --> dependence of shape on T depends on stat. significance

        P = data_oe_prec
        T = data_oe_temp
        thr = thr
        init_g = self.init_param_guess
        alpha = self.alpha

        if b_set:
            if b_exp:
                min_phat_bset = minimize(
                    lambda theta: -wbl_leftcensor_loglik_bset_bexp(
                        theta, P, T, thr, b_set
                    ),
                    init_g,
                    method='Nelder-Mead')
                phat_bset = min_phat_bset.x
                loglik_bset = wbl_leftcensor_loglik_bset_bexp(
                    phat_bset, P, T, thr, b_set
                    )
                phat_bset[1] = b_set
                phat = phat_bset
                loglik = loglik_bset
                # TODO: figure this out, do we need these outputs?
                loglik_H1, loglik_H0shape = None, None
            else:
                min_phat_bset = minimize(
                    lambda theta: -wbl_leftcensor_loglik_bset(
                        theta, P, T, thr, b_set
                    ),
                    init_g,
                    method='Nelder-Mead')
                phat_bset = min_phat_bset.x
                loglik_bset = wbl_leftcensor_loglik_bset(
                    phat_bset, P, T, thr, b_set
                    )
                phat_bset[1] = b_set
                phat = phat_bset
                loglik = loglik_bset
                # TODO: figure this out, do we need these outputs?
                loglik_H1, loglik_H0shape = None, None

        elif b_exp:
            min_phat_H1 = minimize(
                lambda theta: -wbl_leftcensor_loglik_exp(
                    theta, P, T, thr
                ),
                init_g,
                method='Nelder-Mead')
            phat_H1 = min_phat_H1.x

            min_phat_H0shape = minimize(
                lambda theta: -wbl_leftcensor_loglik_H0shape(
                    theta, P, T, thr
                ),
                init_g,
                method='Nelder-Mead',
                options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 1000})

            phat_H0shape = min_phat_H0shape.x
            phat_H0shape[1] = 0

            loglik_H1 = wbl_leftcensor_loglik_exp(
                phat_H1, P, T, thr
                )
            loglik_H0shape = wbl_leftcensor_loglik_H0shape(
                phat_H0shape, P, T, thr
                )
            lambda_LR_shape = -2*(loglik_H0shape - loglik_H1)
            pval = chi2.sf(lambda_LR_shape, df=1)

            if alpha == 0:  # dependence of shape on T is always allowed
                phat = phat_H1
                loglik = loglik_H1
            elif alpha == 1:  # dependence of shape on T is never allowed
                phat = phat_H0shape
                loglik = loglik_H0shape
            elif pval <= alpha:  # depends on stat. significance
                phat = phat_H1
                loglik = loglik_H1
            else:
                phat = phat_H0shape
                loglik = loglik_H0shape

        else:
            min_phat_H1 = minimize(
                lambda theta: -wbl_leftcensor_loglik(
                    theta, P, T, thr
                ),
                init_g,
                method='Nelder-Mead')
            phat_H1 = min_phat_H1.x

            min_phat_H0shape = minimize(
                lambda theta: -wbl_leftcensor_loglik_H0shape(
                    theta, P, T, thr
                ),
                init_g,
                method='Nelder-Mead',
                options={'xatol': 1e-8, 'fatol': 1e-8, 'maxiter': 1000})

            phat_H0shape = min_phat_H0shape.x
            phat_H0shape[1] = 0

            loglik_H1 = wbl_leftcensor_loglik(
                phat_H1, P, T, thr
                )
            loglik_H0shape = wbl_leftcensor_loglik_H0shape(
                phat_H0shape, P, T, thr
                )
            lambda_LR_shape = -2*(loglik_H0shape - loglik_H1)
            pval = chi2.sf(lambda_LR_shape, df=1)

            if alpha == 0:  # dependence of shape on T is always allowed
                phat = phat_H1
                loglik = loglik_H1
            elif alpha == 1:  # dependence of shape on T is never allowed
                phat = phat_H0shape
                loglik = loglik_H0shape
            elif pval <= alpha:  # depends on stat. significance
                phat = phat_H1
                loglik = loglik_H1
            else:
                phat = phat_H0shape
                loglik = loglik_H0shape

        return phat, loglik, loglik_H1, loglik_H0shape

    def temperature_model(
        self,
        data_oe_temp,
        beta=0,
        method="norm"
    ):
        """Fit temperature data to the TENAX temperature model.

        Parameters
        ----------
        data_oe_temp : np.ndarray
            Temperature data.
        beta : float, optional
            Shape parameter of the generalised normal distribution. If 0,
            uses ``self.beta``. Defaults to 0.
        method : str, optional
            Distribution to fit. ``"norm"`` uses the generalised normal;
            ``"skewnorm"`` uses a skewed normal. Defaults to ``"norm"``.

        Returns
        -------
        g_phat : np.ndarray
            Fitted parameters. ``[mu, sigma]`` for ``"norm"``;
            ``[alpha, loc, scale]`` for ``"skewnorm"``.
        """
        if beta == 0:
            beta = self.beta
        else:
            beta = beta

        if method == "norm":
            mu, sigma = norm.fit(data_oe_temp)
            init_g = [mu, sigma]

            g_phat = minimize(
                lambda par: -gen_norm_loglik(data_oe_temp, par, beta),
                init_g,
                method="Nelder-Mead",
            ).x

        elif method == "skewnorm":
            # Fit the skew-normal distribution
            # Initial guess for the parameters
            # Compute histogram data
            def skewnorm_pdf(x, alpha, loc, scale):
                return skewnorm.pdf(x, alpha, loc=loc, scale=scale)

            hist, bin_edges = np.histogram(
                data_oe_temp, bins=100, density=True
                )
            # Bin centers for xdata
            xdata = (bin_edges[:-1] + bin_edges[1:]) / 2
            initial_guess = [
                -3,
                np.mean(data_oe_temp),
                np.std(data_oe_temp),
            ]  # Guess for alpha, loc, scale
            g_phat, _ = curve_fit(
                skewnorm_pdf, xdata, hist, p0=initial_guess, maxfev=10000
            )
            g_phat = tuple(g_phat.reshape(1, -1)[0])
            # g_phat = skewnorm.fit(data_oe_temp) #returns loc, scale, shape

        else:
            print("not given method - temperature model")
            g_phat = []

        return g_phat

    def model_inversion(
        self,
        F_phat,
        g_phat,
        n,
        Ts,
        gen_P_mc=False,
        gen_RL=True,
        temp_method="norm",
        method_root_scalar="brentq",
        b_exp=False
    ):
        """
        Inversion of the TENAX model to predict return levels or plot model.

        Parameters
        ----------
        F_phat : numpy.ndarray
            distribution values. F_phat = [kappa_0,b,lambda_0,a].
        g_phat : numpy.ndarray
            [mu, sigma] of temperature distribution.
        n : float
            Mean number of ordinary events per year.
        Ts : numpy.ndarray
            Array of T values to use in the Monte Carlo.
        gen_P_mc : bool, optional
            Specify whether to generate Monte Carlo values for precipitation.
            The default is False.
        gen_RL : bool, optional
            Specify whether to generate return levels. The default is True.
        temp_method : str, optional
            Type of fit used for the temperature model. The default is "norm".
        method_root_scalar : str, optional
            method used for inversion. The default is "brentq".
        b_exp : bool
            If True, uses the exponential rather than linear fit for b.

        Returns
        -------
        ret_lev : np.ndarray or list
            Return levels at periods specified in ``self.return_period``.
            Empty list ``[]`` if ``gen_RL=False``.
        T_mc : np.ndarray
            Monte Carlo generated temperature values, shape
            ``(n_monte_carlo, 1)``.
        P_mc : np.ndarray or list
            Monte Carlo generated precipitation values. Empty list ``[]``
            if ``gen_P_mc=False``.
        """

        P_mc = []
        ret_lev = []

        if temp_method == "skewnorm":
            pdf_values = skewnorm.pdf(Ts, *g_phat)
            df = np.vstack([pdf_values, Ts])
        elif temp_method == "norm":
            pdf_values = gen_norm_pdf(Ts, g_phat[0], g_phat[1], self.beta)
            df = np.vstack([pdf_values, Ts])
        else:
            print("not given correct method - temperature model")

        # Generates random T values according to the temperature model
        T_mc = randdf(self.n_monte_carlo, df, "pdf").T

        # Generates random P according to the magnitude model
        if b_exp:
            # exponential model for b
            wbl_phat = np.column_stack((
                                        F_phat[2] * np.exp(F_phat[3] * T_mc),
                                        F_phat[0] * np.exp(F_phat[1] * T_mc)
                                        ))

        else:
            # linear model for b
            wbl_phat = np.column_stack((
                                        F_phat[2] * np.exp(F_phat[3] * T_mc),
                                        F_phat[0] + F_phat[1] * T_mc
                                        ))
        # old vguess
        # vguess = 10 ** np.arange(np.log10(F_phat[2]), np.log10(5e2), 0.05
        # test new vguess
        vguess = 10 ** np.arange(np.log10(0.05), np.log10(5e2), 0.05)

        if gen_RL:
            ret_lev = SMEV_Mc_inversion(
                wbl_phat,
                n,
                self.return_period,
                vguess,
                method_root_scalar=method_root_scalar,
            )
        else:
            pass

        # Generate P_mc if needed
        if gen_P_mc:
            start = time.time()
            P_mc = weibull_min.ppf(
                np.random.rand(self.n_monte_carlo),
                c=wbl_phat[:, 1],
                scale=wbl_phat[:, 0],
            )
            end = time.time() - start
            print(f"mc {self.n_monte_carlo}: {end}")

        else:
            pass

        return ret_lev, T_mc, P_mc

    def TNX_tenax_bootstrap_uncertainty(
        self,
        P,
        T,
        blocks_id,
        Ts,
        temp_method="norm",
        method_root_scalar="brentq"
    ):
        """Bootstrap uncertainty estimation for the TENAX model.

        Parameters
        ----------
        P : np.ndarray
            Precipitation ordinary events data.
        T : np.ndarray
            Temperature ordinary events data.
        blocks_id : np.ndarray
            Block identifiers (e.g., years) for each event.
        Ts : np.ndarray
            Array of temperature values for the Monte Carlo integration.
        temp_method : str, optional
            Distribution used for the temperature model. Defaults to
            ``"norm"``.
        method_root_scalar : str, optional
            Root-finding method for model inversion. Defaults to
            ``"brentq"``.

        Returns
        -------
        F_phat_unc : np.ndarray
            Magnitude model parameters from each bootstrap sample,
            shape ``(niter, 4)``.
        g_phat_unc : np.ndarray
            Temperature model parameters from each bootstrap sample,
            shape ``(niter, 2)`` for ``"norm"`` or ``(niter, 3)`` for
            ``"skewnorm"``.
        RL_unc : np.ndarray
            Return levels from each bootstrap sample,
            shape ``(niter, len(return_period))``.
        n_unc : np.ndarray
            Mean number of events per block from each bootstrap sample,
            shape ``(niter,)``.
        n_err : int
            Number of iterations where model fitting failed.
        """

        perc_thres = self.left_censoring[1]
        niter = self.niter_tenax
        RP = self.return_period

        blocks = np.unique(blocks_id)
        M = len(blocks)
        randy = np.random.randint(0, M, size=(M, niter))

        # Initialize variables
        F_phat_unc = np.full((niter, 4), np.nan)
        if temp_method == "norm":
            g_phat_unc = np.full((niter, 2), np.nan)
        elif temp_method == "skewnorm":
            g_phat_unc = np.full((niter, 3), np.nan)
        else:
            g_phat_unc = []
        RL_unc = np.full((niter, len(RP)), np.nan)
        n_unc = np.full(niter, np.nan)
        n_err = 0

        _t_resample = 0.0
        _t_magnitude = 0.0
        _t_temperature = 0.0
        _t_inversion = 0.0

        # Random sampling iterations
        for ii in range(niter):
            Pr = []
            Tr = []
            Bid = []

            # Create bootstrapped data sample and
            # corresponding 'fake' blocks id
            _t0 = time.perf_counter()
            for iy in range(M):
                selected = blocks_id == blocks[randy[iy, ii]]
                Pr.append(P[selected])
                Tr.append(T[selected])
                Bid.append(
                    np.full(np.sum(selected), iy + 1)
                )  # MATLAB indexing starts at 1

            # Concatenate the resampled data
            Pr = np.concatenate(Pr)
            Tr = np.concatenate(Tr)
            Bid = np.concatenate(Bid)
            _t_resample += time.perf_counter() - _t0

            try:
                # Left-censoring threshold
                thr = np.quantile(Pr, perc_thres)

                # TENAX model components
                # Magnitude model
                _t0 = time.perf_counter()
                F_phat_temporary, loglik_temp, _, _ = self.magnitude_model(
                    Pr,
                    Tr,
                    thr
                )
                _t_magnitude += time.perf_counter() - _t0

                # Temperature model
                _t0 = time.perf_counter()
                g_phat_temporary = self.temperature_model(
                    Tr,
                    method=temp_method
                )
                _t_temperature += time.perf_counter() - _t0

                # Mean number of events per block
                n_temporary = len(Pr) / M

                # Estimate return levels using Monte Carlo samples
                _t0 = time.perf_counter()
                RL_temporary, _, _ = self.model_inversion(
                    F_phat_temporary,
                    g_phat_temporary,
                    n_temporary,
                    Ts,
                    temp_method=temp_method,
                    method_root_scalar=method_root_scalar,
                )
                _t_inversion += time.perf_counter() - _t0

                # Store results
                F_phat_unc[ii, :] = F_phat_temporary
                g_phat_unc[ii, :] = g_phat_temporary
                RL_unc[ii, :] = RL_temporary
                n_unc[ii] = n_temporary
            except Exception:
                n_err += 1

        print("\n--- bootstrap_uncertainty internal timings ---")
        for label, elapsed in [
            ("resample", _t_resample),
            ("magnitude_model", _t_magnitude),
            ("temperature_model", _t_temperature),
            ("model_inversion", _t_inversion),
        ]:
            print(f"  {label:<20} {elapsed:7.3f} s")

        return F_phat_unc, g_phat_unc, RL_unc, n_unc, n_err


def wbl_leftcensor_loglik(
    theta,
    x,
    t,
    thr
):
    """Compute log-likelihood for a left-censored Weibull distribution.

    Shape and scale parameters depend linearly on temperature. Observations
    below the threshold are left-censored.

    Parameters
    ----------
    theta : array-like
        Parameter vector ``[kappa_0, b, lambda_0, a]``.
    x : np.ndarray
        Precipitation values.
    t : np.ndarray
        Temperature values.
    thr : float
        Left-censoring threshold.

    Returns
    -------
    float
        Log-likelihood value.
    """
    a_w, b_w, a_C, b_C = theta
    mask = x < thr
    t0, t1, x1 = t[mask], t[~mask], x[~mask]
    shapes0 = a_w + b_w * t0
    scales0 = a_C * np.exp(b_C * t0)
    shapes1 = a_w + b_w * t1
    scales1 = a_C * np.exp(b_C * t1)
    return (
        np.sum(_wbl_logcdf(shapes0, scales0, thr))
        + np.sum(_wbl_logpdf(x1, shapes1, scales1))
    )


def wbl_leftcensor_loglik_H0shape(
    theta,
    x,
    t,
    thr
):
    """
    Compute log-likelihood for a left-censored Weibull
    with constant shape (H0).

    Same as `wbl_leftcensor_loglik` but with ``b=0``, i.e. shape does not
    depend on temperature. Used as the null hypothesis in the likelihood-ratio
    test.

    Parameters
    ----------
    theta : array-like
        Parameter vector ``[kappa_0, b, lambda_0, a]`` (``b`` is ignored).
    x : np.ndarray
        Precipitation values.
    t : np.ndarray
        Temperature values.
    thr : float
        Left-censoring threshold.

    Returns
    -------
    float
        Log-likelihood value.
    """
    a_w, _, a_C, b_C = theta
    mask = x < thr
    t0, t1, x1 = t[mask], t[~mask], x[~mask]
    scales0 = a_C * np.exp(b_C * t0)
    scales1 = a_C * np.exp(b_C * t1)
    return (
        np.sum(_wbl_logcdf(np.full(len(t0), a_w), scales0, thr))
        + np.sum(_wbl_logpdf(x1, np.full(len(t1), a_w), scales1))
    )


def wbl_leftcensor_loglik_bset(theta, x, t, thr, b_set):
    """Compute log-likelihood for a left-censored Weibull with fixed b.

    Same as `wbl_leftcensor_loglik` but ``b`` is fixed to ``b_set`` and not
    optimised.

    Parameters
    ----------
    theta : array-like
        Parameter vector ``[kappa_0, b, lambda_0, a]`` (``b`` is overridden
        by ``b_set``).
    x : np.ndarray
        Precipitation values.
    t : np.ndarray
        Temperature values.
    thr : float
        Left-censoring threshold.
    b_set : float
        Fixed value of ``b`` (shape temperature-dependence parameter).

    Returns
    -------
    float
        Log-likelihood value.
    """
    a_w, _, a_C, b_C = theta
    b_w = b_set
    mask = x < thr
    t0, t1, x1 = t[mask], t[~mask], x[~mask]
    shapes0 = a_w + b_w * t0
    scales0 = a_C * np.exp(b_C * t0)
    shapes1 = a_w + b_w * t1
    scales1 = a_C * np.exp(b_C * t1)
    return (
        np.sum(_wbl_logcdf(shapes0, scales0, thr))
        + np.sum(_wbl_logpdf(x1, shapes1, scales1))
    )


def wbl_leftcensor_loglik_exp(
    theta,
    x,
    t,
    thr
):
    """Compute log-likelihood for a left-censored Weibull
    with exponential shape.

    Like `wbl_leftcensor_loglik` but shape depends exponentially on
    temperature: ``shape = kappa_0 * exp(b * T)``.

    Parameters
    ----------
    theta : array-like
        Parameter vector ``[kappa_0, b, lambda_0, a]``.
    x : np.ndarray
        Precipitation values.
    t : np.ndarray
        Temperature values.
    thr : float
        Left-censoring threshold.

    Returns
    -------
    float
        Log-likelihood value.
    """
    a_w, b_w, a_C, b_C = theta
    mask = x < thr
    t0, t1, x1 = t[mask], t[~mask], x[~mask]
    shapes0 = a_w * np.exp(b_w * t0)
    scales0 = a_C * np.exp(b_C * t0)
    shapes1 = a_w * np.exp(b_w * t1)
    scales1 = a_C * np.exp(b_C * t1)
    return (
        np.sum(_wbl_logcdf(shapes0, scales0, thr))
        + np.sum(_wbl_logpdf(x1, shapes1, scales1))
    )


def wbl_leftcensor_loglik_bset_bexp(
    theta,
    x,
    t,
    thr,
    b_set
):
    """Compute log-likelihood for a left-censored Weibull
    with exponential shape, fixed b.

    Combines `wbl_leftcensor_loglik_exp` and `wbl_leftcensor_loglik_bset`:
    shape depends exponentially on temperature and ``b`` is fixed to
    ``b_set``.

    Parameters
    ----------
    theta : array-like
        Parameter vector ``[kappa_0, b, lambda_0, a]`` (``b`` is overridden
        by ``b_set``).
    x : np.ndarray
        Precipitation values.
    t : np.ndarray
        Temperature values.
    thr : float
        Left-censoring threshold.
    b_set : float
        Fixed value of ``b``.

    Returns
    -------
    float
        Log-likelihood value.
    """
    a_w, _, a_C, b_C = theta
    b_w = b_set
    mask = x < thr
    t0, t1, x1 = t[mask], t[~mask], x[~mask]
    shapes0 = a_w * np.exp(b_w * t0)
    scales0 = a_C * np.exp(b_C * t0)
    shapes1 = a_w * np.exp(b_w * t1)
    scales1 = a_C * np.exp(b_C * t1)
    return (
        np.sum(_wbl_logcdf(shapes0, scales0, thr))
        + np.sum(_wbl_logpdf(x1, shapes1, scales1))
    )


def _wbl_logcdf(shapes, scales, thr):
    """log(CDF) of Weibull at thr: log(1 - exp(-(thr/scale)^shape))."""
    z = (thr / scales) ** shapes
    return np.log(-np.expm1(-z))


def _wbl_logpdf(x, shapes, scales):
    """log(PDF) of Weibull: log(c/scale) + (c-1)*log(x/scale) - (x/scale)^c."""
    z = x / scales
    return (
        np.log(shapes) - np.log(scales)
        + (shapes - 1) * np.log(z) - z ** shapes
    )


def gen_norm_pdf(
    x: np.ndarray,
    mu: float,
    sigma: float,
    beta: float
) -> np.ndarray:
    """Compute the Generalized normal distribution PDF.

    Parameters
    ----------
    x : np.ndarray
        Data points.
    mu : float
        Location parameter.
    sigma : float
        Scale parameter.
    beta : float
        Shape parameter.

    Returns
    -------
    np.ndarray
        Generalized normal distribution PDF values.
    """
    coeff = beta / (2 * sigma * gamma(1 / beta))
    exponent = -((np.abs(x - mu) / sigma) ** beta)
    return coeff * np.exp(exponent)


def gen_norm_loglik(
    x: np.ndarray,
    par: list,
    beta: float
) -> float:
    """Compute the log-likelihood for the Generalized normal distribution.

    Parameters
    ----------
    x : np.ndarray
        Data points.
    par : list
        Parameters ``[mu, sigma]``.
    beta : float
        Shape parameter.

    Returns
    -------
    float
        Log-likelihood value.
    """
    # Compute the log-likelihood
    pdf = gen_norm_pdf(x, par[0], par[1], beta)
    n = len(pdf[pdf == 0])
    if n > 5:
        print(f"warning: {n} zero values")

    pdf[pdf == 0] = 1e-10  # stops issue if zero generated
    loglik = np.sum(np.log(pdf))

    return loglik


def randdf(
    size,
    df,
    flag
):
    """Generate random numbers from a user-defined PDF or CDF.

    Pythonised version of MATLAB's randdf coded by halleyhit on Aug. 15th,
    2018. Email: halleyhit@sjtu.edu.cn or halleyhit@163.com

    Parameters
    ----------
    size : int or tuple
        Output size. ``10`` → 1-D array of 10; ``(10, 2)`` → 10×2 matrix.
    df : np.ndarray
        2-row matrix: first row is function values (PDF or CDF), second row
        is the corresponding sampling points.
    flag : str
        ``"pdf"`` or ``"cdf"``.

    Returns
    -------
    np.ndarray
        Random samples drawn according to the defined distribution.
    """

    # Determine output dimensions
    if isinstance(size, int):
        n, m = 1, size
    elif isinstance(size, tuple) and len(size) == 2:
        n, m = size
    else:
        raise ValueError("Size must be an integer or a tuple of two integers")

    all_samples = n * m

    # Validate input density function
    if df.shape[0] != 2:
        raise ValueError("Density function matrix must have 2 rows")
    if np.any(df[0, :] < 0):
        raise ValueError("Function values must be non-negative")
    if df.shape[1] < 2:
        raise ValueError("Density function must have at least two columns")

    # Normalize pdf or cdf
    if flag == "pdf":
        df[0, :] = np.cumsum(df[0, :]) / np.sum(df[0, :])
    elif flag == "cdf":
        if np.any(np.diff(df[0, :]) < 0):
            raise ValueError("CDF values must be non-decreasing")
        df[0, :] = df[0, :] / df[0, -1]
    else:
        raise ValueError("Flag must be 'pdf' or 'cdf'")

    # Add a small epsilon to ensure no repeated values
    df[0, :] += np.arange(df.shape[1]) * np.finfo(float).eps

    # Generate random samples
    temp = np.random.rand(all_samples)

    # Interpolate to get the corresponding values
    try:
        result = np.interp(temp, df[0, :], df[1, :])
    except ValueError:
        # Handle repeated x-values by taking unique values
        _, unique_indices = np.unique(df[0, :], return_index=True)
        df_unique = df[:, unique_indices]
        result = np.interp(temp, df_unique[0, :], df_unique[1, :])

    return result.reshape((n, m))


def MC_tSMEV_cdf(
    y,
    wbl_phat,
    n
):
    """Evaluate the Monte Carlo SMEV CDF at given values.

    Parameters
    ----------
    y : float or array-like
        Value(s) at which to evaluate the CDF.
    wbl_phat : np.ndarray
        Weibull parameters, shape ``(N, 2)``: columns are ``[scale, shape]``.
    n : float
        Power applied to the average probability.

    Returns
    -------
    np.ndarray
        CDF value(s) for input ``y``.
    """
    y = np.atleast_1d(y)  # Ensure y is array
    scale = wbl_phat[:, 0]
    shape = wbl_phat[:, 1]

    # Reshape to broadcast over y and MC samples
    y_grid = y[:, None]        # shape: (len(y), 1)
    scale_grid = scale[None, :]  # shape: (1, N)
    shape_grid = shape[None, :]  # shape: (1, N)

    cdf_vals = 1 - np.exp(-(y_grid / scale_grid) ** shape_grid)
    p_avg = np.mean(cdf_vals, axis=1)  # Average over MC samples

    return p_avg ** n


def SMEV_Mc_inversion(
    wbl_phat,
    n,
    target_return_periods,
    vguess,
    method_root_scalar
):
    """Invert the MC-SMEV CDF to find quantiles for target return periods.

    Parameters
    ----------
    wbl_phat : np.ndarray
        Weibull parameters, shape ``(N, 2)``: columns are ``[scale, shape]``.
    n : float
        Power applied to the average probability.
    target_return_periods : list or array-like
        Desired return periods.
    vguess : np.ndarray
        Initial value grid for root-finding.
    method_root_scalar : str
        Root-finding method passed to ``scipy.optimize.root_scalar``.

    Returns
    -------
    np.ndarray
        Quantiles corresponding to each target return period.
    """

    # if n is numpy or panda series, this should give u just float
    if not isinstance(n, float):
        n = float(n.values[0])
    else:
        pass

    pr = 1 - 1 / np.array(
        target_return_periods
        )  # Probabilities associated with target_return_periods
    pv = MC_tSMEV_cdf(
        vguess, wbl_phat, n
        )       # Probabilities associated with vguess values
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
            if len(last_valid_idx) > 0:
                first_guess = vguess[last_valid_idx[-1]]
            else:
                first_guess = vguess[-1]

        # Define the function for root finding
        def func(y):
            return MC_tSMEV_cdf(y, wbl_phat, n) - pr[t]

        # Use root_scalar as an alternative to MATLAB's fzero
        result = root_scalar(
            func,
            bracket=[vguess[0], vguess[-1]],
            x0=first_guess,
            method=method_root_scalar
        )

        if result.converged:
            qnt[t] = result.root

    return qnt


def inverse_magnitude_model(
    F_phat,
    eT,
    qs,
    b_exp=False
):
    """
    Calculate percentiles from the Weibell magnitude model

    Parameters
    ----------
    F_phat : numpy.ndarray
        distribution values. F_phat = [kappa_0,b,lambda_0,a].
    eT : numpy.ndarray
        Temperature values from which to produce distribution.
    qs : list
        list of percentiles to calculate (between 0 and 1).
        e.g. [0.85,0.95,0.99].
    b_exp : bool
        If True, uses the exponential rather than linear fit for b.

    Returns
    -------
    percentile_lines : numpy.ndarray
        array with shape length(qs) by length(eT) giving
        the magnitudes for each eT. percentile_lines[0] are
        the values for qs[0].

    """

    percentile_lines = np.zeros((len(qs), len(eT)))
    if b_exp:
        for iq, q in enumerate(qs):
            scale = F_phat[2] * np.exp(F_phat[3] * eT)
            # b is multiplicatve
            shape = F_phat[0] * np.exp(F_phat[1] * eT)
            percentile_lines[iq, :] = scale * (-np.log(1 - q)) ** (1 / shape)

    else:
        for iq, q in enumerate(qs):
            scale = F_phat[2] * np.exp(F_phat[3] * eT)
            # b is additive
            shape = F_phat[0] + F_phat[1] * eT
            percentile_lines[iq, :] = scale * (-np.log(1 - q)) ** (1 / shape)

    return percentile_lines
