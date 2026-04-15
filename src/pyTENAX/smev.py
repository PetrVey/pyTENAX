import math
import numpy as np
import pandas as pd
from typing import Union, Tuple, Dict
import statsmodels.api as sm

try:
    from numba import njit as _njit, prange as _prange

    @_njit
    def _smev_inner_loop_numba_seq(data, start_indices, end_indices, window_size, n_events):
        max_vals = np.empty(n_events, dtype=np.int64)
        max_global_idx = np.empty(n_events, dtype=np.int64)
        for i in range(n_events):
            si = start_indices[i]
            ei = end_indices[i]
            if si == ei:
                max_vals[i] = data[si]
                max_global_idx[i] = si
            else:
                slice_len = ei - si + 1
                output_len = slice_len if slice_len > window_size else window_size
                min_len = slice_len if slice_len < window_size else window_size
                offset = (min_len - 1) // 2
                best_val = np.int64(-9223372036854775807)
                best_idx = 0
                for j in range(output_len):
                    full_idx = j + offset
                    start_k = full_idx - (window_size - 1)
                    if start_k < 0:
                        start_k = 0
                    end_k = full_idx + 1
                    if end_k > slice_len:
                        end_k = slice_len
                    s = np.int64(0)
                    for k in range(start_k, end_k):
                        s += data[si + k]
                    if s > best_val:
                        best_val = s
                        best_idx = j
                max_vals[i] = best_val
                max_global_idx[i] = si + best_idx
        return max_vals, max_global_idx

    @_njit(parallel=True)
    def _smev_inner_loop_numba(data, start_indices, end_indices, window_size, n_events):
        # data must be int64 (scaled by 10000) so sums are exact — no floating-point ties
        max_vals = np.empty(n_events, dtype=np.int64)
        max_global_idx = np.empty(n_events, dtype=np.int64)
        for i in _prange(n_events):
            si = start_indices[i]
            ei = end_indices[i]
            if si == ei:
                max_vals[i] = data[si]
                max_global_idx[i] = si
            else:
                slice_len = ei - si + 1
                # np.convolve 'same' returns max(n, m) elements; numpy's start offset is (min(n,m)-1)//2
                output_len = slice_len if slice_len > window_size else window_size
                min_len = slice_len if slice_len < window_size else window_size
                offset = (min_len - 1) // 2
                best_val = np.int64(-9223372036854775807)
                best_idx = 0
                for j in range(output_len):
                    full_idx = j + offset
                    start_k = full_idx - (window_size - 1)
                    if start_k < 0:
                        start_k = 0
                    end_k = full_idx + 1
                    if end_k > slice_len:
                        end_k = slice_len
                    s = np.int64(0)
                    for k in range(start_k, end_k):
                        s += data[si + k]
                    if s > best_val:
                        best_val = s
                        best_idx = j
                max_vals[i] = best_val
                max_global_idx[i] = si + best_idx
        return max_vals, max_global_idx

    _NUMBA_AVAILABLE = True

except ImportError:
    _NUMBA_AVAILABLE = False


class SMEV:
    def __init__(
        self,
        return_period: list[Union[int, float]],
        durations: list[int],
        time_resolution: int,
        tolerance: float = 0.1,
        min_event_duration: int = 30,
        storm_separation_time: int = 24,
        left_censoring: list = [0, 1],
        min_rain: Union[float, int] = 0,
        
        
    ):
        """Initiates SMEV class.

        Args:
            return_period (list[Union[int, float]]): List of return periods of interest [years].
            durations (list[Union[int]]): List of durations of interest [min].
            time_resolution (int): Temporal resolution of the precipitation data [min].
            tolerance (float, optional): Maximum allowed fraction of missing data in one year. \
                If exceeded, year will be disregarded from samples. Defaults to 0.1.
            min_event_duration (int, optional): Minimum event duration [min]. Defaults to 30.
            storm_separation_time (int, optional): Separation time between independent storms [hours]. \
                Defaults to 24.
            left_censoring (list, optional): 2-elements list with the limits in probability \
                of the data to be used for the parameters estimation. Defaults to [0, 1].
            min_rain (Union[float, int], optional): Minimum rainfall value. Defaults to 0.
        """
        
        self.return_period = return_period
        self.durations = durations
        self.time_resolution = time_resolution
        self.tolerance = tolerance
        self.min_event_duration = min_event_duration
        self.storm_separation_time = storm_separation_time
        self.left_censoring = left_censoring
        self.min_rain = min_rain

        self.__incomplete_years_removed__ = False
        
        
    def remove_incomplete_years(
        self, data_pr: pd.DataFrame, name_col="value", nan_to_zero=True
    ) -> pd.DataFrame:
        """Function that delete incomplete years in precipitation data.
        An incomplete year is defined as a year where observations are missing above a given threshold.

        Args:
            data_pr (pd.DataFrame): Dataframe containing (hourly) precipitation values.
            name_col (str, optional): Column name in `data_pr` with precipitation values. Defaults to "value".
            nan_to_zero (bool, optional): Set `nan` to zero. Defaults to True.

        Returns:
            pd.DataFrame: Dataframe containing (hourly) precipitation values with incomplete years removed.
        """
        # Step 1: get resolution of dataset (MUST BE SAME in whole dataset!!!)
        time_res = (data_pr.index[-1] - data_pr.index[-2]).total_seconds() / 60
        # Validate: if user provided time_resolution, it must match the data
        if self.time_resolution is not None and self.time_resolution != time_res:
            raise ValueError(
                f"time_resolution provided ({self.time_resolution} min) does not match "
                f"the resolution detected from data ({time_res} min)."
            )
        # Step 2: Resample by year and count total and NaN values
        yearly_valid = data_pr.resample("YE").apply(
            lambda x: x.notna().sum()
        )  # Count not NaNs per year
        # Step 3: Estimate expected lenght of yearly timeseries
        expected = pd.DataFrame(index=yearly_valid.index)
        expected["Total"] = 1440 / time_res * 365  # 1440 stands for the number of minutes in a day
        # Step 4: Calculate percentage of missing data per year by aligning the dimensions
        valid_percentage = yearly_valid[name_col] / expected["Total"]
        # Step 5: Filter out years where more than tolerance% of the values are NaN
        years_to_remove = valid_percentage[valid_percentage < 1 - self.tolerance].index
        # Step 6: Remove data for those years from the original DataFrame
        data_cleanded = data_pr[~data_pr.index.year.isin(years_to_remove.year)]
        # Replace NaN values with 0 in the specific column
        if nan_to_zero:
            data_cleanded.loc[:, name_col] = data_cleanded[name_col].fillna(0)

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

        Args:
            data (Union[pd.DataFrame, np.ndarray]): Precipitation values.
            dates (np.ndarray): Timestamps of the precipitation data.
                dtype must be datetime64[ns].
            name_col (str, optional): Column name to use when ``data`` is a
                DataFrame. Defaults to "value".
            check_gaps (bool, optional): Remove events that fall within
                ``storm_separation_time`` of the dataset boundaries or internal
                data gaps. Defaults to True.

        Returns:
            list: List of np.ndarray, each containing the timestamps of one
                ordinary event (values >= ``self.min_rain`` separated by more
                than ``self.storm_separation_time`` hours).
        """
        if not self.__incomplete_years_removed__:
            raise ValueError(
                "You must run 'remove_incomplete_years' before running this function. "
                "If you are sure your data is complete, set "
                "self.__incomplete_years_removed__ = True to bypass this check."
            )

        if isinstance(data, pd.DataFrame):
            data = np.array(data[name_col])

        above_threshold_indices = np.where(data >= self.min_rain)[0]

        if len(above_threshold_indices) == 0:
            return []
 
        # Get dates at above-threshold positions
        above_dates = dates[above_threshold_indices]
 
        # Compute time differences between consecutive above-threshold timesteps (in nanoseconds)
        time_diffs_above = np.diff(above_dates).astype(np.int64)
 
        # Find where gaps exceed separation time
        separation_ns = int(self.storm_separation_time * 3.6e12)  # hours to nanoseconds
        gap_mask = time_diffs_above > separation_ns
 
        # Split indices at gap locations
        split_points = np.where(gap_mask)[0] + 1
 
        # Split into groups of indices, then map back to dates
        index_groups = np.split(above_threshold_indices, split_points)
 
        # Convert to list of date arrays (same format as original)
        consecutive_values = [dates[group] for group in index_groups]
 
        if check_gaps:
            # remove event that starts before dataset starts in regard of separation time
            if (consecutive_values[0][0] - dates[0]).item() < (
                self.storm_separation_time * 3.6e12
            ):  # this numpy dt, so still in nanoseconds
                consecutive_values.pop(0)
            else:
                pass
 
            # remove event that ends before dataset ends in regard of separation time
            if (dates[-1] - consecutive_values[-1][-1]).item() < (
                self.storm_separation_time * 3.6e12
            ):  # this numpy dt, so still in nanoseconds
                consecutive_values.pop()
            else:
                pass
 
            # Locate OE that ends before gaps in data starts.
            # Calculate the differences between consecutive elements
            time_diffs = np.diff(dates)
            # difference of first element is time resolution
            time_res = time_diffs[0]
            # Identify gaps (where the difference is greater than separation time)
            gap_indices_end = np.where(
                time_diffs
                > np.timedelta64(int(self.storm_separation_time * 3.6e12), "ns")
            )[0]
            # extend by another index in gap cause we need to check if there is OE there too
            gap_indices_start = gap_indices_end + 1
 
            match_info = []
            for gap_idx in gap_indices_end:
                end_date = dates[gap_idx]
                start_date = end_date - np.timedelta64(
                    int(self.storm_separation_time * 3.6e12), "ns"
                )
                temp_date_array = np.arange(start_date, end_date, time_res)
 
                for i, sub_array in enumerate(consecutive_values):
                    match_indices = np.where(np.isin(sub_array, temp_date_array))[0]
                    if match_indices.size > 0:
                        match_info.append(i)
 
            for gap_idx in gap_indices_start:
                start_date = dates[gap_idx]
                end_date = start_date + np.timedelta64(
                    int(self.storm_separation_time * 3.6e12), "ns"
                )
                temp_date_array = np.arange(start_date, end_date, time_res)
 
                for i, sub_array in enumerate(consecutive_values):
                    match_indices = np.where(np.isin(sub_array, temp_date_array))[0]
                    if match_indices.size > 0:
                        match_info.append(i)
 
            for del_index in sorted(match_info, reverse=True):
                del consecutive_values[del_index]
 
        return consecutive_values
        
        
        
    def remove_short(
        self, list_ordinary: list
    ) -> Tuple[np.ndarray, np.ndarray, pd.Series]:
        """Function that removes ordinary events that are too short.

        Args:
            list_ordinary (list): list of ordinary events as returned by
                `get_ordinary_events()`. Each event may contain pd.Timestamp
                or np.datetime64 values.

        Returns:
            arr_vals (np.ndarray): Array with indices of events that are not too short.
            arr_dates (np.ndarray): Array with tuple consisting of start and end dates of events that are not too short.
            n_ordinary_per_year (pd.Series): Series with the number of ordinary events per year.
        """
        if not self.__incomplete_years_removed__:
            raise ValueError(
                "You must run 'remove_incomplete_years' before running this function. "
                "If you are sure your data is complete, set "
                "self.__incomplete_years_removed__ = True to bypass this check."
            )

        # Convert pd.Timestamp events to np.datetime64 if needed
        if isinstance(list_ordinary[0][0], pd.Timestamp):
            list_ordinary = [
                np.array([t.to_datetime64() for t in ev]) for ev in list_ordinary
            ]

        min_duration = np.timedelta64(int(self.min_event_duration), "m")
        time_res = np.timedelta64(int(self.time_resolution), "m")

        ll_short = [
            (ev[-1] - ev[0]).astype("timedelta64[m]") + time_res >= min_duration
            for ev in list_ordinary
        ]
        ll_dates = [
            (ev[-1], ev[0]) if keep else (np.nan, np.nan)
            for ev, keep in zip(list_ordinary, ll_short)
        ]

        arr_vals = np.array(ll_short)[ll_short]
        arr_dates = np.array(ll_dates)[ll_short]

        filtered_list = [ev for ev, keep in zip(list_ordinary, ll_short) if keep]
        list_year = pd.DataFrame(
            [ev[0].astype("datetime64[Y]").item().year for ev in filtered_list],
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
            End and start times of ordinary events as returned by remove_short.
        method : str, optional
            Backend used for the sliding-window maximum search. Defaults to
            ``"vectorized"``. One of:

            - ``"vectorized"``    — pure numpy, ``np.convolve`` per event.
            - ``"njit"``          — numba JIT-compiled loop, single-threaded.
            - ``"njit_parallel"`` — numba JIT-compiled loop, parallelised over
              events. Requires ``numba`` to be installed.

        Notes
        -----
        When using ``method="njit"`` or ``method="njit_parallel"``, the first
        call in a Python session triggers JIT compilation and can take several
        seconds. Run a warmup call before any timed or production code::

            # Warmup — compile both kernels once at session start
            S_SMEV.get_ordinary_events_values(
                data=df_arr, dates=df_dates, arr_dates_oe=arr_dates,
                method="njit"
            )
            S_SMEV.get_ordinary_events_values(
                data=df_arr, dates=df_dates, arr_dates_oe=arr_dates,
                method="njit_parallel"
            )
            # Subsequent calls are fast
            dict_ordinary, dict_AMS = S_SMEV.get_ordinary_events_values(
                data=df_arr, dates=df_dates, arr_dates_oe=arr_dates,
                method="njit_parallel"
            )

        Returns
        -------
        dict_ordinary : dict
            Key is duration (str), value is a ``pd.DataFrame`` with columns
            ``year``, ``oe_time``, ``ordinary`` (event depth/intensity).
            Example: ``{"10": pd.DataFrame(columns=['year', 'oe_time', 'ordinary'])}``.
        dict_AMS : dict
            Key is duration (str), value is a ``pd.DataFrame`` with columns
            ``year`` and ``AMS`` (annual maximum value).
        """
        if method in ("njit", "njit_parallel") and not _NUMBA_AVAILABLE:
            raise ImportError("numba is required for method='njit'/'njit_parallel'. Install with: pip install numba")

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
                        arr_conv = np.convolve(data_int[si:ei + 1], ones_kernel, "same")
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
            ams_vals = np.array([np.max(max_vals[mask]) for yr, mask in year_masks.items()])

            df_ams = pd.DataFrame({"year": unique_years, "AMS": ams_vals})
            df_oe = pd.DataFrame({
                "year": ll_yrs,
                "oe_time": ll_dates_arr,
                "ordinary": max_vals,
            })
            dict_AMS[f"{self.durations[d]}"] = df_ams
            dict_ordinary[f"{self.durations[d]}"] = df_oe

        return dict_ordinary, dict_AMS


    def estimate_smev_parameters(
        self, ordinary_events: Union[np.ndarray, pd.Series, list], data_portion: list[Tuple[int, float]]
    ) -> list[float]:
        """Function that estimates shape and scale parameters of the Weibull distribution.

        Args:
            ordinary_events ([np.ndarray, pd.Series, list): values of ordinary events.
            data_portion (list): Lower and upper limits of the probabilities of data \
                to be used for the parameters estimation.

        Returns:
            list[float]: Shape and scale parameters of the Weibull distribution.
        """

        sorted_df = np.sort(ordinary_events)
        ECDF = np.arange(1, 1 + len(sorted_df)) / (1 + len(sorted_df))
        #fidx: first index of data to keep
        fidx = max(1, math.floor((len(sorted_df)) * data_portion[0]))
        #tidx: last index of data to keep
        tidx = math.ceil(len(sorted_df) * data_portion[1])
        if fidx == 1: #this is check basically if censoring set to [0,1], if so, we take all values
            to_use = np.arange(fidx-1, tidx) # Create an array of indices from fidx-1 up to tidx (inclusive)
        else: # else, we take only from this fidx, eg. [0.5,1] out of 1000 samples will take 500-999 indexes (top 500)
            to_use = np.arange(fidx, tidx) # Create an array of indices from fidx up to tidx (inclusive)
        # Select only the subset of sorted values corresponding to the chosen quantile range
        to_use_array = sorted_df[to_use]

        X = np.log(np.log(1 / (1 - ECDF[to_use])))
        Y = np.log(to_use_array)
        X = sm.add_constant(X)
        model = sm.OLS(Y, X)
        results = model.fit()
        param = results.params

        slope = float(param[1])
        intercept = float(param[0])
        shape = 1 / slope
        scale = np.exp(intercept)
        weibull_param = [shape, scale]

        return weibull_param

    def smev_return_values(
        self, return_period: int, shape: float, scale: float, n: float
    ) -> float:
        """Function that calculates return values (here, rainfall intensity) acoording to parameters of the Weibull distribution.

        Args:
            return_period (int): Return period of interest.
            shape (float): Shape parameter value.
            scale (float): Scale parameter value.
            n (float): SMEV parameter `n`.

        Returns:
            float: Rainfall intensity value.
        """

        return_period = np.asarray(return_period)
        quantile = 1 - (1 / return_period)
        if shape == 0 or n == 0:
            intensity = 0
        else:
            intensity = scale * ((-1) * (np.log(1 - quantile ** (1 / n)))) ** (
                1 / shape
            )

        return intensity

    def do_smev_all(
        self,
        dict_ordinary: Dict[str, pd.DataFrame],
        n: float,
    ) -> Dict[str, pd.DataFrame]:
        """Run SMEV parameter estimation and return level computation for all durations.

        Args:
            dict_ordinary (Dict[str, pd.DataFrame]): Dictionary of ordinary events per duration,
                as returned by get_ordinary_events_values.
            n (float): Mean number of ordinary events per year.

        Returns:
            Dict[str, pd.DataFrame]: Dictionary with SMEV parameters and return levels per duration.
                Each entry has keys 'SMEV_phat' (list[shape, scale]) and 'RLs' (return levels).
        """
        dict_smev_outputs = {}
        for d in range(len(self.durations)):
            P = dict_ordinary[f"{self.durations[d]}"]["ordinary"]

            # Estimate shape and scale parameters of weibull distribution
            smev_shape, smev_scale = self.estimate_smev_parameters(
                P, self.left_censoring
            )

            # Estimate return period (quantiles) with SMEV
            smev_RL = self.smev_return_values(
                self.return_period, smev_shape, smev_scale, n
            )

            dict_smev_outputs[f"{self.durations[d]}"] = {
                "SMEV_phat": [smev_shape, smev_scale],
                "RLs": smev_RL,
            }

        return dict_smev_outputs

    def _run_smev_all_durations(
        self,
        dict_ordinary: Dict[str, pd.DataFrame],
        n: float,
    ) -> pd.DataFrame:
        """Estimate SMEV parameters and return levels for all durations.

        Convenience wrapper around :meth:`estimate_smev_parameters` and
        :meth:`smev_return_values` that iterates over all configured durations
        and collects results into a single summary DataFrame.

        Parameters
        ----------
        dict_ordinary : Dict[str, pd.DataFrame]
            Ordinary events per duration as returned by
            :meth:`get_ordinary_events_values`.
        n : float
            Mean number of ordinary events per year (from
            :meth:`remove_short`).

        Returns
        -------
        pd.DataFrame
            One row per duration (index e.g. ``"10 min"``), columns:
            ``N_oe``, ``n_mean``, ``shape``, ``scale``,
            and one column per return period (``"RP 2yr"``, ``"RP 5yr"``, …).
        """
        rows = {}
        for dur in [str(d) for d in self.durations]:
            P = dict_ordinary[dur]["ordinary"].to_numpy()
            shape, scale = self.estimate_smev_parameters(P, self.left_censoring)
            RL = self.smev_return_values(self.return_period, shape, scale, n)
            rows[f"{dur} min"] = (
                [len(P), round(n, 2), round(shape, 4), round(scale, 4)]
                + [round(v, 2) for v in RL]
            )

        col_names = ["N_oe", "n_mean", "shape", "scale"] + [
            f"RP {rp}yr" for rp in self.return_period
        ]
        return pd.DataFrame(rows, index=col_names).T

    @staticmethod
    def get_stats(
        df: pd.DataFrame,
    ) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """Computes statistics of precipitation values.
        Statistics are total percipitation per year, mean precipitation per year,
        standard deviation of precipitation per year, and count of precipitation events per year.

        Args:
            df (pd.DataFrame): Dataframe with precipitation values.

        Returns:
            pd.Series: Total percipitation per year.
            pd.Series: Mean percipitation per year.
            pd.Series: Standard deviation of percipitation per year.
            pd.Series: Count of percipitation events per year.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df is not a pandas dataframe")

        total_prec = df.groupby(df.index.year)["value"].sum()
        mean_prec = (
            df[df.value > 0].groupby(df[df.value > 0].index.year)["value"].mean()
        )
        sd_prec = df[df.value > 0].groupby(df[df.value > 0].index.year)["value"].std()
        count_prec = (
            df[df.value > 0].groupby(df[df.value > 0].index.year)["value"].count()
        )

        return total_prec, mean_prec, sd_prec, count_prec

    def SMEV_bootstrap_uncertainty(
        self, P: np.ndarray, blocks_id: np.ndarray, niter: int, n: float
    ):
        """Function that bootstraps uncertainty of SMEV return values.

        Args:
            P (np.ndarray): Array of precipitation data.
            blocks_id (np.ndarray): Array of block identifiers (e.g., years).
            niter (int): Number of bootstrap iterations.
            n (float): SMEV parameter `n`.

        Returns:
            np.ndarray: Array with bootstrapped return value uncertainty.
        """
        RP = self.return_period

        blocks = np.unique(blocks_id)
        M = len(blocks)
        randy = np.random.randint(0, M, size=(M, niter))

        # Initialize variables
        RL_unc = np.full((niter, len(RP)), np.nan)
        n_err = 0

        # Random sampling iterations
        for ii in range(niter):
            Pr = []
            Bid = []

            # Create bootstrapped data sample and corresponding 'fake' blocks id
            for iy in range(M):
                selected = blocks_id == blocks[randy[iy, ii]]
                Pr.append(P[selected])
                Bid.append(
                    np.full(np.sum(selected), iy + 1)
                )  # MATLAB indexing starts at 1

            # Concatenate the resampled data
            Pr = np.concatenate(Pr)
            Bid = np.concatenate(Bid)

            try:
                # estimate shape and  scale parameters of weibull distribution
                SMEV_shape, SMEV_scale = self.estimate_smev_parameters(
                    Pr, self.left_censoring
                )
                # estimate return period (quantiles) with SMEV
                smev_RP = self.smev_return_values(
                    self.return_period, SMEV_shape, SMEV_scale, n
                )
                # Store results
                RL_unc[ii, :] = smev_RP

            except Exception:
                n_err += 1
        return RL_unc