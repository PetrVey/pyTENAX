import math
import numpy as np
import pandas as pd
from typing import Union, Tuple, Dict
import statsmodels.api as sm

try:
    from numba import njit as _njit

    @_njit
    def _smev_inner_loop_numba(data, start_indices, end_indices, window_size, n_events):
        # data must be int64 (scaled by 10000) so sums are exact — no floating-point ties
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
        dates: list,
        name_col="value",
        check_gaps=True,
    ) -> list:
        """Function that extracts ordinary precipitation events out of the entire data.
        This also checks and deletes ordinary events with unknown start/end if check_gaps = True.

        Args:
            data (Union[pd.DataFrame, np.ndarray]): Data with precipitation values.
            dates (list): List with dates of precipitation values. dtype must be datetime64[ns].\
                Only relevant if `data` is an array or if `check_gaps==True`.
            name_col (str, optional): Column name in `data` for precipitation values.\
                Only relevant if `data` is a dataframe. Defaults to "value".
            check_gaps (bool, optional): Check for gaps in precipitation time series. \
                Defaults to True.

        Returns:
            list: Consecutive values above `self.min_rain` separated by more than `self.storm_separation_time`.
        """
        if not self.__incomplete_years_removed__:
            raise ValueError(
                "You must run 'remove_incomplete_years' before running this function. "
                "If you are sure your data is complete, set "
                "self.__incomplete_years_removed__ = True to bypass this check."
            )
            
        if isinstance(data, pd.DataFrame):
            # Find values above threshold
            above_threshold = data[data[name_col] > self.min_rain]
            # Find consecutive values above threshold separated by more than storm_separation_time
            consecutive_values = []
            temp = []
            for index, _ in above_threshold.iterrows():
                if not temp:
                    temp.append(index)
                else:
                    if index - temp[-1] > pd.Timedelta(
                        hours=self.storm_separation_time
                    ):
                        if len(temp) >= 1:
                            consecutive_values.append(temp)
                        temp = []
                    temp.append(index)
            if len(temp) >= 1:
                consecutive_values.append(temp)
        elif isinstance(data, np.ndarray):
            above_threshold_indices = np.where(data > self.min_rain)[0]

            # Find consecutive values above threshold separated by more than storm_separation_time
            consecutive_values = []
            temp = []
            for index in above_threshold_indices:
                if not temp:
                    temp.append(index)
                else:
                    # numpy delta is in nanoseconds, it  might be better to do dates[index] - dates[temp[-1]]).item() / np.timedelta64(1, 'm')
                    if (
                        (dates[index] - dates[temp[-1]]).item()
                        > (self.storm_separation_time * 3.6e12)
                    ):  # Assuming 24 is the number of hours, nanoseconds * 3.6e+12 = hours
                        if len(temp) >= 1:
                            consecutive_values.append(dates[temp])
                        temp = []
                    temp.append(index)
            if len(temp) >= 1:
                consecutive_values.append(dates[temp])

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
            # Identify gaps (where the difference is greater than 1 hour)
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
                # Creating an array from start_date to end_date in hourly intervals
                temp_date_array = np.arange(start_date, end_date, time_res)

                # Checking for matching indices in consecutive_values
                for i, sub_array in enumerate(consecutive_values):
                    match_indices = np.where(np.isin(sub_array, temp_date_array))[0]
                    if match_indices.size > 0:
                        match_info.append(i)

            for gap_idx in gap_indices_start:
                start_date = dates[gap_idx]
                end_date = start_date + np.timedelta64(
                    int(self.storm_separation_time * 3.6e12), "ns"
                )
                # Creating an array from start_date to end_date in hourly intervals
                temp_date_array = np.arange(start_date, end_date, time_res)

                # Checking for matching indices in consecutive_values
                for i, sub_array in enumerate(consecutive_values):
                    match_indices = np.where(np.isin(sub_array, temp_date_array))[0]
                    if match_indices.size > 0:
                        match_info.append(i)

            for del_index in sorted(match_info, reverse=True):
                del consecutive_values[del_index]

        return consecutive_values
        
    def get_ordinary_events_new(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        dates: np.ndarray,
        name_col: str = "value",
        check_gaps=True,
        ) -> list:
        """Vectorized ordinary event extraction using np.diff + np.split.

        Functionally equivalent to `get_ordinary_events` (numpy branch) but
        significantly faster by replacing the Python for-loop with vectorized
        numpy operations.

        Args:
            data (Union[pd.DataFrame, np.ndarray]): Data with precipitation values.
                DataFrame support is deprecated and will be removed in a future version.
            dates (np.ndarray): Array with dates of precipitation values. dtype must be datetime64[ns].
            name_col (str, optional): Column name in `data` for precipitation values.
                Only relevant if `data` is a DataFrame. Defaults to "value".
            check_gaps (bool, optional): Check for gaps in precipitation time series.
                Defaults to True.

        Returns:
            list: Consecutive values above `self.min_rain` separated by more than `self.storm_separation_time`.
        """
        if not self.__incomplete_years_removed__:
            raise ValueError(
                "You must run 'remove_incomplete_years' before running this function. "
                "If you are sure your data is complete, set "
                "self.__incomplete_years_removed__ = True to bypass this check."
            )

        if isinstance(data, pd.DataFrame):
            data = np.array(data[name_col])

        above_threshold_indices = np.where(data > self.min_rain)[0]
 
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
            list_ordinary (list): list of indices of ordinary events as returned by `get_ordinary_events()`.

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

        if isinstance(list_ordinary[0][0], pd.Timestamp):
            # event is multiplied by its lenght to get duration and compared with min_event_duration
            ll_short = [
                True
                if ev[-1] - ev[0] + pd.Timedelta(minutes=self.time_resolution)
                >= pd.Timedelta(minutes=self.min_event_duration)
                else False
                for ev in list_ordinary
            ]
            ll_dates = [
                (
                    ev[-1].strftime("%Y-%m-%d %H:%M:%S"),
                    ev[0].strftime("%Y-%m-%d %H:%M:%S"),
                )
                if ev[-1] - ev[0] + pd.Timedelta(minutes=self.time_resolution)
                >= pd.Timedelta(minutes=self.min_event_duration)
                else (np.nan, np.nan)
                for ev in list_ordinary
            ]
            arr_vals = np.array(ll_short)[ll_short]
            arr_dates = np.array(ll_dates)[ll_short]

            filtered_list = [x for x, keep in zip(list_ordinary, ll_short) if keep]
            list_year = pd.DataFrame(
                [filtered_list[_][0].year for _ in range(len(filtered_list))],
                columns=["year"],
            )
            n_ordinary_per_year = list_year.reset_index().groupby(["year"]).count()

        elif isinstance(list_ordinary[0][0], np.datetime64):
            ll_short = [
                True
                if (ev[-1] - ev[0]).astype("timedelta64[m]")
                + np.timedelta64(int(self.time_resolution), "m")
                >= pd.Timedelta(minutes=self.min_event_duration)
                else False
                for ev in list_ordinary
            ]
            ll_dates = [
                (ev[-1], ev[0])
                if (ev[-1] - ev[0]).astype("timedelta64[m]")
                + np.timedelta64(int(self.time_resolution), "m")
                >= pd.Timedelta(minutes=self.min_event_duration)
                else (np.nan, np.nan)
                for ev in list_ordinary
            ]
            arr_vals = np.array(ll_short)[ll_short]
            arr_dates = np.array(ll_dates)[ll_short]

            filtered_list = [x for x, keep in zip(list_ordinary, ll_short) if keep]
            list_year = pd.DataFrame(
                [
                    filtered_list[_][0].astype("datetime64[Y]").item().year
                    for _ in range(len(filtered_list))
                ],
                columns=["year"],
            )
            n_ordinary_per_year = list_year.reset_index().groupby(["year"]).count()

        return arr_vals, arr_dates, n_ordinary_per_year


    def remove_short_new(
        self, list_ordinary: list
    ) -> Tuple[np.ndarray, np.ndarray, pd.Series]:
        """Function that removes ordinary events that are too short.

        Functionally equivalent to `remove_short` (numpy branch) but handles
        pd.Timestamp input by converting to np.datetime64 upfront, so only
        one code path is needed.

        Args:
            list_ordinary (list): list of ordinary events as returned by
                `get_ordinary_events()` or `get_ordinary_events_new()`.
                Each event may contain pd.Timestamp or np.datetime64 values.

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
        self, data: np.ndarray, dates: np.ndarray, arr_dates_oe
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
        """
        Function that extract ordinary events and annual maximas out of precpitation data. 
        
        Parameters
        ----------
        data (np.ndarray): data of full precipitation dataset
        dates (np.ndarray): time of full precipitation dataset
        arr_dates_oe (np.ndarray): end and start of ordinary event as retruned by remove_short function.

        Returns
        -------
        dict_ordinary (dict): key is duration, value is pd.DataFrame with year, oe_time and value of ordinary event (eg. depth)
            contains ordinary events values per duration.
            example dict_ordinary = {"10" : pd.DataFrame(columns=['year', 'oe_time', 'ordinary'])
        dict_AMS (dict): key is duration, value is pd.DataFrame with year and the annual maxima (AMS) value.
            contains anual maximas for each year per duration.

        """
        dict_ordinary = {}
        dict_AMS = {}
        # Scale data to integers so sliding-window sums are exact (no FP ties).
        # Data is assumed to be rounded to 4 decimal places; multiply by 10000 fits in int64.
        data_int = np.round(data * 10000).astype(np.int64)
        for d in range(len(self.durations)):
            arr_conv = np.convolve(
                data_int,
                np.ones(int(self.durations[d] / self.time_resolution), dtype=np.int64),
                "same",
            )

            # Convert time index to numpy array
            time_index = dates.reshape(-1)

            # Use numpy indexing to get the max values efficiently
            ll_vals = []
            ll_dates = []
            for i in range(arr_dates_oe.shape[0]):
                start_time_idx = np.searchsorted(time_index, arr_dates_oe[i, 1])

                end_time_idx = np.searchsorted(time_index, arr_dates_oe[i, 0])

                # Check if start and end times are the same
                if start_time_idx == end_time_idx:
                    ll_val = arr_conv[start_time_idx] / 10000.0
                    ll_date = time_index[start_time_idx]
                else:
                    # the +1 in end_time_index is because then we search by index but we want to includde last as well,
                    # without, it slices eg. end index is 10, without +1 it slices 0 to 9 instead of 0 to 10 (stops 1 before)
                    # get index of ll_val within the sliced array and perform convolve in this slice
                    arr_conv2 = np.convolve(data_int[start_time_idx : end_time_idx + 1],
                                            np.ones(int(self.durations[d] / self.time_resolution), dtype=np.int64),
                                            "same",
                                        )
                    # get index of max value in convolve vector
                    ll_idx_in_slice = np.nanargmax(arr_conv2)

                    # adjust the index to refer to the original arr_conv
                    ll_idx_in_arr_conv = start_time_idx + ll_idx_in_slice
                    ll_val = arr_conv2[ll_idx_in_slice] / 10000.0
                    ll_date = time_index[ll_idx_in_arr_conv]

                ll_vals.append(ll_val)
                ll_dates.append(ll_date)

            # years  of ordinary events
            ll_yrs = [
                arr_dates_oe[_, 0].astype("datetime64[Y]").item().year
                for _ in range(arr_dates_oe.shape[0])
            ]

            blocks = np.unique(ll_yrs)

            AMS = {}
            for j in blocks:
                indices = [index for index, value in enumerate(ll_yrs) if value == j]
                slice_ll_vals = [ll_vals[i] for i in indices]
                AMS[j] = max(slice_ll_vals)

            df_ams = pd.DataFrame({"year": [*AMS.keys()], "AMS": [*AMS.values()]})
            df_oe = pd.DataFrame(
                {"year": ll_yrs, "oe_time": ll_dates, "ordinary": ll_vals}
            )
            dict_AMS.update({f"{self.durations[d]}": df_ams})
            dict_ordinary.update({f"{self.durations[d]}": df_oe})

        return dict_ordinary, dict_AMS

    def get_ordinary_events_values_new(self, data, dates, arr_dates_oe):
        """Optimized version of get_ordinary_events_values.

        Uses np.convolve per event (for exact match with original),
        but optimizes:
        - searchsorted batched once for all durations
        - years precomputed once
        - pre-allocated output arrays instead of list.append
        - vectorized AMS with numpy groupby-style ops
        """
        dict_ordinary = {}
        dict_AMS = {}

        time_index = dates.reshape(-1)
        n_events = arr_dates_oe.shape[0]

        # Batch searchsorted (once for all durations)
        oe_end = arr_dates_oe[:, 0].astype("datetime64[ns]")
        oe_start = arr_dates_oe[:, 1].astype("datetime64[ns]")
        start_indices = np.searchsorted(time_index, oe_start)
        end_indices = np.searchsorted(time_index, oe_end)

        # Precompute years (once for all durations)
        ll_yrs = np.array([
            oe_end[i].astype("datetime64[Y]").item().year
            for i in range(n_events)
        ], dtype=np.int64)

        # Precompute unique years and masks for AMS
        unique_years = np.unique(ll_yrs)
        year_masks = {yr: ll_yrs == yr for yr in unique_years}

        # Scale data to integers so sliding-window sums are exact (no FP ties).
        data_int = np.round(data * 10000).astype(np.int64)

        for d in range(len(self.durations)):
            window_size = int(self.durations[d] / self.time_resolution)
            ones_kernel = np.ones(window_size, dtype=np.int64)

            # Pre-allocate arrays
            max_vals = np.empty(n_events, dtype=np.float64)
            max_global_idx = np.empty(n_events, dtype=np.int64)

            for i in range(n_events):
                si = start_indices[i]
                ei = end_indices[i]

                if si == ei:
                    max_vals[i] = data_int[si] / 10000.0
                    max_global_idx[i] = si
                else:
                    arr_conv2 = np.convolve(data_int[si:ei + 1], ones_kernel, "same")
                    ll_idx_in_slice = np.nanargmax(arr_conv2)
                    max_vals[i] = arr_conv2[ll_idx_in_slice] / 10000.0
                    max_global_idx[i] = si + ll_idx_in_slice

            ll_dates_arr = time_index[max_global_idx]

            # Vectorized AMS
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


    def get_ordinary_events_values_new_numba(self, data, dates, arr_dates_oe):
        """Numba-accelerated version of get_ordinary_events_values.

        Replaces the per-event np.convolve with a JIT-compiled inner loop
        using cumsum-based sliding window sums (O(1) per position instead of
        O(window_size)). Requires numba to be installed.
        """
        if not _NUMBA_AVAILABLE:
            raise ImportError("numba is required for this function. Install with: pip install numba")

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

        # Scale data to integers so sliding-window sums are exact (no FP ties).
        data_int = np.round(data * 10000).astype(np.int64)

        for d in range(len(self.durations)):
            window_size = int(self.durations[d] / self.time_resolution)

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
            blocks_id = dict_ordinary[f"{self.durations[d]}"]["year"]

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