"""
Copied on 6/11/2024 4:22PM
source code:
https://github.com/luigicesarini/pysmev/blob/main/src/pysmev/smev_class.py
"""

import math
import numpy as np
import pandas as pd
from typing import Union
import statsmodels.api as sm


class SMEV:
    def __init__(
        self,
        threshold,
        separation: int,
        return_period: list[Union[int, float]],
        durations: list[Union[int, float]],
        time_resolution,
        min_duration: Union[None, int] = None,
        left_censoring: Union[None, list] = None,
    ):
        self.threshold = threshold
        self.separation = separation
        self.return_period = return_period
        self.durations = durations
        self.time_resolution = time_resolution  # time resolution
        self.min_duration = (
            min_duration if min_duration is not None else 0
        )  # min duration of precipitation
        self.left_censoring = left_censoring if left_censoring is not None else [0, 1]

    def get_ordinary_events(
        self, data: np.ndarray, dates, name_col="value", check_gaps=True
    ) -> list:
        """

        Function that extracts ordinary precipitation events out of the entire data.

        Parameters
        ----------
        - data np.array: array containing the hourly values of precipitation.
        - separation (int): The number of hours used to define an independet ordianry event. Defult: 24 hours. this is saved in SMEV S class
                        Days with precipitation amounts above this threshold are considered as ordinary events.
        - pr_field (string): The name of the df column with precipitation values.
        - hydro_year_field (string): The name of the df column with hydrological years values.

        Returns
        -------
        - consecutive_values np.array: index of time of consecutive values defining the ordinary events.


        Examples
        --------
        """
        if isinstance(data, pd.DataFrame):
            # Find values above threshold
            above_threshold = data[data[name_col] > self.threshold]
            # Find consecutive values above threshold separated by more than 24 observations
            consecutive_values = []
            temp = []
            for index, row in above_threshold.iterrows():
                if not temp:
                    temp.append(index)
                else:
                    if index - temp[-1] > pd.Timedelta(hours=self.separation):
                        if len(temp) >= 1:
                            consecutive_values.append(temp)
                        temp = []
                    temp.append(index)
            if len(temp) >= 1:
                consecutive_values.append(temp)
        elif isinstance(data, np.ndarray):
            # Assuming data is your numpy array
            # Assuming name_col is the index for comparing threshold
            # Assuming threshold is the value above which you want to filter

            above_threshold_indices = np.where(data > 0)[0]

            # Find consecutive values above threshold separated by more than 24 observations
            consecutive_values = []
            temp = []
            for index in above_threshold_indices:
                if not temp:
                    temp.append(index)
                else:
                    # numpy delta is in nanoseconds, it  might be better to do dates[index] - dates[temp[-1]]).item() / np.timedelta64(1, 'm')
                    if (
                        (dates[index] - dates[temp[-1]]).item()
                        > (self.separation * 3.6e12)
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
                self.separation * 3.6e12
            ):  # this numpy dt, so still in nanoseconds
                consecutive_values.pop(0)
            else:
                pass

            # remove event that ends before dataset ends in regard of separation time
            if (dates[-1] - consecutive_values[-1][-1]).item() < (
                self.separation * 3.6e12
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
                time_diffs > np.timedelta64(int(self.separation * 3.6e12), "ns")
            )[0]
            # extend by another index in gap cause we need to check if there is OE there too
            gap_indices_start = gap_indices_end + 1

            match_info = []
            for gap_idx in gap_indices_end:
                end_date = dates[gap_idx]
                start_date = end_date - np.timedelta64(
                    int(self.separation * 3.6e12), "ns"
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
                    int(self.separation * 3.6e12), "ns"
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

    def remove_short(self, list_ordinary: list, min_duration: int):
        """

        Function that removes ordinary events too short.

        Parameters
        ----------
        - self.time_resolution: Used to calculate lenght of storm
        - list_ordinary list: list of indices of ordinary events as returned by `get_ordinary_events`.
        - min_duration (int): minimun number of minutes tto define an event.
        - pr_field (string): The name of the df column with precipitation values.
        - hydro_year_field (string): The name of the df column with hydrological years values.

        Returns
        -------
        - consecutive_values np.array: index of time of consecutive values defining the ordinary events.


        Examples
        --------
        """
        if isinstance(list_ordinary[0][0], pd.Timestamp):
            ll_short = [
                True
                if ev[-1] - ev[0] + pd.Timedelta(minutes=self.time_resolution)
                >= pd.Timedelta(minutes=min_duration)
                else False
                for ev in list_ordinary
            ]
            ll_dates = [
                (
                    ev[-1].strftime("%Y-%m-%d %H:%M:%S"),
                    ev[0].strftime("%Y-%m-%d %H:%M:%S"),
                )
                if ev[-1] - ev[0] + pd.Timedelta(minutes=self.time_resolution)
                >= pd.Timedelta(minutes=min_duration)
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
            n_ordinary = (
                n_ordinary_per_year.values.mean()
            )  # ordinary events mean value (one of input to smev)
        elif isinstance(list_ordinary[0][0], np.datetime64):
            ll_short = [
                True
                if (ev[-1] - ev[0]).astype("timedelta64[m]")
                + np.timedelta64(int(self.time_resolution), "m")
                >= pd.Timedelta(minutes=min_duration)
                else False
                for ev in list_ordinary
            ]
            ll_dates = [
                (ev[-1], ev[0])
                if (ev[-1] - ev[0]).astype("timedelta64[m]")
                + np.timedelta64(int(self.time_resolution), "m")
                >= pd.Timedelta(minutes=min_duration)
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
            n_ordinary = (
                n_ordinary_per_year.values.mean()
            )  # ordinary events mean value (one of input to smev)

        return arr_vals, arr_dates, n_ordinary_per_year, n_ordinary

    def estimate_smev_parameters(self, ordinary_events_df, data_portion):
        """

        Function that estimates parameters of the Weibull distribution

        Parameters
        ----------
        - ordinary_events_df (dataframe): Pandas dataframe of the ordinary events - withot zeros!!!
        - pr_field (string): The name of the df column with precipitation values.
        - data_portion (list): 2-elements list with the limits in probability of the data to be used for the parameters estimation
        e.g. data_portion = [0.75, 1] uses the largest 25% values in data

        Returns
        -------
        - weibull_param (list): [shape,scale]

        Examples
        --------
        """

        sorted_df = np.sort(ordinary_events_df)
        ECDF = np.arange(1, 1 + len(sorted_df)) / (1 + len(sorted_df))
        fidx = max(1, math.floor((len(sorted_df)) * data_portion[0]))
        tidx = math.ceil(len(sorted_df) * data_portion[1])
        to_use = np.arange(fidx - 1, tidx)
        to_use_array = sorted_df[to_use]

        X = np.log(np.log(1 / (1 - ECDF[to_use])))
        Y = np.log(to_use_array)
        X = sm.add_constant(X)
        model = sm.OLS(Y, X)
        results = model.fit()
        param = results.params

        slope = param[1]
        intercept = param[0]

        shape = 1 / slope
        scale = np.exp(intercept)

        weibull_param = [shape, scale]

        return weibull_param

    def smev_return_values(self, return_period, shape, scale, n):
        """

        Function that calculates return values acoording to parameters of the Weibull distribution

        Parameters
        ----------
        - return_period (int): The desired return period for which intensity is calculated.
        - shape (float): Weibull distribution shape parameter
        - scale (float): Weibull distribution scale parameter
        - n (float): Mean number of ordinary events per year

        Returns
        -------
        - intensity (float): The corresponding intensity value.

        Examples
        --------
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

    def get_stats(df):
        assert isinstance(df, pd.DataFrame), "df is not a pandas dataframe"

        total_prec = df.groupby(df.index.year)["value"].sum()
        mean_prec = (
            df[df.value > 0].groupby(df[df.value > 0].index.year)["value"].mean()
        )
        sd_prec = df[df.value > 0].groupby(df[df.value > 0].index.year)["value"].std()
        count_prec = (
            df[df.value > 0].groupby(df[df.value > 0].index.year)["value"].count()
        )

        return total_prec, mean_prec, sd_prec, count_prec

    def SMEV_bootstrap_uncertainty(self, P, blocks_id, niter, n):
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
