import pandas as pd
import numpy as np
from scipy.special import gamma
from scipy.stats import weibull_min, norm, skewnorm, chi2
from scipy.optimize import root_scalar, minimize
import time
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from typing import Union, Tuple


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
        non-asymptotic theoretical framework that incorporates temperature as a covariate
        in a physically consistent manner.

        Args:
            return_period (list[Union[int, float]]): Return periods [years].
            durations (list[int]): Duration of interest [min].
            beta (Union[float, int], optional): Shape parameter of the Generalized Normal for g(T). Defaults to 4.
            temp_time_hour (int, optional): Time window to compute T [h]. Will be converted to negative if needed. Defaults to 24.
            alpha (float, optional): Unitless significance level for the dependence of the shape on T. Defaults to 0.05.
                - alpha = 0 --> dependence of shape on T is always allowed.
                - alpha = 1 --> dependence of shape on T is never allowed.
                - 0 < alpha < 1 --> dependence of shape on T depends on statistical significance at the alpha-level.
            n_monte_carlo (int, optional): Number of elements in the MC samples. Defaults to int(2e4).
            tolerance (float, optional): Maximum allowed fraction of missing data in one year. If exceeded, year will be disregarded from samples. Defaults to 0.1.
            min_event_duration (int, optional): Minimum event duration [min]. Defaults to 30.
            storm_separation_time (int, optional): Separation time between independent storms [hours]. Defaults to 24.
            left_censoring (list, optional): 2-elements list with the limits in probability of the data to be used for the parameters estimation. Defaults to [0, 1].
            niter_smev (int, optional): Number of iterations for uncertainty for the SMEV model. Defaults to 100.
            niter_tenax (int, optional): Number of iterations for uncertainty for the TENAX model. Defaults to 100.
            temp_res_monte_carlo (float, optional): Resolution in T for the MC samples. Defaults to 0.001.
            temp_delta (int, optional): Range in T of MC samples. Explores temperatures up to Tdelt degrees higher and lower of the observed ones. Defaults to 10.
            init_param_guess (list, optional): Initial values of Weibull parameters for `fminsearch`. Defaults to [0.7, 0, 2, 0].
            min_rain (Union[float, int], optional): Minimum rainfall value. Defaults to 0.
        """
        self.return_period = return_period
        self.durations = durations
        self.time_resolution = time_resolution
        self.beta = beta
        self.temp_time_hour = temp_time_hour if temp_time_hour < 0 else -temp_time_hour
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
        # Step 2: Resample by year and count total and NaN values
        yearly_valid = data_pr.resample("YE").apply(
            lambda x: x.notna().sum()
        )  # Count not NaNs per year
        # Step 3: Estimate expected lenght of yearly timeseries
        expected = pd.DataFrame(index=yearly_valid.index)
        expected["Total"] = 1440 / time_res * 365
        # Step 4: Calculate percentage of missing data per year by aligning the dimensions
        valid_percentage = yearly_valid[name_col] / expected["Total"]
        # Step 3: Filter out years where more than 10% of the values are NaN
        years_to_remove = valid_percentage[valid_percentage < 1 - self.tolerance].index
        # Step 4: Remove data for those years from the original DataFrame
        data_cleanded = data_pr[~data_pr.index.year.isin(years_to_remove.year)]
        # Replace NaN values with 0 in the specific column
        if nan_to_zero:
            data_cleanded.loc[:, name_col] = data_cleanded[name_col].fillna(0)

        self.time_resolution = time_res

        self.__incomplete_years_removed__ = True

        return data_cleanded

    def get_ordinary_events(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        dates,
        name_col="value",
        check_gaps=True,
    ) -> list:
        """

        Function that extracts ordinary precipitation events out of the entire data.

        Parameters
        ----------
        - data np.array: array containing the hourly values of precipitation.
        - separation (int): The number of hours used to define an independet ordianry event. Defult: 24 hours. this is saved in SMEV S class
                        Days with precipitation amounts above this threshold are considered as ordinary events.
        - name_col (string): The name of the df column with precipitation values.
        - check_gaps (bool): This also check for gaps in data and for unknown start/end ordinary events

        Returns
        -------
        - consecutive_values np.array: index of time of consecutive values defining the ordinary events.


        Examples
        --------
        """
        if isinstance(data, pd.DataFrame):
            # Find values above threshold
            above_threshold = data[data[name_col] > self.min_rain]
            # Find consecutive values above threshold separated by more than 24 observations
            consecutive_values = []
            temp = []
            for index, row in above_threshold.iterrows():
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
            # Assuming data is your numpy array
            # Assuming name_col is the index for comparing threshold
            # Assuming threshold is the value above which you want to filter

            above_threshold_indices = np.where(data > self.min_rain)[0]

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

    def remove_short(
        self, list_ordinary: list
    ) -> Tuple[np.ndarray, np.ndarray, pd.Series]:
        """Function that removes ordinary events too short.

        Args:
            list_ordinary (list): list of indices of ordinary events as returned by `get_ordinary_events()`.

        Returns:
            np.ndarray: Array with indices of events that are not too short.
            np.ndarray: Array with tuple consisting of start and end dates of events that are not too short.
            pd.Series: Series with the number of ordinary events per year.
        """
        if not self.__incomplete_years_removed__:
            raise ValueError(
                "You must run 'remove_incomplete_years' before running this function."
            )

        if isinstance(list_ordinary[0][0], pd.Timestamp):
            # event is multiplied by its lenght to get duration and compared with min_duration setup
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

    def get_ordinary_events_values(
        self, data: np.ndarray, dates: np.ndarray, arr_dates_oe
    ):
        """
        Parameters
        ----------
        data : np array
            data of full dataset
        dates : np array
            time of full dataset
        arr_dates_oe : TYPE
            end and start of ordinary event, this is output from remove_short function.

        Returns
        -------
        dict_ordinary : dict of pandas
            ordinary events per duration.
            dict_ordinary = {"10" : pd.DataFrame(columns=['year', 'oe_time', 'ordinary'])
        dict_AMS : dict of pandas
            contains anual maximas for each year per duration.

        """
        dict_ordinary = {}
        dict_AMS = {}
        for d in range(len(self.durations)):
            arr_conv = np.convolve(
                data,
                np.ones(int(self.durations[d] / self.time_resolution), dtype=int),
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
                    ll_val = arr_conv[start_time_idx]
                    ll_date = time_index[start_time_idx]
                else:
                    # the +1 in end_time_index is because then we search by index but we want to includde last as well,
                    # without, it slices eg. end index is 10, without +1 it slices 0 to 9 instead of 0 to 10 (stops 1 before)
                    # get index of ll_val within the sliced array
                    ll_idx_in_slice = np.nanargmax(
                        arr_conv[start_time_idx : end_time_idx + 1]
                    )
                    # adjust the index to refer to the original arr_conv
                    ll_idx_in_arr_conv = start_time_idx + ll_idx_in_slice
                    ll_val = arr_conv[ll_idx_in_arr_conv]
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

    def associate_vars(self, dict_ordinary, data_temperature, dates_temperature):
        """
        Get additional variables for the ordinary events #TODO: have no clue about this one

        Parameters
        ----------
        dict_ordinary : TYPE
            DESCRIPTION.
        data_temperature : TYPE
            DESCRIPTION.
        dates_temperature : TYPE
            DESCRIPTION.

        Returns
        -------
        dict_ordinary : TYPE
            DESCRIPTION.
        dict_dropped_oe : TYPE
            DESCRIPTION.
        n_ordinary_per_year_new : TYPE
            DESCRIPTION.

        """
        # start here
        dict_dropped_oe = {}
        time_index = dates_temperature.reshape(-1)

        delta_time = np.timedelta64(int(self.temp_time_hour), "h")
        for d in self.durations:
            df_oe = dict_ordinary[f"{d}"]
            arr_dates_oe = np.array(df_oe["oe_time"])
            ll_vals = []

            # Use vectorized search for indices using pandas `merge_asof`
            df_time_index = pd.DataFrame({"time_index": time_index})
            df_arr_dates_oe = pd.DataFrame({"oe_time": arr_dates_oe})

            # Use pandas to perform an "as-of" merge that efficiently finds the closest index, 30 are handled to nearest lower
            merged = pd.merge_asof(
                df_arr_dates_oe,
                df_time_index,
                left_on="oe_time",
                right_on="time_index",
                direction="nearest",
            )

            for _, row in merged.iterrows():
                end_time = row["time_index"]

                # Find the index of the closest time directly using `np.searchsorted`
                if end_time is None:
                    continue  # Skip this iteration if no match was found

                # Use `np.searchsorted` to find the index in `time_index`
                closest_idx = np.searchsorted(time_index, np.datetime64(end_time))

                # Calculate end time with delta
                end_time_minus_delta = time_index[closest_idx] + delta_time

                # Find start index efficiently
                start_time_idx = np.searchsorted(time_index, end_time_minus_delta)

                # Slice array more efficiently
                ll_idx_in_slice_vals = data_temperature[
                    start_time_idx : closest_idx + 1
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
                dict_ordinary[f"{d}"].dropna(subset=["T"]).reset_index(drop=True)
            )

        # Recalculate number of OE
        n_ordinary_per_year_new = (
            dict_ordinary[f"{d}"]
            .groupby(["year"])["ordinary"]
            .count()
            .to_frame(name="N_oe")
        )

        return dict_ordinary, dict_dropped_oe, n_ordinary_per_year_new

    def magnitude_model(self, data_oe_prec, data_oe_temp, thr, b_set=None):
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

        Returns
        -------
        phat : numpy.ndarray
            Parameters of the magnitude model. [kappa_0,b,lambda_0,a].
        loglik : numpy.float64
            Log likelihood.
        loglik_H1 : numpy.float64
            Log likelihood of alternative hypothesis.
        loglik_H0shape : numpy.float64
            Log likelihood of null hypothesis.

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
            min_phat_bset = minimize(
                lambda theta: -wbl_leftcensor_loglik_bset(theta, P, T, thr, b_set),
                init_g,
                method="Nelder-Mead",
            )
            phat_bset = min_phat_bset.x
            loglik_bset = wbl_leftcensor_loglik_bset(phat_bset, P, T, thr, b_set)
            phat_bset[1] = b_set
            phat = phat_bset
            loglik = loglik_bset
            loglik_H1, loglik_H0shape = (
                None,
                None,
            )  # TODO: figure this out, do we need these outputs?

        else:
            min_phat_H1 = minimize(
                lambda theta: -wbl_leftcensor_loglik(theta, P, T, thr),
                init_g,
                method="Nelder-Mead",
            )
            phat_H1 = min_phat_H1.x

            min_phat_H0shape = minimize(
                lambda theta: -wbl_leftcensor_loglik_H0shape(theta, P, T, thr),
                init_g,
                method="Nelder-Mead",
                options={"xatol": 1e-8, "fatol": 1e-8, "maxiter": 1000},
            )

            phat_H0shape = min_phat_H0shape.x
            phat_H0shape[1] = 0

            loglik_H1 = wbl_leftcensor_loglik(phat_H1, P, T, thr)
            loglik_H0shape = wbl_leftcensor_loglik_H0shape(phat_H0shape, P, T, thr)
            lambda_LR_shape = -2 * (loglik_H0shape - loglik_H1)
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

    def temperature_model(self, data_oe_temp, beta=0, method="norm"):
        """
        Fits the temperature data to the TENAX temperature model.

        Parameters
        ----------
        data_oe_temp : numpy.ndarray
            Temperature data.
        beta : float, optional
            beta of the generalised normal distribution. if not defined, uses the beta defined in S. The default is 0.
        method : string, optional
            Type of fit. "norm" is for the generalised normal distribution. "skewnorm" is for a skewed normal distribution. The default is "norm".

        Returns
        -------
        g_phat: numpy.array
            parameters of the temperature distribution. if "norm", [shape,scale]. if "skewnorm", also has skew. #TODO: I couldnt actually figure which was which for the skewnorm g_phat

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

            hist, bin_edges = np.histogram(data_oe_temp, bins=100, density=True)
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
            Specify whether to generate Monte Carlo values for precipitation. The default is False.
        gen_RL : bool, optional
            Specify whether to generate return levels. The default is True.
        temp_method : str, optional
            Type of fit used for the temperature model. The default is "norm".
        method_root_scalar : str, optional
            method used for inversion. The default is "brentq".

        Returns
        -------
        ret_lev : list (?) #TODO: check
            Return levels at periods specified in self.return_period.
        T_mc : numpy.ndarray
            Monte Carlo generated temperature values.
        P_mc : numpy.ndarray
            Monte Carlo generated precipitation values.

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
        wbl_phat = np.column_stack(
            (F_phat[2] * np.exp(F_phat[3] * T_mc), F_phat[0] + F_phat[1] * T_mc)
        )

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

    # uncerteinty TENAX MODEL HERE
    def TNX_tenax_bootstrap_uncertainty(
        self, P, T, blocks_id, Ts, temp_method="norm", method_root_scalar="brentq"
    ):
        """
        Bootstrap uncertainty estimation for the TENAX model.

        Parameters:
        - P: numpy array of precipitation data.
        - T: numpy array of temperature data.
        - blocks_id: numpy array of block identifiers (e.g., years).
        - perc_thres: percentile threshold for left-censoring.
        - S: object containing model parameters and methods.
        - RP: return periods (numpy array).
        - N: number of Monte Carlo simulations.
        - Ts: time scales (numpy array).
        - niter: number of bootstrap iterations.

        Returns:
        - F_phat_unc: array of magnitude model parameters from bootstrap samples.
        - g_phat_unc: array of temperature model parameters from bootstrap samples.
        - RL_unc: array of estimated return levels from bootstrap samples.
        - n_unc: array of mean number of events per block from bootstrap samples.
        - n_err: number of iterations where the model fitting failed.
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

        # Random sampling iterations
        for ii in range(niter):
            Pr = []
            Tr = []
            Bid = []

            # Create bootstrapped data sample and corresponding 'fake' blocks id
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

            try:
                # Left-censoring threshold
                # TODO: double check on this, I think it should be from Pr not P
                thr = np.quantile(P, perc_thres)

                # TENAX model components
                # Magnitude model
                F_phat_temporary, loglik_temp, _, _ = self.magnitude_model(Pr, Tr, thr)
                # Temperature model
                g_phat_temporary = self.temperature_model(Tr, method=temp_method)
                # Mean number of events per block
                n_temporary = len(Pr) / M
                # Estimate return levels using Monte Carlo samples
                # TODO: check this cause it is slow...
                RL_temporary, _, _ = self.model_inversion(
                    F_phat_temporary,
                    g_phat_temporary,
                    n_temporary,
                    Ts,
                    temp_method=temp_method,
                    method_root_scalar=method_root_scalar,
                )

                # Store results
                F_phat_unc[ii, :] = F_phat_temporary
                g_phat_unc[ii, :] = g_phat_temporary
                RL_unc[ii, :] = RL_temporary
                n_unc[ii] = n_temporary
            except Exception:
                n_err += 1

        return F_phat_unc, g_phat_unc, RL_unc, n_unc, n_err


def wbl_leftcensor_loglik(theta, x, t, thr):
    """
    TODO: I dont understand these things

    Parameters
    ----------
    theta : float
        initial guess for fit.
    x : numpy.ndarray
        precipitation values.
    t : numpy.ndarray
        temperature values.
    thr : float
        threshold value for left-censoring.

    Returns
    -------
    loglik : TYPE
        DESCRIPTION.

    """
    # theta is init guess
    # x is precipitaon\
    # t is temperature
    # thr is threshold value (exact, no percentual)
    a_w = theta[0]
    b_w = theta[1]
    a_C = theta[2]
    b_C = theta[3]

    # Apply conditions based on the threshold
    t0 = t[x < thr]
    shapes0 = a_w + b_w * t0
    scales0 = a_C * np.exp(b_C * t0)

    x1 = x[x >= thr]
    t1 = t[x >= thr]
    shapes1 = a_w + b_w * t1
    scales1 = a_C * np.exp(b_C * t1)

    # Calculate the log-likelihood components
    loglik1 = np.sum(np.log(weibull_min.cdf(thr, c=shapes0, scale=scales0)))
    loglik2 = np.sum(np.log(weibull_min.pdf(x1, c=shapes1, scale=scales1)))

    # Sum the components for the final log-likelihood
    loglik = loglik1 + loglik2

    return loglik


def wbl_leftcensor_loglik_H0shape(theta, x, t, thr):
    """
    TODO: Documentation

    Parameters
    ----------
    theta : float
        initial guess for fit.
    x : numpy.ndarray
        precipitation values.
    t : numpy.ndarray
        temperature values.
    thr : float
        threshold value for left-censoring.

    Returns
    -------
    loglik : TYPE
        DESCRIPTION.

    """
    # theta is init guess
    # x is precipitaon\
    # t is temperature
    # thr is threshold value (exact, no percentual)

    a_w = theta[0]  # Shape parameter (constant) - lambda_0
    a_C = theta[2]  # Scale parameter base (a)
    b_C = theta[3]  # Scale parameter adjustment based on `t` - k_0

    # Handle data below the threshold
    t0 = t[x < thr]
    shapes0 = a_w * np.ones_like(t0)  # Constant shape parameter
    scales0 = a_C * np.exp(b_C * t0)

    # Handle data above or equal to the threshold
    x1 = x[x >= thr]
    t1 = t[x >= thr]
    shapes1 = a_w * np.ones_like(t1)  # Constant shape parameter
    scales1 = a_C * np.exp(b_C * t1)

    # Calculate the log-likelihood components
    loglik1 = np.sum(np.log(weibull_min.cdf(thr, c=shapes0, scale=scales0)))
    loglik2 = np.sum(np.log(weibull_min.pdf(x1, c=shapes1, scale=scales1)))

    # Sum the components for the final log-likelihood
    loglik = loglik1 + loglik2

    return loglik


def wbl_leftcensor_loglik_bset(theta, x, t, thr, b_set):
    """

    Parameters
    ----------
    theta : float
        initial guess for fit.
    x : numpy.ndarray
        precipitation values.
    t : numpy.ndarray
        temperature values.
    thr : float
        threshold value for left-censoring.
    b_set : float
        chosen b value that will not change

    Returns
    -------
    loglik : TYPE
        DESCRIPTION.

    """
    # theta is init guess
    # x is precipitaon\
    # t is temperature
    # thr is threshold value (exact, no percentual)
    a_w = theta[0]
    b_w = b_set
    a_C = theta[2]
    b_C = theta[3]

    # Apply conditions based on the threshold
    t0 = t[x < thr]
    shapes0 = a_w + b_w * t0
    scales0 = a_C * np.exp(b_C * t0)

    x1 = x[x >= thr]
    t1 = t[x >= thr]
    shapes1 = a_w + b_w * t1
    scales1 = a_C * np.exp(b_C * t1)

    # Calculate the log-likelihood components
    loglik1 = np.sum(np.log(weibull_min.cdf(thr, c=shapes0, scale=scales0)))
    loglik2 = np.sum(np.log(weibull_min.pdf(x1, c=shapes1, scale=scales1)))

    # Sum the components for the final log-likelihood
    loglik = loglik1 + loglik2

    return loglik


def gen_norm_pdf(x: np.ndarray, mu: float, sigma: float, beta: float) -> np.ndarray:
    """Function computing the Generalized normal distribution PDF.

    Args:
        x (np.ndarray): Data points.
        mu (float): Location parameter.
        sigma (float): Scale parameter.
        beta (float): Snape parameter.

    Returns:
        np.ndarray: Generalized normal distribution PDF
    """
    coeff = beta / (2 * sigma * gamma(1 / beta))
    exponent = -((np.abs(x - mu) / sigma) ** beta)
    return coeff * np.exp(exponent)


def gen_norm_loglik(x: np.ndarray, par: list, beta: float) -> np.ndarray:
    """Function computing the Log-likelihood for the Generalized normal distribution.

    Args:
        x (np.ndarray): Data points.
        par (list): List of parameters [mu, sigma].
        beta (float): Snape parameter.

    Returns:
        np.ndarray: Log-likelihood for the Generalized normal distribution.
    """
    # Compute the log-likelihood
    pdf = gen_norm_pdf(x, par[0], par[1], beta)
    n = len(pdf[pdf == 0])
    if n > 5:
        print(f"warning: {n} zero values")

    pdf[pdf == 0] = 1e-10  # stops issue if zero generated
    loglik = np.sum(np.log(pdf))

    return loglik


def randdf(size, df, flag):
    """
    This function generates random numbers according to a user-defined probability
    density function (pdf) or cumulative distribution function (cdf).
    This is pythonized version of Matlab f randdf coded by halleyhit on Aug. 15th, 2018
    % Email: halleyhit@sjtu.edu.cn or halleyhit@163.com

    Parameters:
    size (int or tuple): Size of the output array. E.g., size=10 creates a 10-by-1 array,
                         size=(10, 2) creates a 10-by-2 matrix.
    df (numpy.ndarray): Density function, should be a 2-row matrix where the first row
                        represents the function values and the second row represents
                        sampling points.
    flag (str): Flag to indicate 'pdf' or 'cdf'.

    Returns:
    numpy.ndarray: Array of random samples based on the defined pdf or cdf.
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
    y: Union[float, np.ndarray], wbl_phat: np.ndarray, n: int
) -> Tuple[float, np.ndarray]:
    """
    Calculate the cumulative distribution function (CDF) based on the given Weibull parameters.

    Args:
        y (Union[float, np.ndarray]): Value(s) at which to evaluate the CDF.
        wbl_phat (np.ndarray): Array of Weibull parameters, where each row contains [shape, scale].
        n (int): Power to raise the final probability to.

    Returns:
        Tuple[float, np.ndarray]: Calculated CDF value(s).
    """
    p = 0
    for i in range(wbl_phat.shape[0]):
        p += 1 - np.exp(-((y / wbl_phat[i, 0]) ** wbl_phat[i, 1]))
    p = (p / wbl_phat.shape[0]) ** n
    return p


def SMEV_Mc_inversion(
    wbl_phat: np.ndarray,
    n: Union[int, float, pd.Series],
    target_return_periods: Union[list, np.ndarray],
    vguess: np.ndarray,
    method_root_scalar: Union[str, None],
) -> np.ndarray:
    """
    Invert to find quantiles corresponding to the target return periods.

    Args:
        wbl_phat (numpy.ndarray): Array of Weibull parameters, where each row contains [shape, scale].
        n (int): Power to raise the final probability to.
        target_return_periods (list or array-like): Desired target return periods.
        vguess (numpy.ndarray): Initial guesses for inversion.

    Returns:
        np.ndarray: Quantiles corresponding to the target return periods.
    """
    if isinstance(n, pd.Series):
        n = float(n.values[0])

    pr = 1 - 1 / np.array(
        target_return_periods
    )  # Probabilities associated with target_return_periods
    pv = MC_tSMEV_cdf(
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
            return MC_tSMEV_cdf(y, wbl_phat, n) - pr[t]

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


def TNX_FIG_temp_model(
    T,
    g_phat,
    beta,
    eT,
    obscol="r",
    valcol="b",
    obslabel="observations",
    vallabel="temperature model g(T)",
    xlimits=[-15, 30],
    ylimits=[0, 0.06],
    method="norm",
):
    """
    Plots the observational and model temperature pdf

    Parameters
    ----------
    T : numpy.ndarray
        Array of observed temperatures.
    g_phat : numpy.ndarray
        [mu, sigma] of temperature distribution.
    beta : float
        value of beta in generalised normal distribution.
    eT : numpy.ndarray
        x (temperature) values to produce distribution.
    obscol : string, optional
        color to plot observations. The default is 'r'.
    valcol : string, optional
        color to plot magnitude model. The default is 'b'.
    obslabel : string, optional
        Label for observations. The default is 'observations'.
    vallabel : string, optional
        Label for model plot. The default is 'temperature model g(T)'.
    xlimits : list, optional
        limits for the x axis [lower_x_limit, upper_x_limit]. The default is [-15,30].
    ylimits : list, optional
        limits for the y axis [lower_y_limit, upper_y_limit]. The default is [0,0.06].

    Returns
    -------
    hist : numpy.ndarray
        pdf values of observed distribution.
    pdf_values : numpy.ndarray
        pdf values of fitted model.

    """

    # Plot empirical PDF of T
    eT_edges = np.concatenate(
        [np.array([eT[0] - (eT[1] - eT[0]) / 2]), (eT + (eT[1] - eT[0]) / 2)]
    )  # convert bin centres into bin edges
    hist, bin_edges = np.histogram(T, bins=eT_edges, density=True)
    plt.plot(eT, hist, "--", color=obscol, label=obslabel)

    # Plot analytical PDF of T (validation)
    if method == "skewnorm":
        pdf_values = skewnorm.pdf(eT, *g_phat)
    elif method == "norm":
        pdf_values = gen_norm_pdf(eT, g_phat[0], g_phat[1], beta)

    plt.plot(eT, pdf_values, "-", color=valcol, label=vallabel)

    # Set plot parameters
    # ax.set_xlim(Tlims)
    plt.xlabel("T [°C]", fontsize=14)
    plt.ylabel("pdf", fontsize=14)
    plt.ylim(ylimits[0], ylimits[1])
    plt.xlim(xlimits[0], xlimits[1])
    plt.legend(
        fontsize=8
    )  # NEED TO SET LOCATION OF THIS, maybe fontsize is too small as well
    plt.tick_params(axis="both", which="major", labelsize=14)

    return hist, pdf_values


def inverse_magnitude_model(F_phat, eT, qs):
    """
    Calculate percentiles from the Weibell magnitude model

    Parameters
    ----------
    F_phat : numpy.ndarray
        distribution values. F_phat = [kappa_0,b,lambda_0,a].
    x : numpy.ndarray
        x (temperature) values from which to produce distribution.
    qs : list
        list of percentiles to calculate (between 0 and 1). e.g. [0.85,0.95,0.99].

    Returns
    -------
    percentile_lines : numpy.ndarray
        array with shape length(qs) by length(eT) giving the magnitudes for each eT. percentile_lines[0] are the values for qs[0].

    """

    percentile_lines = np.zeros((len(qs), len(eT)))
    for iq, q in enumerate(qs):
        percentile_lines[iq, :] = (
            F_phat[2]
            * np.exp(F_phat[3] * eT)
            * (-np.log(1 - q)) ** (1 / (F_phat[0] + F_phat[1] * eT))
        )

    return percentile_lines


def TNX_obs_scaling_rate(P, T, qs, niter):
    """
    calculate quantile regression parameters.

    Parameters
    ----------
    P : numpy.ndarray
        precipitation values
    T : numpy.ndarray
        temperature values
    qs : float
        percentile.

    Returns
    -------
    qhat : numpy.ndarray
        [something, scaling rate]. #TODO: I dont know what this is

    """
    T = sm.add_constant(T)  # Add a constant (intercept) term
    model = sm.QuantReg(np.log(P), T)
    qhat = model.fit(q=qs).params

    qhat_unc = np.zeros([2, niter])
    for iter in np.arange(0, niter):
        rr = np.random.randint(0, len(T), size=(niter))
        model = sm.QuantReg(np.log(P[rr]), T[rr])
        qhat_unc[:, iter] = model.fit(q=qs).params

    return qhat, qhat_unc


def TNX_FIG_scaling(
    P,
    T,
    P_mc,
    T_mc,
    F_phat,
    niter_smev,
    eT,
    iTs,
    qs=[0.99],
    obscol="r",
    valcol="b",
    xlimits=[-15, 30],
    ylimits=[0.4, 1000],
):
    """
    Plots figure 5.

    Parameters
    ----------
    P : numpy.ndarray
        precipitation values
    T : numpy.ndarray
        temperature values
    P_mc : numpy.ndarray
        Monte Carlo generated precipitation values.
    T_mc : numpy.ndarray
        Monte Carlo generated temperature values.
    F_phat : numpy.ndarray
        distribution values. F_phat = [kappa_0,b,lambda_0,a].
    niter_smev : int
        Number of iterations for uncertainty for the SMEV model .
    eT : numpy.ndarray
        x (temperature) values to produce distribution for magnitude model.
    iTs : numpy.ndarray
        x (temperature) values to produce distribution for quantile regression, binning, and TENAX.
    qs : list
        percentiles to calculate W.
    obscol : string, optional
        color code to plot observations. The default is 'r'.
    valcol : string, optional
        color code to plot model. The default is 'b'.
    xlimits : list, optional
        [min_x,max_x]. x limits to plot. The default is [-15,30].
    ylimits : list, optional
        [min_y,max_y]. y limits to plot. The default is [0.4,1000].

    Returns
    -------
    scaling_rate : float
        scaling rate of

    """
    percentile_lines = inverse_magnitude_model(F_phat, eT, qs)
    scaling_rate_W = (np.exp(F_phat[3]) - 1) * 100

    # TODO: this doesn't seem quite right ... uncertainty is way off compared to paper
    qhat, qhat_unc = TNX_obs_scaling_rate(P, T, qs[0], niter_smev)
    scaling_rate_q = (np.exp(qhat[1]) - 1) * 100

    # quantile regression uncertainties
    q_reg_full_unc = np.zeros([len(iTs), niter_smev])
    for i in np.arange(0, len(iTs)):
        q_reg_full_unc[i, :] = np.exp(qhat_unc[0, :]) * np.exp(iTs[i] * qhat_unc[1, :])

    q_up = np.quantile(q_reg_full_unc, 0.95, axis=1)
    q_low = np.quantile(q_reg_full_unc, 0.05, axis=1)

    plt.figure(figsize=(5, 5))
    plt.scatter(T, P, s=1.5, color=obscol, alpha=0.3, label="observations")
    plt.plot(
        iTs[0:-7],
        np.exp(qhat[0]) * np.exp(iTs[0:-7] * qhat[1]),
        "--k",
        label="Quantile regression method",
    )  # need uncertainty on this too...
    plt.fill_between(
        iTs[0:-7], q_low[0:-7], q_up[0:-7], color="k", alpha=0.2
    )  # quantile regression uncertainty

    ############################################################### PUT THIS ELSEWHERE
    T_mc_bins = np.reshape(T_mc, [np.size(T), niter_smev])
    P_mc_bins = np.reshape(P_mc, [np.size(P), niter_smev])

    qperc_model = np.zeros([np.size(iTs), niter_smev])
    qperc_obs = np.zeros([np.size(iTs), niter_smev])

    for nit in range(niter_smev):
        for i in range(np.size(iTs) - 1):
            tmpP = P_mc_bins[:, nit]
            mask_model = (T_mc_bins[:, nit] > iTs[i]) & (
                T_mc_bins[:, nit] <= iTs[i + 1]
            )
            if np.any(mask_model):
                qperc_model[i, nit] = np.quantile(
                    tmpP[mask_model], qs[0]
                )  # binning monte carlos to get TENAX model

            mask_obs = (T > iTs[i]) & (T <= iTs[i + 1])
            if np.any(mask_obs):
                qperc_obs[i] = np.quantile(P[mask_obs], qs[0])  # binning observations

    qperc_obs_med = np.median(qperc_obs, axis=1)
    qperc_model_med = np.median(qperc_model, axis=1)

    qperc_model_up = np.quantile(qperc_model, 0.95, axis=1)
    qperc_model_low = np.quantile(qperc_model, 0.05, axis=1)

    #####################################################################################

    # plot uncertainty
    plt.fill_between(
        iTs[1:-6] + (iTs[2] - iTs[1]) / 2,
        qperc_model_low[1:-6],
        qperc_model_up[1:-6],
        color="m",
        alpha=0.2,
    )  # TENAX

    plt.plot(
        iTs[1:-7] + (iTs[2] - iTs[1]) / 2,
        qperc_obs_med[1:-7],
        "-xr",
        label="Binning method",
    )  # don't really know why we cut off at the end like this
    plt.plot(
        iTs[1:-6] + (iTs[2] - iTs[1]) / 2,
        qperc_model_med[1:-6],
        "-om",
        label="The TENAX model",
    )

    n = 0
    while n < np.size(qs):
        plt.plot(eT, percentile_lines[n], color=valcol, label="Magnitude model W(x,T)")
        n = n + 1

    plt.yscale("log")
    plt.ylim(ylimits[0], ylimits[1])
    plt.xlim(xlimits[0], xlimits[1])
    plt.legend(title=str(qs[0] * 100) + "th percentile lines computed by:")

    return scaling_rate_W, scaling_rate_q
