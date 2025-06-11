# -*- coding: utf-8 -*-
"""
Created on Sat May 17 18:31:24 2025

@author: Petr

NOTE:
These tests are tightly coupled to the specific input dataset currently in use -> "prec_data_Aadorf.parquet".
If the underlying time series data changes (e.g., different date range, missing years, values), ALL TESTS WILL FAIL.
Make sure to revisit and update test expectations accordingly when the dataset changes.

IMPORTANT:
Many tests in this suite are chained — they rely on the same preprocessing steps (e.g., removing incomplete years,
detecting and filtering ordinary events, etc.). If one method fails or the data structure changes, it can cause 
multiple downstream test failures. This is by design to ensure the integrated workflow is working as expected.
"""

import unittest
import pandas as pd
import numpy as np
from importlib.resources import files
from pyTENAX import tenax


class TestTENAX(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """
        Class-level setup method that runs once before all tests in this class.

        Initializes the TENAX class instance and loads the dataset.  
        Using @classmethod allows us to share this setup among all test methods without repeating it,  
        improving test execution efficiency.
        
        Args:
            cls (TestTENAX): The test class itself, used to set class variables.
        """
        # Initialize TENAX class (runs once for the whole test class)
        cls.S = tenax.TENAX(
            return_period=[2, 5, 10, 20, 50, 100, 200],
            durations=[10,],
            time_resolution=5,
            left_censoring=[0, 0.90],
            alpha=0.05,
        )
        # Load data
        file_path_input = files("pyTENAX.res").joinpath("prec_data_Aadorf.parquet")
        cls.data = pd.read_parquet(file_path_input)
        cls.data["prec_time"] = pd.to_datetime(cls.data["prec_time"])
        cls.data.set_index("prec_time", inplace=True)
        
        # Push small values to zero
        name_col = "prec_values"
        cls.data.loc[cls.data[name_col] < cls.S.min_rain, name_col] = 0
        
        # Load temperature data
        file_path_temperature = files("pyTENAX.res").joinpath("temp_data_Aadorf.parquet")
        cls.t_data = pd.read_parquet(file_path_temperature)
        cls.t_data["temp_time"] = pd.to_datetime(cls.t_data["temp_time"])
        cls.t_data.set_index("temp_time", inplace=True)
        
        cls.temp_name_col = "temp_values"
        cls.df_arr_t_data = np.array(cls.t_data[cls.temp_name_col])
        cls.df_dates_t_data = np.array(cls.t_data.index)

    def test_remove_incomplete_years(self):
        name_col = "prec_values"
        
        # Get initial number of years
        initial_years = len(self.data.index.year.unique())
        
        # Apply the method
        cleaned_data = self.S.remove_incomplete_years(self.data.copy(), name_col)
        
        # Get the number of years after cleaning
        final_years = len(cleaned_data.index.year.unique())
        
        # Check if one year was removed
        self.assertEqual(initial_years, 39, "Initial year count should be 39")
        self.assertEqual(final_years, 38, "After cleaning, it should be 38")
        
    def test_get_ordinary_events(self):
        """
        Test extracting ordinary events.
        """
        name_col = "prec_values"
        
        # Clean the data first
        cleaned_data = self.S.remove_incomplete_years(self.data.copy(), name_col)
        
        # Convert to numpy arrays
        df_arr = np.array(cleaned_data[name_col])
        df_dates = np.array(cleaned_data.index)

        # Get the ordinary events
        idx_ordinary = self.S.get_ordinary_events(
            data=df_arr, dates=df_dates, name_col=name_col, check_gaps=False
        )

        # Assert the expected number of ordinary events
        self.assertEqual(len(idx_ordinary), 2848, "Expected 2848 ordinary events")
        
    def test_remove_short(self):
        """
        Test removing short ordinary events.
        """
        name_col = "prec_values"
        
        # First, clean the data by removing incomplete years
        cleaned_data = self.S.remove_incomplete_years(self.data.copy(), name_col)
        
        # Convert to numpy arrays
        df_arr = np.array(cleaned_data[name_col])
        df_dates = np.array(cleaned_data.index)
        
        # Get ordinary events indexes
        idx_ordinary = self.S.get_ordinary_events(
            data=df_arr, dates=df_dates, name_col=name_col, check_gaps=False
        )
        
        # Now run remove_short with those ordinary events
        arr_vals, arr_dates, n_ordinary_per_year = self.S.remove_short(idx_ordinary)
        
        # Assert the expected count of True values in arr_vals (i.e., events that remain after filtering)
        self.assertEqual(sum(arr_vals), 2634, "Expected 2634 events after removing short ones")

    def test_get_ordinary_events_values(self):
        """
        Test extraction of ordinary event values and AMS for given durations.
        As we cannot test if all values are equal to expected values, I simply calculcate sum of ordinary events and annual maximas.
        """
        name_col = "prec_values"
        
        # Clean the data first
        cleaned_data = self.S.remove_incomplete_years(self.data.copy(), name_col)
        
        # Convert to numpy arrays
        df_arr = np.array(cleaned_data[name_col])
        df_dates = np.array(cleaned_data.index)
        
        # Get ordinary events
        idx_ordinary = self.S.get_ordinary_events(
            data=df_arr, dates=df_dates, name_col=name_col, check_gaps=False
        )
        
        # Remove short events
        arr_vals, arr_dates, n_ordinary_per_year = self.S.remove_short(idx_ordinary)
        
        # Get ordinary events values and AMS
        dict_ordinary, dict_AMS = self.S.get_ordinary_events_values(
            data=df_arr, dates=df_dates, arr_dates_oe=arr_dates
        )
        
        # Assert sum of the 'ordinary' column for duration "10" matches expected value
        total_ordinary = dict_ordinary["10"].ordinary.sum()
        self.assertAlmostEqual(total_ordinary, 4042.0, places=2,
                               msg="Sum of ordinary events for duration 10 should be approximately 4042.0")
        
        # Assert sum of the 'AMS' column for duration "10" matches expected value
        total_AMS = dict_AMS["10"].AMS.sum()
        self.assertAlmostEqual(total_AMS, 474.2, places=2,
                               msg="Sum of AMS for duration 10 should be approximately 474.2")


    def test_associate_vars(self):
        """
        Test the associate_vars method which associates auxiliary variables (like temperature)
        with the ordinary events dictionary.
        """
        name_col = "prec_values"
    
        # Step 1: Clean the precipitation data first
        cleaned_data = self.S.remove_incomplete_years(self.data.copy(), name_col)
    
        # Step 2: Convert to numpy arrays
        df_arr = np.array(cleaned_data[name_col])
        df_dates = np.array(cleaned_data.index)
    
        # Step 3: Get ordinary events
        idx_ordinary = self.S.get_ordinary_events(
            data=df_arr, dates=df_dates, name_col=name_col, check_gaps=False
        )
    
        # Step 4: Remove short events
        arr_vals, arr_dates, _ = self.S.remove_short(idx_ordinary)
    
        # Step 5: Get ordinary events values dictionary
        dict_ordinary, _ = self.S.get_ordinary_events_values(
            data=df_arr, dates=df_dates, arr_dates_oe=arr_dates
        )
    
        # Step 6: Now test associate_vars with temperature data from setUpClass
        dict_ordinary_updated, _, n_ordinary_per_year = self.S.associate_vars(
            dict_ordinary, self.df_arr_t_data, self.df_dates_t_data
        )
    
        # Basic assertions
        self.assertIn("10", dict_ordinary_updated)
        self.assertTrue(hasattr(dict_ordinary_updated["10"], "T"), "Expected Temperature ('T') attribute in dict_ordinary['10']")
        
        # Assert Temperature sum is close to expected
        temp_sum = dict_ordinary_updated["10"]["T"].sum()
        self.assertAlmostEqual(temp_sum, 27096.004, places=3, msg="Temperature sum does not match expected")
    
        # Assert sum of n_ordinary_per_year equals expected 2634
        self.assertEqual(n_ordinary_per_year.sum().item(), 2634, "Sum of ordinary events per year does not match expected")

    def test_magnitude_model(self):
        """
        Test the magnitude_model method to ensure the magnitude model is fitted correctly.
        """
        # Re-run the chain of methods needed for this function
        name_col = "prec_values"
        cleaned_data = self.S.remove_incomplete_years(self.data.copy(), name_col)
        df_arr = np.array(cleaned_data[name_col])
        df_dates = np.array(cleaned_data.index)
        idx_ordinary = self.S.get_ordinary_events(data=df_arr, dates=df_dates, name_col=name_col, check_gaps=False)
        arr_vals, arr_dates, _ = self.S.remove_short(idx_ordinary)
        dict_ordinary, _ = self.S.get_ordinary_events_values(data=df_arr, dates=df_dates, arr_dates_oe=arr_dates)
        dict_ordinary, _, _ = self.S.associate_vars(dict_ordinary, self.df_arr_t_data, self.df_dates_t_data)
    
        # Extract values for duration "10"
        P = dict_ordinary["10"]["ordinary"].to_numpy()
        T = dict_ordinary["10"]["T"].to_numpy()
        
        # Calculate the threshold
        thr = dict_ordinary["10"]["ordinary"].quantile(self.S.left_censoring[1])

        # Run magnitude model
        F_phat, loglik, _, _ = self.S.magnitude_model(P, T, thr)
    
        # Basic checks
        self.assertIsNotNone(F_phat, "F_phat should not be None")
        self.assertTrue(np.isfinite(loglik), "loglik should be a finite number")
        self.assertGreater(loglik, -1e6, "loglik should be reasonably large (not extreme negative)")
    
        # Check the exact values of F_phat
        expected_F_phat = np.array([0.8655, 0.0, 0.3131, 0.1112])
        np.testing.assert_array_almost_equal(F_phat.round(4), expected_F_phat, decimal=4, err_msg="F_phat values do not match expected output")

    def test_temperature_model(self):
        """
        Test the temperature_model method to ensure the temperature model is fitted correctly.
        """
        # Re-run the full chain of methods needed for this function
        name_col = "prec_values"
        cleaned_data = self.S.remove_incomplete_years(self.data.copy(), name_col)
        df_arr = np.array(cleaned_data[name_col])
        df_dates = np.array(cleaned_data.index)
        idx_ordinary = self.S.get_ordinary_events(data=df_arr, dates=df_dates, name_col=name_col, check_gaps=False)
        arr_vals, arr_dates, _ = self.S.remove_short(idx_ordinary)
        dict_ordinary, _ = self.S.get_ordinary_events_values(data=df_arr, dates=df_dates, arr_dates_oe=arr_dates)
        dict_ordinary, _, _ = self.S.associate_vars(dict_ordinary, self.df_arr_t_data, self.df_dates_t_data)
    
        # Extract temperature values from the associated dictionary
        T = dict_ordinary["10"]["T"].to_numpy()
    
        # Run the temperature model
        g_phat = self.S.temperature_model(T)
    
        # Basic checks
        self.assertIsNotNone(g_phat, "g_phat should not be None")
        self.assertEqual(len(g_phat), 2, "g_phat should have exactly two parameters")
    
        # Check the exact values of g_phat
        expected_g_phat = np.array([9.8198, 12.3587])
        np.testing.assert_array_almost_equal(g_phat.round(4), expected_g_phat, decimal=4, err_msg="g_phat values do not match expected output")

    def test_model_inversion(self):
        """
        Test the model_inversion method with Monte Carlo variability in mind.
        """
        # Re-run the full chain of methods needed for this function
        name_col = "prec_values"
        cleaned_data = self.S.remove_incomplete_years(self.data.copy(), name_col)
        df_arr = np.array(cleaned_data[name_col])
        df_dates = np.array(cleaned_data.index)
        idx_ordinary = self.S.get_ordinary_events(data=df_arr, dates=df_dates, name_col=name_col, check_gaps=False)
        arr_vals, arr_dates, n_ordinary_per_year = self.S.remove_short(idx_ordinary)
        dict_ordinary, _ = self.S.get_ordinary_events_values(data=df_arr, dates=df_dates, arr_dates_oe=arr_dates)
        dict_ordinary, _, _ = self.S.associate_vars(dict_ordinary, self.df_arr_t_data, self.df_dates_t_data)
    
        # Prepare inputs for model_inversion
        P = dict_ordinary["10"]["ordinary"].to_numpy()
        T = dict_ordinary["10"]["T"].to_numpy()
        thr = dict_ordinary["10"]["ordinary"].quantile(self.S.left_censoring[1])
        Ts = np.arange(np.min(T) - self.S.temp_delta, np.max(T) + self.S.temp_delta, self.S.temp_res_monte_carlo)
        F_phat, _, _, _ = self.S.magnitude_model(P, T, thr)
        g_phat = self.S.temperature_model(T)
        n = n_ordinary_per_year.sum() / len(n_ordinary_per_year)
    
        # Run the model_inversion
        RL, _, _ = self.S.model_inversion(F_phat, g_phat, n, Ts)
    
        # Expected range (based on typical outputs)
        expected_RL = np.array([11.2, 16.4, 20.3, 24.4, 30.1, 34.7, 39.6])
        buffer = 0.1  # 10% buffer
    
        # Check each RL element individually
        for i, (rl, expected) in enumerate(zip(RL, expected_RL)):
            lower_bound = expected * (1 - buffer)
            upper_bound = expected * (1 + buffer)
            self.assertTrue(
                lower_bound <= rl <= upper_bound,
                f"RL[{i}] = {rl:.4f} not within expected range ({lower_bound:.4f} to {upper_bound:.4f})"
                )

    @classmethod
    def tearDownClass(cls):
        """
        Class-level teardown method that runs once after all tests in this class.

        Cleans up class variables to free memory or reset state after tests finish.

        Args:
            cls (TestTENAX): The test class itself.
        """
        # Clean up, if needed
        del cls.data
        del cls.S


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestTENAX)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)