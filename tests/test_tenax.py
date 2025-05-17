# -*- coding: utf-8 -*-
"""
Created on Sat May 17 18:31:24 2025

@author: Petr
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