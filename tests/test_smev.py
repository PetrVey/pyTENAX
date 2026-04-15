# -*- coding: utf-8 -*-
"""
Tests for the SMEV class in pyTENAX.smev.

NOTE:
These tests are tightly coupled to the specific input dataset -> "prec_data_Aadorf.parquet".
If the underlying time series data changes, ALL TESTS WILL FAIL.
Revisit and update test expectations accordingly when the dataset changes.

IMPORTANT:
Tests are chained — each builds on the same preprocessing steps. If an early
method fails or the data structure changes, downstream tests may fail too.
This is by design to ensure the integrated workflow is correct.

ON TOLERANCES (atol / places):
Expected reference values were produced in MATLAB. Small discrepancies between
Python and MATLAB outputs are expected and do NOT indicate a bug. Known sources:

  1. OLS fitting (estimate_smev_parameters): Python's statsmodels and MATLAB's
     regress() use the same least-squares algorithm but differ in floating-point
     accumulation order, which shifts shape/scale by ~0.001–0.01.

  2. Return level formula (smev_return_values): small errors in shape/scale
     propagate through log/power operations, producing differences of ~0.05–0.1 mm
     at the higher return periods (100–200 yr) where the curve is steepest.

  3. Integer scaling (get_ordinary_events_values): data is multiplied by 10000
     and cast to int64 for exact sliding-window sums. Rounding at that step can
     shift convolution argmax by one timestep in tied events, producing tiny value
     differences vs MATLAB's floating-point convolution.

Tolerances are therefore set to:
  - shape / scale : places=1  (±0.05)
  - return levels : atol=0.1  (±0.1 mm)
These are tight enough to catch real regressions while ignoring MATLAB/Python
floating-point noise.
"""

import unittest
import numpy as np
import pandas as pd
from importlib.resources import files
from pyTENAX import smev


class TestSMEV(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Class-level setup: runs once before all tests.
        Initialises SMEV, loads data, and runs the full pipeline so that
        downstream tests can reuse shared state without repeating work.
        """
        cls.S = smev.SMEV(
            return_period=[2, 5, 10, 20, 50, 100, 200],
            durations=[10, 60, 180, 360, 720, 1440],
            time_resolution=10,
            min_rain=0.1,
            storm_separation_time=24,
            min_event_duration=30,
            left_censoring=[0.9, 1],
        )

        file_path_input = files("pyTENAX.res").joinpath("prec_data_Aadorf.parquet")
        data = pd.read_parquet(file_path_input)
        data["prec_time"] = pd.to_datetime(data["prec_time"])
        data.set_index("prec_time", inplace=True)
        cls.name_col = "prec_values"

        data = cls.S.remove_incomplete_years(data, cls.name_col)
        cls.df_arr   = np.array(data[cls.name_col])
        cls.df_dates = np.array(data.index)

        cls.idx_ordinary = cls.S.get_ordinary_events(
            data=cls.df_arr, dates=cls.df_dates, check_gaps=False
        )
        cls.arr_vals, cls.arr_dates, cls.n_ordinary_per_year = cls.S.remove_short(
            cls.idx_ordinary
        )
        cls.n = (cls.n_ordinary_per_year.sum() / len(cls.n_ordinary_per_year)).item()

        cls.dict_ordinary, cls.dict_AMS = cls.S.get_ordinary_events_values(
            data=cls.df_arr, dates=cls.df_dates, arr_dates_oe=cls.arr_dates
        )

    # -------------------------------------------------------------------------
    # get_ordinary_events
    # -------------------------------------------------------------------------
    def test_get_ordinary_events_count(self):
        """Total number of storm events extracted from the dataset."""
        self.assertEqual(len(self.idx_ordinary), 2848,
                         "Expected 2848 ordinary events")

    # -------------------------------------------------------------------------
    # remove_short
    # -------------------------------------------------------------------------
    def test_remove_short_count(self):
        """Events remaining after removing those shorter than min_event_duration."""
        self.assertEqual(len(self.arr_vals), 2634,
                         "Expected 2634 events after removing short ones")

    def test_remove_short_n_mean(self):
        """Mean number of ordinary events per year."""
        self.assertAlmostEqual(self.n, 69.32, places=1,
                               msg="Mean N of ordinary events per year should be ~69.32")

    # -------------------------------------------------------------------------
    # get_ordinary_events_values — shape / scale / return levels
    # -------------------------------------------------------------------------
    # Expected values (MATLAB reference):
    #   Duration  scale    shape
    #     10 min  0.8803   0.6005
    #     60 min  2.3396   0.6776
    #    180 min  5.0060   0.8697
    #    360 min  7.1135   0.9501
    #    720 min  9.5669   0.9914
    #   1440 min 12.7219   1.0342
    #
    #   Return levels [mm]:
    #     10 min:  11.22  16.16  19.83  23.62  28.92  33.17  37.63
    #     60 min:  22.32  30.85  36.97  43.18  51.66  58.33  65.23
    #    180 min:  29.02  37.34  42.99  48.52  55.79  61.33  66.91
    #    360 min:  35.54  44.76  50.93  56.89  64.66  70.51  76.36
    #    720 min:  44.70  55.76  63.11  70.17  79.32  86.18  93.02
    #   1440 min:  55.76  68.92  77.61  85.91  96.62 104.62 112.57

    EXPECTED_PARAMS = {
        "10":   (0.8803, 0.6005),
        "60":   (2.3396, 0.6776),
        "180":  (5.0060, 0.8697),
        "360":  (7.1135, 0.9501),
        "720":  (9.5669, 0.9914),
        "1440": (12.7219, 1.0342),
    }

    EXPECTED_RL = {
        "10":   [11.22, 16.16, 19.83, 23.62, 28.92, 33.17, 37.63],
        "60":   [22.32, 30.85, 36.97, 43.18, 51.66, 58.33, 65.23],
        "180":  [29.02, 37.34, 42.99, 48.52, 55.79, 61.33, 66.91],
        "360":  [35.54, 44.76, 50.93, 56.89, 64.66, 70.51, 76.36],
        "720":  [44.70, 55.76, 63.11, 70.17, 79.32, 86.18, 93.02],
        "1440": [55.76, 68.92, 77.61, 85.91, 96.62, 104.62, 112.57],
    }

    def _params_and_rl(self, dur):
        P = self.dict_ordinary[dur]["ordinary"].to_numpy()
        shape, scale = self.S.estimate_smev_parameters(P, self.S.left_censoring)
        RL = self.S.smev_return_values(self.S.return_period, shape, scale, self.n)
        return shape, scale, RL

    def test_smev_params_10min(self):
        shape, scale, _ = self._params_and_rl("10")
        exp_scale, exp_shape = self.EXPECTED_PARAMS["10"]
        self.assertAlmostEqual(scale, exp_scale, places=1, msg="10 min scale mismatch")
        self.assertAlmostEqual(shape, exp_shape, places=1, msg="10 min shape mismatch")

    def test_smev_params_60min(self):
        shape, scale, _ = self._params_and_rl("60")
        exp_scale, exp_shape = self.EXPECTED_PARAMS["60"]
        self.assertAlmostEqual(scale, exp_scale, places=1, msg="60 min scale mismatch")
        self.assertAlmostEqual(shape, exp_shape, places=1, msg="60 min shape mismatch")

    def test_smev_params_180min(self):
        shape, scale, _ = self._params_and_rl("180")
        exp_scale, exp_shape = self.EXPECTED_PARAMS["180"]
        self.assertAlmostEqual(scale, exp_scale, places=1, msg="180 min scale mismatch")
        self.assertAlmostEqual(shape, exp_shape, places=1, msg="180 min shape mismatch")

    def test_smev_params_360min(self):
        shape, scale, _ = self._params_and_rl("360")
        exp_scale, exp_shape = self.EXPECTED_PARAMS["360"]
        self.assertAlmostEqual(scale, exp_scale, places=1, msg="360 min scale mismatch")
        self.assertAlmostEqual(shape, exp_shape, places=1, msg="360 min shape mismatch")

    def test_smev_params_720min(self):
        shape, scale, _ = self._params_and_rl("720")
        exp_scale, exp_shape = self.EXPECTED_PARAMS["720"]
        self.assertAlmostEqual(scale, exp_scale, places=1, msg="720 min scale mismatch")
        self.assertAlmostEqual(shape, exp_shape, places=1, msg="720 min shape mismatch")

    def test_smev_params_1440min(self):
        shape, scale, _ = self._params_and_rl("1440")
        exp_scale, exp_shape = self.EXPECTED_PARAMS["1440"]
        self.assertAlmostEqual(scale, exp_scale, places=1, msg="1440 min scale mismatch")
        self.assertAlmostEqual(shape, exp_shape, places=1, msg="1440 min shape mismatch")

    def test_smev_return_levels_10min(self):
        _, _, RL = self._params_and_rl("10")
        np.testing.assert_allclose(
            RL, self.EXPECTED_RL["10"], atol=0.1,
            err_msg="10 min return levels mismatch"
        )

    def test_smev_return_levels_60min(self):
        _, _, RL = self._params_and_rl("60")
        np.testing.assert_allclose(
            RL, self.EXPECTED_RL["60"], atol=0.1,
            err_msg="60 min return levels mismatch"
        )

    def test_smev_return_levels_180min(self):
        _, _, RL = self._params_and_rl("180")
        np.testing.assert_allclose(
            RL, self.EXPECTED_RL["180"], atol=0.1,
            err_msg="180 min return levels mismatch"
        )

    def test_smev_return_levels_360min(self):
        _, _, RL = self._params_and_rl("360")
        np.testing.assert_allclose(
            RL, self.EXPECTED_RL["360"], atol=0.1,
            err_msg="360 min return levels mismatch"
        )

    def test_smev_return_levels_720min(self):
        _, _, RL = self._params_and_rl("720")
        np.testing.assert_allclose(
            RL, self.EXPECTED_RL["720"], atol=0.1,
            err_msg="720 min return levels mismatch"
        )

    def test_smev_return_levels_1440min(self):
        _, _, RL = self._params_and_rl("1440")
        np.testing.assert_allclose(
            RL, self.EXPECTED_RL["1440"], atol=0.1,
            err_msg="1440 min return levels mismatch"
        )

    @classmethod
    def tearDownClass(cls):
        del cls.S
        del cls.df_arr
        del cls.df_dates


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestSMEV)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
