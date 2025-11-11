# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 17:28:25 2025

@author: Petr

Unit tests for the weibul_test_MC function from pyTENAX package.

This test suite validates the Weibull tail detection functionality using four different 
synthetic datasets to ensure the function correctly identifies when precipitation data 
follows a Weibull distribution in the tail region.

Test Overview:
=============

T1 (test_weibull_tail_with_censor):
    Tests Weibull tail detection WITH annual maximum censoring enabled.
    - Generates mixed data: bulk values (0.2) below quantile q + Weibull tail above q
    - Tests multiple quantile thresholds from 0.5 to 0.99 (step 0.05)
    - Expected: Function should correctly identify the quantile threshold q
    - Purpose: Validates censored AM approach can handle different tail proportions

T2 (test_weibull_tail_without_censor):
    Tests Weibull tail detection WITHOUT annual maximum censoring.
    - Same synthetic data generation as T1 (mixed bulk + Weibull tail)
    - Tests same range of quantile thresholds (0.5 to 0.99)
    - Expected: Function should correctly identify the quantile threshold q
    - Purpose: Validates uncensored approach performance vs censored approach

T3 (test_weibull_single_distribution):
    Tests with pure Weibull-distributed precipitation data.
    - Generates data entirely from Weibull distribution (shape=0.7, scale=2)
    - No artificial bulk/tail mixing - naturally Weibull throughout
    - Expected: Optimal threshold < 1 (indicating Weibull tail detected)
    - Purpose: Validates function correctly identifies genuine Weibull behavior

T4 (test_uniform_distribution_no_weibull_tail):
    Tests with uniform distribution (negative control).
    - Generates uniformly distributed precipitation values (0-800mm)
    - No Weibull characteristics in any portion of the data
    - Expected: Optimal threshold = 1 (indicating NO Weibull tail detected)
    - Purpose: Validates function correctly rejects non-Weibull distributions

Synthetic Data Details:
======================
- Sample size: 1750 events across ~25 years (realistic for hydrological analysis)
- Years: Random integers 2000-2025 (simulating modern precipitation records)
- Weibull parameters: shape=0.7, scale=10 
- Value ranges: Clipped to reasonable limits (300 weibull - 800mm uniform)
- Random seed: Fixed at 7 for reproducible results

Statistical Parameters:
======================
- Monte Carlo simulations: 500 synthetic records
- Confidence level: 90% (p_confidence=0.1)
- All tests use consistent random seeding for reproducibility

"""
import unittest
import numpy as np
import pandas as pd
from pyTENAX.wbl_tail_test import weibul_test_MC

class TestWeibullTail(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.random_seed = 7
        
    def test_weibull_tail_with_censor(self):
        random_seed = self.random_seed
        ############################################################################
        # Test 1 for Weibull where tail can be described from weibull distribution
        # This is with given quantile, which will define tail. to test if function is working all good with censored AM
        for q in np.arange(0.5, 0.99, 0.05):
            # From which quantile the Weibull should be generated
            q = q.round(2)
            # Seed for reproducibility   
            np.random.seed(random_seed)

            # Given of the number of years is 25, generating 1750 events means that one year has 70 events (reasonable amoount)
            # Number of samples
            n_samples = 1750 
            # Generate 'year': random integers between 2000 and 2025
            years = np.random.randint(2000, 2026, size=n_samples)
            
            # Assuming that all values larger than 10 comes from the parent weibull ditribution.
            # Parameters
            shape = 0.7    # Weibull shape
            scale = 10  # Weibull scale 
            pr_weibull = scale * np.random.weibull(shape, size=n_samples) #generate 2x more than needed for the smoother tail
            pr_weibull = np.clip(pr_weibull, 0, 300) # Clip to avoid extreme outliers
            
            # Number of low and high samples
            n_low = int(n_samples * q)
            n_tail = n_samples - n_low
            
            # Tail = top fraction of Weibull
            thr = np.quantile(pr_weibull, q)
            tail_pool = pr_weibull[pr_weibull >= thr]
            tail = np.random.choice(tail_pool, size=n_tail, replace=False)
            
            # Bulk = fixed value 0.2
            bulk = np.full(n_low, 0.2)
            
            # Mix and shuffle (optional)
            pr_mixed = np.concatenate([bulk, tail])
            np.random.shuffle(pr_mixed)
            
            df_example = pd.DataFrame({
                'year': years,
                'pr': pr_mixed
            }).sort_values(by="year").reset_index(drop=True)
            
            opt_threshold, estimated_params, range_optimal, _ = weibul_test_MC(ordinary_events_df=df_example,
                                                                            pr_field='pr',
                                                                            hydro_year_field='year',
                                                                            seed_random=random_seed,
                                                                            synthetic_records_amount=500,
                                                                            p_confidence=0.1,
                                                                            make_plot=False,
                                                                            censor_AM=True)
            self.assertEqual(opt_threshold, q, f"Failed for q={q} with censor_AM=True")
    
    def test_weibull_tail_without_censor(self):
        random_seed = self.random_seed    
        # Test 2 for Weibull where tail can be described from weibull distribution
        # This is with given quantile, which will define tail. to test if function is working all good
        for q in np.arange(0.5, 0.99, 0.05):
            # From which quantile the Weibull should be generated
        
            q = q.round(2)
            # Seed for reproducibility
            
            np.random.seed(random_seed)

            # Given of the number of years is 25, generating 1750 events means that one year has 70 events (reasonable amoount)
            # Number of samples
            n_samples = 1750 
            # Generate 'year': random integers between 2000 and 2025
            years = np.random.randint(2000, 2026, size=n_samples)
            
            # Assuming that all values larger than 10 comes from the parent weibull ditribution.
            # Parameters
            shape = 0.7    # Weibull shape
            scale = 10  # Weibull scale 
            pr_weibull = scale * np.random.weibull(shape, size=n_samples) #generate 2x more than needed for the smoother tail
            pr_weibull = np.clip(pr_weibull, 0, 300) # Clip to avoid extreme outliers
            
            # Number of low and high samples
            n_low = int(n_samples * q)
            n_tail = n_samples - n_low
            
            # Tail = top fraction of Weibull
            thr = np.quantile(pr_weibull, q)
            tail_pool = pr_weibull[pr_weibull >= thr]
            tail = np.random.choice(tail_pool, size=n_tail, replace=False)
            
            # Bulk = fixed value 0.2
            bulk = np.full(n_low, 0.2)
            
            # Mix and shuffle (optional)
            pr_mixed = np.concatenate([bulk, tail])
            np.random.shuffle(pr_mixed)
            
            df_example = pd.DataFrame({
                'year': years,
                'pr': pr_mixed
            }).sort_values(by="year").reset_index(drop=True)
            
            opt_threshold, estimated_params, range_optimal, _ = weibul_test_MC(ordinary_events_df=df_example,
                                                                            pr_field='pr',
                                                                            hydro_year_field='year',
                                                                            seed_random=random_seed,
                                                                            synthetic_records_amount=500,
                                                                            p_confidence=0.1,
                                                                            make_plot=False,
                                                                            censor_AM=False)
            self.assertEqual(opt_threshold, q, f"Failed for q={q} with censor_AM=False")
    
    def test_weibull_single_distribution(self):
        random_seed = self.random_seed
        ############################################################################
        # Test 3 for Weibull where tail can be described from weibull distribution
        # Set random seed for reproducibility
        np.random.seed(random_seed)
        # Number of samples
        n_samples = 1750
        # Generate 'year': random integers between 2000 and 2025
        years = np.random.randint(2000, 2026, size=n_samples)
        # Generate 'pr': Weibull-distributed values scaled to 0–800
        shape = 0.7   # Weibull shape parameter
        scale = 10 # Weibull scale parameter
        pr_values = scale * np.random.weibull(shape, size=n_samples)
        pr_values = np.clip(pr_values, 0, 800) # Clip to avoid extreme outliers
        # Create DataFrame
        df_example = pd.DataFrame({
            'year': years,
            'pr': pr_values
        }).sort_values(by="year").reset_index(drop=True)
        
        optimal_threshold, estimated_params, range_optimal, _ = weibul_test_MC(ordinary_events_df=df_example,
                                                                pr_field='pr',
                                                                hydro_year_field='year',
                                                                seed_random=random_seed,
                                                                synthetic_records_amount=500,
                                                                p_confidence=0.1,
                                                                make_plot=False)
        self.assertLess(optimal_threshold, 1, "Optimal threshold should be < 1 for Weibull tail")
        
    def test_uniform_distribution_no_weibull_tail(self):
        random_seed = self.random_seed
        ############################################################################
        # Test for Weibull where tail can NOT be described from weibull distribution
        # Set random seed for reproducibility
        np.random.seed(random_seed )
        # Number of samples
        n_samples = 1750
        # Generate 'year': random integers between 2000 and 2025
        years = np.random.randint(2000, 2026, size=n_samples)
        pr_values = np.random.uniform(0, 800, size=n_samples) # Clip to avoid extreme outliers
        
        # Create DataFrame
        df_example = pd.DataFrame({
            'year': years,
            'pr': pr_values
        }).sort_values(by="year").reset_index(drop=True)
        
        optimal_threshold, estimated_params, range_optimal, _ = weibul_test_MC(ordinary_events_df=df_example,
                                                                pr_field='pr',
                                                                hydro_year_field='year',
                                                                seed_random=random_seed ,
                                                                synthetic_records_amount=500,
                                                                p_confidence=0.1,
                                                                make_plot=False)
        
        self.assertEqual(optimal_threshold, 1, "Optimal threshold should be 1 for uniform distribution")
        

if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestWeibullTail)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)