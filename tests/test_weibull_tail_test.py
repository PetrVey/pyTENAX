# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 17:28:25 2025

@author: Petr
"""
import unittest
import numpy as np
import pandas as pd
from pyTENAX.wbl_tail_test import weibul_test_MC
from pyTENAX import smev
import matplotlib.pyplot as plt

############################################################################
# Test for Weibull where tail can be described from weibull distribution
# This is with given quantile, which will define tail. to test if function is working all good
#for q in np.arange(0.5, 0.95, 0.05):
    # From which quantile the Weibull should be generated
q = 0.8   # splice quantile
#q = q.round(2)
# Seed for reproducibility

np.random.seed(42)
rng = np.random.default_rng(42)

# Given of the number of years is 25, generating 1000 events means that one year has 40 events (reasonable amoount)
# Number of samples
n_samples = 1000
# Generate 'year': random integers between 2000 and 2025
years = np.random.randint(2000, 2026, size=n_samples)

# Assuming that all values larger than 10 comes from the parent weibull ditribution.
min_tail_thr = 5
# Parameters
shape = 0.7    # Weibull shape
scale = 10  # Weibull scale 
pr_weibull = scale * np.random.weibull(shape, size=n_samples*1000) #generate 1000x more than needed for the smooher tail
pr_weibull = np.clip(pr_weibull, 0, 300) # Clip to avoid extreme outliers

# Number of low and high samples
n_low = int(n_samples * q) - 1 # Remove 1
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

ordinary_events_df = df_example
ordinary_events = ordinary_events_df["pr"]
smev.SMEV.estimate_smev_parameters(
                    None, # dummy class
                    ordinary_events, 
                    [q,1])


opt_threshold, estimated_params, range_optimal = weibul_test_MC(ordinary_events_df=df_example,
                                                                pr_field='pr',
                                                                hydro_year_field='year',
                                                                seed_random=7,
                                                                synthetic_records_amount=500,
                                                                p_confidence=0.1,
                                                                make_plot=True,
                                                                censor_AM=False)
expected_q = q
print(estimated_params)


print(f"expected { expected_q}, given {opt_threshold}")
    
""" 

############################################################################
# Test for Weibull where tail can be described from weibull distribution
# Set random seed for reproducibility
np.random.seed(42)
# Number of samples
n_samples = 1000
# Generate 'year': random integers between 2000 and 2025
years = np.random.randint(2000, 2026, size=n_samples)
# Generate 'pr': Weibull-distributed values scaled to 0–800
shape = 0.7   # Weibull shape parameter
scale = 2 # Weibull scale parameter
pr_values = scale * np.random.weibull(shape, size=n_samples)
pr_values = np.clip(pr_values, 0, 800) # Clip to avoid extreme outliers
# Create DataFrame
df_example = pd.DataFrame({
    'year': years,
    'pr': pr_values
}).sort_values(by="year").reset_index(drop=True)

optimal_threshold, estimated_params = weibul_test_MC(ordinary_events_df=df_example,
                                                        pr_field='pr',
                                                        hydro_year_field='year',
                                                        seed_random=7,
                                                        synthetic_records_amount=500,
                                                        p_confidence=0.1,
                                                        make_plot=True)

############################################################################
# Test for Weibull where tail can NOT be described from weibull distribution
# Set random seed for reproducibility
np.random.seed(42)
# Number of samples
n_samples = 1000
# Generate 'year': random integers between 2000 and 2025
years = np.random.randint(2000, 2026, size=n_samples)
pr_values = np.random.uniform(0, 800, size=n_samples) # Clip to avoid extreme outliers

# Create DataFrame
df_example = pd.DataFrame({
    'year': years,
    'pr': pr_values
}).sort_values(by="year").reset_index(drop=True)

Monte_Carlo(ordinary_events_df=df_example,
                pr_field='pr',
                hydro_year_field='year',
                seed_random=7,
                synthetic_records_amount=500,
                p_confidence=0.1,
                make_plot=True)"""
