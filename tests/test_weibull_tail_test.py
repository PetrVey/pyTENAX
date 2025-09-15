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


############################################################################
# Test for Weibull where tail can be described from weibull distribution
# This is with given quantile, which will define tail. to test if function is working all good
#for q in np.arange(0.5, 0.95, 0.05):
    # From which quantile the Weibull should be generated
    
q = 0.85   # splice quantile
#q = q.round(2)
# Seed for reproducibility
np.random.seed(42)
rng = np.random.default_rng(42)
# Given of the number of years is 25, generating 1000 events means that one year has 40 events (reasonable amoount)
# Number of samples
n_samples = 1000
low_idx = int(q * n_samples)
top_idx = 1000 - int(q * n_samples)
# Generate 'year': random integers between 2000 and 2025
years = np.random.randint(2000, 2026, size=n_samples)


# Assuming that all values larger than 10 comes from the parent weibull ditribution.
min_tail_thr = 5
# Parameters
shape = 0.7    # Weibull shape
scale = 10  # Weibull scale 
pr_weibull = scale * np.random.weibull(shape, size=n_samples*10) #generate 10x more than needed
pr_weibull = np.clip(pr_weibull, 0, 300) # Clip to avoid extreme outliers
pr_weibull_tail =  pr_weibull[ pr_weibull > min_tail_thr]

pr_weibull_tail = rng.choice(pr_weibull_tail, size=top_idx, replace=False)

# Generate exp and Weibull samples
#pr_exp = np.random.exponential(scale=10, size=n_samples*5) #scale 10 is medium slow decay where last value is around 11
#pr_exp = rng.beta(a=0.5, b=5, size=n_samples*10)  # Skewed toward 0
##r_exp = pr_exp * (min_tail_thr - 0.2) + 0.2
#pr_exp =  pr_exp[  pr_exp < min_tail_thr] # get exp which ends in min_tail_thr-5, this create sort of jump between distribtuons
#pr_exp = rng.choice(pr_exp, size=low_idx, replace=False)
pr_exp= np.full(low_idx, 0.2)
# Take bulk from uniform (top q%) and tail from Weibull (top 15%)
pr_mixed = np.concatenate([
    pr_exp,
    pr_weibull_tail,
])


# Randomly shuffle values
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
                    [0.85,1])


opt_threshold, estimated_params = weibul_test_MC(ordinary_events_df=df_example,
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

optimal_threshold, estimated_params = weibul_tail_test_MC(ordinary_events_df=df_example,
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
