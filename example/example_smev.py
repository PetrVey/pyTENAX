from importlib.resources import files
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Union
# Import pyTENAX
from pyTENAX import smev, plotting



# Initiate SMEV class with customized setup following TENAX
S_SMEV = smev.SMEV(
    return_period=[
        2,
        5,
        10,
        20,
        50,
        100,
        200,
    ],
    durations=[10, 60, 180, 360, 720, 1440],
    time_resolution=10,  # time resolution in minutes
    min_rain=0.1,
    storm_separation_time=24,
    min_event_duration=30,
    left_censoring=[0.9, 1],
)


# Load precipitation data
# Create input path file for the test file
file_path_input = files('pyTENAX.res').joinpath('prec_data_Aadorf.parquet')
# Load data from csv file
data = pd.read_parquet(file_path_input)
# Convert 'prec_time' column to datetime, if it's not already
data["prec_time"] = pd.to_datetime(data["prec_time"])
# Set 'prec_time' as the index
data.set_index("prec_time", inplace=True)
name_col = "prec_values"  # name of column containing data to extract


# Clean data from incomplete years
data = S_SMEV.remove_incomplete_years(data, name_col)

# get data from pandas to numpy array
# SMEV now supports only numpy inputs
df_arr = np.array(data[name_col])
df_dates = np.array(data.index)


# extract indexes of ordinary events
# these are time-wise indexes =>returns list of np arrays with np.timeindex
idx_ordinary = S_SMEV.get_ordinary_events(data=df_arr,
                                     dates=df_dates,
                                     name_col=name_col,
                                     check_gaps=False)

# get ordinary events by removing too short events
# returns boolean array, dates of OE in TO, FROM format, and count of OE in each years
arr_vals, arr_dates, n_ordinary_per_year = S_SMEV.remove_short(idx_ordinary)

# assign ordinary events values by given durations, values are in depth per duration, NOT in intensity mm/h
dict_ordinary, dict_AMS = S_SMEV.get_ordinary_events_values(data=df_arr,
                                                       dates=df_dates,
                                                       arr_dates_oe=arr_dates)

# We can SMEV for single duration, with extracing certain P data
# Your data (P, T arrays) and threshold thr=3.8
P = dict_ordinary["10"]["ordinary"].to_numpy()  # Replace with your actual data
blocks_id = dict_ordinary["10"]["year"].to_numpy()  # Replace with your actual data
#  mean n of ordinary events
n = n_ordinary_per_year.sum() / len(n_ordinary_per_year)
AMS = dict_AMS["10"] 

smev_shape, smev_scale = S_SMEV.estimate_smev_parameters(P, S_SMEV.left_censoring)
# estimate return period (quantiles) with SMEV
smev_RL = S_SMEV.smev_return_values(
    S_SMEV.return_period, smev_shape, smev_scale, n.item()
)

smev_RL_unc = S_SMEV.SMEV_bootstrap_uncertainty(P, blocks_id, 1000, n.item())


plotting.SMEV_FIG_valid(
                        AMS,
                        S_SMEV.return_period,
                        smev_RL,
                        smev_RL_unc
                        )
    
    
    
    
# it should also run as full smev
dict_smev_outputs = S_SMEV.do_smev_all(dict_ordinary, n.item())
#inside each duration from S.SMEV.durations is inside of dictionary
dict_smev_outputs['10']
dict_smev_outputs['60']
dict_smev_outputs['360']
dict_smev_outputs['720']
 dict_smev_outputs['1440']