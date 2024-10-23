"""
Created on Thu Oct 17 14:53:36 2024

@author: Petr
@Riccardo Ciceri, riccardo.ciceri@studenti.unipd.it
# Developed starting from https://zenodo.org/records/11935026
"""

import numpy as np
import pandas as pd
from pyTENAX.pyTENAX import TENAX
import time 

S = TENAX(
        return_period = [2,5,10,20,50,100, 200],
        durations = [10, 60, 180, 360, 720, 1440],
        left_censoring = [0, 0.90]
    )

file_path_input ="prec_data.csv"
#Load data from csv file
data=pd.read_csv(file_path_input, parse_dates=True, index_col='prec_time')
name_col = "prec_values" #name of column containing data to extract

start_time = time.time()

#push values belows 0.1 to 0 in prec due to 
data.loc[data[name_col] < S.min_rain, name_col] = 0
data = S.remove_incomplete_years(data, name_col)


#get data from pandas to numpy array
df_arr = np.array(data[name_col])
df_dates=np.array(data.index)

#extract indexes of ordinary events
#these are time-wise indexes =>returns list of np arrays with np.timeindex
idx_ordinary=S.get_ordinary_events(data=df_arr,dates=df_dates, name_col=name_col,  check_gaps=False)
    

#get ordinary events by removing too short events
#returns boolean array, dates of OE in TO, FROM format, and count of OE in each years
arr_vals,arr_dates,n_ordinary_per_year=S.remove_short(idx_ordinary)

#assign ordinary events values by given durations, values are in depth per duration, NOT in intensity mm/h
dict_ordinary, dict_AMS = S.get_ordinary_events_values(data=df_arr,dates=df_dates, arr_dates_oe=arr_dates)

elapsed_time = time.time() - start_time
# Print the elapsed time
print(f"Elapsed time get OE: {elapsed_time:.4f} seconds")

#load temperature data
t_data=pd.read_csv("temp_data.csv", parse_dates=True, index_col='temp_time')


start_time = time.time()
temp_name_col = "temp_values"
df_arr_t_data = np.array(t_data[temp_name_col])
df_dates_t_data = np.array( t_data.index)

dict_ordinary, _ , n_ordinary_per_year = S.associate_vars(dict_ordinary, df_arr_t_data, df_dates_t_data)

elapsed_time = time.time() - start_time
# Print the elapsed time
print(f"Elapsed time : {elapsed_time:.4f} seconds")


start_time = time.time()
# Your data (P, T arrays) and threshold thr=3.8
P = dict_ordinary["10"]["ordinary"].to_numpy() # Replace with your actual data
T = dict_ordinary["10"]["T"].to_numpy()  # Replace with your actual data



# Number of threshold 
thr = dict_ordinary["10"]["ordinary"].quantile(S.left_censoring[1])

# Sampling intervals for the Montecarlo
Ts = np.arange(np.min(T) - S.temp_delta, np.max(T) + S.temp_delta, S.temp_res_monte_carlo)

#TENAX MODEL HERE
#magnitude model
F_phat, loglik, _, _ = S.magnitude_model(P, T, thr)
#temperature model
g_phat = S.temperature_model(T)
# M is mean n of ordinary events
n = n_ordinary_per_year.sum() / len(n_ordinary_per_year)  
#estimates return levels using MC samples
RL, _, __ = S.model_inversion(F_phat, g_phat, n, Ts)
print(RL)