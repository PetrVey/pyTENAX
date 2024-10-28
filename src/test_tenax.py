"""
Created on Thu Oct 17 14:53:36 2024

@author: Petr
@Riccardo Ciceri, riccardo.ciceri@studenti.unipd.it
# Developed starting from https://zenodo.org/records/11935026
"""
import os
# os.environ['USE_PYGEOS'] = '0'
from os.path import dirname, abspath, join
from os import getcwd
import sys
#run this fro src folder, otherwise it doesn't work
THIS_DIR = dirname(getcwd())
CODE_DIR = join(THIS_DIR, 'src')
RES_DIR =  join(THIS_DIR, 'res')
sys.path.append(CODE_DIR)
sys.path.append(RES_DIR)
import numpy as np
import pandas as pd
from pyTENAX.pyTENAX import *
import time 
import sys


S = TENAX(
        return_period = [2,5,10,20,50,100, 200],
        durations = [10, 60, 180, 360, 720, 1440],
        left_censoring = [0, 0.90],
    )

file_path_input =f"{RES_DIR}/prec_data_Aadorf.parquet"
#Load data from csv file
data=pd.read_parquet(file_path_input)
# Convert 'prec_time' column to datetime, if it's not already
data['prec_time'] = pd.to_datetime(data['prec_time'])
# Set 'prec_time' as the index
data.set_index('prec_time', inplace=True)
name_col = "prec_values" #name of column containing data to extract

start_time = time.time()

#push values belows 0.1 to 0 in prec due to 
data.loc[data[name_col] < S.min_rain, name_col] = 0


data = S.remove_incomplete_years(data, name_col)


#get data from pandas to numpy array
df_arr = np.array(data[name_col])
df_dates = np.array(data.index)

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
file_path_temperature = f"{RES_DIR}/temp_data_Aadorf.parquet"
t_data=pd.read_parquet(file_path_temperature)
# Convert 'temp_time' column to datetime if it's not already in datetime format
t_data['temp_time'] = pd.to_datetime(t_data['temp_time'])
# Set 'temp_time' as the index
t_data.set_index('temp_time', inplace=True)

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