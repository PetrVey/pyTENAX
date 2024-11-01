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
import matplotlib.pyplot as plt
from scipy.stats import chi2



S = TENAX(
        return_period = [1.1,1.2,1.5,2,5,10,20,50,100, 200],  #for some reason it doesnt like calculating RP =<1
        durations = [10, 60, 180, 360, 720, 1440],
        left_censoring = [0, 0.90],
        alpha = 0.05,
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
RL, T_mc, P_mc = S.model_inversion(F_phat, g_phat, n, Ts,n_mc = np.size(P)*S.niter_smev) 
print(RL)



#PLOTTING THE GRAPHS

eT = np.arange(np.min(T),np.max(T)+4,1) # define T values to calculate distributions. +4 to go beyond graph end

# fig 2a
qs = [.85,.95,.99,.999]
TNX_FIG_magn_model(P,T,F_phat,thr,eT,qs)
plt.show()

#fig 2b
TNX_FIG_temp_model(T=T, g_phat=g_phat,beta=4,eT=eT)
plt.show()

#fig 4 (without SMEV and uncertainty) 
AMS = dict_AMS['10'] # yet the annual maxima
TNX_FIG_valid(AMS,S.return_period,RL)
plt.show()


#fig 5 
iTs = np.arange(-2.5,37.5,1.5) #idk why we need a different T range here 

TNX_FIG_scaling(P,T,P_mc,T_mc,F_phat,S.niter_smev,eT,iTs)
plt.show()

#SPLITTING INTO SUMMER/WINTER
season_separations = [5, 10]
months = dict_ordinary["10"]["oe_time"].dt.month
winter_inds = months.index[(months>season_separations[1]) | (months<season_separations[0])]
summer_inds = months.index[(months<season_separations[1]+1)&(months>season_separations[0]-1)]
T_winter = T[winter_inds]
T_summer = T[summer_inds]


g_phat_winter = temperature_model_free(2, T_winter)
g_phat_summer = temperature_model_free(2, T_summer)


winter_pdf = gen_norm_pdf(eT, g_phat_winter[0], g_phat_winter[1], 2)
summer_pdf = gen_norm_pdf(eT, g_phat_summer[0], g_phat_summer[1], 2)

combined_pdf = (winter_pdf*np.size(T_winter)+summer_pdf*np.size(T_summer))/(np.size(T_winter)+np.size(T_summer))


#fig 3


TNX_FIG_temp_model(T=T_summer, g_phat=g_phat_summer,beta=2,eT=eT,obscol='r',valcol='r',xlimits = [-15,30],ylimits = [0,0.1])
TNX_FIG_temp_model(T=T_winter, g_phat=g_phat_winter,beta=2,eT=eT,obscol='b',valcol='b',xlimits = [-15,30],ylimits = [0,0.1])
TNX_FIG_temp_model(T=T, g_phat=g_phat,beta=4,eT=eT,obscol='k',valcol='k',xlimits = [-15,30],ylimits = [0,0.1])
plt.plot(eT,combined_pdf,'m',label = 'Combined summer and winter')
plt.show()


#TENAX MODEL VALIDATION
yrs = dict_ordinary["10"]["oe_time"].dt.year
yrs_unique = np.unique(yrs)
midway = yrs_unique[int(np.ceil(np.size(yrs_unique)/2))]

#DEFINE FIRST PERIOD
P1 = P[yrs<=midway]
T1 = T[yrs<=midway]
AMS1 = AMS[AMS['year']<=midway]
n_ordinary_per_year1 = n_ordinary_per_year[n_ordinary_per_year.index<=midway]
n1 = n_ordinary_per_year1.sum() / len(n_ordinary_per_year1)

#DEFINE SECOND PERIOD
P2 = P[yrs>midway]
T2 = T[yrs>midway]
AMS2 = AMS[AMS['year']>midway]
n_ordinary_per_year2 = n_ordinary_per_year[n_ordinary_per_year.index>midway]
n2 = n_ordinary_per_year2.sum() / len(n_ordinary_per_year2)


g_phat1 = S.temperature_model(T1)
g_phat2 = S.temperature_model(T2)


F_phat1, loglik1, _, _ = S.magnitude_model(P1, T1, thr)
RL1, T_mc1, P_mc1 = S.model_inversion(F_phat1, g_phat1, n1, Ts,n_mc = np.size(P1)*S.niter_smev)
   
F_phat2, loglik2, _, _ = S.magnitude_model(P2, T2, thr)
RL2, T_mc2, P_mc2 = S.model_inversion(F_phat2, g_phat2, n2, Ts,n_mc = np.size(P2)*S.niter_smev)   

if F_phat[2]==0:
    dof=3
    alpha1=1; # b parameter is not significantly different from 0; 3 degrees of freedom for the LR test
else: 
    dof=4
    alpha1=0  # b parameter is significantly different from 0; 4 degrees of freedom for the LR test




#check magnitude model the same in both periods
lambda_LR = -2*( loglik - (loglik1+loglik2) )
pval = chi2.sf(lambda_LR, dof)

#modelling second model based on first magnitude and changes in mean/std
mu_delta = np.mean(T2)-np.mean(T1)
sigma_factor = np.std(T2)/np.std(T1)

g_phat2_predict = [g_phat1[0]+mu_delta, g_phat1[1]*sigma_factor]
RL2_predict, _,_ = S.model_inversion(F_phat1,g_phat2_predict,n2,Ts)


#fig 7a

TNX_FIG_temp_model(T=T1, g_phat=g_phat1,beta=4,eT=eT,obscol='b',valcol='b')
TNX_FIG_temp_model(T=T2, g_phat=g_phat2_predict,beta=4,eT=eT,obscol='r',valcol='r') # model based on temp ave and std changes
plt.show() #this is slightly different in code and paper I think.. using predicted T vs fitted T

#fig 7b

TNX_FIG_valid(AMS1,S.return_period,RL1,TENAXcol='b',obscol_shape = 'b+')
TNX_FIG_valid(AMS2,S.return_period,RL2_predict,TENAXcol='r',obscol_shape = 'r+')

plt.show()



# SENSITIVITY ANALYSIS

# changes in T mean, std, and n to chekc sensitivity
delta_Ts = [-1, 1, 2, 3]
delta_as = [.9, .95, 1.05, 1.1, 1.2]
delta_ns = [.5, .75, 1.3, 2]

# T mean sensitivity
T_sens = np.zeros([np.size(delta_Ts),np.size(S.return_period)])
i=0
while i<np.size(delta_Ts):
    
    T_sens[i,:],_,_ = S.model_inversion(F_phat, [g_phat[0]+delta_Ts[i],g_phat[1]], n, Ts) 
    i=i+1

# T std sensitivity

as_sens = np.zeros([np.size(delta_as),np.size(S.return_period)])
i=0
while i<np.size(delta_as):
    
    as_sens[i,:],_,_ = S.model_inversion(F_phat, [g_phat[0],g_phat[1]*delta_as[i]], n, Ts) 
    i=i+1

# n sensitivity
n_sens = np.zeros([np.size(delta_ns),np.size(S.return_period)])
i=0
while i<np.size(delta_ns):
    
    n_sens[i,:],_,_ = S.model_inversion(F_phat, g_phat, n*delta_ns[i], Ts) 
    i=i+1


#fig 6
fig = plt.figure(figsize = (15,5))
ax1 = fig.add_subplot(1,3,1)
i = 0
while i< np.size(delta_Ts):  
    ax1.plot(S.return_period,T_sens[i],'k',alpha = 0.7,label = str(delta_Ts[i]))
    
    i=i+1
plt.xscale('log')
ax1.plot(S.return_period,RL,'b')
ax1.set_title('Sensitivity to changes in mean temp')
plt.legend()
plt.xscale('log')
plt.xlim(1,200)
plt.ylim(0.60)

ax2 = fig.add_subplot(1,3,2)
i = 0
while i< np.size(delta_as):  
    ax2.plot(S.return_period,as_sens[i],'k',alpha = 0.7,label = str(delta_as[i]))
    
    i=i+1
plt.xscale('log')
ax2.plot(S.return_period,RL,'b',label = 'The TENAX MODEL')
ax2.set_title('Sensitivity to changes in temp std')
plt.legend()
plt.xscale('log')
plt.xlim(1,200)
plt.ylim(0.60)

ax3 = fig.add_subplot(1,3,3)
i = 0
while i< np.size(delta_ns):  
    ax3.plot(S.return_period,n_sens[i],'k',alpha = 0.7,label = str(delta_ns[i]))
    
    i=i+1
plt.xscale('log')
ax3.plot(S.return_period,RL,'b')
ax3.set_title('Sensitivity to changes in mean events per year (n)')
plt.legend()
plt.xscale('log')
plt.xlim(1,200)
plt.ylim(0.60)



plt.show()




