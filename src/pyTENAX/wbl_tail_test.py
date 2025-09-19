# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 15:53:57 2025

@author1: Yaniv yaniv.goldschmidt@unipd.it
@author2: PetrVey

"""
import math
import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import List, Tuple, Union, Optional
import os
import matplotlib.pyplot as plt
from pyTENAX import smev
import warnings

#subgraph Core_Process["Monte_Carlo Hypothesis Test weibull tail test"]
#    A[weibul_test_MC] --> B[estimate_smev_param_without_AM]
#    A --> C[create_synthetic_records]
#    A --> D[check_confidence_interval]
#    A --> E[find_optimal_threshold]
#    A -->|optional| F[plot_curve]
#end

def estimate_smev_param_without_AM(ordinary_events: Union[np.ndarray, pd.Series, list], 
                                   censor_value, annual_max_indexes):
    '''--------------------------------------------------------------------------
    Function that estimates parameters of the Weibull distribution, excluding annual maxima (block maxima) values. 
    
    Arguments:
    - ordinary_events ([np.ndarray, pd.Series, list): values of ordinary events - without zeros!!!
    - censor_value (float): The threshold for left censoring the record
    - annual_max_indexes (list): List of indexes in the record of the annual/block maxima, COMING ALREADY FROM SORTED ARRAY!
    
    Returns:
    - shape, scale (floats): Weibull distribution parameters 
    -----------------------------------------------------------------------------'''
    
    sorted_df = np.sort(ordinary_events)
    record_size = len(sorted_df) #The number of ordinary events in the record
    ECDF = (np.arange(1, 1 + record_size) / (1 + record_size))
    data_portion=[censor_value,1]
    #fidx: first index of data to keep
    fidx = max(1, math.floor(record_size * data_portion[0]))  
    #tidx: last index of data to keep
    tidx = math.ceil(record_size * data_portion[1])
    if fidx == 1:
        to_use = np.arange(fidx-1, tidx) # Create an array of indices from fidx-1 up to tidx (inclusive)
    else:
        to_use = np.arange(fidx, tidx) # Create an array of indices from fidx up to tidx (inclusive)
        
    to_use_without_am = [index for index in to_use if index not in annual_max_indexes]
    events_without_am = sorted_df[to_use_without_am]


    X = (np.log(np.log(1 / (1 - ECDF[to_use_without_am]))))  
    Y = (np.log(events_without_am))  
    X = sm.add_constant(X)  
    model = sm.OLS(Y, X)
    results = model.fit()
    param = results.params

    slope = param[1]
    intercept = param[0]

    shape = 1 / slope
    scale = np.exp(intercept)
    
    return shape, scale


def create_synthetic_records(seed_random: int,
                           synthetic_records_amount: int,
                           record_size: int,
                           shape: float,
                           scale: float) -> pd.DataFrame:
    '''--------------------------------------------------------------------------
    Function that generates synthetic records using the Weibull parameters which were
    estimated based on original record (without AM).
    The synthetic records contain random ordinary events sampled uniformly from the Weibull distribution.  
    These synthetic records use as the basis for extracting the confidence interval.
    
    Parameters
    ----------
    seed_random : int
        Value that determines the starting point for the pseudorandom number generator => due to reproducibility
    synthetic_records_amount : int
        Value that determines how many synthetic records to generate => number of stochastic realizations
    record_size : int
        The number of ordinary events in the record
    shape : float
        Weibull distribution parameter
    scale : float
        Weibull distribution parameter
    
    Returns
    -------
    pd.DataFrame
        DataFrame with all the synthetic records. Each row represents separate synthetic record.
    -----------------------------------------------------------------------------'''
    
    # Create a local random state
    rng = np.random.RandomState(seed_random)
    
    # Generate random array of probability values between 0 to 1, uniformly sampled
    random_array = rng.uniform(0, 1, synthetic_records_amount * record_size) 

    # Calculate quantiles & create records matrix
    random_ordinary_events = scale * ((-1) * np.log(1 - random_array)) ** (1 / shape)
    
    #old
    #random_ordinary_events = []  
    #for p in random_array:
    #    intensity = scale * ((-1) * (np.log(1 - p))) ** (1 / shape) 
    #    random_ordinary_events.append(intensity)

    records_matrix = np.array(random_ordinary_events).reshape(synthetic_records_amount, record_size)
    records_matrix = np.sort(records_matrix, axis=1)  # sort each row
    records_df = pd.DataFrame(records_matrix) 
    
    return records_df

def check_confidence_interval(annual_max_indexes, records_df, p_confidence, annual_max, censor_value, p_out_dicts_lst):
    '''--------------------------------------------------------------------------
    Function that checks the fraction of the annual/block maxima that are out of the confidence interval.
    
    Arguments:
    - annual_max_indexes (list): List of indexes in the record of the annual/block maxima
    - records_df (dataframe): df with all the synthetic records
    - p_confidence (float): Probability to be used for the test. confidence interval = 1-p_confidence 
    - annual_max (list): List of values over which the hypothesis is tested, i.e. block maxima
    - censor_value (float): The threshold for left censoring the record 
    - p_out_dicts_lst (list): List of dicts - Each censor value tested gets a dict as follow: {censor_value:p_out}
    
    Returns:
    - p_out_dicts_lst (list): Same list as in the arguments, after appending dict for the tested censor_value
    -----------------------------------------------------------------------------'''
    
    p_lo = 0
    p_hi = 0

    counter_index = 0
    # Iterate over AM values
    for index in annual_max_indexes: 
        column =  records_df.iloc[:,index] # Select from each synthetic record the value in the position of the AM tested value
        
        # Create confidence interval
        lower = p_confidence/2
        upper = 1-(p_confidence/2)
        quantiles = column.quantile([lower, upper])
        quantile_lower = quantiles.iloc[0] # Lower value of confidence interval
        quantile_upper = quantiles.iloc[1] # Upper value of confidence interval
        
        # Select AM value to test
        annual_max_value = annual_max[counter_index]

        counter_index +=1 
        
        # Test if AM is within the confidence interval - i.e count how many times AM value is out of confidence interval  
        if annual_max_value<quantile_lower :
            p_lo += 1
        elif annual_max_value>quantile_upper :
            p_hi += 1

    p_out = p_hi/len(annual_max) + p_lo/len(annual_max) # fraction of block maxima out of the (1-p) CI        

    p_out_dict = {round(censor_value,2):round(p_out,2)}

    p_out_dicts_lst.append(p_out_dict)
    
    return p_out_dicts_lst

def find_optimal_threshold(p_out_dicts_lst, p_confidence):
    '''--------------------------------------------------------------------------
    Function that finds the optimal threshold out of the list of dicts.
    The function returns the minimal threshold from which p_out <= p_confidence for all bigger thresholds.
    If all threshold rejected - it will return 1
    
    Arguments:
    - p_out_dicts_lst (list): List of dicts for each of the censor values tested, as follow: {censor_value:p_out}
    - p_confidence (float): Probability to be used for the test. confidence interval = 1-p_confidence 
    
    Returns:
    - optimal_threshold (float): The minimal threshold from which p_out <= p_confidence for all bigger thresholds.
                                 If all threshold rejected - it will return 1. If not all thresholds rejected,
                                 (1-optimal_threshold) is the portion of the record that can be assumed to be 
                                 distributed Weibull.
      
    -----------------------------------------------------------------------------'''
    
    p_out_lst = []
    thresholds_lst = []
    
    # Get values from p_out_dicts - thresholds and their corresponding p_out  
    for dic in p_out_dicts_lst:
        p_out_lst.append(list(dic.values())[0])
        thresholds_lst.append(list(dic.keys())[0])
    
    # Get indexes of all thresholds that are rejected
    indexes_rejected = [index for index, p_out in enumerate(p_out_lst) if p_out > p_confidence]
    
    if len(indexes_rejected)>0 : 
        if len(indexes_rejected)==len(thresholds_lst): 
            optimal_threshold = 1
            #All thresholds rejected    
        else:
            #some thresholds rejected and some not
            index_to_use = indexes_rejected[-1]+1 # Select the next threshold after the biggest one that was rejected
            
            if index_to_use<len(thresholds_lst):
                optimal_threshold = thresholds_lst[index_to_use]
            else:
                optimal_threshold = 1

    else:
        optimal_threshold = thresholds_lst[0] 
        # No threshold rejected
        
    thr_below = [
    list(d.keys())[0]  # extract the single key
    for d in p_out_dicts_lst
    if list(d.values())[0] < p_confidence
    ]    
        
    return optimal_threshold, thr_below

def plot_curve(p_out_dicts_lst, p_confidence, optimal_threshold):
    '''--------------------------------------------------------------------------
    Function that plots a curve that shows for each threshold its corresponding fraction of block maxima out of
    the confidence interval.
    
    Arguments:
    - p_out_dicts_lst (list): List of dicts for each of the censor values tested, as follow: {censor_value:p_out}
    - p_confidence (float): Probability to be used for the test. confidence interval = 1-p_confidence
    - optimal_threshold (float): The optimal left censoring threshold
    - csv_filename (str): Name of the input CSV file (without extension) to use for the output plot name
    
    Returns:
    - Saves figure to monte_carlo/monte_carlo_output directory
    -----------------------------------------------------------------------------'''
    
    # Create figure with larger size to accommodate legend
    plt.figure(figsize=(10, 6))
    
    # Extract keys and values from the dictionaries
    keys = [list(d.keys())[0] for d in p_out_dicts_lst]
    values = [list(d.values())[0] for d in p_out_dicts_lst]
    # Create a DataFrame
    df_p_out = pd.DataFrame({'censor_value': keys, 'p_out': values})
    
    # Plotting the line plot
    plt.plot(df_p_out['censor_value'], df_p_out['p_out'], label=f'Fraction of\nblock maxima\nout of {100*p_confidence}% CI')

    # Adding a dashed line at p_confidence
    plt.axhline(y=p_confidence, color='gray', linestyle='--')
    
    # Finding the optimal threshold point 
    intersection = df_p_out.loc[df_p_out['p_out'] <= p_confidence, ['censor_value','p_out']]
    
    # Plotting a point at optimal threshold point 
    if intersection.size > 0:
        optimal_censor_value = optimal_threshold  
        optimal_p_out_loc = intersection.loc[intersection['censor_value'] == optimal_censor_value, 'p_out'].max()
        plt.scatter(optimal_censor_value, optimal_p_out_loc, color='k', marker='o')
        plt.text(optimal_censor_value, p_confidence+.05, f'{optimal_censor_value}',color='red', ha='center')
        plt.axvline(x=optimal_censor_value, ymax=p_confidence+.04, color='red', linestyle='--')

    plt.xlabel('Left-censoring threshold',size=13)
    plt.ylabel(f'Fraction of block maxima out of {100*p_confidence}% CI', size=13)
    
    # Place legend inside the plot at top right
    plt.legend(loc='upper right')
    
    # Save the plot with tight layout to prevent legend cutoff
    plt.tight_layout()
    plt.show()


def weibul_test_MC(ordinary_events_df: pd.DataFrame,
                pr_field: str,
                hydro_year_field: str,
                seed_random: int= 42,
                synthetic_records_amount: int=500,
                p_confidence: float =0.1,
                make_plot: bool = True,
                censor_AM: bool = True,
                censor_values: Union[np.ndarray, list] = np.arange(0, 1, 0.05)):
    '''--------------------------------------------------------------------------
    Function that tests the hypothesis that block maxima are samples from a parent distribution with Weibull tail.
    The tail is defined by a given left censoring threshold. 
    This function will return the optimal left censoring threshold. If all threshold rejected - it will return 1.
    If not all thresholds rejected, (1-optimal_threshold) is the portion of the record that can be assumed to be
    distributed Weibull.
    
    ⚠️ Warning:
    If the returned optimal_threshold == 1.0, this does not necessarily mean that 
    all thresholds failed. It may indicate that your tested threshold range 
    extended too high (e.g., up to 0.95 or higher), where results are dominated 
    by stochastic sampling noise of extremes.  
    In such cases, you should check whether some lower thresholds worked.  
    As a rule of thumb, thresholds above ~0.95 are usually unreliable, though 
    the exact cutoff is case dependent.
    
    Parameters
    ----------
    ordinary_events_df : pd.DataFrame
        One column pandas dataframe of the ordinary events - without zeros!!!
    pr_field : str
        The name of the column with the precipitation values
    hydro_year_field : str
        The name of the column with the hydrological-years / blocks values
    seed_random : int
        Value that determines the starting point for the pseudorandom number generator => due to reproducibility
    synthetic_records_amount : int
        Value that determines how many synthetic records to generate. => number of stochastic realizations
    p_confidence : float
        Probability to be used for the test. confidence interval = 1-p_confidence
    make_plot : bool
        Choose whether or not to include the plot
    censor_AM: bool, natively True
        Choose whether or not the annual maximas should be included in ordinary events and test
    censor_values_range: np.ndarray or list
        The censoring thresholds which should be tested, nativally range from 0 to 1 in 0.05 step

    Returns
    -------
    optimal_threshold: Union[float, int]
        The optimal left censoring threshold, or 1 if all rejected, or 1111 if there is a problem with Weibull parameters fit
    estimated_params: list
        Estimated weibull parameters of the optimal threshold (None if optimal==1)
    range_of_optimal: list
        List of thresholds where p_out < p_confidence.
    p_out_dicts_lst: list
        Fraction of block maxima outside of the Y = 1-p_out confidence interval 
    -----------------------------------------------------------------------------'''
    
    ordinary_events_df = ordinary_events_df.sort_values(by=pr_field) 
    ordinary_events_df = ordinary_events_df.reset_index(drop=True)
    annual_max = sorted(list(ordinary_events_df.groupby(hydro_year_field)[pr_field].max().values))
    annual_max_indexes = sorted(list(ordinary_events_df.groupby(hydro_year_field)[pr_field].idxmax().values)) 
    record_size = len(ordinary_events_df)
    p_out_dicts_lst = []
    ordinary_events = ordinary_events_df[pr_field]
    
    shape_scale_dict = {}
    
    # Loop over censor values
    for censor_value in censor_values: 
        censor_value = censor_value.round(2)
        try:
            if censor_AM == True:
                shape, scale = estimate_smev_param_without_AM(
                    ordinary_events, 
                    censor_value, #data_portion is auto created in this func
                    annual_max_indexes
                ) 
            else:
                shape, scale = smev.SMEV.estimate_smev_parameters(
                    None, # dummy class
                    ordinary_events, 
                    [censor_value,1]) # in smev the input is data_portion [x, 1]
        except Exception as e:
            print(f"Error occurred: {e}") 
            print("The parameters of SMEV cannot be estimated, usually due to too small number of events after censoring.") 
            return 1111
        
        records_df = create_synthetic_records(
            seed_random, 
            synthetic_records_amount, 
            record_size, 
            shape, 
            scale
        ) 
        
        p_out_dicts_lst = check_confidence_interval(
            annual_max_indexes, 
            records_df, 
            p_confidence, 
            annual_max, 
            censor_value, 
            p_out_dicts_lst) 
        
        shape_scale_dict[censor_value.round(2)] = [shape, scale]

    optimal_threshold, range_of_optimal = find_optimal_threshold(p_out_dicts_lst, p_confidence)
    if optimal_threshold != 1:
        estimated_params = shape_scale_dict[optimal_threshold]
    else:
        estimated_params= None
    
    
    if make_plot:
        plot_curve(p_out_dicts_lst, p_confidence, optimal_threshold)
        
    # Warning if optimal threshold is 1.0
    if (optimal_threshold == 1.0) and (len(range_of_optimal) != 0):
        warnings.warn(
            "Optimal threshold reached the upper limit (0.95) but not all thresholds were rejected.\n"
            "Consider checking your tested threshold range:\n"
            "too high thresholds may fall into stochastic noise of extremes\n"
            "(typically values > 0.95 are unreliable).\n"
            "Check if lower thresholds provided valid results.\n",
            UserWarning
        )
    
    return optimal_threshold, estimated_params, range_of_optimal, p_out_dicts_lst

