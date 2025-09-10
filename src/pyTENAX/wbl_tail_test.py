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

#subgraph Core_Process["Monte_Carlo Hypothesis Test weibull tail test"]
#    A[Monte_Carlo] --> B[estimate_smev_param_without_AM]
#    A --> C[create_synthetic_records]
#    A --> D[check_confidence_interval]
#    A --> E[find_optimal_threshold]
#    A -->|optional| F[plot_curve]
#end

def estimate_smev_parameters(ordinary_events_df: pd.DataFrame, 
                           pr_field: str, 
                           data_portion: List[float]) -> List[float]:
    '''--------------------------------------------------------------------------
    Function that estimates parameters of the Weibull distribution
    
    Parameters
    ----------
    ordinary_events_df : pd.DataFrame
        Pandas dataframe of the ordinary events - without zeros!!!
    pr_field : str
        The name of the df column with precipitation values.
    data_portion : List[float]
        2-elements list with the limits in probability of the data to be used for the parameters estimation
        e.g. data_portion = [0.75, 1] uses the largest 25% values in data 
    
    Returns
    -------
    List[float]
        [shape, scale] parameters of the Weibull distribution
    -----------------------------------------------------------------------------'''
    sorted_df = np.sort(ordinary_events_df[pr_field].values)
    ECDF = (np.arange(1, 1 + len(sorted_df)) / (1 + len(sorted_df)))
    fidx = max(1, math.floor((len(sorted_df)) * data_portion[0]))  
    tidx = math.ceil(len(sorted_df) * data_portion[1])  
    to_use = np.arange(fidx - 1, tidx)  
    to_use_array = sorted_df[to_use]

    X = (np.log(np.log(1 / (1 - ECDF[to_use]))))  
    Y = (np.log(to_use_array))  
    X = sm.add_constant(X)  
    model = sm.OLS(Y, X)
    results = model.fit()
    param = results.params

    slope = param[1]
    intercept = param[0]

    shape = 1 / slope
    scale = np.exp(intercept)

    weibull_param = [shape, scale]

    return weibull_param

def smev_return_values(return_period, shape, scale, n):
    '''--------------------------------------------------------------------------
    Function that calculates return values according to parameters of the Weibull distribution
    
    arguments:
    - return_period (int): The desired return period for which intensity is calculated.
    - shape (float): Weibull distribution shape parameter
    - scale (float): Weibull distribution scale parameter
    - n (float): Mean number of ordinary events per year 
    
    returns:
    - intensity (float): The corresponding intensity value. 
    -----------------------------------------------------------------------------'''
    return_period = np.asarray(return_period)
    quantile = (1 - (1 / return_period))
    if shape == 0 or n == 0:
        intensity = 0
    else:
        intensity = scale * ((-1) * (np.log(1 - quantile ** (1 / n)))) ** (1 / shape)

    return intensity

def get_ordinary_events(data_df, zero, pr_field, hydro_year_field):
    '''--------------------------------------------------------------------------
    Function that extracts ordinary precipitation events out of the entire data.
    
    Parameters:
    - data_df (dataframe): df with 2 columns - precipitation values, hydrological year values
    - zero (float): The threshold value for precipitation amount.
                    Days with precipitation amounts above this threshold are considered as ordinary events.
    - pr_field (string): The name of the df column with precipitation values.
    - hydro_year_field (string): The name of the df column with hydrological years values.
    
    Returns:
    - ordinary_events_df (dataframe): Two columns df of ordinary events, and their corresponding hydrological year. 
    -----------------------------------------------------------------------------'''
    
    data_df = data_df.dropna()
    ordinary_events_df = data_df.loc[data_df[pr_field]>zero]
    
    return ordinary_events_df


def estimate_smev_param_without_AM(ordinary_events_df, pr_field, record_size, censor_value, annual_max_indexes):
    '''--------------------------------------------------------------------------
    Function that estimates parameters of the Weibull distribution, excluding annual maxima (block maxima) values. 
    
    Arguments:
    - ordinary_events_df (dataframe): pandas dataframe of the ordinary events - withot zeros!!!
    - pr_field (str): The name of the column with the precipitation values 
    - record_size (int): The number of ordinary events in the record
    - censor_value (float): The threshold for left censoring the record
    - annual_max_indexes (list): List of indexes in the record of the annual/block maxima
    
    Returns:
    - shape, scale (floats): Weibull distribution parameters 
    -----------------------------------------------------------------------------'''
    sorted_df = np.sort(ordinary_events_df[pr_field].values)
    ECDF = (np.arange(1, 1 + record_size) / (1 + record_size))
    data_portion=[censor_value,1]
    fidx = max(1, math.floor(record_size * data_portion[0]))  
    tidx = math.ceil(record_size * data_portion[1])  
    to_use = np.arange(fidx - 1, tidx)
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
        Value that determines the starting point for the pseudorandom number generator
    synthetic_records_amount : int
        Value that determines how many synthetic records to generate
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
    random_ordinary_events = []  
    for p in random_array:
        intensity = scale * ((-1) * (np.log(1 - p))) ** (1 / shape) 
        random_ordinary_events.append(intensity)

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
    If all threshold rejected - it will return 0.95 
    
    Arguments:
    - p_out_dicts_lst (list): List of dicts for each of the censor values tested, as follow: {censor_value:p_out}
    - p_confidence (float): Probability to be used for the test. confidence interval = 1-p_confidence 
    
    Returns:
    - optimal_threshold (float): The minimal threshold from which p_out <= p_confidence for all bigger thresholds.
                                 If all threshold rejected - it will return 0.95. If not all thresholds rejected,
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
            optimal_threshold = 0.95 
            #All thresholds rejected    
        else:
            #some thresholds rejected and some not
            index_to_use = indexes_rejected[-1]+1 # Select the next threshold after the biggest one that was rejected
            
            if index_to_use<len(thresholds_lst):
                optimal_threshold = thresholds_lst[index_to_use]
            else:
                optimal_threshold = 0.95

    else:
        optimal_threshold = thresholds_lst[0] 
        # No threshold rejected
        
    return optimal_threshold 

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
    
    plot_filename = 'monte_carlo_plot.png'
    
    # Save the plot with tight layout to prevent legend cutoff
    plt.tight_layout()
    plt.show()


def Monte_Carlo(ordinary_events_df: pd.DataFrame,
                pr_field: str,
                hydro_year_field: str,
                seed_random: int,
                synthetic_records_amount: int,
                p_confidence: float,
                make_plot: bool):
    '''--------------------------------------------------------------------------
    Function that tests the hypothesis that block maxima are samples from a parent distribution with Weibull tail.
    The tail is defined by a given left censoring threshold. 
    This function will return the optimal left censoring threshold. If all threshold rejected - it will return 0.95.
    If not all thresholds rejected, (1-optimal_threshold) is the portion of the record that can be assumed to be
    distributed Weibull.
    
    Parameters
    ----------
    ordinary_events_df : pd.DataFrame
        One column pandas dataframe of the ordinary events - without zeros!!!
    pr_field : str
        The name of the column with the precipitation values
    hydro_year_field : str
        The name of the column with the hydrological-years / blocks values
    seed_random : int
        Value that determines the starting point for the pseudorandom number generator
    synthetic_records_amount : int
        Value that determines how many synthetic records to generate
    p_confidence : float
        Probability to be used for the test. confidence interval = 1-p_confidence
    make_plot : bool
        Choose whether or not to include the plot
    csv_filename : str, optional
        Name of the input CSV file to use for the output plot name
    
    Returns
    -------
    Union[float, int]
        The optimal left censoring threshold, or 999999 if there is a problem with Weibull parameters fit
    -----------------------------------------------------------------------------'''
    
    censor_values = np.arange(0, 1, 0.05)
    
    ordinary_events_df = ordinary_events_df.sort_values(by=pr_field) 
    ordinary_events_df = ordinary_events_df.reset_index(drop=True)
    annual_max = sorted(list(ordinary_events_df.groupby(hydro_year_field)[pr_field].max().values))
    annual_max_indexes = sorted(list(ordinary_events_df.groupby(hydro_year_field)[pr_field].idxmax().values))

    record_size = len(ordinary_events_df)
    p_out_dicts_lst = []
    
    # Loop over censor values
    for censor_value in censor_values:    
        try:
            shape, scale = estimate_smev_param_without_AM(
                ordinary_events_df, 
                pr_field, 
                record_size, 
                censor_value, 
                annual_max_indexes
            ) 
        except Exception as e:
            return 999999
        
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

    optimal_threshold = find_optimal_threshold(p_out_dicts_lst, p_confidence)
    
    if make_plot:
        plot_curve(p_out_dicts_lst, p_confidence, optimal_threshold)
    
    return optimal_threshold


# Set random seed for reproducibility
np.random.seed(42)

# Number of samples
n_samples = 500

# Generate 'year': random integers between 2000 and 2025
years = np.random.randint(2000, 2026, size=n_samples)

# Generate 'pr': Weibull-distributed values scaled to 0–800
shape = 2.0   # Weibull shape parameter
scale = 300.0  # Weibull scale parameter
pr_values = scale * np.random.weibull(shape, size=n_samples)
pr_values = np.clip(pr_values, 0, 800)

# Create DataFrame
df_example = pd.DataFrame({
    'year': years,
    'pr': pr_values
})

print(df_example.head())

########################################################################


optimal_threshold = Monte_Carlo(ordinary_events_df=df_example,
                                pr_field='pr',
                                hydro_year_field='year',
                                seed_random=7,
                                synthetic_records_amount=10,
                                p_confidence=0.1,
                                make_plot=True)


# Set random seed for reproducibility
np.random.seed(42)

# Number of samples
n_samples = 500

# Generate 'year': random integers between 2000 and 2025
years = np.random.randint(2000, 2026, size=n_samples)

pr_values = np.random.uniform(0, 800, size=n_samples)


# Create DataFrame
df_example = pd.DataFrame({
    'year': years,
    'pr': pr_values
})

print(df_example.head())

########################################################################


Monte_Carlo(ordinary_events_df=df_example,
                pr_field='pr',
                hydro_year_field='year',
                seed_random=7,
                synthetic_records_amount=10,
                p_confidence=0.1,
                make_plot=True)