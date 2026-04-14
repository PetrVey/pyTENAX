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

import timeit

N = 100

# Run both functions N times
time_old = timeit.timeit(
    lambda: S_SMEV.get_ordinary_events(data=df_arr, dates=df_dates, check_gaps=False),
    number=N
)
time_new = timeit.timeit(
    lambda: S_SMEV.get_ordinary_events_new(data=df_arr, dates=df_dates, check_gaps=False),
    number=N
)

idx_ordinary_old = S_SMEV.get_ordinary_events(data=df_arr, dates=df_dates, check_gaps=False)
idx_ordinary_new = S_SMEV.get_ordinary_events_new(data=df_arr, dates=df_dates, check_gaps=False)

# Check 1:1 equality
assert len(idx_ordinary_old) == len(idx_ordinary_new), \
    f"Length mismatch: old={len(idx_ordinary_old)}, new={len(idx_ordinary_new)}"

all_equal = all(
    np.array_equal(old, new)
    for old, new in zip(idx_ordinary_old, idx_ordinary_new)
)

print(f"--- get_ordinary_events ---")
print(f"Number of events - old: {len(idx_ordinary_old)}, new: {len(idx_ordinary_new)}")
print(f"Outputs are identical: {all_equal}")
print(f"Mean time old: {time_old/N*1000:.3f} ms")
print(f"Mean time new: {time_new/N*1000:.3f} ms")
print(f"Speedup: {time_old/time_new:.1f}x")

# --- remove_short comparison ---
time_rs_old = timeit.timeit(
    lambda: S_SMEV.remove_short(idx_ordinary_old),
    number=N
)
time_rs_new = timeit.timeit(
    lambda: S_SMEV.remove_short_new(idx_ordinary_new),
    number=N
)

arr_vals_old, arr_dates_old, n_per_year_old = S_SMEV.remove_short(idx_ordinary_old)
arr_vals_new, arr_dates_new, n_per_year_new = S_SMEV.remove_short_new(idx_ordinary_new)

assert np.array_equal(arr_vals_old, arr_vals_new), "arr_vals mismatch"
assert np.array_equal(arr_dates_old, arr_dates_new), "arr_dates mismatch"
assert n_per_year_old.equals(n_per_year_new), "n_per_year mismatch"

print(f"\n--- remove_short ---")
print(f"Events kept - old: {len(arr_vals_old)}, new: {len(arr_vals_new)}")
print(f"  arr_vals exact:   {np.array_equal(arr_vals_old, arr_vals_new)}")
print(f"  arr_dates exact:  {np.array_equal(arr_dates_old, arr_dates_new)}")
print(f"  n_per_year exact: {n_per_year_old.equals(n_per_year_new)}")
print(f"Mean time old: {time_rs_old/N*1000:.3f} ms")
print(f"Mean time new: {time_rs_new/N*1000:.3f} ms")
print(f"Speedup: {time_rs_old/time_rs_new:.1f}x")

# --- get_ordinary_events_values ---
dict_ordinary_old, dict_AMS_old = S_SMEV.get_ordinary_events_values(
    data=df_arr, dates=df_dates, arr_dates_oe=arr_dates_old
)
dict_ordinary_new, dict_AMS_new = S_SMEV.get_ordinary_events_values_new(
    data=df_arr, dates=df_dates, arr_dates_oe=arr_dates_new
)

time_gev_old = timeit.timeit(
    lambda: S_SMEV.get_ordinary_events_values(data=df_arr, dates=df_dates, arr_dates_oe=arr_dates_old),
    number=N
)
time_gev_new = timeit.timeit(
    lambda: S_SMEV.get_ordinary_events_values_new(data=df_arr, dates=df_dates, arr_dates_oe=arr_dates_new),
    number=N
)

print(f"\n--- get_ordinary_events_values ---")
for dur in [str(d) for d in S_SMEV.durations]:
    oe_old = dict_ordinary_old[dur]['ordinary'].to_numpy()
    oe_new = dict_ordinary_new[dur]['ordinary'].to_numpy()
    ams_old = dict_AMS_old[dur]['AMS'].to_numpy()
    ams_new = dict_AMS_new[dur]['AMS'].to_numpy()

    oe_exact     = np.array_equal(oe_old, oe_new)
    ams_exact    = np.array_equal(ams_old, ams_new)
    oe_max_diff  = np.max(np.abs(oe_old - oe_new))
    ams_max_diff = np.max(np.abs(ams_old - ams_new))

    print(f"  duration {dur:>4} min — "
          f"ordinary exact: {oe_exact} (max diff: {oe_max_diff:.2e}), "
          f"AMS exact: {ams_exact} (max diff: {ams_max_diff:.2e})")

print(f"Mean time old: {time_gev_old/N*1000:.3f} ms")
print(f"Mean time new: {time_gev_new/N*1000:.3f} ms")
print(f"Speedup: {time_gev_old/time_gev_new:.1f}x")

n_old = n_per_year_old.sum().item() / len(n_per_year_old)
n_new = n_per_year_new.sum().item() / len(n_per_year_new)

# --- SMEV parameters and return levels for all durations (new pipeline) ---
def _run_smev_all_durations():
    for dur in [str(d) for d in S_SMEV.durations]:
        P = dict_ordinary_new[dur]["ordinary"].to_numpy()
        shape, scale = S_SMEV.estimate_smev_parameters(P, S_SMEV.left_censoring)
        S_SMEV.smev_return_values(S_SMEV.return_period, shape, scale, n_new)

time_smev = timeit.timeit(_run_smev_all_durations, number=N)

rows = {}
for dur in [str(d) for d in S_SMEV.durations]:
    P = dict_ordinary_new[dur]["ordinary"].to_numpy()
    shape, scale = S_SMEV.estimate_smev_parameters(P, S_SMEV.left_censoring)
    RL = S_SMEV.smev_return_values(S_SMEV.return_period, shape, scale, n_new)
    rows[f"{dur} min"] = [round(shape, 4), round(scale, 4)] + [round(v, 2) for v in RL]

col_names = ["shape", "scale"] + [f"RP {rp}yr" for rp in S_SMEV.return_period]
df_table = pd.DataFrame(rows, index=col_names).T
print("\n--- SMEV parameters & return levels (mm) ---")
print(df_table.to_string())
print(f"Mean time (all durations): {time_smev/N*1000:.3f} ms")

# --- SMEV fit comparison plot: 10, 60, 1440 min ---
plot_durations = ["10", "60", "1440"]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, dur in zip(axes, plot_durations):
    # old pipeline
    P_old = dict_ordinary_old[dur]["ordinary"].to_numpy()
    AMS_old = dict_AMS_old[dur]
    shape_old, scale_old = S_SMEV.estimate_smev_parameters(P_old, S_SMEV.left_censoring)
    RL_old = S_SMEV.smev_return_values(S_SMEV.return_period, shape_old, scale_old, n_old)

    # new pipeline
    P_new = dict_ordinary_new[dur]["ordinary"].to_numpy()
    AMS_new = dict_AMS_new[dur]
    shape_new, scale_new = S_SMEV.estimate_smev_parameters(P_new, S_SMEV.left_censoring)
    RL_new = S_SMEV.smev_return_values(S_SMEV.return_period, shape_new, scale_new, n_new)

    # observed AMS plot positions (same for both)
    AMS_sort = AMS_old.sort_values(by=["AMS"])["AMS"]
    plot_pos = np.arange(1, len(AMS_sort) + 1) / (1 + len(AMS_sort))
    eRP = 1 / (1 - plot_pos)

    ax.plot(eRP, AMS_sort, "g+", label="Observed AMS")
    ax.plot(S_SMEV.return_period, RL_old, "--r", linewidth=2.5, label="SMEV old")
    ax.plot(S_SMEV.return_period, RL_new, "--b", linewidth=1.0, label="SMEV new")
    ax.set_xscale("log")
    ax.set_xlabel("Return period (years)")
    ax.set_ylabel("Depth (mm)")
    ax.set_title(f"Duration: {dur} min")
    ax.legend()

plt.tight_layout()
plt.show()

# --- Timing summary ---
print("\n" + "="*65)
print(f"{'TIMING SUMMARY':^65}")
print("="*65)
print(f"{'Function':<35} {'Old (ms)':>9} {'New (ms)':>9} {'Speedup':>8}")
print("-"*65)
print(f"{'get_ordinary_events':<35} {time_old/N*1000:>9.3f} {time_new/N*1000:>9.3f} {time_old/time_new:>7.1f}x")
print(f"{'remove_short':<35} {time_rs_old/N*1000:>9.3f} {time_rs_new/N*1000:>9.3f} {time_rs_old/time_rs_new:>7.1f}x")
print(f"{'get_ordinary_events_values':<35} {time_gev_old/N*1000:>9.3f} {time_gev_new/N*1000:>9.3f} {time_gev_old/time_gev_new:>7.1f}x")
print(f"{'estimate_params + return_levels':<35} {'N/A':>9} {time_smev/N*1000:>9.3f} {'—':>8}")
print("-"*65)
total_old = (time_old + time_rs_old + time_gev_old) / N * 1000
total_new = (time_new + time_rs_new + time_gev_new + time_smev) / N * 1000
print(f"{'TOTAL (pipeline)':<35} {total_old:>9.3f} {total_new:>9.3f} {total_old/total_new:>7.1f}x")
print("="*65)
