from importlib.resources import files
import numpy as np
import pandas as pd
import timeit
from pyTENAX import smev
from decimal import Decimal

# --- Setup ---
S_SMEV = smev.SMEV(
    return_period=[2, 5, 10, 20, 50, 100, 200],
    durations=[10, 60, 180, 360, 720, 1440],
    time_resolution=10,
    min_rain=0.1,
    storm_separation_time=24,
    min_event_duration=30,
    left_censoring=[0.9, 1],
)

file_path_input = files('pyTENAX.res').joinpath('prec_data_Aadorf.parquet')
data = pd.read_parquet(file_path_input)
data["prec_time"] = pd.to_datetime(data["prec_time"])
data.set_index("prec_time", inplace=True)
name_col = "prec_values"

data = S_SMEV.remove_incomplete_years(data, name_col)
df_arr = np.round(np.array(data[name_col]), 4) 
df_dates = np.array(data.index)

# --- Storm separation (new pipeline) ---
idx_ordinary = S_SMEV.get_ordinary_events_new(data=df_arr, dates=df_dates, check_gaps=False)
arr_vals, arr_dates, n_per_year = S_SMEV.remove_short_new(idx_ordinary)
print(f"Storm separation done: {len(arr_vals)} events")

# --- Warmup numba (first call triggers JIT compilation) ---
print("Warming up numba JIT (first call compiles)...")
_ = S_SMEV.get_ordinary_events_values_new_numba(data=df_arr, dates=df_dates, arr_dates_oe=arr_dates)
print("Warmup done.")

N = 1

# --- Benchmark ---
time_old = timeit.timeit(
    lambda: S_SMEV.get_ordinary_events_values(data=df_arr, dates=df_dates, arr_dates_oe=arr_dates),
    number=N
)
time_new = timeit.timeit(
    lambda: S_SMEV.get_ordinary_events_values_new(data=df_arr, dates=df_dates, arr_dates_oe=arr_dates),
    number=N
)
time_numba = timeit.timeit(
    lambda: S_SMEV.get_ordinary_events_values_new_numba(data=df_arr, dates=df_dates, arr_dates_oe=arr_dates),
    number=N
)

# --- Get outputs for comparison ---
dict_old,   dict_AMS_old   = S_SMEV.get_ordinary_events_values(data=df_arr, dates=df_dates, arr_dates_oe=arr_dates)
dict_new,   dict_AMS_new   = S_SMEV.get_ordinary_events_values_new(data=df_arr, dates=df_dates, arr_dates_oe=arr_dates)
dict_numba, dict_AMS_numba = S_SMEV.get_ordinary_events_values_new_numba(data=df_arr, dates=df_dates, arr_dates_oe=arr_dates)

# --- Exact diff check ---
print(f"\n--- get_ordinary_events_values: output check ---")
for dur in [str(d) for d in S_SMEV.durations]:
    oe_old   = dict_old[dur]['ordinary'].to_numpy()
    oe_new   = dict_new[dur]['ordinary'].to_numpy()
    oe_numba = dict_numba[dur]['ordinary'].to_numpy()
    time_old_arr   = dict_old[dur]['oe_time'].to_numpy()
    time_new_arr   = dict_new[dur]['oe_time'].to_numpy()
    time_numba_arr = dict_numba[dur]['oe_time'].to_numpy()

    new_val_diff   = np.max(np.abs(oe_old - oe_new))
    numba_val_diff = np.max(np.abs(oe_old - oe_numba))
    new_time_match   = np.array_equal(time_old_arr, time_new_arr)
    numba_time_match = np.array_equal(time_old_arr, time_numba_arr)
    numba_time_mismatch_count = np.sum(time_old_arr != time_numba_arr)

    print(f"  {dur:>4} min — "
          f"new: val diff={new_val_diff:.2e} time match={new_time_match} | "
          f"numba: val diff={numba_val_diff:.2e} time match={numba_time_match} (mismatches={numba_time_mismatch_count}/{len(time_old_arr)})")

# --- Debug: find first mismatching event for 60min ---
dur_dbg = "60"
time_new_dbg   = dict_new[dur_dbg]['oe_time'].to_numpy()
time_numba_dbg = dict_numba[dur_dbg]['oe_time'].to_numpy()
mismatch_idx = np.where(time_new_dbg != time_numba_dbg)[0]

if len(mismatch_idx) > 0:
    ev_i = mismatch_idx[0]
    oe_end_arr   = arr_dates[:, 0].astype("datetime64[ns]")
    oe_start_arr = arr_dates[:, 1].astype("datetime64[ns]")
    time_index = df_dates.reshape(-1)
    si = int(np.searchsorted(time_index, oe_start_arr[ev_i]))
    ei = int(np.searchsorted(time_index, oe_end_arr[ev_i]))
    window_size = int(60 / S_SMEV.time_resolution)
    ones_int   = np.ones(window_size, dtype=np.int64)
    ones_float = np.ones(window_size, dtype=np.float64)

    event_slice = df_arr[si:ei + 1].astype(np.float64)
    conv_int   = np.convolve(event_slice, ones_int,   'same')
    conv_float = np.convolve(event_slice, ones_float, 'same')

    half_left  = window_size // 2
    half_right = (window_size - 1) // 2
    n = len(event_slice)
    conv_manual = np.zeros(n)
    for j in range(n):
        start_k = max(0, j - half_left)
        end_k   = min(n, j + half_right + 1)
        s = 0.0
        for k in range(start_k, end_k):
            s += event_slice[k]
        conv_manual[j] = s

    idx_int    = int(np.argmax(conv_int))
    idx_float  = int(np.argmax(conv_float))
    idx_manual = int(np.argmax(conv_manual))

    print(f"\n--- Debug mismatch event {ev_i} (60min, slice len={n}) ---")
    print(f"np.convolve int64 kernel  argmax: {idx_int}  val={conv_int[idx_int]:.15f}")
    print(f"np.convolve float64 kernel argmax: {idx_float}  val={conv_float[idx_float]:.15f}")
    print(f"manual loop argmax:               {idx_manual}  val={conv_manual[idx_manual]:.15f}")
    print(f"--- values at both indices ---")
    for idx in sorted(set([idx_int, idx_float, idx_manual])):
        diff_int_manual = conv_int[idx] - conv_manual[idx]
        print(f"  idx {idx}: int={conv_int[idx]:.15f}  float={conv_float[idx]:.15f}  manual={conv_manual[idx]:.15f}  (int-manual={diff_int_manual:+.3e})")
    print(f"event slice values: {event_slice}")
    print(f"--- checking if this is truly a flat/tied event ---")
    print(f"  max conv value: {np.max(conv_int):.15f}")
    print(f"  2nd max:        {np.sort(conv_int)[-2]:.15f}")
    print(f"  difference:     {np.max(conv_int) - np.sort(conv_int)[-2]:.3e}")

# --- First 10 events for 1440min (MATLAB comparison) ---
print(f"\n--- First 10 ordinary events (1440 min) ---")
df_1440 = dict_old['1440']
print(df_1440[['oe_time', 'ordinary']].head(10).to_string(index=False))

# --- Timing summary ---
print(f"\n{'='*60}")
print(f"{'TIMING SUMMARY — get_ordinary_events_values':^60}")
print(f"{'='*60}")
print(f"{'Version':<20} {'Mean time (ms)':>15} {'Speedup vs old':>15}")
print(f"{'-'*60}")
print(f"{'old':<20} {time_old/N*1000:>15.3f} {'—':>15}")
print(f"{'new':<20} {time_new/N*1000:>15.3f} {time_old/time_new:>14.1f}x")
print(f"{'numba':<20} {time_numba/N*1000:>15.3f} {time_old/time_numba:>14.1f}x")
print(f"{'='*60}")
