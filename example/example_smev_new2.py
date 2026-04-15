from importlib.resources import files
import numpy as np
import pandas as pd
import time
from pyTENAX import smev

try:
    import psutil
    import os
    _HAS_PSUTIL = True
    _proc = psutil.Process(os.getpid())
except ImportError:
    _HAS_PSUTIL = False
    print("psutil not installed — per-core CPU stats unavailable. Run: pip install psutil")

# --- Setup ---
S_SMEV = smev.SMEV(
    return_period=[2, 5, 10, 20, 50, 100, 200],
    durations=[10, 60, 180, 360, 720, 1440],
    time_resolution=10,
    min_rain=0,
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

# --- Storm separation ---
idx_ordinary = S_SMEV.get_ordinary_events_new(data=df_arr, dates=df_dates, check_gaps=False)
arr_vals, arr_dates, n_per_year = S_SMEV.remove_short_new(idx_ordinary)
print(f"Storm separation done: {len(arr_vals)} events")

# --- Warmup numba (first call triggers JIT compilation) ---
print("Warming up numba JIT (first call compiles)...")
_ = S_SMEV.get_ordinary_events_values_v2(data=df_arr, dates=df_dates, arr_dates_oe=arr_dates, method="njit")
_ = S_SMEV.get_ordinary_events_values_v2(data=df_arr, dates=df_dates, arr_dates_oe=arr_dates, method="njit_parallel")
print("Warmup done.")

N = 100

def bench(fn, n):
    """Run fn n times; return (wall_s, cpu_s, cpu_pct)."""
    if _HAS_PSUTIL:
        _proc.cpu_percent()  # discard first dummy reading
    t_wall0 = time.perf_counter()
    t_cpu0  = time.process_time()
    for _ in range(n):
        fn()
    t_cpu1  = time.process_time()
    t_wall1 = time.perf_counter()
    cpu_pct = _proc.cpu_percent(interval=None) if _HAS_PSUTIL else float("nan")
    return t_wall1 - t_wall0, t_cpu1 - t_cpu0, cpu_pct

# --- Benchmark ---
wall_old,      cpu_old,      pct_old      = bench(lambda: S_SMEV.get_ordinary_events_values(data=df_arr, dates=df_dates, arr_dates_oe=arr_dates), N)
wall_vec,      cpu_vec,      pct_vec      = bench(lambda: S_SMEV.get_ordinary_events_values_v2(data=df_arr, dates=df_dates, arr_dates_oe=arr_dates, method="vectorized"), N)
wall_njit,     cpu_njit,     pct_njit     = bench(lambda: S_SMEV.get_ordinary_events_values_v2(data=df_arr, dates=df_dates, arr_dates_oe=arr_dates, method="njit"), N)
wall_njit_par, cpu_njit_par, pct_njit_par = bench(lambda: S_SMEV.get_ordinary_events_values_v2(data=df_arr, dates=df_dates, arr_dates_oe=arr_dates, method="njit_parallel"), N)

# --- Get outputs for comparison ---
dict_old     , _ = S_SMEV.get_ordinary_events_values(data=df_arr, dates=df_dates, arr_dates_oe=arr_dates)
dict_vec     , _ = S_SMEV.get_ordinary_events_values_v2(data=df_arr, dates=df_dates, arr_dates_oe=arr_dates, method="vectorized")
dict_njit    , _ = S_SMEV.get_ordinary_events_values_v2(data=df_arr, dates=df_dates, arr_dates_oe=arr_dates, method="njit")
dict_njit_par, _ = S_SMEV.get_ordinary_events_values_v2(data=df_arr, dates=df_dates, arr_dates_oe=arr_dates, method="njit_parallel")

# --- Output check vs old ---
print(f"\n--- Output check vs get_ordinary_events_values (old) ---")
for dur in [str(d) for d in S_SMEV.durations]:
    oe_old = dict_old[dur]['ordinary'].to_numpy()
    t_old  = dict_old[dur]['oe_time'].to_numpy()
    for label, d in [("vectorized", dict_vec), ("njit", dict_njit), ("njit_par", dict_njit_par)]:
        val_diff   = np.max(np.abs(oe_old - d[dur]['ordinary'].to_numpy()))
        time_match = np.array_equal(t_old, d[dur]['oe_time'].to_numpy())
        print(f"  {dur:>4} min  {label:<12} val diff={val_diff:.2e}  time match={time_match}")

# --- First 10 events for 1440min (MATLAB comparison) ---
print(f"\n--- First 10 ordinary events (1440 min) ---")
print(dict_old['1440'][['oe_time', 'ordinary']].head(10).to_string(index=False))

print(f"\n--- First 10 ordinary events (10 min) ---")
print(dict_old['10'][['oe_time', 'ordinary']].head(10).to_string(index=False))

# --- Timing summary ---
print(f"\n{'='*72}")
print(f"{'TIMING SUMMARY — get_ordinary_events_values':^72}")
print(f"{'='*72}")
print(f"{'Version':<16} {'Wall ms':>10} {'CPU ms':>10} {'CPU/Wall':>10} {'CPU%':>8} {'Speedup':>10}")
print(f"{'-'*72}")
for label, wall, cpu, pct in [
    ("old",           wall_old,      cpu_old,      pct_old),
    ("v2 vectorized", wall_vec,      cpu_vec,      pct_vec),
    ("v2 njit",       wall_njit,     cpu_njit,     pct_njit),
    ("v2 njit_par",   wall_njit_par, cpu_njit_par, pct_njit_par),
]:
    speedup = f"{wall_old/wall:.1f}x" if label != "old" else "—"
    cpu_wall_ratio = cpu / wall if wall > 0 else float("nan")
    print(f"{label:<16} {wall/N*1000:>10.3f} {cpu/N*1000:>10.3f} {cpu_wall_ratio:>10.2f} {pct:>7.1f}% {speedup:>10}")
print(f"{'='*72}")
print("CPU/Wall > 1.0 means multiple threads/cores used (njit_parallel).")
