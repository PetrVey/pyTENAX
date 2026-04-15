from importlib.resources import files
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from pyTENAX import smev

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

# --- Load data ---
file_path_input = files('pyTENAX.res').joinpath('prec_data_Aadorf.parquet')
data = pd.read_parquet(file_path_input)
data["prec_time"] = pd.to_datetime(data["prec_time"])
data.set_index("prec_time", inplace=True)
name_col = "prec_values"

t0 = time.perf_counter()
data = S_SMEV.remove_incomplete_years(data, name_col)
print(f"remove_incomplete_years:     {(time.perf_counter()-t0)*1000:.1f} ms")

df_arr = np.array(data[name_col])
df_dates = np.array(data.index)

# --- Storm separation ---
t0 = time.perf_counter()
idx_ordinary = S_SMEV.get_ordinary_events(data=df_arr, dates=df_dates, check_gaps=False)
print(f"get_ordinary_events:         {(time.perf_counter()-t0)*1000:.1f} ms  →  {len(idx_ordinary)} events")

t0 = time.perf_counter()
arr_vals, arr_dates, n_ordinary_per_year = S_SMEV.remove_short(idx_ordinary)
print(f"remove_short:                {(time.perf_counter()-t0)*1000:.1f} ms  →  {len(arr_vals)} events kept")

n = (n_ordinary_per_year.sum() / len(n_ordinary_per_year)).item()

# --- Extract values per duration ---
t0 = time.perf_counter()
dict_ordinary, dict_AMS = S_SMEV.get_ordinary_events_values(
    data=df_arr, dates=df_dates, arr_dates_oe=arr_dates
)
print(f"get_ordinary_events_values:  {(time.perf_counter()-t0)*1000:.1f} ms")

# --- SMEV for all durations ---
t0 = time.perf_counter()
df_results = S_SMEV._run_smev_all_durations(dict_ordinary, n)
print(f"_run_smev_all_durations:     {(time.perf_counter()-t0)*1000:.1f} ms")

# --- Pretty print summary table ---
print(f"\n{'='*85}")
print(f"{'SMEV RESULTS — Aadorf':^85}")
print(f"{'='*85}")
print(df_results.to_string())
print(f"{'='*85}")

# --- Bootstrap uncertainty for all durations ---
t0 = time.perf_counter()
boot_results = {}
for dur in [str(d) for d in S_SMEV.durations]:
    P        = dict_ordinary[dur]["ordinary"].to_numpy()
    blocks   = dict_ordinary[dur]["year"].to_numpy()
    shape, scale = S_SMEV.estimate_smev_parameters(P, S_SMEV.left_censoring)
    RL       = S_SMEV.smev_return_values(S_SMEV.return_period, shape, scale, n)
    RL_unc   = S_SMEV.SMEV_bootstrap_uncertainty(P, blocks, 1000, n)
    boot_results[dur] = {"shape": shape, "scale": scale, "RL": RL, "RL_unc": RL_unc}
print(f"Bootstrap uncertainty (all durations, 1000 iter): {(time.perf_counter()-t0)*1000:.1f} ms")

# --- All-durations plot: 2 rows x 3 columns ---
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for ax, dur in zip(axes, [str(d) for d in S_SMEV.durations]):
    AMS    = dict_AMS[dur]
    shape  = boot_results[dur]["shape"]
    scale  = boot_results[dur]["scale"]
    RL     = boot_results[dur]["RL"]
    RL_unc = boot_results[dur]["RL_unc"]

    RL_up  = np.quantile(RL_unc, 0.95, axis=0)
    RL_low = np.quantile(RL_unc, 0.05, axis=0)

    AMS_sort = AMS.sort_values(by=["AMS"])["AMS"]
    plot_pos = np.arange(1, len(AMS_sort) + 1) / (1 + len(AMS_sort))
    eRP = 1 / (1 - plot_pos)

    ax.fill_between(S_SMEV.return_period, RL_low, RL_up, color="r", alpha=0.2, label="90% CI")
    ax.plot(eRP, AMS_sort, "g+", label="Observed AMS")
    ax.plot(S_SMEV.return_period, RL, "--r", linewidth=2, label="SMEV")
    ax.set_xscale("log")
    ax.set_xlim(1, 200)
    ax.set_xticks(S_SMEV.return_period)
    ax.set_xticklabels(S_SMEV.return_period)
    ax.set_xlabel("Return period (years)")
    ax.set_ylabel("Depth (mm)")
    ax.set_title(f"{dur} min  |  shape={shape:.3f}  scale={scale:.3f}")
    ax.legend(fontsize=8)

plt.suptitle("SMEV fit — Aadorf, all durations", fontsize=13)
plt.tight_layout()
plt.show()
