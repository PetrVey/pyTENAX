from importlib.resources import files
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from pyTENAX import smev, tenax, plotting

# Initiate TENAX class with customized setup
S = tenax.TENAX(
    return_period=[2, 5, 10, 20, 50, 100, 200],
    durations=[10, 60, 180, 360, 720, 1440],
    time_resolution=10,
    left_censoring=[0, 0.90],
    alpha=0.05,
)

timings = {}

# Load precipitation data
t0 = time.perf_counter()
file_path_input = files('pyTENAX.res').joinpath('prec_data_Aadorf.parquet')
data = pd.read_parquet(file_path_input)
data["prec_time"] = pd.to_datetime(data["prec_time"])
data.set_index("prec_time", inplace=True)
name_col = "prec_values"
data.loc[data[name_col] < S.min_rain, name_col] = 0
data = S.remove_incomplete_years(data, name_col)
timings["load_precipitation"] = time.perf_counter() - t0

df_arr = np.array(data[name_col])
df_dates = np.array(data.index)

t0 = time.perf_counter()
idx_ordinary = S.get_ordinary_events(
    data=df_arr, dates=df_dates, name_col=name_col, check_gaps=False
)
timings["get_ordinary_events"] = time.perf_counter() - t0

t0 = time.perf_counter()
arr_vals, arr_dates, n_ordinary_per_year = S.remove_short(idx_ordinary)
timings["remove_short"] = time.perf_counter() - t0

t0 = time.perf_counter()
dict_ordinary, dict_AMS = S.get_ordinary_events_values(
    data=df_arr, dates=df_dates, arr_dates_oe=arr_dates
)
timings["get_ordinary_events_values"] = time.perf_counter() - t0

# Load temperature data
t0 = time.perf_counter()
file_path_temperature = files('pyTENAX.res').joinpath('temp_data_Aadorf.parquet')
t_data = pd.read_parquet(file_path_temperature)
t_data["temp_time"] = pd.to_datetime(t_data["temp_time"])
t_data.set_index("temp_time", inplace=True)
temp_name_col = "temp_values"
df_arr_t_data = np.array(t_data[temp_name_col])
df_dates_t_data = np.array(t_data.index)
timings["load_temperature"] = time.perf_counter() - t0

t0 = time.perf_counter()
dict_ordinary, _, n_ordinary_per_year = S.associate_vars(
    dict_ordinary, df_arr_t_data, df_dates_t_data
)
timings["associate_vars"] = time.perf_counter() - t0

# magnitude_model optimizer comparison
methods = ['Nelder-Mead', "L-BFGS-B"]
ref_phat = None

for mm in methods:
    print(f"\n--- magnitude_model  method={mm} ---")
    print(f"  {'dur':>5}  {'t':>8}  {'max_dphat':>10}  phat")
    t_total = 0.0
    for dur in S.durations:
        key = str(dur)
        P_d = dict_ordinary[key]["ordinary"].to_numpy()
        T_d = dict_ordinary[key]["T"].to_numpy()
        thr_d = dict_ordinary[key]["ordinary"].quantile(
            S.left_censoring[1]
        )
        try:
            t0 = time.perf_counter()
            F_phat_d, loglik_d, _, _ = S.magnitude_model(
                P_d, T_d, thr_d, minimize_method=mm
            )
            t_dur = time.perf_counter() - t0
        except Exception as e:
            print(f"  {dur:>5}  FAILED: {e}")
            continue
        t_total += t_dur

        if ref_phat is None:
            ref_phat = F_phat_d.copy()
            diff = 0.0
        else:
            diff = np.max(np.abs(F_phat_d - ref_phat))

        print(f"  {dur:>5}  {t_dur:>8.3f}  {diff:>10.2e}"
              f"  {np.round(F_phat_d, 6)}")

        if mm == 'Nelder-Mead' and dur == 10:
            P, T = P_d, T_d
            thr = thr_d
            F_phat, loglik = F_phat_d, loglik_d
            timings["magnitude_model"] = t_dur
            ref_phat = F_phat_d.copy()

    print(f"  {'TOTAL':>5}  {t_total:>8.3f}")

blocks_id = dict_ordinary["10"]["year"].to_numpy()
Ts = np.arange(
    np.min(T) - S.temp_delta, np.max(T) + S.temp_delta, S.temp_res_monte_carlo
)

t0 = time.perf_counter()
g_phat = S.temperature_model(T)
timings["temperature_model"] = time.perf_counter() - t0

t0 = time.perf_counter()
n = n_ordinary_per_year.sum() / len(n_ordinary_per_year)
RL, _, _ = S.model_inversion(F_phat, g_phat, n, Ts)
timings["model_inversion"] = time.perf_counter() - t0

# TENAX uncertainty
S.n_monte_carlo = 20000
t0 = time.perf_counter()
F_phat_unc, g_phat_unc, RL_unc, n_unc, n_err = S.TNX_tenax_bootstrap_uncertainty(
    P, T, blocks_id, Ts, "norm", "brentq", "L-BFGS-B"
)
timings["tenax_bootstrap_uncertainty"] = time.perf_counter() - t0

# SMEV and its uncertainty
S_SMEV = smev.SMEV(
    return_period=S.return_period,
    durations=S.durations,
    time_resolution=S.time_resolution,
    left_censoring=[S.left_censoring[1], 1],
)

t0 = time.perf_counter()
smev_shape, smev_scale = S_SMEV.estimate_smev_parameters(P, S_SMEV.left_censoring)
smev_RL = S_SMEV.smev_return_values(
    S_SMEV.return_period, smev_shape, smev_scale, n.item()
)
timings["smev_fit"] = time.perf_counter() - t0

t0 = time.perf_counter()
smev_RL_unc = S_SMEV.smev_bootstrap_uncertainty(P, blocks_id, S.niter_smev, n.item())
timings["smev_bootstrap_uncertainty"] = time.perf_counter() - t0

# Pretty-print timings
print("\n--- TENAX step timings ---")
total = sum(timings.values())
for step, elapsed in timings.items():
    print(f"  {step:<35} {elapsed:6.3f} s")
print(f"  {'TOTAL':<35} {total:6.3f} s")

# fig 4
AMS = dict_AMS["10"]
plotting.TNX_FIG_valid(AMS, S.return_period, RL, smev_RL, RL_unc, smev_RL_unc)
plt.title("fig 4")
plt.ylabel("10-minute precipitation (mm)")
plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2))
plt.show()
