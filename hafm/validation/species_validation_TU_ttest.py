import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.interpolate import PchipInterpolator

# ============================================================
# User settings
# ============================================================

window = 12.0
t_start_exp = 20.0  # heating onset in experimental time record

# Matched radial locations
r_mm = np.array([0, 1, 2, 4], dtype=float)
r_m = r_mm * 1e-3
keys = ["R0", "R1", "R2", "R4"]
rs = "r0124"

outdir = "figures/species_transport/TU_ttest"
csv_outdir = "figures/species_transport/TU_ttest"
os.makedirs(outdir, exist_ok=True)
os.makedirs(csv_outdir, exist_ok=True)

# ============================================================
# Uncertainty settings
# ============================================================

u_piv = 0.1348   # m/s
u_therm = 0.4 #0.57  #0.55   # K
u_D_sim_num = 0.0  # K*m^3, optional absolute numerical uncertainty on simulation dose

confidence = 0.95
alpha = 0.05

# Use the independent realizations, not the cross-combination count.
n_exp_independent = 3
n_sim_independent = 10

# Fine radial grid for smooth integration
r_fine_m = np.linspace(r_m.min(), r_m.max(), 300)

# ============================================================
# File paths
# ============================================================

temp_exp_file = "data/data_exp/Raw_Temp_Data/Air_Temperature_Measurements_1W_0.5LPM.csv"
sim_temp_files = [f"data/data_sim/outputSample{i}.csv" for i in range(1, 11)]
exp_vel_files = [
    "data/velocities/S1_Low_Exp_Outlet.csv",
    "data/velocities/S2_Low_Exp_Outlet.csv",
    "data/velocities/S3_Low_Exp_Outlet.csv",
]
sim_vel_files = [
    "data/velocities/S1_Model.csv",
    "data/velocities/S2_Model.csv",
    "data/velocities/S3_Model.csv",
]

# ============================================================
# Column mappings
# ============================================================

exp_cols = {
    "R0": ["T1T2", "T2T2", "T3T2"],
    "R1": ["T1T3", "T2T3", "T3T3"],
    "R2": ["T1T1", "T2T1", "T3T1"],
    "R4": ["T1T4", "T2T4", "T3T4"],
}

sim_cols = {
    "R0": "Monitor Point: Mouthpiece1mm0mmTempX0YZ (Temperature) [K]",
    "R1": "Monitor Point: Mouthpiece1mm1mmTempX0YZ (Temperature) [K]",
    "R2": "Monitor Point: Mouthpiece1mm2mmTempX0YZ (Temperature) [K]",
    "R4": "Monitor Point: Mouthpiece1mm4mmTempX0YZ (Temperature) [K]",
}

# ============================================================
# Functions
# ============================================================

def crop_and_shift_time(df, time_col="time", start=0.0, duration=12.0):
    tmp = df[(df[time_col] >= start) & (df[time_col] <= start + duration)].copy()
    tmp["time_shift"] = tmp[time_col] - start
    return tmp


def read_velocity_profile(filepath, source="exp"):
    df = pd.read_csv(filepath)

    if source == "exp":
        x_mm = df["x (mm)"].to_numpy(dtype=float)
        u = df["V (m/s)"].to_numpy(dtype=float)
    elif source == "sim":
        x_mm = df[" Y [ m ]"].to_numpy(dtype=float) * 1e3
        u = df[" Velocity [ m s^-1 ]"].to_numpy(dtype=float)
    else:
        raise ValueError("source must be 'exp' or 'sim'")

    center_mm = 0.5 * (np.nanmin(x_mm) + np.nanmax(x_mm))
    r_vel_mm = np.abs(x_mm - center_mm)
    order = np.argsort(r_vel_mm)
    return r_vel_mm[order] * 1e-3, u[order]


def interpolate_velocity_to_temperature_radii(r_vel_m, u_vel, r_target_m):
    vel_df = pd.DataFrame({"r_m": r_vel_m, "u": u_vel})
    vel_df["r_round"] = vel_df["r_m"].round(7)
    vel_avg = vel_df.groupby("r_round", as_index=False)["u"].mean().sort_values("r_round")

    r_unique = vel_avg["r_round"].to_numpy()
    u_unique = vel_avg["u"].to_numpy()

    return np.interp(r_target_m, r_unique, u_unique, left=u_unique[0], right=0.0)


def profile_pchip(r_data, y_data, r_eval):
    r_data = np.asarray(r_data, dtype=float)
    y_data = np.asarray(y_data, dtype=float)
    order = np.argsort(r_data)
    return PchipInterpolator(r_data[order], y_data[order], extrapolate=False)(r_eval)


def compute_temperature_weighted_flux_fit(time, T_df, u_profile, u_v=None, u_T=None):
    """
    F_T(t) = 2*pi*int DeltaT(r,t)*u(r)*r dr
    D_T    = int F_T(t) dt

    If u_v and u_T are provided, returns the first-order propagated
    measurement-uncertainty envelope for F_T(t) and D_T.
    """
    time = np.asarray(time, dtype=float)
    u_fit = profile_pchip(r_m, u_profile, r_fine_m)

    flux_t = []
    u_flux_t = []

    for _, row in T_df.iterrows():
        dT = row[keys].to_numpy(dtype=float)
        T_fit = profile_pchip(r_m, dT, r_fine_m)

        F_t = 2 * np.pi * np.trapezoid(T_fit * u_fit * r_fine_m, r_fine_m)
        flux_t.append(F_t)

        if u_v is not None and u_T is not None:
            u_integrand = (np.abs(T_fit) * u_v + np.abs(u_fit) * u_T) * r_fine_m
            u_F_t = 2 * np.pi * np.trapezoid(u_integrand, r_fine_m)
            u_flux_t.append(u_F_t)

    flux_t = np.asarray(flux_t)
    dose = np.trapezoid(flux_t, time)

    if u_v is not None and u_T is not None:
        u_flux_t = np.asarray(u_flux_t)
        u_dose = np.trapezoid(u_flux_t, time)
        return flux_t, dose, u_flux_t, u_dose

    return flux_t, dose


def summarize_values(values):
    values = np.asarray(values, dtype=float)
    return {
        "n": len(values),
        "mean": np.mean(values),
        "std": np.std(values, ddof=1) if len(values) > 1 else np.nan,
        "min": np.min(values),
        "median": np.median(values),
        "max": np.max(values),
    }


def mean_ci_from_uncertainty(mean, U):
    return mean - U, mean + U

# ============================================================
# Load experimental temperature samples
# ============================================================

tempE = pd.read_csv(temp_exp_file)

exp_T_samples = []
for i in range(3):
    tmp = pd.DataFrame()
    tmp["time"] = tempE["Time"]

    for key, cols in exp_cols.items():
        tmp[key] = tempE[cols[i]]

    baseline_window = tmp["time"] <= 5.0
    for key in keys:
        tmp[key] = tmp[key] - tmp.loc[baseline_window, key].mean()

    exp_T_samples.append(crop_and_shift_time(tmp, time_col="time", start=t_start_exp, duration=window))

# ============================================================
# Load simulation temperature samples
# ============================================================

sim_T_samples = []
for f in sim_temp_files:
    df = pd.read_csv(f, skiprows=4)

    tmp = pd.DataFrame()
    tmp["time"] = df["Time [s]"]

    for key, col in sim_cols.items():
        tmp[key] = df[col]

    for key in keys:
        tmp[key] = tmp[key] - tmp[key].iloc[0]

    sim_T_samples.append(crop_and_shift_time(tmp, time_col="time", start=0.0, duration=window))

# ============================================================
# Load velocity profiles
# ============================================================

u_exp_samples = []
for f in exp_vel_files:
    r_vel, u_vel = read_velocity_profile(f, source="exp")
    u_exp_samples.append(interpolate_velocity_to_temperature_radii(r_vel, u_vel, r_m))

u_sim_samples = []
for f in sim_vel_files:
    r_vel, u_vel = read_velocity_profile(f, source="sim")
    u_sim_samples.append(interpolate_velocity_to_temperature_radii(r_vel, u_vel, r_m))

u_exp_mean = np.mean(u_exp_samples, axis=0)
u_sim_mean = np.mean(u_sim_samples, axis=0)

velocity_summary = pd.DataFrame({
    "Radius_mm": r_mm,
    "Key": keys,
    "U_exp_mean_m_per_s": u_exp_mean,
    "U_sim_mean_m_per_s": u_sim_mean,
})
velocity_summary.T.to_csv(f"{csv_outdir}/velocity_at_temperature_radii_{rs}.csv") #, index=False)

# ============================================================
# Mean temperature profiles
# ============================================================

exp_T_mean = exp_T_samples[0][["time_shift"] + keys].copy()
for key in keys:
    exp_T_mean[key] = np.mean([tmp[key].to_numpy() for tmp in exp_T_samples], axis=0)

sim_T_mean = sim_T_samples[0][["time_shift"] + keys].copy()
for key in keys:
    sim_T_mean[key] = np.mean([tmp[key].to_numpy() for tmp in sim_T_samples], axis=0)

# ============================================================
# Mean flux histories and dose
# ============================================================

F_exp_mean, D_exp_mean_fit, uF_exp_mean, uD_exp_mean_meas = compute_temperature_weighted_flux_fit(
    exp_T_mean["time_shift"].to_numpy(),
    exp_T_mean,
    u_exp_mean,
    u_v=u_piv,
    u_T=u_therm,
)

F_sim_mean, D_sim_mean_fit = compute_temperature_weighted_flux_fit(
    sim_T_mean["time_shift"].to_numpy(),
    sim_T_mean,
    u_sim_mean,
)

percent_difference_fit = 100 * (D_sim_mean_fit - D_exp_mean_fit) / D_exp_mean_fit

pd.DataFrame({
    "time_s": exp_T_mean["time_shift"].to_numpy(),
    "F_exp_mean_K_m3_per_s": F_exp_mean,
    "u_F_exp_mean_meas_K_m3_per_s": uF_exp_mean,
    "F_exp_low_K_m3_per_s": F_exp_mean - uF_exp_mean,
    "F_exp_high_K_m3_per_s": F_exp_mean + uF_exp_mean,
}).to_csv(f"{csv_outdir}/mean_experimental_flux_history_fit_{rs}.csv", index=False)

pd.DataFrame({
    "time_s": sim_T_mean["time_shift"].to_numpy(),
    "F_sim_mean_K_m3_per_s": F_sim_mean,
}).to_csv(f"{csv_outdir}/mean_simulation_flux_history_fit_{rs}.csv", index=False)

# ============================================================
# Dose spread from internal realizations
# ============================================================
# These internal estimates are used only to estimate replicate/UQ spread.
# The final uncertainty-aware statistical comparison uses independent counts:
# n_exp = 3 and n_sim = 10.

exp_records = []
for iT, T_exp in enumerate(exp_T_samples, start=1):
    for iU, u_exp in enumerate(u_exp_samples, start=1):
        _, D, _, uD_meas = compute_temperature_weighted_flux_fit(
            T_exp["time_shift"].to_numpy(),
            T_exp,
            u_exp,
            u_v=u_piv,
            u_T=u_therm,
        )
        exp_records.append({
            "Source": "Experiment",
            "Temperature_sample": f"T{iT}",
            "Velocity_sample": f"U{iU}",
            "Dose_K_m3": D,
            "u_D_meas_K_m3": uD_meas,
        })

sim_records = []
for iT, T_sim in enumerate(sim_T_samples, start=1):
    for iU, u_sim in enumerate(u_sim_samples, start=1):
        _, D = compute_temperature_weighted_flux_fit(
            T_sim["time_shift"].to_numpy(),
            T_sim,
            u_sim,
        )
        sim_records.append({
            "Source": "Simulation",
            "Temperature_sample": f"T{iT}",
            "Velocity_sample": f"U{iU}",
            "Dose_K_m3": D,
            "u_D_meas_K_m3": np.nan,
        })

dose_df = pd.DataFrame(exp_records + sim_records)
D_exp_all = dose_df.loc[dose_df["Source"] == "Experiment", "Dose_K_m3"].to_numpy()
D_sim_all = dose_df.loc[dose_df["Source"] == "Simulation", "Dose_K_m3"].to_numpy()

# Replicate/UQ spread from internal dose estimates
u_D_exp_rep = np.std(D_exp_all, ddof=1)
u_D_sim_rep = np.std(D_sim_all, ddof=1)

# Measurement and numerical uncertainty contributions
u_D_exp_meas = dose_df.loc[dose_df["Source"] == "Experiment", "u_D_meas_K_m3"].mean()
u_D_exp_total = np.sqrt(u_D_exp_rep**2 + u_D_exp_meas**2)
u_D_sim_total = np.sqrt(u_D_sim_rep**2 + u_D_sim_num**2)

D_exp_low, D_exp_high = mean_ci_from_uncertainty(D_exp_mean_fit, u_D_exp_total)
D_sim_low, D_sim_high = mean_ci_from_uncertainty(D_sim_mean_fit, u_D_sim_total)

Delta_D = D_sim_mean_fit - D_exp_mean_fit
u_Delta_D = np.sqrt(u_D_exp_total**2 + u_D_sim_total**2)

# ============================================================
# Uncertainty-aware t-test style comparison
# ============================================================

se_delta = u_Delta_D
t_unc = Delta_D / se_delta if se_delta > 0 else np.nan

# Conservative degrees of freedom based on independent realization counts.
df_unc = min(n_exp_independent - 1, n_sim_independent - 1)
p_unc = 2 * stats.t.sf(np.abs(t_unc), df=df_unc) if np.isfinite(t_unc) else np.nan
intervals_overlap = D_exp_high >= D_sim_low

uncertainty_ttest_summary = pd.DataFrame([{
    "test": "Uncertainty-aware dose comparison",
    "quantity_tested": "temperature-weighted dose with propagated uncertainty",
    "alpha": alpha,
    "n_exp_independent": n_exp_independent,
    "n_sim_independent": n_sim_independent,
    "D_exp_mean_K_m3": D_exp_mean_fit,
    "D_exp_propagated_uncertainty_K_m3": u_D_exp_total,
    "D_exp_low_K_m3": D_exp_low,
    "D_exp_high_K_m3": D_exp_high,
    "D_sim_mean_K_m3": D_sim_mean_fit,
    "D_sim_propagated_uncertainty_K_m3": u_D_sim_total,
    "D_sim_low_K_m3": D_sim_low,
    "D_sim_high_K_m3": D_sim_high,
    "percent_difference": percent_difference_fit,
    "difference_D_sim_minus_D_exp_K_m3": Delta_D,
    "standard_uncertainty_of_difference_K_m3": u_Delta_D,
    "t_statistic_uncertainty_aware": t_unc,
    "degrees_of_freedom_conservative": df_unc,
    "p_value_uncertainty_aware": p_unc,
    "statistically_distinguishable_alpha_0_05": p_unc < alpha if np.isfinite(p_unc) else np.nan,
    "intervals_overlap": intervals_overlap,
    "interpretation": (
        "Statistically distinguishable at alpha=0.05"
        if np.isfinite(p_unc) and p_unc < alpha
        else "Not statistically distinguishable at alpha=0.05 when propagated dose uncertainty is included"
    ),
    "note": (
        "This comparison uses the propagated dose uncertainties shown in the dose interval plots. "
        "The independent realization counts are n_exp=3 and n_sim=10. "
        "The 9 experimental and 30 simulation internal combinations are not treated as independent samples."
    ),
}])
uncertainty_ttest_summary.T.to_csv(f"{csv_outdir}/dose_uncertainty_aware_ttest_summary_{rs}.csv") #,index=False)

uncertainty_summary = pd.DataFrame([{
    "D_exp_mean_fit_K_m3": D_exp_mean_fit,
    "D_sim_mean_fit_K_m3": D_sim_mean_fit,
    "percent_difference_fit": percent_difference_fit,
    "u_piv_m_per_s": u_piv,
    "u_therm_K": u_therm,
    "u_D_exp_rep_K_m3": u_D_exp_rep,
    "u_D_exp_meas_K_m3": u_D_exp_meas,
    "u_D_exp_total_K_m3": u_D_exp_total,
    "u_D_sim_rep_K_m3": u_D_sim_rep,
    "u_D_sim_num_K_m3": u_D_sim_num,
    "u_D_sim_total_K_m3": u_D_sim_total,
    "Delta_D_K_m3": Delta_D,
    "u_Delta_D_K_m3": u_Delta_D,
    "intervals_overlap": intervals_overlap,
    "p_value_uncertainty_aware": p_unc,
}])
uncertainty_summary.T.to_csv(f"{csv_outdir}/dose_fit_uncertainty_summary_{rs}.csv") #, index=False)

# ============================================================
# Plot 1: mean flux histories with experimental uncertainty envelope
# ============================================================

plt.figure(figsize=(7, 5))
plt.plot(
    exp_T_mean["time_shift"],
    F_exp_mean,
    "-",
    color="C0",
    linewidth=2.5,
    label="Experiment mean",
)
plt.fill_between(
    exp_T_mean["time_shift"],
    F_exp_mean - uF_exp_mean,
    F_exp_mean + uF_exp_mean,
    color="C0",
    alpha=0.2,
    label="Experiment measurement uncertainty",
)
plt.plot(
    sim_T_mean["time_shift"],
    F_sim_mean,
    "--",
    color="C1",
    linewidth=2.5,
    label="Model mean",
)
plt.xlabel("Time [s]")
plt.ylabel(r"Temperature-weighted flux, $F_T(t)$ [K m$^3$/s]")
plt.title(fr"Smooth-fit Mean Percent Difference: {percent_difference_fit:.2f}%")
plt.legend()
plt.tight_layout()
plt.savefig(f"{outdir}/mean_temperature_weighted_flux_fit_uncertainty_{rs}.png", dpi=300, bbox_inches="tight")
plt.close()

# ============================================================
# Plot 2: vertical dose confidence/uncertainty intervals
# ============================================================

plot_df = pd.DataFrame({
    "Source": ["Experiment", "Simulation"],
    "Mean": [D_exp_mean_fit, D_sim_mean_fit],
    "Uncertainty": [u_D_exp_total, u_D_sim_total],
})

x = np.arange(len(plot_df))
plt.figure(figsize=(6, 5))
plt.errorbar(
    x,
    plot_df["Mean"].to_numpy(),
    yerr=plot_df["Uncertainty"].to_numpy(),
    fmt="o",
    capsize=8,
    linewidth=2.5,
    markersize=8,
)
plt.xticks(x, plot_df["Source"])
plt.ylabel(r"Cumulative dose, $D_T$ [K m$^3$]")
plt.title("Smooth-Fit Dose With Propagated Uncertainty")
plt.tight_layout()
plt.savefig(f"{outdir}/dose_interval_comparison_fit_uncertainty_{rs}.png", dpi=300, bbox_inches="tight")
plt.close()

# ============================================================
# Plot 3: Hariharan-style dose interval plot
# ============================================================

fig, ax = plt.subplots(figsize=(7, 2.2))
y = 0.0
xmax = max(D_exp_high, D_sim_high) * 1.15

ax.hlines(y, 0, xmax, color="k", linewidth=2)
ax.annotate(
    "",
    xy=(xmax, y),
    xytext=(xmax * 0.97, y),
    arrowprops=dict(arrowstyle="->", color="k", lw=2),
)

ax.errorbar(
    D_exp_mean_fit,
    y,
    xerr=[[D_exp_mean_fit - D_exp_low], [D_exp_high - D_exp_mean_fit]],
    fmt="o",
    color="C0",
    ecolor="C0",
    elinewidth=3,
    capsize=10,
    markersize=8,
    label=r"Experiment, $D_e$",
)

ax.errorbar(
    D_sim_mean_fit,
    y,
    xerr=[[D_sim_mean_fit - D_sim_low], [D_sim_high - D_sim_mean_fit]],
    fmt="o",
    color="C1",
    ecolor="C1",
    elinewidth=3,
    capsize=10,
    markersize=8,
    label=r"Model, $D_c$",
)

ax.text(D_exp_mean_fit, y - 0.12, r"$D_e$", ha="center", va="top", fontsize=14)
ax.text(D_sim_mean_fit, y + 0.20, r"$D_c$", ha="center", va="bottom", fontsize=14)

ax.set_xlabel(r"Temperature-weighted dose, $D_T$ [K m$^3$]")
ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
ax.set_yticks([])
ax.set_ylim(-0.25, 0.35)
ax.spines[["left", "right", "top"]].set_visible(False)
ax.legend(loc="upper left", frameon=True)

plt.tight_layout()
plt.savefig(f"{outdir}/dose_interval_hariharan_style_fit_uncertainty_{rs}.png", dpi=300, bbox_inches="tight")
plt.close()

# ============================================================
# Console summary
# ============================================================

print("\nUncertainty-aware dose comparison:")
print(uncertainty_ttest_summary.T)

print("\nSaved CSV outputs:")
print(f"  {csv_outdir}/mean_experimental_flux_history_fit_{rs}.csv")
print(f"  {csv_outdir}/mean_simulation_flux_history_fit_{rs}.csv")
print(f"  {csv_outdir}/dose_fit_uncertainty_summary_{rs}.csv")
print(f"  {csv_outdir}/dose_uncertainty_aware_ttest_summary_{rs}.csv")

print("\nSaved figures:")
print(f"  {outdir}/mean_temperature_weighted_flux_fit_uncertainty_{rs}.png")
print(f"  {outdir}/dose_interval_comparison_fit_uncertainty_{rs}.png")
print(f"  {outdir}/dose_interval_hariharan_style_fit_uncertainty_{rs}.png")
