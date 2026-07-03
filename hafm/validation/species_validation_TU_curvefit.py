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
t_start_exp = 20.0  # adjust if needed based on heating onset

# Matched radial locations
r_mm = np.array([0, 1, 2, 4], dtype=float)
r_m = r_mm * 1e-3
keys = ["R0", "R1", "R2", "R4"]
rs = "r0124"  # what will be saved to the output files

outdir = "figures/species_transport/TU_final"
csv_outdir = "figures/species_transport/TU_final"

os.makedirs(outdir, exist_ok=True)
os.makedirs(csv_outdir, exist_ok=True)

# ============================================================
# Uncertainty settings
# ============================================================

# Measurement uncertainty values.
# Update these after confirming PIV and thermocouple-difference uncertainty.
u_piv = 0.1348      # m/s; replace if you decide to use another PIV value
u_therm = 0.4 #0.42      # K; placeholder within 0.2--0.6 K suggested range

# Optional numerical uncertainty on dose for simulation.
# If you have an absolute dose numerical uncertainty, enter it here.
u_D_sim_num = 0.0   # K*m^3

# Approximate significance level for uncertainty-overlap z-test
confidence = 0.95
zcrit = stats.norm.ppf((1 + confidence) / 2)

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
    "R0": ["T1T2", "T2T2", "T3T2"],  # center
    "R1": ["T1T3", "T2T3", "T3T3"],  # 1 mm
    "R2": ["T1T1", "T2T1", "T3T1"],  # 2 mm
    "R4": ["T1T4", "T2T4", "T3T4"],  # 4 mm
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
    r_vel_mm = r_vel_mm[order]
    u = u[order]

    return r_vel_mm * 1e-3, u


def interpolate_velocity_to_temperature_radii(r_vel_m, u_vel, r_target_m):
    """Average duplicate radial locations and interpolate velocity to TC radii."""
    vel_df = pd.DataFrame({"r_m": r_vel_m, "u": u_vel})
    vel_df["r_round"] = vel_df["r_m"].round(7)

    vel_avg = (
        vel_df.groupby("r_round", as_index=False)["u"]
        .mean()
        .sort_values("r_round")
    )

    r_unique = vel_avg["r_round"].to_numpy()
    u_unique = vel_avg["u"].to_numpy()

    u_target = np.interp(
        r_target_m,
        r_unique,
        u_unique,
        left=u_unique[0],
        right=0.0,
    )

    return u_target


def profile_pchip(r_data, y_data, r_eval):
    """
    Shape-preserving radial interpolation.

    This is safer than fitting a high-order polynomial through only 4 temperature
    points. It uses all 4 points and avoids oscillatory cubic-polynomial behavior.
    """
    r_data = np.asarray(r_data, dtype=float)
    y_data = np.asarray(y_data, dtype=float)
    order = np.argsort(r_data)
    r_sorted = r_data[order]
    y_sorted = y_data[order]
    return PchipInterpolator(r_sorted, y_sorted, extrapolate=False)(r_eval)


def compute_temperature_weighted_flux_fit(time, T_df, u_profile, u_v=None, u_T=None):
    """
    Smooth-fit dose calculation.

    F_T(t) = 2*pi*int DeltaT(r,t)*u(r)*r dr
    D_T    = int F_T(t) dt

    If u_v and u_T are provided, also computes first-order measurement
    uncertainty contribution:

    u_F(t) = 2*pi*int (|DeltaT(r,t)|*u_v + |u(r)|*u_T)*r dr
    u_D    = int u_F(t) dt

    Units:
    F_T: K*m^3/s
    D_T: K*m^3
    """
    time = np.asarray(time, dtype=float)
    u_fit = profile_pchip(r_m, u_profile, r_fine_m)

    flux_t = []
    u_flux_t = []

    for _, row in T_df.iterrows():
        dT = row[keys].to_numpy(dtype=float)
        T_fit = profile_pchip(r_m, dT, r_fine_m)

        integrand = T_fit * u_fit * r_fine_m
        F_t = 2 * np.pi * np.trapezoid(integrand, r_fine_m)
        flux_t.append(F_t)

        if u_v is not None and u_T is not None:
            uncertainty_integrand = (np.abs(T_fit) * u_v + np.abs(u_fit) * u_T) * r_fine_m
            u_F_t = 2 * np.pi * np.trapezoid(uncertainty_integrand, r_fine_m)
            u_flux_t.append(u_F_t)

    flux_t = np.asarray(flux_t)
    dose = np.trapezoid(flux_t, time)

    if u_v is not None and u_T is not None:
        u_flux_t = np.asarray(u_flux_t)
        u_dose_meas = np.trapezoid(u_flux_t, time)
        return flux_t, dose, u_flux_t, u_dose_meas

    return flux_t, dose


def summarize_values(values):
    values = np.asarray(values, dtype=float)
    return {
        "n": len(values),
        "mean": np.mean(values),
        "std": np.std(values, ddof=1) if len(values) > 1 else np.nan,
        "min": np.min(values),
        "p2_5": np.percentile(values, 2.5),
        "median": np.median(values),
        "p97_5": np.percentile(values, 97.5),
        "max": np.max(values),
    }


def mean_ci_t(values, confidence=0.95):
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = np.mean(values)
    sem = stats.sem(values)  # standard error of the mean = std/sqrt(n)
    h = sem * stats.t.ppf((1 + confidence) / 2, n - 1)
    return mean, mean - h, mean + h

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

    # Baseline correction using pre-heating portion
    baseline_window = tmp["time"] <= 5.0
    for key in keys:
        baseline = tmp.loc[baseline_window, key].mean()
        tmp[key] = tmp[key] - baseline

    tmp = crop_and_shift_time(tmp, time_col="time", start=t_start_exp, duration=window)
    exp_T_samples.append(tmp)

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

    # Convert absolute K to temperature rise
    for key in keys:
        tmp[key] = tmp[key] - tmp[key].iloc[0]

    tmp = crop_and_shift_time(tmp, time_col="time", start=0.0, duration=window)
    sim_T_samples.append(tmp)

# ============================================================
# Load velocity samples
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

print(f"u_exp_mean = {u_exp_mean}")
print(f"u_sim_mean = {u_sim_mean}")

velocity_summary = pd.DataFrame({
    "Radius_mm": r_mm,
    "Key": keys,
    "U_exp_mean_m_per_s": u_exp_mean,
    "U_sim_mean_m_per_s": u_sim_mean,
})
velocity_summary.to_csv(f"{csv_outdir}/velocity_at_temperature_radii_{rs}.csv", index=False)

print("\nVelocity at temperature radii:")
print(velocity_summary)

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
# Mean flux comparison using smooth radial fits
# ============================================================

F_exp_mean, D_exp_mean, uF_exp_mean, uD_exp_mean_meas = compute_temperature_weighted_flux_fit(
    exp_T_mean["time_shift"].to_numpy(),
    exp_T_mean,
    u_exp_mean,
    u_v=u_piv,
    u_T=u_therm,
)

F_sim_mean, D_sim_mean = compute_temperature_weighted_flux_fit(
    sim_T_mean["time_shift"].to_numpy(),
    sim_T_mean,
    u_sim_mean,
)

percent_diff_mean = 100 * (D_sim_mean - D_exp_mean) / D_exp_mean

mean_flux_df = pd.DataFrame({
    "time_s": exp_T_mean["time_shift"].to_numpy(),
    "F_exp_mean_K_m3_per_s": F_exp_mean,
    "u_F_exp_mean_meas_K_m3_per_s": uF_exp_mean,
})

mean_flux_df.to_csv(f"{csv_outdir}/mean_experimental_flux_history_fit_{rs}.csv", index=False)

sim_flux_df = pd.DataFrame({
    "time_s": sim_T_mean["time_shift"].to_numpy(),
    "F_sim_mean_K_m3_per_s": F_sim_mean,
})
sim_flux_df.to_csv(f"{csv_outdir}/mean_simulation_flux_history_fit_{rs}.csv", index=False)

mean_summary = pd.DataFrame([{
    "D_exp_mean_fit_K_m3": D_exp_mean,
    "D_sim_mean_fit_K_m3": D_sim_mean,
    "percent_difference_mean_fit": percent_diff_mean,
    "u_D_exp_mean_meas_K_m3": uD_exp_mean_meas,
}])
mean_summary.to_csv(f"{csv_outdir}/mean_flux_dose_summary_fit_{rs}.csv", index=False)

print("\nMean smooth-fit flux comparison:")
print(mean_summary)

# ============================================================
# All sample combinations using smooth radial fits
# ============================================================

records = []
flux_records = []

# Experimental: 3 T samples x 3 U samples = 9
for iT, T_exp in enumerate(exp_T_samples, start=1):
    for iU, u_exp in enumerate(u_exp_samples, start=1):
        F, D, uF, uD_meas = compute_temperature_weighted_flux_fit(
            T_exp["time_shift"].to_numpy(),
            T_exp,
            u_exp,
            u_v=u_piv,
            u_T=u_therm,
        )

        records.append({
            "Source": "Experiment",
            "Temperature_sample": f"T{iT}",
            "Velocity_sample": f"U{iU}",
            "Dose_K_m3": D,
            "u_D_meas_K_m3": uD_meas,
        })

        for t, fval, ufval in zip(T_exp["time_shift"], F, uF):
            flux_records.append({
                "Source": "Experiment",
                "Temperature_sample": f"T{iT}",
                "Velocity_sample": f"U{iU}",
                "time_s": t,
                "Flux_K_m3_per_s": fval,
                "u_F_meas_K_m3_per_s": ufval,
            })

# Simulation: 10 T samples x 3 U samples = 30
for iT, T_sim in enumerate(sim_T_samples, start=1):
    for iU, u_sim in enumerate(u_sim_samples, start=1):
        F, D = compute_temperature_weighted_flux_fit(
            T_sim["time_shift"].to_numpy(),
            T_sim,
            u_sim,
        )

        records.append({
            "Source": "Simulation",
            "Temperature_sample": f"T{iT}",
            "Velocity_sample": f"U{iU}",
            "Dose_K_m3": D,
            "u_D_meas_K_m3": np.nan,
        })

        for t, fval in zip(T_sim["time_shift"], F):
            flux_records.append({
                "Source": "Simulation",
                "Temperature_sample": f"T{iT}",
                "Velocity_sample": f"U{iU}",
                "time_s": t,
                "Flux_K_m3_per_s": fval,
                "u_F_meas_K_m3_per_s": np.nan,
            })

dose_df = pd.DataFrame(records)
flux_all_df = pd.DataFrame(flux_records)

dose_df.to_csv(f"{csv_outdir}/all_temperature_velocity_dose_combinations_fit_{rs}.csv", index=False)
flux_all_df.to_csv(f"{csv_outdir}/all_temperature_velocity_flux_histories_fit_{rs}.csv", index=False)

# ============================================================
# Dose summaries and percent differences
# ============================================================

D_exp_all = dose_df.loc[dose_df["Source"] == "Experiment", "Dose_K_m3"].to_numpy()
D_sim_all = dose_df.loc[dose_df["Source"] == "Simulation", "Dose_K_m3"].to_numpy()

summary_rows = []
for source, values in [("Experiment", D_exp_all), ("Simulation", D_sim_all)]:
    s = summarize_values(values)
    s["Source"] = source
    summary_rows.append(s)

dose_summary_df = pd.DataFrame(summary_rows)
dose_summary_df = dose_summary_df[["Source", "n", "mean", "std", "min", "p2_5", "median", "p97_5", "max"]]
dose_summary_df.to_csv(f"{csv_outdir}/dose_distribution_summary_fit_{rs}.csv", index=False)

percent_diff_values = []
for D_sim in D_sim_all:
    for D_exp in D_exp_all:
        percent_diff_values.append(100 * (D_sim - D_exp) / D_exp)
percent_diff_values = np.asarray(percent_diff_values)

percent_summary_df = pd.DataFrame([summarize_values(percent_diff_values)])
percent_summary_df.to_csv(f"{csv_outdir}/percent_difference_distribution_summary_fit_{rs}.csv", index=False)

print("\nSmooth-fit dose distribution summary:")
print(dose_summary_df)

print("\nSmooth-fit percent difference distribution summary:")
print(percent_summary_df)

# ============================================================
# Uncertainty propagation summary
# ============================================================

D_exp_mean_fit = np.mean(D_exp_all)
D_sim_mean_fit = np.mean(D_sim_all)

# Replicate/combination spread from the sampled combinations
u_D_exp_rep = np.std(D_exp_all, ddof=1)
u_D_sim_rep = np.std(D_sim_all, ddof=1)

# Measurement uncertainty contribution to experimental dose
u_D_exp_meas = dose_df.loc[dose_df["Source"] == "Experiment", "u_D_meas_K_m3"].mean()

# RMS total uncertainties
u_D_exp_total = np.sqrt(u_D_exp_rep**2 + u_D_exp_meas**2)
u_D_sim_total = np.sqrt(u_D_sim_rep**2 + u_D_sim_num**2)

# Difference uncertainty
Delta_D = D_sim_mean_fit - D_exp_mean_fit
u_Delta_D = np.sqrt(u_D_exp_total**2 + u_D_sim_total**2)
z_score_difference = Delta_D / u_Delta_D if u_Delta_D > 0 else np.nan
p_two_sided_approx = 2 * (1 - stats.norm.cdf(abs(z_score_difference))) if np.isfinite(z_score_difference) else np.nan
not_statistically_distinguishable_approx95 = abs(z_score_difference) < zcrit if np.isfinite(z_score_difference) else np.nan

percent_difference_fit = 100 * Delta_D / D_exp_mean_fit
relative_uncertainty_exp_percent = 100 * u_D_exp_total / D_exp_mean_fit
relative_uncertainty_difference_percent = 100 * u_Delta_D / D_exp_mean_fit

uncertainty_summary = pd.DataFrame([{
    "D_exp_mean_fit_K_m3": D_exp_mean_fit,
    "D_sim_mean_fit_K_m3": D_sim_mean_fit,
    "percent_difference_fit": percent_difference_fit,
    "u_piv_m_per_s": u_piv,
    "u_therm_K": u_therm,
    "u_D_exp_rep_K_m3": u_D_exp_rep,
    "u_D_exp_meas_K_m3": u_D_exp_meas,
    "u_D_exp_total_K_m3": u_D_exp_total,
    "u_D_exp_total_percent_of_exp_mean": relative_uncertainty_exp_percent,
    "u_D_sim_rep_K_m3": u_D_sim_rep,
    "u_D_sim_num_K_m3": u_D_sim_num,
    "u_D_sim_total_K_m3": u_D_sim_total,
    "Delta_D_K_m3": Delta_D,
    "u_Delta_D_K_m3": u_Delta_D,
    "u_Delta_D_percent_of_exp_mean": relative_uncertainty_difference_percent,
    "z_score_difference": z_score_difference,
    "p_two_sided_approx": p_two_sided_approx,
    "not_statistically_distinguishable_approx95": not_statistically_distinguishable_approx95,
}])
uncertainty_summary.to_csv(f"{csv_outdir}/dose_fit_uncertainty_summary_{rs}.csv", index=False)

print("\nSmooth-fit propagated uncertainty summary:")
print(uncertainty_summary)

# ============================================================
# Welch t-test on smooth-fit combination distributions
# ============================================================
# This only tests the sampled dose distributions. It does NOT include measurement
# uncertainty unless measurement uncertainty is explicitly simulated as random draws.
# Keep it as supplementary information, not the main uncertainty conclusion.

t_statistic, p_value = stats.ttest_ind(D_sim_all, D_exp_all, equal_var=False)
exp_mean, exp_ci_low, exp_ci_high = mean_ci_t(D_exp_all, confidence=confidence)
sim_mean, sim_ci_low, sim_ci_high = mean_ci_t(D_sim_all, confidence=confidence)

stats_summary = pd.DataFrame([{
    "D_exp_mean_fit_K_m3": exp_mean,
    "D_exp_CI95_low_fit_K_m3": exp_ci_low,
    "D_exp_CI95_high_fit_K_m3": exp_ci_high,
    "D_sim_mean_fit_K_m3": sim_mean,
    "D_sim_CI95_low_fit_K_m3": sim_ci_low,
    "D_sim_CI95_high_fit_K_m3": sim_ci_high,
    "percent_difference_fit": percent_difference_fit,
    "welch_t_statistic_fit_only": t_statistic,
    "welch_p_value_fit_only": p_value,
    "n_exp": len(D_exp_all),
    "n_sim": len(D_sim_all),
}])
stats_summary.to_csv(f"{csv_outdir}/dose_welch_ttest_summary_fit_{rs}.csv", index=False)

print("\nWelch t-test summary on smooth-fit dose combinations only:")
print(stats_summary)

# ============================================================
# Plot 1: mean flux histories from smooth fits
# ============================================================

plt.figure()
plt.plot(
    sim_T_mean["time_shift"],
    F_sim_mean,
    "--",
    color="C1",
    linewidth=2.5,
    label="Model mean",
)
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
plt.xlabel("Time [s]")
plt.ylabel(r"Temperature-weighted flux, $F_T(t)$ [K m$^3$/s]")
plt.title(fr"Smooth-fit Mean Percent Difference: {percent_diff_mean:.2f}%")
plt.legend()
plt.tight_layout()
plt.savefig(f"{outdir}/mean_temperature_weighted_flux_fit_uncertainty_{rs}.png", dpi=300, bbox_inches="tight")
plt.close()

# ============================================================
# Plot 2: all smooth-fit flux histories
# ============================================================

plt.figure()
for _, group in flux_all_df[flux_all_df["Source"] == "Experiment"].groupby(["Temperature_sample", "Velocity_sample"]):
    plt.plot(group["time_s"], group["Flux_K_m3_per_s"], color="C0", alpha=0.35, linewidth=1.5)

for _, group in flux_all_df[flux_all_df["Source"] == "Simulation"].groupby(["Temperature_sample", "Velocity_sample"]):
    plt.plot(group["time_s"], group["Flux_K_m3_per_s"], color="C1", alpha=0.25, linewidth=1.5, linestyle="--")

plt.plot(exp_T_mean["time_shift"], F_exp_mean, color="C0", linewidth=3, label="Experiment mean")
plt.plot(sim_T_mean["time_shift"], F_sim_mean, color="C1", linewidth=3, linestyle="--", label="Model mean")
plt.xlabel("Time [s]")
plt.ylabel(r"Temperature-weighted flux, $F_T(t)$ [K m$^3$/s]")
plt.title("All Smooth-Fit Temperature-Velocity Flux Combinations")
plt.legend()
plt.tight_layout()
plt.savefig(f"{outdir}/all_temperature_weighted_flux_histories_fit_{rs}.png", dpi=300, bbox_inches="tight")
plt.close()

# ============================================================
# Plot 3: uncertainty-based dose interval comparison
# ============================================================

plot_df = pd.DataFrame({
    "Source": ["Experiment", "Simulation"],
    "Mean": [D_exp_mean_fit, D_sim_mean_fit],
    "Uncertainty": [u_D_exp_total, u_D_sim_total],
})

x = np.arange(len(plot_df))
plt.figure()
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
# Plot 4: smooth-fit dose distributions
# ============================================================

plt.figure()
plt.hist(D_exp_all, bins=8, alpha=0.6, label="Experiment", color="C0")
plt.hist(D_sim_all, bins=8, alpha=0.6, label="Model", color="C1")
plt.xlabel(r"Cumulative dose, $D_T$ [K m$^3$]")
plt.ylabel("Count")
plt.title("Smooth-Fit Dose Distribution From Sample Combinations")
plt.legend()
plt.tight_layout()
plt.savefig(f"{outdir}/dose_distribution_histogram_fit_{rs}.png", dpi=300, bbox_inches="tight")
plt.close()

# ============================================================
# Plot 5: percent difference distribution
# ============================================================

plt.figure()
plt.hist(percent_diff_values, bins=20, color="C2", alpha=0.75)
plt.axvline(np.mean(percent_diff_values), color="k", linestyle="--", linewidth=2)
plt.xlabel("Percent difference [%]")
plt.ylabel("Count")
plt.title("Smooth-Fit Percent Difference Distribution")
plt.tight_layout()
plt.savefig(f"{outdir}/percent_difference_distribution_fit_{rs}.png", dpi=300, bbox_inches="tight")
plt.close()

# ============================================================
# Plot 6: Hariharan-style interval plot with propagated uncertainty
# ============================================================

fig, ax = plt.subplots(figsize=(7, 2.2))

y_exp = 0.0
y_sim = 0.0

De_mean = D_exp_mean_fit
De_low = D_exp_mean_fit - u_D_exp_total
De_high = D_exp_mean_fit + u_D_exp_total

Dc_mean = D_sim_mean_fit
Dc_low = D_sim_mean_fit - u_D_sim_total
Dc_high = D_sim_mean_fit + u_D_sim_total

xmax = max(De_high, Dc_high) * 1.15
ax.hlines(y_exp, 0, xmax, color="k", linewidth=2)
ax.annotate(
    "",
    xy=(xmax, y_exp),
    xytext=(xmax * 0.97, y_exp),
    arrowprops=dict(arrowstyle="->", color="k", lw=2),
)

ax.errorbar(
    De_mean,
    y_exp,
    xerr=[[De_mean - De_low], [De_high - De_mean]],
    fmt="o",
    color="C0",
    ecolor="C0",
    elinewidth=3,
    capsize=10,
    markersize=8,
    label=r"Experiment, $D_e$",
)

ax.errorbar(
    Dc_mean,
    y_sim,
    xerr=[[Dc_mean - Dc_low], [Dc_high - Dc_mean]],
    fmt="o",
    color="C1",
    ecolor="C1",
    elinewidth=3,
    capsize=10,
    markersize=8,
    label=r"Model, $D_c$",
)

ax.text(De_mean, y_exp - 0.12, r"$D_e$", ha="center", va="top", fontsize=14)
ax.text(Dc_mean, y_sim + 0.20, r"$D_c$", ha="center", va="bottom", fontsize=14)

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
# Final console summary
# ============================================================

print("\nSaved CSV files to:", csv_outdir)
print("Saved figures to:", outdir)
print("\nMain files to inspect:")
print(f"  {csv_outdir}/dose_fit_uncertainty_summary_{rs}.csv")
print(f"  {outdir}/dose_interval_hariharan_style_fit_uncertainty_{rs}.png")
print(f"  {outdir}/mean_temperature_weighted_flux_fit_uncertainty_{rs}.png")


# ##### new analysis 
# # ============================================================
# # Statistical testing summary for dose comparison
# # ============================================================
# # Welch t-test tests:
# # H0: mean experimental dose = mean simulation dose
# # HA: means are different
# #
# # Important: this is NOT an equivalence test.
# # If p > 0.05, report "not statistically distinguishable"
# # or "failed to reject equal means", not "proved equivalent."

# alpha = 0.05

# welch_statistically_distinguishable = p_value < alpha

# # Effect size: Hedges' g, small-sample corrected Cohen's d
# n_exp = len(D_exp_all)
# n_sim = len(D_sim_all)

# var_exp = np.var(D_exp_all, ddof=1)
# var_sim = np.var(D_sim_all, ddof=1)

# pooled_sd = np.sqrt(
#     ((n_exp - 1) * var_exp + (n_sim - 1) * var_sim)
#     / (n_exp + n_sim - 2)
# )

# cohens_d = (np.mean(D_sim_all) - np.mean(D_exp_all)) / pooled_sd

# # Hedges correction
# J = 1 - (3 / (4 * (n_exp + n_sim) - 9))
# hedges_g = J * cohens_d

# # Welch-Satterthwaite degrees of freedom
# welch_df = (
#     (var_sim / n_sim + var_exp / n_exp) ** 2
#     / (
#         ((var_sim / n_sim) ** 2 / (n_sim - 1))
#         + ((var_exp / n_exp) ** 2 / (n_exp - 1))
#     )
# )

# dose_statistical_interpretation = (
#     "Statistically distinguishable at alpha=0.05"
#     if welch_statistically_distinguishable
#     else "Not statistically distinguishable at alpha=0.05"
# )

# dose_stats_extended = pd.DataFrame([{
#     "test": "Welch two-sample t-test",
#     "null_hypothesis": "mean experimental dose equals mean simulation dose",
#     "alternative_hypothesis": "means are different",
#     "alpha": alpha,
#     "n_exp": n_exp,
#     "n_sim": n_sim,
#     "D_exp_mean_K_m3": np.mean(D_exp_all),
#     "D_sim_mean_K_m3": np.mean(D_sim_all),
#     "D_exp_std_K_m3": np.std(D_exp_all, ddof=1),
#     "D_sim_std_K_m3": np.std(D_sim_all, ddof=1),
#     "welch_t_statistic": t_statistic,
#     "welch_degrees_of_freedom": welch_df,
#     "welch_p_value": p_value,
#     "statistically_distinguishable_alpha_0_05": welch_statistically_distinguishable,
#     "interpretation": dose_statistical_interpretation,
#     "cohens_d": cohens_d,
#     "hedges_g": hedges_g,
#     "note": (
#         "This test compares sampled dose distributions only. "
#         "Dose values are generated from repeated temperature-velocity combinations, "
#         "so the propagated uncertainty comparison should be treated as the primary "
#         "uncertainty-aware validation result."
#     ),
# }])

# dose_stats_extended.to_csv(
#     f"{csv_outdir}/dose_welch_ttest_extended_summary_fit_{rs}.csv",
#     index=False
# )

# print("\nExtended Welch t-test and effect-size summary:")
# print(dose_stats_extended)


# ============================================================
# Dose integration + Welch t-test
# ============================================================

import numpy as np
import pandas as pd
from scipy import stats

def integrate_dose(time_s, flux_samples):
    """
    Compute one time-integrated dose per realization.

    Parameters
    ----------
    time_s : array-like, shape (n_time,)
        Time vector in seconds.
    flux_samples : array-like, shape (n_samples, n_time)
        Temperature-weighted flux histories for each realization.

    Returns
    -------
    dose : ndarray, shape (n_samples,)
        Integrated dose for each realization.
    """
    time_s = np.asarray(time_s, dtype=float)
    flux_samples = np.asarray(flux_samples, dtype=float)

    if flux_samples.ndim == 1:
        flux_samples = flux_samples[None, :]

    return np.trapezoid(flux_samples, x=time_s, axis=1)


def mean_ci95(x):
    """
    Mean and 95% confidence interval.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    mean = np.mean(x)
    sd = np.std(x, ddof=1)

    if n > 1:
        tcrit = stats.t.ppf(0.975, df=n-1)
        half_width = tcrit * sd / np.sqrt(n)
    else:
        half_width = np.nan

    return mean, mean - half_width, mean + half_width, sd


def welch_ttest_summary(exp_dose, sim_dose):
    """
    Welch two-sample t-test comparing experimental and simulation dose samples.
    """
    exp_dose = np.asarray(exp_dose, dtype=float)
    sim_dose = np.asarray(sim_dose, dtype=float)

    exp_mean, exp_low, exp_high, exp_sd = mean_ci95(exp_dose)
    sim_mean, sim_low, sim_high, sim_sd = mean_ci95(sim_dose)

    t_stat, p_value = stats.ttest_ind(
        exp_dose,
        sim_dose,
        equal_var=False,
        nan_policy="omit"
    )

    n_exp = len(exp_dose)
    n_sim = len(sim_dose)

    # Welch-Satterthwaite degrees of freedom
    s1 = np.var(exp_dose, ddof=1)
    s2 = np.var(sim_dose, ddof=1)

    welch_df = ((s1/n_exp + s2/n_sim)**2) / (
        ((s1/n_exp)**2 / (n_exp - 1)) +
        ((s2/n_sim)**2 / (n_sim - 1))
    )

    percent_difference = (sim_mean - exp_mean) / exp_mean * 100

    result = {
        "test": "Welch two-sample t-test",
        "quantity_tested": "time-integrated temperature-weighted dose",
        "null_hypothesis": "mean experimental dose equals mean simulation dose",
        "alternative_hypothesis": "means are different",
        "alpha": 0.05,
        "n_exp": n_exp,
        "n_sim": n_sim,
        "D_exp_mean": exp_mean,
        "D_exp_CI95_low": exp_low,
        "D_exp_CI95_high": exp_high,
        "D_exp_std": exp_sd,
        "D_sim_mean": sim_mean,
        "D_sim_CI95_low": sim_low,
        "D_sim_CI95_high": sim_high,
        "D_sim_std": sim_sd,
        "percent_difference": percent_difference,
        "welch_t_statistic": t_stat,
        "welch_degrees_of_freedom": welch_df,
        "welch_p_value": p_value,
        "statistically_distinguishable_alpha_0_05": p_value < 0.05,
    }

    return result


# ============================================================
# REQUIRED INPUTS
# ============================================================
# Replace these with the arrays already created in your script.
#
# time_s should be shape: (n_time,)
# exp_flux_samples should be shape: (n_exp_samples, n_time)
# sim_flux_samples should be shape: (n_sim_samples, n_time)
#
# Example:
# time_s = np.array([0.1, 1, 2, 4, 5, 10, 11, 12])
# exp_flux_samples = ...
# sim_flux_samples = ...

D_exp = integrate_dose(time_s, exp_flux_samples)
D_sim = integrate_dose(time_s, sim_flux_samples)

dose_ttest_results = welch_ttest_summary(D_exp, D_sim)

dose_ttest_df = pd.DataFrame([dose_ttest_results])

print("\nExperimental doses:")
print(D_exp)

print("\nSimulation doses:")
print(D_sim)

print("\nDose Welch t-test summary:")
print(dose_ttest_df.T)

dose_ttest_df.to_csv(
    "dose_welch_ttest_results.csv",
    index=False
)

pd.DataFrame({
    "source": ["Experiment"] * len(D_exp) + ["Simulation"] * len(D_sim),
    "dose": np.concatenate([D_exp, D_sim])
}).to_csv(
    "dose_samples.csv",
    index=False
)