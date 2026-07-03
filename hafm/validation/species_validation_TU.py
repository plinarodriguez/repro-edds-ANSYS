import pandas as pd 
import numpy as np
from matplotlib import pyplot as plt

# Radius to compare
r_mm = np.array([0, 1, 2, 4]) #np.array([0,4]) #
r_m = r_mm * 1e-3
keys = ["R0", "R1", "R2", "R4"] #["R0", "R4"]#

# Extract Velocities
def read_velocity_profile(filepath, source="exp"):
    df = pd.read_csv(filepath)
    if source == "exp":
        x_mm = df["x (mm)"].to_numpy(dtype=float)
        u = df["V (m/s)"].to_numpy(dtype=float)
    elif source == "sim":
        # Model format: X [ m ], Y [ m ], Z [ m ], Velocity [ m s^-1 ]
        x_mm = df[" Y [ m ]"].to_numpy(dtype=float) * 1e3
        u = df[" Velocity [ m s^-1 ]"].to_numpy(dtype=float)
    center_mm = 0.5 * (np.nanmin(x_mm) + np.nanmax(x_mm))
    r_vel_mm = np.abs(x_mm - center_mm)
    order = np.argsort(r_vel_mm)
    r_vel_mm = r_vel_mm[order]
    u = u[order]
    return r_vel_mm * 1e-3, u

# Extract Matching velocities to Temperature radial locations
def interpolate_velocity_to_temperature_radii(r_vel_m, u_vel, r_target_m):
    vel_df = pd.DataFrame({"r_m": r_vel_m, "u": u_vel})
    # Round avoids tiny floating-point differences between symmetric points
    vel_df["r_round"] = vel_df["r_m"].round(7)
    vel_avg = (vel_df.groupby("r_round", as_index=False)["u"].mean().sort_values("r_round"))
    r_unique = vel_avg["r_round"].to_numpy()
    u_unique = vel_avg["u"].to_numpy()
    u_target = np.interp(r_target_m,r_unique,u_unique,left=u_unique[0],right=0.0)
    return u_target

# compute integrals with trapezoid rule 
def compute_temperature_weighted_flux(time, T_df,u_profile):
    """
    Computes F_T(t) = 2*pi*integral DeltaT(r,t)*u(r)*r dr
    and D_T = integral F_T(t) dt.
    Units:
    F_T(t): K*m^3/s
    D_T: K*m^3
    """
    flux_t = []
    for _, row in T_df.iterrows():
        dT = row[keys].to_numpy(dtype=float)
        integrand = dT * u_profile * r_m
        F_t = 2 * np.pi * np.trapezoid(integrand, r_m)
        flux_t.append(F_t)
    flux_t = np.array(flux_t)
    dose = np.trapezoid(flux_t, time)
    return flux_t, dose

# Extract Experiment Velocity Data 
exp_vel_files = [
    "data/velocities/S1_Low_Exp_Outlet.csv",
    "data/velocities/S2_Low_Exp_Outlet.csv",
    "data/velocities/S3_Low_Exp_Outlet.csv",
]

# Extract Simulation Velocity data
sim_vel_files = [
    "data/velocities/S1_Model.csv",
    "data/velocities/S2_Model.csv",
    "data/velocities/S3_Model.csv",
]

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

print("Velocity at temperature radii:")
for key, radius, ue, us in zip(keys, r_mm, u_exp_mean, u_sim_mean):
    print(f"{key} ({radius} mm): U_exp={ue:.4f} m/s, U_sim={us:.4f} m/s")


# Extract Experiment data Temperature
tempE = pd.read_csv("data/data_exp/Raw_Temp_Data/Air_Temperature_Measurements_1W_0.5LPM.csv")
# Experimental data mapping to simulation radii
exp_cols = {
    "R0": ["T1T2", "T2T2", "T3T2"],  # center
    "R1": ["T1T3", "T2T3", "T3T3"],  # 1 mm
    "R2": ["T1T1", "T2T1", "T3T1"],  # 2 mm
    "R4": ["T1T4", "T2T4", "T3T4"],  # 4 mm
}

# Extract Simulation data Temperature 
sim_data = []
for i in range(1, 11):
    df = pd.read_csv(f"data/data_sim/outputSample{i}.csv",skiprows=4)
    sim_data.append(df)

# Simulation samples
sim_cols = {
    "R0": "Monitor Point: Mouthpiece1mm0mmTempX0YZ (Temperature) [K]",
    "R1": "Monitor Point: Mouthpiece1mm1mmTempX0YZ (Temperature) [K]",
    "R2": "Monitor Point: Mouthpiece1mm2mmTempX0YZ (Temperature) [K]",
    # "R2": "Monitor Point: Mouthpiece1mm3mmTempX0YZ (Temperature) [K]",
    "R4": "Monitor Point: Mouthpiece1mm4mmTempX0YZ (Temperature) [K]",
}

# Mean experiment results 
exp_T_mean = pd.DataFrame()
exp_T_mean["time"] = tempE["Time"] # time

# Temp data
for key, cols in exp_cols.items():
    exp_T_mean[key] = tempE[cols].mean(axis=1)
# Use pre-heating baseline from first 5 seconds
baselineT = exp_T_mean["time"] <= 5.0
for key in keys:
    baseline = exp_T_mean.loc[baselineT, key].mean()
    exp_T_mean[key] = exp_T_mean[key] - baseline

# Temperature rise data
# for key in keys:
#     exp_T_mean[key] = exp_T_mean[key] - exp_T_mean[key].iloc[0]

# mean simulation results 
sim_T_samples = []
for df in sim_data:
    tmp = pd.DataFrame()
    tmp["time"] = df["Time [s]"] # time
    for key, col in sim_cols.items():
        tmp[key] = df[col]
    for key in keys:
        tmp[key] = tmp[key] - tmp[key].iloc[0]
    sim_T_samples.append(tmp)
# Combine into mean simulation profile
sim_T_mean = pd.DataFrame()
sim_T_mean["time"] = sim_T_samples[0]["time"]
for key in keys:
    sim_T_mean[key] = np.mean([tmp[key].to_numpy() for tmp in sim_T_samples], axis=0)

    
# get the matching data for the same times for the experiments and model

t_start_exp = 20 #18.5 #30.0      # adjust based on heating onset in experiment
range = 12.0           # compare same 12 s window as simulation
exp_filtered = exp_T_mean[(exp_T_mean["time"] >= t_start_exp) &(exp_T_mean["time"] <= t_start_exp + range)].copy()
# shift experimental time to match the filtered data 
exp_filtered["time_shift"] = exp_filtered["time"] - t_start_exp
# for key in keys:
#     exp_filtered[key] = exp_filtered[key] - exp_filtered[key].iloc[0]

sim_filtered = sim_T_mean[(sim_T_mean["time"] >= 0.0) &(sim_T_mean["time"] <= range)].copy()
sim_filtered["time_shift"] = sim_filtered["time"]

# Computet the exposure
F_exp, D_exp = compute_temperature_weighted_flux(exp_filtered["time_shift"].to_numpy(),exp_filtered,u_exp_mean)
F_sim, D_sim = compute_temperature_weighted_flux(sim_filtered["time_shift"].to_numpy(),sim_filtered,u_sim_mean)

percent_diff = 100 * np.abs(D_sim - D_exp) / D_exp

print(f"Experimental temperature-weighted dose: {D_exp:.6e} K*m^3")
print(f"Simulation temperature-weighted dose:   {D_sim:.6e} K*m^3")
print(f"Percent difference: {percent_diff:.2f}%")

plt.figure() #figsize=(7, 5))
plt.plot(sim_filtered["time_shift"], F_sim, "--", label=fr"Model") # ({D_exp:.1e} K*m^2*s)")
plt.plot(exp_filtered["time_shift"], F_exp, label=fr"Experiment") # ({D_sim:.1e} K*m^2*s)")
plt.xlabel("Time [s]")
plt.ylabel(r"Temperature-weighted flux, $F_T(t)$ [K m$^3$/s]")
plt.title(fr'Precent Difference: {percent_diff:.2f}%')
plt.legend()
plt.tight_layout()
plt.savefig('figures/species_transport/TU/species_transport_r0124.png', dpi=300, bbox_inches='tight')
plt.close()

# plt.figure() #figsize=(8,6))
# lines = ["-","--",":","-."]
# for key,l in zip(keys,lines):
#     plt.plot(exp_T_mean["time"], exp_T_mean[key], l ,label=key, color='C0')
# plt.xlim(20,32)
# plt.xlabel("Time [s]")
# plt.ylabel("Temperature [C]")
# plt.title('Experiment')
# plt.legend()
# plt.savefig('figures/species_transport/TU/species_transport_ExperimentONLY_r0124.png', dpi=300, bbox_inches='tight')
# plt.close()

# plt.figure() #figsize=(8,6))
# for key,l in zip(keys,lines):
#     plt.plot(sim_T_mean["time"], sim_T_mean[key], l ,label=key, color='C1')
# plt.xlabel("Time [s]")
# plt.ylabel("Temperature [C]")
# plt.title('Model')
# plt.legend()
# plt.savefig('figures/species_transport/TU/species_transport_ModelONLY_r0134.png', dpi=300, bbox_inches='tight')
# plt.close()

# for key in keys:
#     idx = exp_T_mean[key].idxmax()
#     print(key)
#     print("Peak temp =", exp_T_mean[key].max())
#     print("Peak time =", exp_T_mean.loc[idx,"time"])
#     print()