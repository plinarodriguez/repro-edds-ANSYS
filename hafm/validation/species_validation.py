import pandas as pd 
import numpy as np
from matplotlib import pyplot as plt

# Radius to compare
r_mm = np.array([0,1,2,4]) #np.array([0, 1, 2, 4])
r_m = r_mm * 1e-3
keys = ["R0", "R1", "R2", "R4"]

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
    # "R2": "Monitor Point: Mouthpiece1mm2mmTempX0YZ (Temperature) [K]",
    "R2": "Monitor Point: Mouthpiece1mm3mmTempX0YZ (Temperature) [K]",
    "R4": "Monitor Point: Mouthpiece1mm4mmTempX0YZ (Temperature) [K]",
}

# analysis starts here 
def compute_area_integrated_exposure(time, T_df):
    """
    Computes E_T(t) = 2*pi*integral DeltaT(r,t)*r dr
    and D_T = integral E_T(t) dt.

    Units:
    E_T(t): K*m^2
    D_T: K*m^2*s
    """
    exposure_t = []

    for _, row in T_df.iterrows():
        dT = row[keys].to_numpy(dtype=float)
        integrand = dT * r_m
        E_t = 2 * np.pi * np.trapezoid(integrand, r_m)
        exposure_t.append(E_t)

    exposure_t = np.array(exposure_t)
    dose = np.trapezoid(exposure_t, time)

    return exposure_t, dose

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
E_exp, D_exp = compute_area_integrated_exposure(exp_filtered["time_shift"].to_numpy(),exp_filtered)
E_sim, D_sim = compute_area_integrated_exposure(sim_filtered["time_shift"].to_numpy(),sim_filtered)

# print(exp_filtered[["R0", "R1", "R2", "R4"]].head())
# print(exp_filtered[["R0", "R1", "R2", "R4"]].max())
# print(f'***************************')
percent_diff = 100 * np.abs(D_sim - D_exp) / D_exp
print(f"Experimental area-integrated temperature-rise dose: {D_exp:.6e} K*m^2*s")
print(f"Model area-integrated temperature-rise dose:   {D_sim:.6e} K*m^2*s")
print(f"Percent difference: {percent_diff:.2f}%")

plt.figure() #figsize=(7, 5))
plt.plot(sim_filtered["time_shift"], E_sim, "--", label=fr"Model") # ({D_exp:.1e} K*m^2*s)")
plt.plot(exp_filtered["time_shift"], E_exp, label=fr"Experiment") # ({D_sim:.1e} K*m^2*s)")
plt.xlabel("Time [s]")
plt.ylabel(r"Area-integrated temperature rise, $E_T(t)$ [K m$^2$]")
plt.title(fr'Precent Difference: {percent_diff:.2f}%')
plt.legend()
plt.tight_layout()
plt.savefig('figures/species_transport/T/species_transport_r012-34.png', dpi=300, bbox_inches='tight')
plt.close()

plt.figure() #figsize=(8,6))
lines = ["-","--",":","-."]
for key,l in zip(keys,lines):
    plt.plot(exp_T_mean["time"], exp_T_mean[key], l ,label=key, color='C0')
plt.xlim(20,32)
plt.xlabel("Time [s]")
plt.ylabel("Temperature [C]")
plt.title('Experiment')
plt.legend()
plt.savefig('figures/species_transport/T/species_transport_ExperimentONLY_r0124.png', dpi=300, bbox_inches='tight')
plt.close()

plt.figure() #figsize=(8,6))
for key,l in zip(keys,lines):
    plt.plot(sim_T_mean["time"], sim_T_mean[key], l ,label=key, color='C1')
plt.xlabel("Time [s]")
plt.ylabel("Temperature [C]")
plt.title('Model')
plt.legend()
plt.savefig('figures/species_transport/T/species_transport_ModelONLY_r0134.png', dpi=300, bbox_inches='tight')
plt.close()

# for key in keys:
#     idx = exp_T_mean[key].idxmax()
#     print(key)
#     print("Peak temp =", exp_T_mean[key].max())
#     print("Peak time =", exp_T_mean.loc[idx,"time"])
#     print()