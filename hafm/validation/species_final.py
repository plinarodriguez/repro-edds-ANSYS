import numpy as np 
import pandas as pd 
from matplotlib import pyplot as plt
from scipy.interpolate import PchipInterpolator
from scipy.stats import t, ttest_ind_from_stats

# radial locations
r_mm = np.array([0, 1, 2, 4])
r_m = r_mm * 1e-3
r_label_sim = ['R0','R1','R2','R3','R4']
# uncertainties
u_piv   = 0.1348 # m/s
u_therm = 0.6    #0.4 range from 0.2-0.6K
u_haigen = 1e-5

# Probability t-test 
confidence = 0.95  

########################################
######## Extract Velocity Data 
########################################
exp_S1 = pd.read_csv("data/velocities/S1_Low_Exp_Outlet.csv")
exp_S1['Sample']= 1
exp_S2 = pd.read_csv("data/velocities/S2_Low_Exp_Outlet.csv")
exp_S2['Sample']= 2
exp_S3 = pd.read_csv("data/velocities/S3_Low_Exp_Outlet.csv")
exp_S3['Sample']= 3
sim_S1 = pd.read_csv("data/velocities/S1_Model.csv")
sim_S1['Sample']= 1
sim_S2 = pd.read_csv("data/velocities/S2_Model.csv")
sim_S2['Sample']= 2
sim_S3 = pd.read_csv("data/velocities/S3_Model.csv")
sim_S3['Sample']= 3

U_exp =pd.concat([exp_S1,exp_S2,exp_S3], ignore_index=True) 
U_exp = U_exp.rename(columns={'x (mm)':'r (m)'})  
U_exp['r (m)'] = (U_exp['r (m)'])*1e-3 #-5.50668
u_s1 = U_exp[U_exp['Sample']==2]
idx = u_s1["V (m/s)"].idxmax()
rexp_center = U_exp.loc[idx, "r (m)"]
U_exp['r (m)'] = U_exp['r (m)']-rexp_center

U_sim =pd.concat([sim_S1,sim_S2,sim_S3])
U_sim = U_sim.drop(columns=["X [ m ]"," Z [ m ]"])
U_sim = U_sim.rename(columns={' Y [ m ]':'r (m)' ," Velocity [ m s^-1 ]":"V (m/s)"})
rsim_center = (U_sim["r (m)"].min() + U_sim["r (m)"].max()) / 2
U_sim['r (m)'] = U_sim['r (m)']-rsim_center
U_sim = U_sim[U_sim["r (m)"].between(-0.002753, 0.002753)].reset_index(drop=True)
# print(f'u_exp max = {U_exp}')
# print(f"U_exp = \n {U_exp[U_exp['Sample']==1]}" ) 
# print(f'U_sim = \n {U_sim[U_sim['Sample']==1]}')


########################################
######## Extract Temperature Data
########################################
Tsim_S1 = pd.read_csv("data/data_sim/outputSample1.csv",skiprows=4)
Tsim_S1['Sample']= 1
Tsim_S2 = pd.read_csv("data/data_sim/outputSample2.csv",skiprows=4)
Tsim_S2['Sample']= 2
Tsim_S3 = pd.read_csv("data/data_sim/outputSample3.csv",skiprows=4)
Tsim_S3['Sample']= 3
Tsim_S4 = pd.read_csv("data/data_sim/outputSample4.csv",skiprows=4)
Tsim_S4['Sample']= 4
Tsim_S5 = pd.read_csv("data/data_sim/outputSample5.csv",skiprows=4)
Tsim_S5['Sample']= 5
Tsim_S6 = pd.read_csv("data/data_sim/outputSample6.csv",skiprows=4)
Tsim_S6['Sample']= 6
Tsim_S7 = pd.read_csv("data/data_sim/outputSample7.csv",skiprows=4)
Tsim_S7['Sample']= 7
Tsim_S8 = pd.read_csv("data/data_sim/outputSample8.csv",skiprows=4)
Tsim_S8['Sample']= 8
Tsim_S9 = pd.read_csv("data/data_sim/outputSample9.csv",skiprows=4)
Tsim_S9['Sample']= 9
Tsim_S10 = pd.read_csv("data/data_sim/outputSample10.csv",skiprows=4)
Tsim_S10['Sample']= 10

T_sim = pd.concat([Tsim_S1,Tsim_S2,Tsim_S3,Tsim_S4,Tsim_S5,Tsim_S6,Tsim_S7,Tsim_S8,Tsim_S9,Tsim_S10])
keep_columns = ['Time [s]',
                'Monitor Point: Mouthpiece1mm0mmTempX0YZ (Temperature) [K]',
                'Monitor Point: Mouthpiece1mm1mmTempX0YZ (Temperature) [K]',
                'Monitor Point: Mouthpiece1mm2mmTempX0YZ (Temperature) [K]',
                'Monitor Point: Mouthpiece1mm3mmTempX0YZ (Temperature) [K]',
                'Monitor Point: Mouthpiece1mm4mmTempX0YZ (Temperature) [K]',
                'Sample']
T_sim = T_sim[keep_columns]
T_sim = T_sim.rename(columns={
    'Monitor Point: Mouthpiece1mm0mmTempX0YZ (Temperature) [K]': "R0",
    'Monitor Point: Mouthpiece1mm1mmTempX0YZ (Temperature) [K]': "R1",
    'Monitor Point: Mouthpiece1mm2mmTempX0YZ (Temperature) [K]': "R2",
    'Monitor Point: Mouthpiece1mm3mmTempX0YZ (Temperature) [K]': "R3",
    'Monitor Point: Mouthpiece1mm4mmTempX0YZ (Temperature) [K]': "R4"
    })

for r in r_label_sim:
    T_sim[r] = T_sim[r]- 273.15 # Convert from K to C
T_exp = pd.read_csv('data/data_exp/Raw_Temp_Data/Air_Temperature_Measurements_1W_0.5LPM.csv')
T_exp = T_exp.rename(columns={
    "T1T2": "R0_S1","T2T2": "R0_S2","T3T2": "R0_S3",
    "T1T3": "R1_S1","T2T3": "R1_S2","T3T3": "R1_S3",
    "T1T1": "R2_S1","T2T1": "R2_S2","T3T1": "R2_S3",
    "T1T4": "R4_S1","T2T4": "R4_S2","T3T4": "R4_S3"
    })
T_exp_S1 = T_exp[["Time","R0_S1","R1_S1","R2_S1","R4_S1"]]
T_exp_S1 = T_exp_S1.rename(columns={"Time":"Time [s]","R0_S1": "R0","R1_S1":"R1","R2_S1":"R2","R4_S1":"R4"})
T_exp_S1["Sample"] = 1
T_exp_S2 = T_exp[["Time","R0_S2","R1_S2","R2_S2","R4_S2"]]
T_exp_S2 = T_exp_S2.rename(columns={"Time":"Time [s]","R0_S2":"R0","R1_S2":"R1","R2_S2":"R2","R4_S2":"R4"})
T_exp_S2["Sample"] = 2
T_exp_S3 = T_exp[["Time","R0_S3","R1_S3","R2_S3","R4_S3"]]
T_exp_S3 = T_exp_S3.rename(columns={"Time":"Time [s]","R0_S3":"R0","R1_S3":"R1","R2_S3":"R2","R4_S3":"R4"})
T_exp_S3["Sample"] = 3

T_exp = pd.concat([T_exp_S1,T_exp_S2,T_exp_S3]) #,ignore_index=True)
# print(f'T_sim = {T_sim.head()}')
# print(f'T_exp = {T_exp.head()}')
# print(f'T_sim unique = {T_sim['Sample'].unique()}')

# ###########################################################################
# # ### Sanity Check to make sure samples and profiles are the same
# # # ### Velocity Plots
# plt.figure()
# plt.title(f'Velocity m/s')
# line_shape = ['--',':',"-",'--',':',"-",'--',':',"-",'--']
# counter = 0 
# for s,l in zip([1,2,3],line_shape):
#     U_sim_S1 = U_sim[U_sim['Sample']==s]
#     U_exp_S1 = U_exp[U_exp['Sample']==s]
#     plt.plot(U_sim_S1['r (m)']*1e3,U_sim_S1['V (m/s)'],l, color='C0',linewidth=2.5,label=f'Sim - S{s}')
#     plt.plot(U_exp_S1['r (m)']*1e3,U_exp_S1['V (m/s)'],l,color='C1',linewidth=2.5,label=f'Exp -S{s}')
#     counter +=1
# plt.xlabel("Radius [m]")
# plt.ylabel('Velocity m/s')
# # plt.xlim(-0.1,12)
# plt.legend()
# plt.tight_layout()
# plt.savefig(f'figures/species_transport/TU_final/U_SimVsExp_diameter.jpg',dpi=300,bbox_inches="tight")
# plt.close()


# ### Temperature Plots
# for r in ['R0','R1','R2','R4']:
#     plt.figure()
#     plt.title(f'Radius = {r}')
#     line_shape = ['--',':',"-",'--',':',"-",'--',':',"-",'--']
#     counter = 0 
#     for s,l in zip(np.arange(1,13,1),line_shape):
#         if counter < 3:
#             T_sim_S1 = T_sim[T_sim['Sample']==s]
#             T_exp_S1 = T_exp[T_exp['Sample']==s]
#             plt.plot(T_sim_S1['Time [s]'],T_sim_S1[r]-T_sim_S1[r][0],l, color='C0',linewidth=2.5,label=f'Sim - S{s}')
#             plt.plot(T_exp_S1['Time [s]'][79:128]-19.75,T_exp_S1[r][79:128]-T_exp_S1[r][0],l,color='C1',linewidth=2.5,label=f'Exp -S{s}')
#         else: 
#             T_sim_S1 = T_sim[T_sim['Sample']==s]
#             plt.plot(T_sim_S1['Time [s]'],T_sim_S1[r]-T_sim_S1[r][0],l, color='C0',linewidth=2.5) 
#         counter +=1
#     plt.xlabel("Time [s]")
#     plt.ylabel(r'Temperature Rise $\Delta T$')
#     plt.xlim(-0.1,12)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(f'figures/species_transport/TU_final/T_SimVsExp_{r}.jpg',dpi=300,bbox_inches="tight")
#     plt.close()
##########################################  

# # Create Curve fits for Temperature and Velocity
# Problem is Temp is at only 4 radial points ad velocity is at an entire diameter
# convert to absolute value so there will be 2 of each radius

rows = []
samplesAll = np.arange(1,11,1) 
counter = 0 
for s in samplesAll: 
    if counter < 3:
        samples = s
        U_exp["r_abs_m"] = np.abs(U_exp["r (m)"])
        U_sim["r_abs_m"] = np.abs(U_sim["r (m)"])
        # print(f'U_exp = {U_exp[U_exp['Sample']==samples]}')
        # print(f'U_sim = {U_sim}')
        ################################################
        ###### for a single sample
        #####################################
        U_expS = U_exp[U_exp['Sample']==samples]
        U_simS = U_sim[U_sim['Sample']==samples]
        # take the average of the two radii 
        U_exp_radial = U_expS.groupby("r_abs_m", as_index=False)["V (m/s)"].mean()
        U_sim_radial = U_simS.groupby("r_abs_m", as_index=False)["V (m/s)"].mean()
        # Interpolate for now
        n = 200 # number of points for continuous representation
        # r_fit = np.linspace(0,0.004,n) # [m] there are differnt radial positions
        r_max = min(
            0.004,
            U_exp_radial["r_abs_m"].max(),
            U_sim_radial["r_abs_m"].max()
        )
        r_fit = np.linspace(0, r_max, n)
        U_exp_interp = PchipInterpolator(U_exp_radial["r_abs_m"], U_exp_radial["V (m/s)"])
        U_sim_interp = PchipInterpolator(U_sim_radial["r_abs_m"], U_sim_radial["V (m/s)"])
        # print(U_sim_radial.tail(10))
        # print(U_sim_radial["r_abs_m"].is_unique)
        # print(U_sim_radial["r_abs_m"].is_monotonic_increasing)
        # print(U_sim_radial["V (m/s)"].describe())
        U_exp_fit = U_exp_interp(r_fit)
        U_sim_fit = U_sim_interp(r_fit)
        # print(U_sim_fit.min())
        # print(U_sim_fit.max())
        idx = np.argmin(U_sim_fit)
        # print(r_fit[idx], U_sim_fit[idx])
        # plt.figure()
        # plt.plot(r_fit, U_exp_fit, label="exp")
        # plt.plot(r_fit, U_sim_fit, label="sim")
        # plt.legend()
        # plt.savefig(f'figures/species_transport/TU_final/Curve_fit_Test_U.jpg',dpi=300,bbox_inches="tight")
        ### added challenge of samples and time steps!!!! 
        # since so few points will use interpolation
        r_T_m = np.array([0, 1, 2, 4]) * 1e-3 
        # get the single sample 
        T_expS = T_exp[T_exp['Sample']==samples]
        T_simS = T_sim[T_sim['Sample']==samples]
        ###########################
        #get a single time not they are different for sims and experiments
        ##########################
        # index_t = 0
        # timesS = T_sim['Time [s]'][index_t].unique()
        # timesE = T_exp['Time [s]'][index_t].unique()
        # T_expS = T_expS.iloc[79:128].copy()
        # T_expS["Time [s]"] = T_expS["Time [s]"] - 19.75

        # fix the time mismatch since there are too many time datapoints
        T_expS = T_exp[T_exp["Sample"] == samples].copy()
        T_expS = T_expS.iloc[79:128].copy()
        T_expS["Time [s]"] = T_expS["Time [s]"] - 19.75

        timesS = np.sort(T_simS["Time [s]"].unique())
        timesE = np.sort(T_expS["Time [s]"].unique())
        # time = np.arange(0,12.25, 0.25)
        time = np.arange(0,13,0.251)
        for ts in time: 
            target_t = ts
            tS = timesS[np.argmin(np.abs(timesS - target_t))]
            tE = timesE[np.argmin(np.abs(timesE - target_t))]
            # print("matched sim time =", tS)
            # print("matched exp time =", tE)
            T_simSt = T_simS[np.isclose(T_simS["Time [s]"], tS)]
            T_expSt = T_expS[np.isclose(T_expS["Time [s]"], tE)]
            T_expVals = T_expSt[["R0", "R1", "R2", "R4"]].iloc[0].to_numpy()
            T_simVals = T_simSt[["R0", "R1", "R2", "R4"]].iloc[0].to_numpy()
            # print(f'T_expST =  {T_expSt}')
            r_cols = ["R0", "R1", "R2", "R4"]
            T_exp_base = T_expS[np.isclose(T_expS["Time [s]"], timesE[0])][r_cols].iloc[0].to_numpy()
            T_sim_base = T_simS[np.isclose(T_simS["Time [s]"], timesS[0])][r_cols].iloc[0].to_numpy()
            T_expVals = T_expSt[r_cols].iloc[0].to_numpy() - T_exp_base
            T_simVals = T_simSt[r_cols].iloc[0].to_numpy() - T_sim_base
            # Temperature interpolation 
            T_exp_interp = PchipInterpolator(r_T_m, T_expVals)
            T_sim_interp = PchipInterpolator(r_T_m, T_simVals)
            T_exp_fit = T_exp_interp(r_fit)
            T_sim_fit = T_sim_interp(r_fit)
            # print(f'T_sim_fit = {T_sim_fit}')
            # print(f'T_exp_fit = {T_exp_fit}')
            # plt.figure()
            # plt.xlabel('Radius [mm]')
            # plt.ylabel(r'Temperature Rise $\Delta T$')
            # plt.title(f'Curve Fits at Time {t} sec')
            # plt.plot(r_fit*1e3, T_sim_fit,color='C0', label="Model")
            # plt.plot(r_fit*1e3, T_exp_fit,color='C1', label="Experiment")
            # plt.legend()
            # plt.tight_layout()
            # plt.savefig(f'figures/species_transport/TU_final/Curve_fit_t{t}.jpg',dpi=300,bbox_inches="tight")
            # plt.close()
            # inside your loops, after you have T_fit and U_fit
            for ri, Ti_exp, Ui_exp, Ti_sim, Ui_sim in zip(r_fit, T_exp_fit, U_exp_fit, T_sim_fit, U_sim_fit):
                rows.append({
                    "time": ts,
                    "sample": samples,
                    "r": ri,
                    "T_exp": Ti_exp,
                    "U_exp": Ui_exp,
                    "T_sim": Ti_sim,
                    "U_sim": Ui_sim
                })
            profiles_df = pd.DataFrame(rows)
    else:
        samples = s
        n = 200 # number of points for continuous representation
        r_fit = np.linspace(0,0.004,n) # [m] 
        r_T_m = np.array([0, 1, 2, 4]) * 1e-3 
        T_simS = T_sim[T_sim['Sample']==samples]
        timesS = np.sort(T_simS["Time [s]"].unique())
        time = np.arange(0,13,0.251)
        for ts in time: 
            target_t = ts
            tS = timesS[np.argmin(np.abs(timesS - target_t))]
            T_simSt = T_simS[np.isclose(T_simS["Time [s]"], tS)]
            T_simVals = T_simSt[["R0", "R1", "R2", "R4"]].iloc[0].to_numpy()
            r_cols = ["R0", "R1", "R2", "R4"]
            T_sim_base = T_simS[np.isclose(T_simS["Time [s]"], timesS[0])][r_cols].iloc[0].to_numpy()
            T_simVals = T_simSt[r_cols].iloc[0].to_numpy() - T_sim_base
            T_sim_interp = PchipInterpolator(r_T_m, T_simVals)
            T_sim_fit = T_sim_interp(r_fit)
            for ri, Ti_exp, Ui_exp, Ti_sim, Ui_sim in zip(r_fit, T_exp_fit, U_exp_fit, T_sim_fit, U_sim_fit):
                rows.append({
                    "time": ts,
                    "sample": samples,
                    "r": ri,
                    "T_exp": np.nan,
                    "U_exp": np.nan,
                    "T_sim": Ti_sim,
                    "U_sim": np.nan
                })
    counter +=1 
profiles_df = pd.DataFrame(rows)
profiles_df.to_csv('figures/species_transport/TU_final/data_curvefit_all.csv',index=False)

summary_df = (
    profiles_df
    .groupby(["time", "r"], as_index=False)
    .agg(
        Tm_exp=("T_exp", "mean"),
        Um_exp=("U_exp", "mean"),
        sdT_exp=("T_exp", lambda x: x.std(ddof=1)),
        sdU_exp=("U_exp", lambda x: x.std(ddof=1)),
        Tm_sim=("T_sim", "mean"),
        Um_sim=("U_sim", "mean"),
        sdT_sim=("T_sim", lambda x: x.std(ddof=1)),
        sdU_sim=("U_sim", lambda x: x.std(ddof=1))
    )
)
u_therms = [0.2,0.4,0.6]
for u_therm  in u_therms:

    summary_df["dT_exp"] = np.sqrt(summary_df["sdT_exp"]**2 + u_therm**2)
    summary_df["dU_exp"] = np.sqrt(summary_df["sdU_exp"]**2 + u_piv**2)
    summary_df["dT_sim"] = summary_df["sdT_sim"]
    summary_df["dU_sim"] = np.sqrt(summary_df["sdU_sim"]**2 + u_haigen**2) 
    summary_df.to_csv(f'figures/species_transport/TU_final/data_summary_stats_u{u_therm}.csv',index=False)

    flux_rows = []
    for tss, df_t in summary_df.groupby("time"):
        # Experiment only
        df_exp = df_t.dropna(subset=["Tm_exp", "Um_exp", "dT_exp", "dU_exp"]).sort_values("r")
        r_exp = df_exp["r"].to_numpy()
        Tm_exp = df_exp["Tm_exp"].to_numpy()
        Um_exp = df_exp["Um_exp"].to_numpy()
        dT_exp = df_exp["dT_exp"].to_numpy()
        dU_exp = df_exp["dU_exp"].to_numpy()
        Fm_exp = np.trapezoid(Tm_exp * Um_exp * 2*np.pi*r_exp, r_exp)
        F_exp_high = np.trapezoid((Tm_exp + dT_exp) * (Um_exp + dU_exp) * 2*np.pi*r_exp, r_exp)
        F_exp_low  = np.trapezoid((Tm_exp - dT_exp) * (Um_exp - dU_exp) * 2*np.pi*r_exp, r_exp)

        # Simulation only
        df_sim = df_t.dropna(subset=["Tm_sim", "Um_sim", "dT_sim", "dU_sim"]).sort_values("r")
        r_sim = df_sim["r"].to_numpy()
        Tm_sim = df_sim["Tm_sim"].to_numpy()
        Um_sim = df_sim["Um_sim"].to_numpy()
        dT_sim = df_sim["dT_sim"].to_numpy()
        dU_sim = df_sim["dU_sim"].to_numpy()
        Fm_sim = np.trapezoid(Tm_sim * Um_sim * 2*np.pi*r_sim, r_sim)
        F_sim_high = np.trapezoid((Tm_sim + dT_sim) * (Um_sim + dU_sim) * 2*np.pi*r_sim, r_sim)
        F_sim_low  = np.trapezoid((Tm_sim - dT_sim) * (Um_sim - dU_sim) * 2*np.pi*r_sim, r_sim)
        flux_rows.append({
            "time": tss,
            "F_exp_mean": Fm_exp,
            "F_exp_low": F_exp_low,
            "F_exp_high": F_exp_high,
            "F_sim_mean": Fm_sim,
            "F_sim_low": F_sim_low,
            "F_sim_high": F_sim_high
        })
    flux_df = pd.DataFrame(flux_rows)
    flux_df["Percent_Difference"] = (100* (flux_df["F_sim_mean"] - flux_df["F_exp_mean"])/ flux_df["F_exp_mean"])
    # mean_percent_diff = flux_df["Percent_Difference"].mean()

    eps = 1e-12  # small threshold
    valid = np.abs(flux_df["F_exp_mean"]) > eps
    flux_df["Percent_Difference"] = np.nan
    flux_df.loc[valid, "Percent_Difference"] = (100* (flux_df.loc[valid, "F_sim_mean"] - flux_df.loc[valid, "F_exp_mean"])/ flux_df.loc[valid, "F_exp_mean"])
    mean_percent_diff = flux_df["Percent_Difference"].mean(skipna=True)
    flux_df.to_csv(f'figures/species_transport/TU_final/data_flux_u{u_therm}.csv',index=False)

    ## Flux Plots 
    plt.figure()
    plt.xlabel('Time [s]')
    plt.ylabel(r"Temperature-weighted flux, $F_T(t)$ [K m$^3$/s]")
    plt.title(fr"Mean Percent Difference: {mean_percent_diff:.2f}%")
    plt.fill_between(flux_df['time'],flux_df['F_sim_low'],flux_df['F_sim_high'],alpha=0.2,color='C0') #, label="Model")
    plt.fill_between(flux_df['time'],flux_df['F_exp_low'],flux_df['F_exp_high'],alpha=0.2,color='C1') #, label="Experiment")
    plt.plot(flux_df['time'],flux_df['F_sim_mean'],color='C0',linewidth=2.5, label="Model Mean")
    plt.plot(flux_df['time'],flux_df['F_exp_mean'],color='C1',linewidth=2.5, label="Experiment Mean")
    plt.xlim(-0.01,12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'figures/species_transport/TU_final/flux_curves_u{u_therm}.jpg',dpi=300,bbox_inches="tight")
    plt.close()

    ###################################
    ###### Interval Plot & Dose 
    ###################################
    dose_rows  = []
    De_mean = np.trapezoid(flux_df["F_exp_mean"],flux_df["time"])
    De_low  = np.trapezoid(flux_df["F_exp_low"] ,flux_df["time"])
    De_high = np.trapezoid(flux_df["F_exp_high"],flux_df["time"])
    Dc_mean = np.trapezoid(flux_df["F_sim_mean"],flux_df["time"])
    Dc_low  = np.trapezoid(flux_df["F_sim_low"] ,flux_df["time"])
    Dc_high = np.trapezoid(flux_df["F_sim_high"],flux_df["time"])
    print(f'De_mean = {De_mean}')
    print(f'De_low  = {De_low}') 
    print(f'De_high = {De_high}')
    print(f'Dc_mean = {Dc_mean}')
    print(f'Dc_low  = {Dc_low}') 
    print(f'Dc_high = {Dc_high}')

    # Dose interval plot in Hariharan-style format
    fig, ax = plt.subplots(figsize=(7, 2.2))
    y_exp = 0.0
    y_sim = 0.0
    # horizontal axis
    xmax = max(De_high, Dc_high) * 1.15
    ax.hlines(y_exp, 0, xmax, color="k", linewidth=2)
    ax.annotate("",xy=(xmax, y_exp),xytext=(xmax * 0.97, y_exp),arrowprops=dict(arrowstyle="->", color="k", lw=2))
    # experimental interval
    ax.errorbar(De_mean, y_exp,xerr=[[De_mean - De_low], [De_high - De_mean]],fmt="o",color="C0",ecolor="C0",elinewidth=3,capsize=10,markersize=8,label=r"Experiment, $D_e$")
    # computational interval
    ax.errorbar(Dc_mean, y_sim,xerr=[[Dc_mean - Dc_low], [Dc_high - Dc_mean]],fmt="o",color="C1",ecolor="C1",elinewidth=3,capsize=10,markersize=8,label=r"Model, $D_c$")
    ax.text(De_mean, y_exp - 0.12, r"$D_e$", ha="center", va="top", fontsize=14)
    ax.text(Dc_mean, y_sim + 0.20, r"$D_c$", ha="center", va="bottom", fontsize=14)
    ax.set_xlabel(r"Temperature-weighted dose, $D_T$ [K m$^3$]")
    ax.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
    ax.set_yticks([])
    ax.set_ylim(-0.25, 0.35)
    ax.spines[["left", "right", "top"]].set_visible(False)
    ax.legend(loc="upper left", frameon=True)
    plt.tight_layout()
    plt.savefig(f'figures/species_transport/TU_final/dose_interval_plots_u{u_therm}.jpg',dpi=300,bbox_inches="tight")
    plt.close()

    #########################
    ##### T-test 
    #########################
    n_De = 3
    n_Dc = 3

    # def std_from_95ci(low, high, n):
    #     halfwidth = 0.5*(high - low)
    #     tcrit = t.ppf(0.975, df=n-1)  # 95% CI
    #     se = halfwidth / tcrit
    #     s = se * np.sqrt(n)
    #     return s, halfwidth, tcrit, se
    # De_std,De_halfwidth,De_tcrit,De_se = std_from_95ci(De_low, De_high, n_De)
    # Dc_std,Dc_halfwidth,Dc_tcrit,Dc_se = std_from_95ci(Dc_low, Dc_high, n_Dc)
    
    def std_from_1sd_bounds(low, high):
        halfwidth = 0.5*(high - low)
        std = halfwidth
        return std, halfwidth 

    De_std, De_halfwidth = std_from_1sd_bounds(De_low, De_high)
    Dc_std, Dc_halfwidth = std_from_1sd_bounds(Dc_low, Dc_high)

    tstat, pval = ttest_ind_from_stats(
        mean1=De_mean, std1=De_std, nobs1=n_De,
        mean2=Dc_mean, std2=Dc_std, nobs2=n_Dc,
        equal_var=False
    )

    print("De_std =", De_std)
    print("Dc_std =", Dc_std)
    print("Welch t =", tstat)
    print("p-value =", pval)
    print(f'####################################')

    
    # Also compute Welch df explicitly (so you can store it)
    v1 = (De_std**2) / n_De
    v2 = (Dc_std**2) / n_Dc
    welch_df = (v1 + v2)**2 / ((v1**2)/(n_De-1) + (v2**2)/(n_Dc-1))

    # ---- Build DataFrames ----
    df_doses = pd.DataFrame([
        {
            "group": "De (experimental)",
            "n": n_De,
            "mean": De_mean,
            "ci95_low": De_low,
            "ci95_high": De_high,
            "ci_halfwidth": De_halfwidth,
            # "tcrit_0.975_df(n-1)": De_tcrit,
            # "SE_from_CI": De_se,
            "std_from_CI": De_std,
        },
        {
            "group": "Dc (computational)",
            "n": n_Dc,
            "mean": Dc_mean,
            "ci95_low": Dc_low,
            "ci95_high": Dc_high,
            "ci_halfwidth": Dc_halfwidth,
            # "tcrit_0.975_df(n-1)": Dc_tcrit,
            # "SE_from_CI": Dc_se,
            "std_from_CI": Dc_std,
        },
    ])
    alpha = 0.05
    df_ttest = pd.DataFrame([{
        "test": "Welch two-sample t-test",
        "alpha": alpha,
        "mean_diff_(De-Dc)": De_mean - Dc_mean,
        "t_stat": float(tstat),
        "welch_df": float(welch_df),
        "p_value_two_sided": float(pval),
    }])

    # "Pass" = no statistically significant difference at alpha
    df_ttest["passes_test"] = df_ttest["p_value_two_sided"] >= df_ttest["alpha"]
    df_ttest["decision"] = np.where(df_ttest["passes_test"],
                                    "PASS (no significant difference)",
                                    "FAIL (significant difference)")

    df_ttest.to_csv(f"figures/species_transport/TU_final/dose_ttest_u{u_therm}.csv", index=False)
    df_doses.to_csv(f"figures/species_transport/TU_final/dose_summary_u{u_therm}.csv", index=False)


