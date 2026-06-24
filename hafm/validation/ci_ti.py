# Libraries
import pandas as pd
import numpy as np
from scipy.stats import t 
from scipy.stats import norm, chi2
from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.pyplot as plt

# Set the font family and size to use for Matplotlib figures.
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 20


### Confidence Intervals 95%
def confidence_interval(data, alpha=0.10): # for 90% CI -----alpha=0.05): 3 for 95% confidence interval
    data = np.asarray(data)
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    tval = t.ppf(1-alpha/2, n-1)
    half_width = tval * std / np.sqrt(n)
    return mean, mean-half_width, mean+half_width

# # 95/95 tolerance interval 
# def tolerance_interval(data, k=4.94):
#     # For n=3, a two-sided 95/95 TI has approximately k=4.94
#     mean = np.mean(data)
#     std = np.std(data, ddof=1)
#     return mean-k*std, mean+k*std

def tolerance_interval(data):
    data = np.asarray(data)
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    if n == 3:
        k = 4.94      # 95/95
    elif n == 10:
        k = 2.91      # 95/95
    return mean-k*std, mean+k*std


def plot_intervals_side_by_side(summary, low_col, high_col, title, figname,metrics_to_plot):
    # metrics = summary["Metric"].unique()
    metrics = metrics_to_plot
    # fig, axes = plt.subplots(1, 2, figsize=(12,6), sharey=True)
    fig, axes = plt.subplots(1, len(metrics), figsize=(6*len(metrics), 6), sharey=False)
    for ax, metric in zip(axes, metrics):
        tmp = summary[summary["Metric"] == metric].copy()
        tmp["Source"] = pd.Categorical(tmp["Source"],categories=["Experiment", "Simulation"],ordered=True)
        tmp = tmp.sort_values("Source")
        labels = ['Experiment','Model'] #tmp["Source"].to_numpy()
        x = np.array([0.0, 0.2])
        means = tmp["Mean"].to_numpy()
        lows = tmp[low_col].to_numpy()
        highs = tmp[high_col].to_numpy()
        yerr = np.vstack([means - lows,highs - means])
        ax.errorbar(x,means,yerr=yerr,fmt="o",capsize=6,elinewidth=3,linewidth=2)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlim(-0.25, 0.60)
        ax.set_ylim(0.4, 3.5) #(0.4, 3.5)
        ax.set_title(metric)
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("Velocity (m/s)")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(figname, dpi=300)
    plt.show()
    plt.close()

# Extract Simulation data 
sim_data = []
for i in range(1, 11):
    df = pd.read_csv(f"data/data_sim/outputSample{i}.csv",skiprows=4)
    sim_data.append(df)

# Extract Experiment data
tempE = pd.read_csv("data/data_exp/Raw_Temp_Data/Air_Temperature_Measurements_1W_0.5LPM.csv")

# Simulation samples
sim_cols = {
    "R0": "Monitor Point: Mouthpiece1mm0mmTempX0YZ (Temperature) [K]",
    "R1": "Monitor Point: Mouthpiece1mm1mmTempX0YZ (Temperature) [K]",
    "R2": "Monitor Point: Mouthpiece1mm2mmTempX0YZ (Temperature) [K]",
    "R3": "Monitor Point: Mouthpiece1mm3mmTempX0YZ (Temperature) [K]",
    "R4": "Monitor Point: Mouthpiece1mm4mmTempX0YZ (Temperature) [K]",
}

# Experimental data mapping to simulation radii
exp_cols = {
    "R0": ["T1T2", "T2T2", "T3T2"],  # center
    "R1": ["T1T3", "T2T3", "T3T3"],  # 1 mm
    "R2": ["T1T1", "T2T1", "T3T1"],  # 2 mm
    "R4": ["T1T4", "T2T4", "T3T4"],  # 4 mm
}

summary_rows = []
for loc in ["R0", "R1", "R2", "R4"]:
    # Simulation: 10 samples
    sim_values = []
    for df in sim_data:
        T = df[sim_cols[loc]].to_numpy()
        dT = T -  T[0]          # temperature rise
        sim_values.append(np.max(dT))
    # Experiment: 3 replicates
    exp_values = []
    for col in exp_cols[loc]:
        T = tempE[col].to_numpy()
        dT = T - T[0]          # temperature rise
        exp_values.append(np.max(dT))
    for source, values in [("Experiment", exp_values),("Simulation", sim_values)]:
        mean, ci_low, ci_high = confidence_interval(values)
        ti_low, ti_high = tolerance_interval(values)
        summary_rows.append({"Source": source,"Metric": loc,"Mean": mean,"CI Low": ci_low,"CI High": ci_high,"TI Low": ti_low,"TI High": ti_high})
summary_temp = pd.DataFrame(summary_rows)
# print(summary_temp)
summary_temp.to_csv("ci_ti_temperature.csv", index=False)

plot_intervals_side_by_side(
    summary_temp,
    low_col="CI Low",
    high_col="CI High",
    title="95% Confidence Intervals for Maximum Temperature Rise",
    figname="figures/confidence_interval_temperature.jpg",
    metrics_to_plot=["R1", "R4"]
)

plot_intervals_side_by_side(
    summary_temp,
    low_col="TI Low",
    high_col="TI High",
    title="95/95 Tolerance Intervals for Maximum Temperature Rise",
    figname="figures/tolerance_interval_temperature.jpg",
    metrics_to_plot=["R1", "R4"]
)

# *********************************
#  By time & temp rise

target_times = [0, 2, 5, 10, 11, 12]
heating_start_exp = 20.0  # seconds
summary_rows = []
for target_time in target_times:
    for loc in ["R0", "R1", "R2", "R4"]:
        # -----------------------
        # Simulation: 10 samples
        # -----------------------
        sim_values = []
        for df in sim_data:
            time = df["Time [s]"].to_numpy()
            T = df[sim_cols[loc]].to_numpy()
            T0 = T[0]
            dT = T - T0
            idx = np.argmin(np.abs(time - target_time))
            sim_values.append(dT[idx])

        # -----------------------
        # Experiment: 3 replicates
        # -----------------------
        exp_values = []
        exp_time = tempE["Time"].to_numpy()

        for col in exp_cols[loc]:
            T = tempE[col].to_numpy()
            idx0 = np.argmin(np.abs(exp_time - heating_start_exp))
            T0 = T[idx0]
            exp_target_time = heating_start_exp + target_time
            idx = np.argmin(np.abs(exp_time - exp_target_time))
            dT = T[idx] - T0
            exp_values.append(dT)

        # -----------------------
        # CI/TI
        # -----------------------
        for source, values in [("Experiment", exp_values),("Simulation", sim_values)]:
            mean, ci_low, ci_high = confidence_interval(values)
            ti_low, ti_high = tolerance_interval(values)
            summary_rows.append({"Time": target_time,"Source": source,"Metric": loc,"Mean": mean,"CI Low": ci_low,"CI High": ci_high,"TI Low": ti_low,"TI High": ti_high})

summary_temp = pd.DataFrame(summary_rows)
# print(summary_temp)
summary_temp.to_csv("ci_ti_temperature_rise_by_time.csv", index=False)


# summary_t10 = summary_temp[summary_temp["Time"] == 10]
# plot_intervals_side_by_side(
#     summary_t10,
#     low_col="CI Low",
#     high_col="CI High",
#     title="95% Confidence Intervals for Temperature Rise at 10 s",
#     figname="figures/confidence_interval_temperature_rise_10s.jpg"
# )

# plot_intervals_side_by_side(
#     summary_t10,
#     low_col="TI Low",
#     high_col="TI High",
#     title="95/95 Tolerance Intervals for Temperature Rise at 10 s",
#     figname="figures/tolerance_interval_temperature_rise_10s.jpg"
# )

summary_t10 = summary_temp[summary_temp["Time"] == 10]
plot_intervals_side_by_side(
    summary_t10,
    low_col="CI Low",
    high_col="CI High",
    title="95% Confidence Intervals for Temperature Rise at 10 s",
    figname="figures/confidence_interval_temperature_rise_10s_R1_R4.jpg",
    metrics_to_plot=["R0", "R4"]
)

plot_intervals_side_by_side(
    summary_t10,
    low_col="TI Low",
    high_col="TI High",
    title="95/95 Tolerance Intervals for Temperature Rise at 10 s",
    figname="figures/tolerance_interval_temperature_rise_10s.jpg",
    metrics_to_plot=["R0", "R4"]
)


############ All in one plot:def plot_intervals_all_together(summary, low_col, high_col, title, figname, metrics_to_plot):
def plot_intervals_all_together(summary, low_col, high_col, title, figname, metrics_to_plot):
    plt.figure(figsize=(12, 7))
    x_positions = []
    labels = []
    x = 0
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for i, metric in enumerate(metrics_to_plot):
        color = default_colors[i]
        tmp = summary[summary["Metric"] == metric].copy()
        tmp["Source"] = pd.Categorical(tmp["Source"],categories=["Experiment", "Simulation"],ordered=True)
        tmp = tmp.sort_values("Source")
        x_local = []
        labels_local = []
        means = []
        lows = []
        highs = []
        for _, row in tmp.iterrows():
            source_label = "Exp" if row["Source"] == "Experiment" else "Sim"
            x_local.append(x)
            labels_local.append(f"{source_label}_{metric}")
            means.append(row["Mean"])
            lows.append(row[low_col])
            highs.append(row[high_col])
            x += 1
        x_local = np.array(x_local)
        means = np.array(means)
        lows = np.array(lows)
        highs = np.array(highs)
        yerr = np.vstack([means - lows,highs - means])
        plt.errorbar(x_local,means,yerr=yerr,fmt="o",markersize=8,capsize=8,elinewidth=3,capthick=3,linewidth=2,color=color,label=metric)
        x_positions.extend(x_local)
        labels.extend(labels_local)
        x += 0.75
    plt.xticks(x_positions, labels, rotation=45, ha="right")
    plt.ylabel(fr"Temperature rise, $\Delta T$")
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend(title="Radial location")
    plt.tight_layout()
    plt.savefig(figname, dpi=300)
    plt.show()
    plt.close()


for t in target_times:
    summary_t = summary_temp[summary_temp["Time"] == t]
    plot_intervals_all_together(
        summary_t,
        low_col="CI Low",
        high_col="CI High",
        title=f"95% Confidence Intervals for Temperature Rise at {t} s",
        figname=f"figures/confidence_interval_temperature_rise_{t}s_all.jpg",
        metrics_to_plot=["R0", "R1", "R2", "R4"]
    )
        
    plot_intervals_all_together(
        summary_t,
        low_col="TI Low",
        high_col="TI High",
        title=f"95/95 Tolerance Intervals for Temperature Rise at {t} s",
        figname=f"figures/tolerance_interval_temperature_rise_{t}s_all.jpg",
        metrics_to_plot=["R0", "R1", "R2", "R4"]
    )


############## Heatmap
def ecdf_values(data, x):
    data = np.sort(np.asarray(data))
    return np.searchsorted(data, x, side="right") / len(data)


def area_metric_ecdf(exp_values, sim_values, normalize=False):
    exp_values = np.asarray(exp_values)
    sim_values = np.asarray(sim_values)
    x = np.linspace(min(exp_values.min(), sim_values.min()),max(exp_values.max(), sim_values.max()),1000)
    F_exp = ecdf_values(exp_values, x)
    F_sim = ecdf_values(sim_values, x)
    area = np.trapezoid(np.abs(F_exp - F_sim), x)
    if normalize:
        area = (area/ np.mean(exp_values))*100 
        # area = area/ np.mean(F_exp)  
        # area = (area/ (x.max() - x.min()))*100
    return area

def area_metric_ecdf_signed(exp_values, sim_values, normalize=False):
    exp_values = np.asarray(exp_values)
    sim_values = np.asarray(sim_values)
    x = np.linspace(min(exp_values.min(), sim_values.min()),max(exp_values.max(), sim_values.max()),1000)
    F_exp = ecdf_values(exp_values, x)
    F_sim = ecdf_values(sim_values, x)
    area = np.trapezoid((F_exp - F_sim), x)
    if normalize:
        area = (area/ np.mean(exp_values))*100 
        # area = area/ np.mean(F_exp)  
        # area = (area/ (x.max() - x.min()))*100
    return area

area_rows = []
for target_time in target_times[1:]:
    for loc in ["R0", "R1", "R2", "R4"]:
        # Simulation values
        sim_values = []
        for df in sim_data:
            time = df["Time [s]"].to_numpy()
            T = df[sim_cols[loc]].to_numpy()
            T0 = T[0]
            dT = T - T0
            idx = np.argmin(np.abs(time - target_time))
            sim_values.append(dT[idx])
        # Experiment values
        exp_values = []
        exp_time = tempE["Time"].to_numpy()

        for col in exp_cols[loc]:
            T = tempE[col].to_numpy()
            idx0 = np.argmin(np.abs(exp_time - heating_start_exp))
            T0 = T[idx0]
            exp_target_time = heating_start_exp + target_time
            idx = np.argmin(np.abs(exp_time - exp_target_time))
            dT = T[idx] - T0
            exp_values.append(dT)
        # print(f'exp_values = {exp_values}')
        area = area_metric_ecdf(exp_values, sim_values, normalize=False)
        area_rows.append({"Time": target_time,"Radius": loc,"Area Metric": area})
area_df = pd.DataFrame(area_rows)
# area_df.to_csv("areaMetric_temperature_rise_by_time.csv", index=False)
area_df.to_csv("areaMetric_temperature_rise_by_time.csv", index=False)
# print(area_df)

heatmap_data = area_df.pivot(index="Time",columns="Radius",values="Area Metric")

# Rename columns for figure labels
heatmap_data = heatmap_data.rename(columns={"R0": "0mm","R1": "1mm","R2": "2mm","R4": "4mm"})
plt.figure(figsize=(10, 8))
im = plt.imshow(heatmap_data.values,aspect="auto",origin="upper")
cbar = plt.colorbar(im)
cbar.set_label(r"Area Metric ($^{\circ}$C)",rotation=90,labelpad=15)
# cbar.set_label("Normalized Area Metric (%)",rotation=90,labelpad=15)
# cbar.set_label("Normalized Area Metric",rotation=90,labelpad=15)
plt.xticks(np.arange(len(heatmap_data.columns)),heatmap_data.columns)
plt.yticks(np.arange(len(heatmap_data.index)),heatmap_data.index)
plt.xlabel("Radius")
plt.ylabel("Time (seconds)")
plt.title("Area Metric")
for i in range(heatmap_data.shape[0]):
    for j in range(heatmap_data.shape[1]):
        plt.text(j,i,f"{heatmap_data.values[i, j]:.2f}",ha="center",va="center",color="white")
plt.tight_layout()
plt.savefig("figures/AreaMetric_HeatMap.png", dpi=300)
plt.show()


#####################Signed errors 
signed_rows = []
for target_time in target_times[1:]:
    for loc in ["R0", "R1", "R2", "R4"]:
        # Simulation values
        sim_values = []
        for df in sim_data:
            time = df["Time [s]"].to_numpy()
            T = df[sim_cols[loc]].to_numpy()
            T0 = T[0]
            dT = T - T0
            idx = np.argmin(np.abs(time - target_time))
            sim_values.append(dT[idx])
        # Experiment values
        exp_values = []
        exp_time = tempE["Time"].to_numpy()

        for col in exp_cols[loc]:
            T = tempE[col].to_numpy()
            idx0 = np.argmin(np.abs(exp_time - heating_start_exp))
            T0 = T[idx0]
            exp_target_time = heating_start_exp + target_time
            idx = np.argmin(np.abs(exp_time - exp_target_time))
            dT = T[idx] - T0
            exp_values.append(dT)
        # print(f'exp_values = {exp_values}')
        area_signed = area_metric_ecdf_signed(exp_values, sim_values, normalize=False)
        signed_rows.append({"Time": target_time,"Radius": loc,"Signed Area Metric": area_signed})
signed_df = pd.DataFrame(signed_rows)
signed_heatmap = signed_df.pivot(index="Time",columns="Radius",values="Signed Area Metric")
signed_heatmap = signed_heatmap.rename(columns={"R0": "0mm", "R1": "1mm", "R2": "2mm", "R4": "4mm"})
vmax = np.nanmax(np.abs(signed_heatmap.values))

plt.figure(figsize=(10, 8))
im = plt.imshow(signed_heatmap.values,aspect="auto",origin="upper",cmap="coolwarm",vmin=-vmax,vmax=vmax)
cbar = plt.colorbar(im)
# cbar.set_label("Normalized Area Metric (%)", rotation=90, labelpad=15)
# cbar.set_label("Area Metric (%)", rotation=90, labelpad=15)
cbar.set_label(r"Area Metric ($^{\circ}$C)",rotation=90,labelpad=15)
plt.xticks(np.arange(len(signed_heatmap.columns)),signed_heatmap.columns)
plt.yticks(np.arange(len(signed_heatmap.index)),signed_heatmap.index)
plt.xlabel("Radius")
plt.ylabel("Time (seconds)")
plt.title("Area Metric")
for i in range(signed_heatmap.shape[0]):
    for j in range(signed_heatmap.shape[1]):
        val = signed_heatmap.values[i, j]
        plt.text(j,i,f"{val:.1f}",ha="center",va="center",color="black")
plt.tight_layout()
# plt.savefig("figures/AreaMetric_HeatMap_normalizedAvgExpPercent_signed.png", dpi=300)
plt.savefig("figures/AreaMetric_HeatMap_signed.png", dpi=300)
plt.show()

# -------

def plot_intervals_by_radius(summary, low_col, high_col, title, figname, metrics_to_plot):

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(metrics_to_plot))
    offset = 0.12

    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    exp_color = default_colors[1]
    sim_color = default_colors[0]

    for source, dx, color, label in [
        ("Experiment", -offset, exp_color, "Experiment"),
        ("Simulation",  offset, sim_color, "Model")
    ]:

        means = []
        lows = []
        highs = []

        for metric in metrics_to_plot:
            row = summary[
                (summary["Metric"] == metric) &
                (summary["Source"] == source)
            ].iloc[0]

            means.append(row["Mean"])
            lows.append(row[low_col])
            highs.append(row[high_col])

        means = np.asarray(means)
        lows = np.asarray(lows)
        highs = np.asarray(highs)

        yerr = np.vstack([
            means - lows,
            highs - means
        ])

        ax.errorbar(
            x + dx,
            means,
            yerr=yerr,
            fmt="o",
            markersize=8,
            capsize=8,
            elinewidth=3,
            capthick=3,
            linewidth=2,
            color=color,
            label=label
        )
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_to_plot)

    ax.set_xlabel("Radial position")
    ax.set_ylabel(r"Temperature rise, $\Delta T$ (K)")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(figname, dpi=300)
    plt.show()
    plt.close()


for t in target_times:
    summary_t = summary_temp[summary_temp["Time"] == t]
    plot_intervals_by_radius(
        summary_t,
        low_col="CI Low",
        high_col="CI High",
        title=f"95% Confidence Intervals at {t} s",
        figname=f"figures/confidence_interval_temperature_rise_{t}s_all_radius.jpg",
        metrics_to_plot=["R0", "R1", "R2", "R4"]
    )
        
    plot_intervals_by_radius(
        summary_t,
        low_col="TI Low",
        high_col="TI High",
        title=f"95/95 Tolerance Intervals at {t} s",
        figname=f"figures/tolerance_interval_temperature_rise_{t}s_all_radius.jpg",
        metrics_to_plot=["R0", "R1", "R2", "R4"]
    )
