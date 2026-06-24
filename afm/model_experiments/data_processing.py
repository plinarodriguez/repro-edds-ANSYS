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

# SIMULATION NOMINAL
# --------------- 
# Nominal Model for low inlet velocity 
nominal = pd.read_csv("data/simulation/nominal/S0_G0_V0_063.csv") 
center = nominal[' Y [ m ]'][0] + 0.008 # Scale to center at 0
# -------------------------------------------------------------------
# SIMULATION SAMPLING 
# --------------- 
# Simulation Samples
sim1 = pd.read_csv("data/simulation/samples/S1_G1_V0_063.csv") 
sim2 = pd.read_csv("data/simulation/samples/S2_G2_V0_063329.csv") 
sim3 = pd.read_csv("data/simulation/samples/S3_G3_V0_0642.csv") 
center1 = sim1[' Y [ m ]'][0] + 0.008 # Scale to center at 0
center2 = sim2[' Y [ m ]'][0] + 0.008 # Scale to center at 0
center3 = sim3[' Y [ m ]'][0] + 0.008 # Scale to center at 0

sim_data = [sim1,sim2,sim3]
# Calculate the Mean, Standard Deviation, and Standard Error for all samples
avgsim,sdsim = [],[]
n,i = 3,0
while i < len(sim1[' Y [ m ]']):
    e = np.array([sim1[' Velocity [ m s^-1 ]'][i],sim2[' Velocity [ m s^-1 ]'][i],sim3[' Velocity [ m s^-1 ]'][i]])
    avgsim.append(np.average(e))
    sdsim.append(np.std(e))
    i+=1
plussim = avgsim + 2*max(sdsim)/np.sqrt(n) # 2 standard deviations
minussim = avgsim - 2*max(sdsim)/np.sqrt(n) # 2 standard deviations

# -------------------------------------------------------------------
# EXPERIMENT SAMPLING
# --------------- 
expS1 = pd.read_csv("data/experiments/S1_Low_Exp_Outlet.csv") 
expS2 = pd.read_csv("data/experiments/S2_Low_Exp_Outlet.csv") 
expS3 = pd.read_csv("data/experiments/S3_Low_Exp_Outlet.csv") 
exp_data = [expS1,expS2,expS3]
n = 3  # Number of Samples

# Calculate the Mean,Standard Deviation, Standard Error
avgexp,sdexp = [],[]
expS3nonoise = np.array(expS3['V (m/s)'][2:])
i = 0
while i < len(expS1['x (mm)']):
    e = np.array([expS1['V (m/s)'][i],expS2['V (m/s)'][i],expS3nonoise[i]])
    avgexp.append(np.average(e))
    sdexp.append(np.std(e))
    i+=1
plus = avgexp + 1*max(sdexp)/np.sqrt(n)
minus = avgexp - 1*max(sdexp)/np.sqrt(n)

# -------------------------------------------------------------------
# Intervals 
# -------------------------------------------------------------------
def peak_avg(data, type, mirror_sim=True):
    if type == 'exp':
        x = data['x (mm)'].to_numpy()
        V = data['V (m/s)'].to_numpy()
    elif type == 'sim':
        y = data[' Y [ m ]'].to_numpy()
        V = data[' Velocity [ m s^-1 ]'].to_numpy()
        # experiment radius from diameter
        radius = 0.00550668 / 2  # m
        # use simulation peak as centerline
        center = y[np.argmax(V)]
        r = np.abs(y - center)
        # keep only same radius around centerline
        mask = r <= radius
        r = r[mask]
        V = V[mask]
        idx = np.argsort(r)
        r = r[idx]
        V = V[idx]
        if mirror_sim:
            x = np.concatenate((-r[:0:-1], r))
            V = np.concatenate((V[:0:-1], V))
        else:
            x = r
    peak = np.max(V)
    # Trapezoidal average since spacing is not uniform
    avg = np.trapezoid(V,x) /(x.max() - x.min())
    return peak,avg

peak_sim = []
avg_sim = []

for df in sim_data:
    peak, avg = peak_avg(df,'sim')
    peak_sim.append(peak)
    avg_sim.append(avg)

peak_exp = []
avg_exp = []

for df in exp_data:
    peak, avg = peak_avg(df,'exp')
    peak_exp.append(peak)
    avg_exp.append(avg)

### Confidence Intervals 95%
def confidence_interval(data, alpha=0.05):
    data = np.asarray(data)
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    tval = t.ppf(1-alpha/2, n-1)
    half_width = tval * std / np.sqrt(n)
    return mean, mean-half_width, mean+half_width

mean_peak_exp, ci_low_exp, ci_high_exp = confidence_interval(peak_exp)
# print(f'Confidence Interval 95% - Experiment')
# print(f'Mean Peak Experiments: {mean_peak_exp}')
# print(f'CI Low: {ci_low_exp},CI High: {ci_high_exp}')
# print('------------------------------------------')
mean_peak_sim, ci_low_sim, ci_high_sim = confidence_interval(peak_sim)
# print(f'Confidence Interval 95% - Simulation')
# print(f'Mean Peak Simulations: {mean_peak_sim}')
# print(f'CI Low: {ci_low_sim},CI High: {ci_high_sim}')


# 95/95 tolerance interval 
def tolerance_interval(data, k=4.94):
    # For n=3, a two-sided 95/95 TI has approximately k=4.94
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    return mean-k*std, mean+k*std

ti_low_exp, ti_high_exp = tolerance_interval(peak_exp)
# print(f'Tolerance Interval 95/95 - Experiment')
# print(f'TI Low: {ti_low_exp},TI High: {ti_high_exp}')
# print('---------------------------------------------')
ti_low_sim, ti_high_sim = tolerance_interval(peak_sim)
# print(f'Tolerance Interval 95/95 - Simulation')
# print(f'TI Low: {ti_low_sim},TI High: {ti_high_sim}')

# Avg intervals
mean_avg_exp, ci_low_avg_exp, ci_high_avg_exp = confidence_interval(avg_exp)
mean_avg_sim, ci_low_avg_sim, ci_high_avg_sim = confidence_interval(avg_sim)
ti_low_avg_exp, ti_high_avg_exp = tolerance_interval(avg_exp)
ti_low_avg_sim, ti_high_avg_sim = tolerance_interval(avg_sim)

summary = pd.DataFrame({
    "Source": ["Exp", "Sim", "Exp", "Sim"],
    "Metric": ["Peak Velocity", "Peak Velocity", "Average Velocity", "Average Velocity"],
    "Mean": [mean_peak_exp,mean_peak_sim,mean_avg_exp,mean_avg_sim],
    "CI Low": [ci_low_exp,ci_low_sim,ci_low_avg_exp,ci_low_avg_sim],
    "CI High": [ci_high_exp,ci_high_sim,ci_high_avg_exp,ci_high_avg_sim],
    "TI Low": [ti_low_exp,ti_low_sim,ti_low_avg_exp,ti_low_avg_sim],
    "TI High": [ti_high_exp,ti_high_sim,ti_high_avg_exp,ti_high_avg_sim]
    })
summary.to_csv('ci_ti_sim_exp.csv', index=False)
# print(summary)

def plot_intervals_side_by_side(summary, low_col, high_col, title, figname):
    metrics = ["Peak Velocity", "Average Velocity"]
    fig, axes = plt.subplots(1, 2, figsize=(12,6), sharey=True)
    for ax, metric in zip(axes, metrics):
        tmp = summary[summary["Metric"] == metric].copy()
        tmp["Source"] = pd.Categorical(tmp["Source"],categories=["Sim","Exp"],ordered=True) #,categories=["Exp", "Sim"],ordered=True)
        tmp = tmp.sort_values("Source")
        # labels = tmp["Source"].to_numpy()
        labels=['Model','Experiment']
        x = np.array([0.0, 0.25])
        means = tmp["Mean"].to_numpy()
        lows = tmp[low_col].to_numpy()
        highs = tmp[high_col].to_numpy()
        yerr = np.vstack([means - lows,highs - means])
        # ax.errorbar(x,means,yerr=yerr,fmt="o",capsize=6,elinewidth=3,linewidth=2)
        default_colors = ["C0", "C1"]
        for i in range(len(x)):  # i=0 Exp, i=1 Model
            ax.errorbar(
                x[i], means[i],  
                yerr=[[yerr[0, i]], [yerr[1, i]]],  # keep asymmetric format (2x1)
                fmt="o", markersize=10,capsize=10, elinewidth=5, linewidth=5,
                color=default_colors[i]
            )
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlim(-0.25, 0.60)
        ax.set_ylim(0.4, 1.20)
        ax.set_title(metric)
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("Velocity (m/s)")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(figname, dpi=300)
    plt.show()
    plt.close()

plot_intervals_side_by_side(summary,low_col="CI Low",high_col="CI High",title="95% Confidence Intervals",figname='figures/confidence_interval_expvssim.jpg')
plot_intervals_side_by_side(summary,low_col="TI Low",high_col="TI High",title="95/95 Tolerance Intervals",figname='figures/tolerance_interval_expvssim.jpg')


# ---------------------------------------
# Area Metric 
# ---------------------------------------

def plot_peak_ecdf_area_metric(peak_exp, peak_sim, figname):
    ecdf_exp = ECDF(peak_exp)
    ecdf_sim = ECDF(peak_sim)
    x = np.linspace(min(min(peak_exp), min(peak_sim)),max(max(peak_exp), max(peak_sim)),1000)
    F_exp = ecdf_exp(x)
    F_sim = ecdf_sim(x)
    area_metric = np.trapezoid(np.abs(F_exp - F_sim),x)
    area_metric_norm = area_metric / np.mean(F_exp) #(x.max() - x.min())
    plt.figure(figsize=(8.0,6.6)) #12,4))
    # Start ECDFs at 0
    x_exp = np.sort(peak_exp)
    y_exp = np.arange(1, len(x_exp)+1) / len(x_exp)
    x_sim = np.sort(peak_sim)
    y_sim = np.arange(1, len(x_sim)+1) / len(x_sim)
    plt.step(np.concatenate(([x_sim[0]], x_sim)),np.concatenate(([0], y_sim)),'--o',linewidth='4',markersize='8',where="post",color='C0',label="Model")
    plt.step(np.concatenate(([x_exp[0]], x_exp)),np.concatenate(([0], y_exp)),'--o',linewidth='4',markersize='8',where="post",color='C1',label="Experiment")
    # Show data points
    # plt.plot(x_exp,y_exp,'o',markersize=10,markeredgewidth=2,label='_nolegend_')
    # plt.plot(x_sim,y_sim,'o',markersize=10,markeredgewidth=2,label='_nolegend_')
    plt.xlabel("Peak velocity (m/s)")
    plt.ylabel("Probability (CDF)")
    plt.ylim([-0.02, 1.05])
    plt.xlim([0.6,1.4]) #[0.8,1.2])
    plt.title(f"Peak Velocity ECDFs\n"f"Area Metric = {area_metric:.4f} m/s ({100*area_metric_norm:.0f}%)")   
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(figname, dpi=300, bbox_inches='tight')
    plt.show()

    return area_metric

area_peak = plot_peak_ecdf_area_metric(peak_exp,peak_sim,"figures/ecdf_peak_velocity.jpg")

print("Peak velocity area metric =", area_peak)

#################### Validation Plots 
plt.figure(figsize=(16,8))
plt.xlabel('Radial Position [mm]')
plt.ylabel('Velocity [m/s]')
plt.title('Model vs. Experiments')
plt.plot(expS2['x (mm)']-5.39655,avgexp, '-k',color="C1",linewidth='5')
# plt.plot(expS2['x (mm)']-5.39655, plus,linestyle='-',color="gray")
# plt.plot(expS2['x (mm)']-5.39655, minus,linestyle='-',color="gray")
plt.xlim(-3,3)
plt.ylim(0,1.2)
plt.fill_between(expS2['x (mm)']-5.39655,plus, minus,label="Experiment",color="C1",alpha=0.25)
plt.plot((sim1[' Y [ m ]']-center1)*1e3, avgsim,label='Model',color='steelblue',linewidth='6')
# plt.plot((sim1[' Y [ m ]']-center1)*1e3, avgsim,'--',label='Model',color='lightskyblue',linewidth='5')
plt.legend()
plt.grid()
plt.savefig('figures/hydrodynamic_validation.png', dpi=300, bbox_inches='tight')