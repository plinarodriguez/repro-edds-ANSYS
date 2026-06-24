from matplotlib import pyplot
import pandas as pd
import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
from experiments import * 
from simulations import *

# Set the font family and size to use for Matplotlib figures.
pyplot.rcParams['font.family'] = 'serif'
pyplot.rcParams['font.size'] = 20

# Plot function to repeat
def sampleplotslabels(point,tempS1,tempS2,tempS3,tempS4,tempS5,tempS6,tempS7,tempS8,tempS9,tempS10,tempSLab):
    samples = [tempS1,tempS2,tempS3,tempS4,tempS5,tempS6,tempS7,tempS8,tempS9,tempS10]
    nameSample = ['Sample 1','Sample 2','Sample 3','Sample 4','Sample 5','Sample 6','Sample 7','Sample 8','Sample 9','Sample 10']
    i = 0
    for sample in samples:
        pyplot.plot(sample[tempSLab[0]], sample[point]-298.152039, label=nameSample[i],linewidth=3)
        i+=1

# Plots Simulation vs. Experiments - by Point Location - Diamter ONLY
def sim_exp_pointlabels(timeE,avg,u,l,tempS1,tempS2,tempS3,tempS4,tempS5,
                  tempS6,tempS7,tempS8,tempS9,tempS10,
                  tempSLab,r,point,save,plotexp):
    pyplot.figure(figsize=(10,8)) #(6.4,4.8))
    pyplot.xlabel('time [s]')
    pyplot.ylabel('Temperature Rise')
    sampleplotslabels(point,tempS1,tempS2,tempS3,tempS4,tempS5,tempS6,tempS7,tempS8,tempS9,tempS10,tempSLab)
    if plotexp == 'YES':
        pyplot.plot(timeE[75:150]-19.650, avg,linestyle='-',label="Experiment",color="black")
        pyplot.fill_between(timeE[75:150]-19.650,u,l, color="lightgray")
    pyplot.grid()
    pyplot.xlim([-0.5,12])
    pyplot.title(f'Model Samples at R{str(r)}')
#     pyplot.title('r='+ str(r)+'mm');
    # pyplot.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0);
    pyplot.legend(loc='lower right') #, borderaxespad=0);
    if save == 'YES' :
        if plotexp == 'YES':
            pyplot.savefig('figures/simulation_exp_radius%imm.png'%r, dpi=300, bbox_inches='tight') 
        else: 
            pyplot.savefig('figures/simulation_radius%imm.png'%r, dpi=300, bbox_inches='tight')
    pyplot.close()


# Point 0
r, point0 = 0, 'Monitor Point: Mouthpiece1mm0mmTempX0YZ (Temperature) [K]' # radius for label, dataframe center labelsave = 'No' # 'YES'
save = 'YES' # 'YES'
plotexp = 'No'
sim_exp_pointlabels( timeE, avg0, u0, l0, tempS1, tempS2, tempS3, tempS4, tempS5, tempS6,
               tempS7, tempS8, tempS9, tempS10, tempSLab,r,point0,save,plotexp)

# Point 1
r, point1 = 1, 'Monitor Point: Mouthpiece1mm1mmTempX0YZ (Temperature) [K]' # radius for label, dataframe center labelsave = 'No' # 'YES'
save = 'YES' # 'YES'
plotexp = 'No'
sim_exp_pointlabels( timeE, avg1, u1, l1, tempS1, tempS2, tempS3, tempS4, tempS5, tempS6,
               tempS7, tempS8, tempS9, tempS10, tempSLab,r,point1,save,plotexp)

# Point 2
r, point2 = 2, 'Monitor Point: Mouthpiece1mm2mmTempX0YZ (Temperature) [K]' # radius for label, dataframe center labelsave = 'No' # 'YES'
save = 'YES' # 'YES'
plotexp = 'No'
sim_exp_pointlabels( timeE, avg2, u2, l2, tempS1, tempS2, tempS3, tempS4, tempS5, tempS6,
               tempS7, tempS8, tempS9, tempS10, tempSLab,r,point2,save,plotexp)

# Point 3
r, point3 = 3, 'Monitor Point: Mouthpiece1mm3mmTempX0YZ (Temperature) [K]' # radius for label, dataframe center labelsave = 'No' # 'YES'
save = 'YES' # 'YES'
plotexp = 'NO' # This data doesn't exist
sim_exp_pointlabels( timeE, avg1, u1, l1, tempS1, tempS2, tempS3, tempS4, tempS5, tempS6,
               tempS7, tempS8, tempS9, tempS10, tempSLab,r,point3,save,plotexp)

# Point 4
r, point4 = 4, 'Monitor Point: Mouthpiece1mm4mmTempX0YZ (Temperature) [K]' # radius for label, dataframe center labelsave = 'No' # 'YES'
save = 'YES' # 'YES'
plotexp = 'No'
sim_exp_pointlabels( timeE, avg4, u4, l4, tempS1, tempS2, tempS3, tempS4, tempS5, tempS6,
               tempS7, tempS8, tempS9, tempS10, tempSLab,r,point4,save,plotexp)


# -------------------------- Combined Plot -------------------------------------------------
def samples10stas(points,tempS1,tempS2,tempS3,tempS4,tempS5,tempS6,tempS7,tempS8,tempS9,tempS10):
    samples = [tempS1,tempS2,tempS3,tempS4,tempS5,tempS6,tempS7,tempS8,tempS9,tempS10]
    i=0;
    meanSamples = []
    stdSamples = []
    while i < len(tempS1['Time [s]']):
        samplesTemp = []
        for point in points:
            f=0
            while f < len(samples):
                samplesTemp.append((samples[f][point])[i]-298.152039)
                f+=1
        meanSamples.append(np.mean(samplesTemp))
        stdSamples.append(np.std(samplesTemp))
        i+=1
    return(meanSamples,stdSamples)

points = ['Monitor Point: Mouthpiece1mm0mmTempXdownYZ (Temperature) [K]','Monitor Point: Mouthpiece1mm0mmTempXupYZ (Temperature) [K]','Monitor Point: Mouthpiece1mm0mmTempX0YZ (Temperature) [K]','Monitor Point: Mouthpiece1mm0mmTempX0YupZ (Temperature) [K]','Monitor Point: Mouthpiece1mm0mmTempX0YdownZ (Temperature) [K]','Monitor Point: MouthpieceUp0mmTempzXdownYZ (Temperature) [K]','Monitor Point: MouthpieceUp0mmTempzXupYZup (Temperature) [K]','Monitor Point: MouthpieceUp0mmTempzX0YZup (Temperature) [K]','Monitor Point: MouthpieceUp0mmTempzX0YupZup (Temperature) [K]','Monitor Point: MouthpieceUp0mmTempzX0YdownZup (Temperature) [K]','Monitor Point: MouthpieceDown0mmTempzXdownYZ (Temperature) [K]','Monitor Point: MouthpieceDown0mmTempzXupYZdown (Temperature) [K]','Monitor Point: MouthpieceDown0mmTempzX0YZdown (Temperature) [K]','Monitor Point: MouthpieceDown0mmTempzX0YupZdown (Temperature) [K]','Monitor Point: MouthpieceDown0mmTempzX0YdownZdown (Temperature) [K]']
meanSamplesP0, stdSamplesP0 = samples10stas(points, tempS1, tempS2, tempS3, tempS4, tempS5, tempS6, tempS7, tempS8, tempS9, tempS10)

points = ['Monitor Point: Mouthpiece1mm1mmTempXdownYZ (Temperature) [K]','Monitor Point: Mouthpiece1mm1mmTempXupYZ (Temperature) [K]','Monitor Point: Mouthpiece1mm1mmTempX0YZ (Temperature) [K]','Monitor Point: Mouthpiece1mm1mmTempX0YupZ (Temperature) [K]','Monitor Point: Mouthpiece1mm1mmTempX0YdownZ (Temperature) [K]','Monitor Point: MouthpieceUp1mmTempzXdownZ (Temperature) [K]','Monitor Point: MouthpieceUp1mmTempzXupYZup (Temperature) [K]','Monitor Point: MouthpieceUp1mmTempzX0YZup (Temperature) [K]','Monitor Point: MouthpieceUp1mmTempzX0YupZup (Temperature) [K]','Monitor Point: MouthpieceUp1mmTempzX0YdownZup (Temperature) [K]','Monitor Point: MouthpieceDown1mmTempzXdownYZ (Temperature) [K]','Monitor Point: MouthpieceDown1mmTempzXupYZdown (Temperature) [K]','Monitor Point: MouthpieceDown1mmTempzX0YZdown (Temperature) [K]','Monitor Point: MouthpieceDown1mmTempzX0YupZdown (Temperature) [K]','Monitor Point: MouthpieceDown1mmTempzX0YdownZdown (Temperature) [K]']
meanSamplesP1, stdSamplesP1 = samples10stas(points, tempS1, tempS2, tempS3, tempS4, tempS5, tempS6, tempS7, tempS8, tempS9, tempS10)

points = ['Monitor Point: Mouthpiece1mm2mmTempXdownYZ (Temperature) [K]','Monitor Point: Mouthpiece1mm2mmTempXupYZ (Temperature) [K]','Monitor Point: Mouthpiece1mm2mmTempX0YZ (Temperature) [K]','Monitor Point: Mouthpiece1mm2mmTempX0YupZ (Temperature) [K]','Monitor Point: Mouthpiece1mm2mmTempX0YdownZ (Temperature) [K]','Monitor Point: MouthpieceUp2mmTempzXdownYZ (Temperature) [K]','Monitor Point: MouthpieceUp2mmTempzXupYZup (Temperature) [K]','Monitor Point: MouthpieceUp2mmTempzX0YZup (Temperature) [K]','Monitor Point: MouthpieceUp2mmTempzX0YupZup (Temperature) [K]','Monitor Point: MouthpieceUp2mmTempzX0YdownZup (Temperature) [K]','Monitor Point: MouthpieceDown2mmTempzXdownYZ (Temperature) [K]','Monitor Point: MouthpieceDown2mmTempzXupYZdown (Temperature) [K]','Monitor Point: MouthpieceDown2mmTempzX0YZdown (Temperature) [K]','Monitor Point: MouthpieceDown2mmTempzX0YupZdown (Temperature) [K]','Monitor Point: MouthpieceDown2mmTempzX0YdownZdpwm (Temperature) [K]']
meanSamplesP2, stdSamplesP2 = samples10stas(points, tempS1, tempS2, tempS3, tempS4, tempS5, tempS6, tempS7, tempS8, tempS9, tempS10)

points = ['Monitor Point: Mouthpiece1mm3mmTempXdownYZ (Temperature) [K]','Monitor Point: Mouthpiece1mm3mmTempXupYZ (Temperature) [K]','Monitor Point: Mouthpiece1mm3mmTempX0YZ (Temperature) [K]','Monitor Point: Mouthpiece1mm3mmTempX0YupZ (Temperature) [K]','Monitor Point: Mouthpiece1mm3mmTempX0YdownZ (Temperature) [K]','Monitor Point: MouthpieceUp3mmTempzXdownYZ (Temperature) [K]','Monitor Point: MouthpieceUp3mmTempzXupYZup (Temperature) [K]','Monitor Point: MouthpieceUp3mmTempzX0YZup (Temperature) [K]','Monitor Point: MouthpieceUp3mmTempzX0YupZup (Temperature) [K]','Monitor Point: MouthpieceUp3mmTempzX0YdownZup (Temperature) [K]','Monitor Point: MouthpieceDown3mmTempzXdownYZ (Temperature) [K]','Monitor Point: MouthpieceDown3mmTempzXupYZdown (Temperature) [K]','Monitor Point: MouthpieceDown3mmTempzX0YZdown (Temperature) [K]','Monitor Point: MouthpieceDown3mmTempzX0YupZdown (Temperature) [K]','Monitor Point: MouthpieceDown3mmTempzX0YdownZdown (Temperature) [K]']
meanSamplesP3, stdSamplesP3 = samples10stas(points, tempS1, tempS2, tempS3, tempS4, tempS5, tempS6, tempS7, tempS8, tempS9, tempS10)

points = ['Monitor Point: Mouthpiece1mm4mmTempXdownYZ (Temperature) [K]','Monitor Point: Mouthpiece1mm4mmTempXupYZ (Temperature) [K]','Monitor Point: Mouthpiece1mm4mmTempX0YZ (Temperature) [K]','Monitor Point: Mouthpiece1mm4mmTempX0YupZ (Temperature) [K]','Monitor Point: Mouthpiece1mm4mmTempX0YdownZ (Temperature) [K]','Monitor Point: MouthpieceUp4mmTempXdownYZ (Temperature) [K]','Monitor Point: MouthpieceUp4mmTempXupYZup (Temperature) [K]','Monitor Point: MouthpieceUp4mmTempX0YZup (Temperature) [K]','Monitor Point: MouthpieceUp4mmTempX0YupZup (Temperature) [K]','Monitor Point: MouthpieceUp4mmTempX0YdownZup (Temperature) [K]','Monitor Point: MouthpieceDown4mmTempzXdownYZ (Temperature) [K]','Monitor Point: MouthpieceDown4mmTempzXupYZdown (Temperature) [K]','Monitor Point: MouthpieceDown4mmTempzX0YZdown (Temperature) [K]','Monitor Point: MouthpieceDown4mmTempzX0YupZdown (Temperature) [K]','Monitor Point: MouthpieceDown4mmTempzX0YdownZdown (Temperature) [K]']
meanSamplesP4, stdSamplesP4 = samples10stas(points, tempS1, tempS2, tempS3, tempS4, tempS5, tempS6, tempS7, tempS8, tempS9, tempS10)

# Plot Simulation and Experiment with Uncertainties for All Point Locations
def sim_exp_uncertainties_ALL(timeE,avg0,u0,l0,avg1,u1,l1,avg2,u2,l2,avg4,u4,l4,
                              tempS1,meanSamplesP0,stdSamplesP0,
                              meanSamplesP1,stdSamplesP1,
                              meanSamplesP2,stdSamplesP2,
                              meanSamplesP3,stdSamplesP3,
                              meanSamplesP4,stdSamplesP4,
                              save,plotexp,plotSradius):
    pyplot.figure(figsize=(10,8)) 
    pyplot.title('Model Statistics')
    if 0 in plotSradius:
        pyplot.plot(tempS1['Time [s]'],meanSamplesP0, color='magenta',linewidth='3')
        pyplot.fill_between(tempS1['Time [s]'],meanSamplesP0+1*np.max(stdSamplesP0),meanSamplesP0-1*np.max(stdSamplesP0), color='purple',alpha=1.0, label='R0 ')
    if 1 in plotSradius:
        pyplot.plot(tempS1['Time [s]'],meanSamplesP1, color='lightgreen',linewidth='3')
        pyplot.fill_between(tempS1['Time [s]'],meanSamplesP1+1*np.max(stdSamplesP1),meanSamplesP1-1*np.max(stdSamplesP1), color='green',alpha=0.6, label='R1')
    if 2 in plotSradius:
        pyplot.plot(tempS1['Time [s]'],meanSamplesP2, color='yellow',linewidth='3')
        pyplot.fill_between(tempS1['Time [s]'],meanSamplesP2+1*np.max(stdSamplesP2),meanSamplesP2-1*np.max(stdSamplesP2), color='orange',alpha=0.6, label='R2')
    if 3 in plotSradius:
        pyplot.plot(tempS1['Time [s]'],meanSamplesP3, color='white',linewidth='3')
        pyplot.fill_between(tempS1['Time [s]'],meanSamplesP3+1*np.max(stdSamplesP3),meanSamplesP3-1*np.max(stdSamplesP3), color='black',alpha=0.6, label='R3')
    if 4 in plotSradius:
        pyplot.plot(tempS1['Time [s]'],meanSamplesP4, color='cyan',linewidth='3')
        pyplot.fill_between(tempS1['Time [s]'],meanSamplesP4+1*np.max(stdSamplesP4),meanSamplesP4-1*np.max(stdSamplesP4), color='cadetblue',alpha=0.6, label='R4')

    if plotexp == 'YES':
        pyplot.title('Simulations & Experiments')
        if 0 in plotSradius:
            pyplot.plot(timeE[75:150]-19.65, avg0,linestyle='-',label="R0",color="darkblue",linewidth='3')
            pyplot.fill_between(timeE[75:150]-19.65,u0, l0, color="lightblue",alpha=0.5)
        if 1 in plotSradius:
            pyplot.plot(timeE[75:150]-19.65, avg1,linestyle='-',label="R1",color="black",linewidth='3')
            pyplot.fill_between(timeE[75:150]-19.65,u1, l1, color="lightgray",alpha=0.75)
        if 2 in plotSradius:
            pyplot.plot(timeE[75:150]-19.65, avg2,linestyle='-',label="R2",color="red",linewidth='3')
            pyplot.fill_between(timeE[75:150]-19.65,u2, l2, color="pink",alpha=0.6)
        if 4 in plotSradius:
            pyplot.plot(timeE[75:150]-19.65, avg4,linestyle='-',label="R4",color="indigo",linewidth='3')
            pyplot.fill_between(timeE[75:150]-19.65,u4, l4, color="blueviolet",alpha=0.25)
    pyplot.grid()
    pyplot.xlabel('Time [s]')
    pyplot.ylabel(r'Temperature Rise, $\Delta T$')
    pyplot.legend(loc='upper left');
    pyplot.xlim(-0.1,12);
    if save == 'YES' :     
        pyplot.savefig('figures/sim_exp_uncert_points0-4.png', dpi=300,bbox_inches='tight')
    pyplot.close()

save = 'YES' # 'YES'
plotexp = 'No' # 'YES'
plotSradius = [0,1,2,4] # [0,1,2,3,4] Point locations you want in the plot
sim_exp_uncertainties_ALL( timeE, avg0, u0, l0, avg1, u1, l1, avg2, u2, l2, avg4, u4, l4,
                           tempS1,meanSamplesP0,stdSamplesP0,
                          meanSamplesP1,stdSamplesP1,meanSamplesP2,stdSamplesP2,
                          meanSamplesP3,stdSamplesP3,meanSamplesP4,stdSamplesP4,
                          save,plotexp,plotSradius)


#---------------------------- Area Metric ECDF Plots ----------------------------------------

# Data for all analysis based on timestep 
def SimSamplesTempbyTime(tempS1,tempS2,tempS3,tempS4,tempS5,tempS6,tempS7,tempS8,tempS9,tempS10,
                         timesId):
    samples = [tempS1,tempS2,tempS3,tempS4,tempS5,tempS6,tempS7,tempS8,tempS9,tempS10]
    points = ['Monitor Point: Mouthpiece1mm0mmTempX0YZ (Temperature) [K]','Monitor Point: Mouthpiece1mm1mmTempX0YZ (Temperature) [K]',
              'Monitor Point: Mouthpiece1mm2mmTempX0YZ (Temperature) [K]', 'Monitor Point: Mouthpiece1mm3mmTempX0YZ (Temperature) [K]',
              'Monitor Point: Mouthpiece1mm4mmTempX0YZ (Temperature) [K]']
    # time 0,2,5,10,12 
    times = [timesId[0],timesId[2],timesId[4],timesId[5],timesId[6],timesId[7]]
    
    # point0 = [[time0.allsamples], time1, ]
    tbypoint = []
    p,t = 0,0
    for point in points: 
        tbytime = []
        for time in times:
            tbysample = []
            for sample in samples:
                value = sample[point][time]-298.152039
                tbysample.append(value)
            tbytime.append(tbysample)
        tbypoint.append(tbytime)
        p+=1   
    return(tbypoint)

timesteps =  [0.100000001,1.0,2.0,4.0,5.0,10.0,11.0,12.0]
headers = tempS1.columns
times = pd.DataFrame([tempS1[headers[0]],tempS2[headers[0]],tempS3[headers[0]],tempS4[headers[0]],tempS5[headers[0]],tempS6[headers[0]],tempS7[headers[0]],tempS8[headers[0]],tempS9[headers[0]],tempS10[headers[0]]])                                                                         
time_list = [tempS1[headers[0]]]
time_list = list(time_list[0][0:])

### These are the means
timesId = [time_list.index(0.00999999978),time_list.index(timesteps[1]),time_list.index(timesteps[2]),
           time_list.index(timesteps[3]),time_list.index(timesteps[4]),time_list.index(timesteps[5]),
           time_list.index(timesteps[6]),time_list.index(timesteps[7])]


# Simulation data process 
avgTS_ts0 = [meanSamplesP0[timesId[0]],meanSamplesP1[timesId[0]],meanSamplesP2[timesId[0]],meanSamplesP4[timesId[0]]]
sdTS_ts0 = [stdSamplesP0[timesId[0]],stdSamplesP1[timesId[0]],stdSamplesP2[timesId[0]],stdSamplesP4[timesId[0]]]
# timestep =  2  
avgTS_ts2 = [meanSamplesP0[timesId[2]],meanSamplesP1[timesId[2]],meanSamplesP2[timesId[2]],meanSamplesP4[timesId[2]]]
sdTS_ts2 = [stdSamplesP0[timesId[2]],stdSamplesP1[timesId[2]],stdSamplesP2[timesId[2]],stdSamplesP4[timesId[2]]]
# timestep =  5  
avgTS_ts4 = [meanSamplesP0[timesId[4]],meanSamplesP1[timesId[4]],meanSamplesP2[timesId[4]],meanSamplesP4[timesId[4]]]
sdTS_ts4 = [stdSamplesP0[timesId[4]],stdSamplesP1[timesId[4]],stdSamplesP2[timesId[4]],stdSamplesP4[timesId[4]]]
# timestep =  10  
avgTS_ts5 = [meanSamplesP0[timesId[5]],meanSamplesP1[timesId[5]],meanSamplesP2[timesId[5]],meanSamplesP4[timesId[5]]]
sdTS_ts5 = [stdSamplesP0[timesId[5]],stdSamplesP1[timesId[5]],stdSamplesP2[timesId[5]],stdSamplesP4[timesId[5]]]
# timestep =  11  
avgTS_ts6 = [meanSamplesP0[timesId[6]],meanSamplesP1[timesId[6]],meanSamplesP2[timesId[6]],meanSamplesP4[timesId[6]]]
sdTS_ts6 = [stdSamplesP0[timesId[6]],stdSamplesP1[timesId[6]],stdSamplesP2[timesId[6]],stdSamplesP4[timesId[6]]]
# timestep =  12  
avgTS_ts7 = [meanSamplesP0[timesId[7]],meanSamplesP1[timesId[7]],meanSamplesP2[timesId[7]],meanSamplesP4[timesId[7]]]
sdTS_ts7 = [stdSamplesP0[timesId[7]],stdSamplesP1[timesId[7]],stdSamplesP2[timesId[7]],stdSamplesP4[timesId[7]]]

tbypoint = SimSamplesTempbyTime(tempS1,tempS2,tempS3,tempS4,tempS5,tempS6,tempS7,tempS8,tempS9,tempS10,timesId)

# Experiments data process 
exp0 = exp_bytime(avg0,avg4,sd0,sd4,tempET1diff,tempET2diff,tempET3diff,tempET4diff,exp_t = 'R0')
exp1 = exp_bytime(avg0,avg4,sd0,sd4,tempET1diff,tempET2diff,tempET3diff,tempET4diff,exp_t = 'R1')
exp2 = exp_bytime(avg0,avg4,sd0,sd4,tempET1diff,tempET2diff,tempET3diff,tempET4diff,exp_t = 'R2')
exp4 = exp_bytime(avg0,avg4,sd0,sd4,tempET1diff,tempET2diff,tempET3diff,tempET4diff,exp_t = 'R4')
nonexist = np.zeros_like(exp0)
exp = [exp0,exp1,exp2,nonexist.tolist(),exp4]

normSim = sim_bytime(avgTS_ts0,avgTS_ts2,avgTS_ts4,avgTS_ts5,avgTS_ts6,avgTS_ts7,sdTS_ts0,sdTS_ts2,sdTS_ts4,sdTS_ts5,sdTS_ts6,sdTS_ts7)


def area_metricPlots(tbypoint,timeEdiff,expS,timeE):
    radii = [0,1,2,4];
    for r in radii:
        # Plots
        # print('r',r)
        timesteps = [0.0,2.0,5.0,10.0,11.0,12.0]
        sim = [tbypoint[r][0],tbypoint[r][1],tbypoint[r][2],tbypoint[r][3],tbypoint[r][4],tbypoint[r][5]]
        exp = [expS[r][0],expS[r][1],expS[r][2],expS[r][3],expS[r][4],expS[r][5]]
        time_E = list(timeE[75:150]-20)
        s = 0 
        for t in timesteps:   
            # print('t',t)
            index = time_E.index(t) # find index at time = 5.0[s]    
            timestep=timeE[index+75]-20
            # pyplot.figure(figsize=(8,8)) 
            pyplot.title(f'R{r} at {timestep}s')
            pyplot.xlabel(rf'Temperature Rise, $\Delta T$') 
            pyplot.ylabel('Probability') 
            ecdfExp = ECDF(exp[s])
#             ecdfExpNorm = ECDF(normExp[s])
            ecdfExp.y[0] = 0
            ecdfExp.x[0] = ecdfExp.x[1]
#             ecdfSimNorm = ECDF(normSim[s]) 
            ecdfSim = ECDF(sim[s])
            pyplot.step(ecdfSim.x,ecdfSim.y,'-o', where='post',label='Model',linewidth=2)
            pyplot.step(ecdfExp.x,ecdfExp.y,'-o', where='post',label='Experiment',linewidth=2)
            if r in (0,4):
                pyplot.legend()
            pyplot.xlim(0, 3.1)
            pyplot.xticks(np.arange(0, 3.01, 0.5))  # start, stop, step
            pyplot.grid()
            times = int(timestep)
            file_name = f'figures/AreaMetric_ECDF_R{r}_Time{times}.png'
            pyplot.savefig(file_name, dpi=300, bbox_inches='tight')
            pyplot.show()
            pyplot.close()
            s+=1

area_metricPlots(tbypoint,timeEdiff,exp,timeEdiff)   


#---------------------------------------------------
# comparison plots with stats 
#----------------------------------------------------
def sim_exp_uncertainties(timeE,avg,u,l,tempS,meanSamplesP,stdSamplesP,r,save,plotexp):
    # pyplot.figure(figsize=(10,8))
    pyplot.title(f'R{r}');
    pyplot.xlabel('time [s]')
    pyplot.ylabel(r'Temperature Rise,$\Delta T$')
    pyplot.plot(tempS['Time [s]'],meanSamplesP,color='C0',linewidth='3', label='Model')
    pyplot.fill_between(tempS['Time [s]'],meanSamplesP+1*np.max(stdSamplesP),meanSamplesP-1*np.max(stdSamplesP), color='C0',alpha=0.25)
    if plotexp == 'YES':
        pyplot.plot(timeE[75:150]-19.650, avg,linestyle='-', label="Experiment",color="C1",linewidth='3')
        pyplot.fill_between(timeE[75:150]-19.650,u,l,color="C1",alpha=0.25)
    pyplot.grid()
    pyplot.legend(loc='upper left') #, facecolor='lightgray');
    pyplot.xlim(0,12);
    pyplot.yticks(np.arange(0, 3.5, 0.5))   
    pyplot.xticks(np.arange(0, 13, 2))   
    pyplot.ylim(-0.25,3.25);
    if save == 'YES' :          
        pyplot.savefig(f'figures/sim_exp_uncert_R{r}.png', dpi=300,bbox_inches='tight')
    pyplot.close()


points = ['Monitor Point: Mouthpiece1mm0mmTempXdownYZ (Temperature) [K]','Monitor Point: Mouthpiece1mm0mmTempXupYZ (Temperature) [K]','Monitor Point: Mouthpiece1mm0mmTempX0YZ (Temperature) [K]','Monitor Point: Mouthpiece1mm0mmTempX0YupZ (Temperature) [K]','Monitor Point: Mouthpiece1mm0mmTempX0YdownZ (Temperature) [K]','Monitor Point: MouthpieceUp0mmTempzXdownYZ (Temperature) [K]','Monitor Point: MouthpieceUp0mmTempzXupYZup (Temperature) [K]','Monitor Point: MouthpieceUp0mmTempzX0YZup (Temperature) [K]','Monitor Point: MouthpieceUp0mmTempzX0YupZup (Temperature) [K]','Monitor Point: MouthpieceUp0mmTempzX0YdownZup (Temperature) [K]','Monitor Point: MouthpieceDown0mmTempzXdownYZ (Temperature) [K]','Monitor Point: MouthpieceDown0mmTempzXupYZdown (Temperature) [K]','Monitor Point: MouthpieceDown0mmTempzX0YZdown (Temperature) [K]','Monitor Point: MouthpieceDown0mmTempzX0YupZdown (Temperature) [K]','Monitor Point: MouthpieceDown0mmTempzX0YdownZdown (Temperature) [K]']
meanSamplesP0, stdSamplesP0 = samples10stas(points,tempS1,tempS2,tempS3,tempS4,tempS5,tempS6,tempS7,tempS8,tempS9,tempS10)
r = 0 # radius for label
save = 'YES' # 'YES'
plotexp = 'YES' # 'YES'
sim_exp_uncertainties(timeE,avg0,u0,l0,tempS1,meanSamplesP0,stdSamplesP0,r,save,plotexp)

points = ['Monitor Point: Mouthpiece1mm1mmTempXdownYZ (Temperature) [K]','Monitor Point: Mouthpiece1mm1mmTempXupYZ (Temperature) [K]','Monitor Point: Mouthpiece1mm1mmTempX0YZ (Temperature) [K]','Monitor Point: Mouthpiece1mm1mmTempX0YupZ (Temperature) [K]','Monitor Point: Mouthpiece1mm1mmTempX0YdownZ (Temperature) [K]','Monitor Point: MouthpieceUp1mmTempzXdownZ (Temperature) [K]','Monitor Point: MouthpieceUp1mmTempzXupYZup (Temperature) [K]','Monitor Point: MouthpieceUp1mmTempzX0YZup (Temperature) [K]','Monitor Point: MouthpieceUp1mmTempzX0YupZup (Temperature) [K]','Monitor Point: MouthpieceUp1mmTempzX0YdownZup (Temperature) [K]','Monitor Point: MouthpieceDown1mmTempzXdownYZ (Temperature) [K]','Monitor Point: MouthpieceDown1mmTempzXupYZdown (Temperature) [K]','Monitor Point: MouthpieceDown1mmTempzX0YZdown (Temperature) [K]','Monitor Point: MouthpieceDown1mmTempzX0YupZdown (Temperature) [K]','Monitor Point: MouthpieceDown1mmTempzX0YdownZdown (Temperature) [K]']
meanSamplesP1, stdSamplesP1 = samples10stas(points,tempS1,tempS2,tempS3,tempS4,tempS5,tempS6,tempS7,tempS8,tempS9,tempS10)
r = 1 # radius for label
save = 'YES' # 'YES'
plotexp = 'YES' # 'YES'
sim_exp_uncertainties(timeE,avg1,u1,l1,tempS1,meanSamplesP1,stdSamplesP1,r,save,plotexp)

points = ['Monitor Point: Mouthpiece1mm2mmTempXdownYZ (Temperature) [K]','Monitor Point: Mouthpiece1mm2mmTempXupYZ (Temperature) [K]','Monitor Point: Mouthpiece1mm2mmTempX0YZ (Temperature) [K]','Monitor Point: Mouthpiece1mm2mmTempX0YupZ (Temperature) [K]','Monitor Point: Mouthpiece1mm2mmTempX0YdownZ (Temperature) [K]','Monitor Point: MouthpieceUp2mmTempzXdownYZ (Temperature) [K]','Monitor Point: MouthpieceUp2mmTempzXupYZup (Temperature) [K]','Monitor Point: MouthpieceUp2mmTempzX0YZup (Temperature) [K]','Monitor Point: MouthpieceUp2mmTempzX0YupZup (Temperature) [K]','Monitor Point: MouthpieceUp2mmTempzX0YdownZup (Temperature) [K]','Monitor Point: MouthpieceDown2mmTempzXdownYZ (Temperature) [K]','Monitor Point: MouthpieceDown2mmTempzXupYZdown (Temperature) [K]','Monitor Point: MouthpieceDown2mmTempzX0YZdown (Temperature) [K]','Monitor Point: MouthpieceDown2mmTempzX0YupZdown (Temperature) [K]','Monitor Point: MouthpieceDown2mmTempzX0YdownZdpwm (Temperature) [K]']
meanSamplesP2, stdSamplesP2 = samples10stas(points,tempS1,tempS2,tempS3,tempS4,tempS5,tempS6,tempS7,tempS8,tempS9,tempS10)
r = 2 # radius for label
save = 'YES' # 'YES'
plotexp = 'YES' # 'YES'
sim_exp_uncertainties(timeE,avg2,u2,l2,tempS1,meanSamplesP2,stdSamplesP2,r,save,plotexp)

points = ['Monitor Point: Mouthpiece1mm3mmTempXdownYZ (Temperature) [K]','Monitor Point: Mouthpiece1mm3mmTempXupYZ (Temperature) [K]','Monitor Point: Mouthpiece1mm3mmTempX0YZ (Temperature) [K]','Monitor Point: Mouthpiece1mm3mmTempX0YupZ (Temperature) [K]','Monitor Point: Mouthpiece1mm3mmTempX0YdownZ (Temperature) [K]','Monitor Point: MouthpieceUp3mmTempzXdownYZ (Temperature) [K]','Monitor Point: MouthpieceUp3mmTempzXupYZup (Temperature) [K]','Monitor Point: MouthpieceUp3mmTempzX0YZup (Temperature) [K]','Monitor Point: MouthpieceUp3mmTempzX0YupZup (Temperature) [K]','Monitor Point: MouthpieceUp3mmTempzX0YdownZup (Temperature) [K]','Monitor Point: MouthpieceDown3mmTempzXdownYZ (Temperature) [K]','Monitor Point: MouthpieceDown3mmTempzXupYZdown (Temperature) [K]','Monitor Point: MouthpieceDown3mmTempzX0YZdown (Temperature) [K]','Monitor Point: MouthpieceDown3mmTempzX0YupZdown (Temperature) [K]','Monitor Point: MouthpieceDown3mmTempzX0YdownZdown (Temperature) [K]']
meanSamplesP3, stdSamplesP3 = samples10stas(points,tempS1,tempS2,tempS3,tempS4,tempS5,tempS6,tempS7,tempS8,tempS9,tempS10)
r = 3 # radius for label
save = 'YES' # 'YES'
plotexp = 'NO' # 'YES'
sim_exp_uncertainties(timeE,avg1,u1,l1,tempS1,meanSamplesP3,stdSamplesP3,r,save,plotexp)

points = ['Monitor Point: Mouthpiece1mm4mmTempXdownYZ (Temperature) [K]','Monitor Point: Mouthpiece1mm4mmTempXupYZ (Temperature) [K]','Monitor Point: Mouthpiece1mm4mmTempX0YZ (Temperature) [K]','Monitor Point: Mouthpiece1mm4mmTempX0YupZ (Temperature) [K]','Monitor Point: Mouthpiece1mm4mmTempX0YdownZ (Temperature) [K]','Monitor Point: MouthpieceUp4mmTempXdownYZ (Temperature) [K]','Monitor Point: MouthpieceUp4mmTempXupYZup (Temperature) [K]','Monitor Point: MouthpieceUp4mmTempX0YZup (Temperature) [K]','Monitor Point: MouthpieceUp4mmTempX0YupZup (Temperature) [K]','Monitor Point: MouthpieceUp4mmTempX0YdownZup (Temperature) [K]','Monitor Point: MouthpieceDown4mmTempzXdownYZ (Temperature) [K]','Monitor Point: MouthpieceDown4mmTempzXupYZdown (Temperature) [K]','Monitor Point: MouthpieceDown4mmTempzX0YZdown (Temperature) [K]','Monitor Point: MouthpieceDown4mmTempzX0YupZdown (Temperature) [K]','Monitor Point: MouthpieceDown4mmTempzX0YdownZdown (Temperature) [K]']
meanSamplesP4, stdSamplesP4 = samples10stas(points,tempS1,tempS2,tempS3,tempS4,tempS5,tempS6,tempS7,tempS8,tempS9,tempS10)
r = 4 # radius for label
save = 'YES' # 'YES'
plotexp = 'YES' # 'YES'
sim_exp_uncertainties(timeE,avg4,u4,l4,tempS1,meanSamplesP4,stdSamplesP4,r,save,plotexp)


################# Experiments

# All Experimental Data
def exp_ALL_2plots(timeE,avg0,avg1,avg2,avg4,
                   u0,u1,u2,u4,
                   l0,l1,l2,l4,
                   tempET1,tempET2,tempET3,tempET4,save):
    pyplot.figure(figsize=(8.0,6.6)) #(10,8))
    pyplot.title('Experiments Statistics')
    pyplot.xlabel('Time [s]')
    pyplot.ylabel(r'Temperature Rise, $\Delta T$')
    pyplot.plot(timeE[75:150]-20, avg0,color='magenta',linewidth=3)
    # pyplot.plot(timeE[75:150]-20, u0,linestyle='-',color="gray")
    # pyplot.plot(timeE[75:150]-20, l0,linestyle='-',color="gray")
    pyplot.fill_between(timeE[75:150]-20,u0, l0, color='purple',alpha=1.0,label="R0")
    
    pyplot.plot(timeE[75:150]-20, avg1, color='lightgreen',linewidth='3')
    # pyplot.plot(timeE[75:150]-20, u1,linestyle='-',color="gray")
    # pyplot.plot(timeE[75:150]-20, l1,linestyle='-',color="gray")
    pyplot.fill_between(timeE[75:150]-20,u1, l1, color='green',alpha=0.6,label="R1")

    pyplot.plot(timeE[75:150]-20, avg2, color='yellow',linewidth='3')
    # pyplot.plot(timeE[75:150]-20, u2,linestyle='-',color="gray")
    # pyplot.plot(timeE[75:150]-20, l2,linestyle='-',color="gray")
    pyplot.fill_between(timeE[75:150]-20,u2, l2, color='orange',alpha=0.6,label="R2")
    
    pyplot.plot(timeE[75:150]-20, avg4,color='cyan',linewidth='3')
    # pyplot.plot(timeE[75:150]-20, u4,linestyle='-',color="gray")
    # pyplot.plot(timeE[75:150]-20, l4,linestyle='-',color="gray")
    pyplot.fill_between(timeE[75:150]-20,u4, l4, color='cadetblue',label='R4')
    pyplot.legend()
    pyplot.grid()
    pyplot.ylim(-0.1,3.0)
    pyplot.xlim(-0.1,15)
    if save == 'YES' :
        pyplot.savefig('figures/experiments_RAll_stats.png', dpi=300, bbox_inches='tight');  
    pyplot.close()
    
    
    pyplot.figure(figsize=(10,8))
    pyplot.title('Experiments Samples')
    pyplot.xlabel('Time [s]')
    pyplot.ylabel(r'Temperature Rise, $\Delta T$')
    pyplot.scatter(timeE[75:150]-20, np.abs(tempET2[0])[75:150],label="Sample 1", color='royalblue')
    pyplot.scatter(timeE[75:150]-20, np.abs(tempET2[1])[75:150],label="Sample 2", color='darkorange')
    pyplot.scatter(timeE[75:150]-20, np.abs(tempET2[2])[75:150],label="Sample 3", color='mediumseagreen')
    pyplot.plot(timeE[75:150]-20, avg0,color='magenta',linewidth='3')

    pyplot.scatter(timeE[75:150]-20, np.abs(tempET3[0])[75:150], color='royalblue')
    pyplot.scatter(timeE[75:150]-20, np.abs(tempET3[1])[75:150], color='darkorange')
    pyplot.scatter(timeE[75:150]-20, np.abs(tempET3[2])[75:150], color='mediumseagreen')
    pyplot.plot(timeE[75:150]-20, avg1, color='lightgreen',linewidth='3')

    pyplot.scatter(timeE[75:150]-20, np.abs(tempET1[0])[75:150], color='royalblue')
    pyplot.scatter(timeE[75:150]-20, np.abs(tempET1[1])[75:150], color='darkorange')
    pyplot.scatter(timeE[75:150]-20, np.abs(tempET1[2])[75:150], color='mediumseagreen')
    pyplot.plot(timeE[75:150]-20, avg2, color='yellow',linewidth='3')

    pyplot.scatter(timeE[75:150]-20, np.abs(tempET4[0])[75:150], color='royalblue')
    pyplot.scatter(timeE[75:150]-20, np.abs(tempET4[1])[75:150], color='darkorange')
    pyplot.scatter(timeE[75:150]-20, np.abs(tempET4[2])[75:150], color='mediumseagreen')
    pyplot.plot(timeE[75:150]-20, avg4, color='cyan',linewidth='3')

    pyplot.legend();
    pyplot.grid();
    pyplot.ylim(-0.1,3.0)
    pyplot.xlim(-0.1,15);
    if save == 'YES' :
        pyplot.savefig('figures/experiments_RAll_samples.png', dpi=300, bbox_inches='tight');  
    pyplot.close()
        
save = 'YES' #'NO'
exp_ALL_2plots(timeE,avg0,avg1,avg2,avg4,u0,u1,u2,u4,l0,l1,l2,l4,tempET1diff,tempET2diff,tempET3diff,tempET4diff,save)