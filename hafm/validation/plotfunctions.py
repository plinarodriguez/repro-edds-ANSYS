from matplotlib import pyplot
import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF
import scipy.integrate as integrate
# Power Profiles and SRQ locations
def powerprof(dataPowerS1,dataPowerS2,dataPowerS3,dataPowerS4,dataPowerS5,dataPowerS6,dataPowerS7,dataPowerS8,dataPowerS9,dataPowerS10, save):
    pyplot.figure(figsize=(14,8))
    pyplot.title('Applied Power Profiles')
    pyplot.xlabel('Time [s]')
    pyplot.ylabel('Power [W]')
    pyplot.plot(dataPowerS1['t[s]'], dataPowerS1['Power[W m^-3]']/((1/13.1533)*(1e9)),label='Sample1') 
    pyplot.plot(dataPowerS2['t[s]'], dataPowerS2['Power[W m^-3]']/((1/13.1533)*(1e9)),label='Sample2') 
    pyplot.plot(dataPowerS3['t[s]'], dataPowerS3['Power[W m^-3]']/((1/13.1533)*(1e9)),label='Sample3') 
    pyplot.plot(dataPowerS4['t[s]'], dataPowerS4['Power[W m^-3]']/((1/13.1533)*(1e9)),label='Sample4') 
    pyplot.plot(dataPowerS5['t[s]'], dataPowerS5['Power[W m^-3]']/((1/13.1533)*(1e9)),label='Sample5')
    pyplot.plot(dataPowerS6['t[s]'], dataPowerS6['Power[W m^-3]']/((1/13.1533)*(1e9)),label='Sample6') 
    pyplot.plot(dataPowerS7['t[s]'], dataPowerS7['Power[W m^-3]']/((1/13.1533)*(1e9)),label='Sample7') 
    pyplot.plot(dataPowerS8['t[s]'], dataPowerS8['Power[W m^-3]']/((1/13.1533)*(1e9)),label='Sample8') 
    pyplot.plot(dataPowerS9['t[s]'], dataPowerS9['Power[W m^-3]']/((1/13.1533)*(1e9)),label='Sample9') 
    pyplot.plot(dataPowerS10['t[s]'], dataPowerS10['Power[W m^-3]']/((1/13.1533)*(1e9)),label='Sample10') 
    pyplot.scatter([0,2,5,10,11,12],[0,1,1,1,0.5,0], marker='x', color='black',s=300, lw=5)
    pyplot.grid()
    pyplot.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    if save == 'YES' :
        pyplot.savefig('figures/appliedPowerProfiles_samples10.png', dpi=300, bbox_inches='tight');
   
# Plot function to repeat
def sampleplotslabels(point,tempS1,tempS2,tempS3,tempS4,tempS5,tempS6,tempS7,tempS8,tempS9,tempS10,tempSLab):
    samples = [tempS1,tempS2,tempS3,tempS4,tempS5,tempS6,tempS7,tempS8,tempS9,tempS10]
    nameSample = ['Sample1','Sample2','Sample3','Sample4','Sample5','Sample6','Sample7','Sample8','Sample9','Sample10']
    i = 0
    for sample in samples:
        pyplot.plot(sample[tempSLab[0]], sample[point]-298.152039, label=nameSample[i])
        i+=1
    
# Plots Simulation vs. Experiments - by Point Location - Diamter ONLY
def sim_exp_pointlabels(timeE,avg,u,l,tempS1,tempS2,tempS3,tempS4,tempS5,
                  tempS6,tempS7,tempS8,tempS9,tempS10,
                  tempSLab,r,point,save,plotexp):
    pyplot.figure(figsize=(12,8))
    pyplot.xlabel('time [s]')
    pyplot.ylabel('Temperature Rise')
    sampleplotslabels(point,tempS1,tempS2,tempS3,tempS4,tempS5,tempS6,tempS7,tempS8,tempS9,tempS10,tempSLab)
    if plotexp == 'YES':
        pyplot.plot(timeE[75:150]-19.650, avg,linestyle='-',label="Experiment",color="black")
        pyplot.fill_between(timeE[75:150]-19.650,u,l, color="lightgray")
    pyplot.grid()
    pyplot.xlim([-0.5,12])
    pyplot.title('Simulations (radius='+str(r)+'mm)')
#     pyplot.title('r='+ str(r)+'mm');
    pyplot.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0);
    if save == 'YES' :
        if plotexp == 'YES':
            pyplot.savefig('figures/simulation_exp_radius%imm.png'%r, dpi=300, bbox_inches='tight') 
        else: 
            pyplot.savefig('figures/simulation_radius%imm.png'%r, dpi=300, bbox_inches='tight')
            
             
# Experimental data plots by raw data vs. mean & SD
def exp_statsvsraw_2plots(radius, time, average,upperbound,lowerbound,rawdata,save): 
    r = str(radius) 
    pyplot.figure(figsize=(20,7)) 
    pyplot.subplot(1, 2, 1) 
    pyplot.title('Experiments (radius='+r+'mm)') 
    pyplot.xlabel('time [s]') 
    pyplot.ylabel('Temperature Rise') 
    pyplot.plot( time[75:150]-20,average,linestyle='-',color= "black",label= "Mean") 
    pyplot.plot( time[75:150]-20,upperbound,linestyle='-',color= "gray",label= "SD" ) 
    pyplot.plot( time[75:150]-20,lowerbound,linestyle='-',color= "gray") 
    pyplot.fill_between( time[75:150]-20, upperbound,lowerbound, color= "lightgray") 
    pyplot.legend()
    pyplot.grid() 
    pyplot.xlim(-0.1,15) 
    pyplot.subplot(1, 2, 2) 
    pyplot.plot( time[75:150]-20, average,linestyle='-',label= "Mean ",color= "black") 
    pyplot.scatter( time[75:150]-20, np.abs(rawdata[0])[75:150],label= "Sample_1") 
    pyplot.scatter( time[75:150]-20, np.abs(rawdata[1])[75:150],label= "Sample_2") 
    pyplot.scatter( time[75:150]-20, np.abs(rawdata[2])[75:150],label= "Sample_3") 
    pyplot.legend()
    pyplot.grid()
    pyplot.xlim(-0.1,15)
    if save == 'YES' :
        pyplot.savefig('figures/experiments_radius'+r+'mm.png', dpi=300, bbox_inches='tight'); 
    return;
# All Experimental Data
def exp_ALL_2plots(timeE,avg0,avg1,avg2,avg4,
                   u0,u1,u2,u4,
                   l0,l1,l2,l4,
                   tempET1,tempET2,tempET3,tempET4,save):
    pyplot.figure(figsize=(20,7))
    pyplot.subplot(1, 2, 1)
    pyplot.title('Experiments')
    pyplot.xlabel('Time [s]')
    pyplot.ylabel('Temperature Rise')
    pyplot.plot(timeE[75:150]-20, avg0,linestyle='dashdot',label="r=0",color="black",linewidth=4)
    pyplot.plot(timeE[75:150]-20, u0,linestyle='-',color="gray")
    pyplot.plot(timeE[75:150]-20, l0,linestyle='-',color="gray")
    pyplot.fill_between(timeE[75:150]-20,u0, l0, color="lightgray")

    pyplot.plot(timeE[75:150]-20, avg1,linestyle='dashed',label="r=1",color="black",linewidth=4)
    pyplot.plot(timeE[75:150]-20, u1,linestyle='-',color="gray")
    pyplot.plot(timeE[75:150]-20, l1,linestyle='-',color="gray")
    pyplot.fill_between(timeE[75:150]-20,u1, l1, color="lightgray")

    pyplot.plot(timeE[75:150]-20, avg2,linestyle='-',label="r=2",color="black",linewidth=4)
    pyplot.plot(timeE[75:150]-20, u2,linestyle='-',color="gray")
    pyplot.plot(timeE[75:150]-20, l2,linestyle='-',color="gray")
    pyplot.fill_between(timeE[75:150]-20,u2, l2, color="lightgray")

    pyplot.plot(timeE[75:150]-20, avg4,linestyle='dotted',label="r=4",color="black",linewidth=4)
    pyplot.plot(timeE[75:150]-20, u4,linestyle='-',color="gray")
    pyplot.plot(timeE[75:150]-20, l4,linestyle='-',label="1SD",color="gray")
    pyplot.fill_between(timeE[75:150]-20,u4, l4, color="lightgray")

    pyplot.legend()
    pyplot.grid()
    pyplot.xlim(-0.1,15)
    pyplot.subplot(1, 2, 2)

    pyplot.scatter(timeE[75:150]-20, np.abs(tempET2[0])[75:150],label="Sample_1", color='royalblue')
    pyplot.scatter(timeE[75:150]-20, np.abs(tempET2[1])[75:150],label="Sample_2", color='darkorange')
    pyplot.scatter(timeE[75:150]-20, np.abs(tempET2[2])[75:150],label="Sample_3", color='mediumseagreen')
    pyplot.plot(timeE[75:150]-20, avg0,linestyle='dashdot',color="black",linewidth=4)

    pyplot.scatter(timeE[75:150]-20, np.abs(tempET3[0])[75:150], color='royalblue')
    pyplot.scatter(timeE[75:150]-20, np.abs(tempET3[1])[75:150], color='darkorange')
    pyplot.scatter(timeE[75:150]-20, np.abs(tempET3[2])[75:150], color='mediumseagreen')
    pyplot.plot(timeE[75:150]-20, avg1,linestyle='dashed',color="black",linewidth=4)

    pyplot.scatter(timeE[75:150]-20, np.abs(tempET1[0])[75:150], color='royalblue')
    pyplot.scatter(timeE[75:150]-20, np.abs(tempET1[1])[75:150], color='darkorange')
    pyplot.scatter(timeE[75:150]-20, np.abs(tempET1[2])[75:150], color='mediumseagreen')
    pyplot.plot(timeE[75:150]-20, avg2,linestyle='-',color="black",linewidth=4)

    pyplot.scatter(timeE[75:150]-20, np.abs(tempET4[0])[75:150], color='royalblue')
    pyplot.scatter(timeE[75:150]-20, np.abs(tempET4[1])[75:150], color='darkorange')
    pyplot.scatter(timeE[75:150]-20, np.abs(tempET4[2])[75:150], color='mediumseagreen')
    pyplot.plot(timeE[75:150]-20, avg4,linestyle='dotted',color="black",linewidth=4)

    pyplot.legend();
    pyplot.grid();
    pyplot.xlim(-0.1,15);
    if save == 'YES' :
        pyplot.savefig('figures/experiments_radiusALL.png', dpi=300, bbox_inches='tight');  
        
####------------------
def sim_exp_uncertainties(timeE,avg,u,l,tempS,meanSamplesP,stdSamplesP,r,save,plotexp):
    pyplot.figure(figsize=(6.4,4.8))
    pyplot.title('Radius='+ str(r)+'mm');
    pyplot.xlabel('time [s]')
    pyplot.ylabel('Temperature Rise')
    pyplot.plot(tempS['Time [s]'],meanSamplesP, label='Sim Mean', color='white',linewidth='2')
    pyplot.fill_between(tempS['Time [s]'],meanSamplesP+1*np.max(stdSamplesP),meanSamplesP-1*np.max(stdSamplesP), label='Sim 1SD', color='steelblue')
    if plotexp == 'YES':
        pyplot.plot(timeE[75:150]-19.650, avg,linestyle='-',label="Exp Mean",color="black")
        pyplot.fill_between(timeE[75:150]-19.650,u,l, color="darkorange",label="Exp 1SD")
    pyplot.grid()
    pyplot.legend(loc='upper left', facecolor='lightgray');
    pyplot.xlim(0,12);
    pyplot.ylim(-0.25,3.25);
    if save == 'YES' :          
        pyplot.savefig('figures/sim_exp_uncert_point%i.png'%r, dpi=300,bbox_inches='tight')

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

# Plot Simulation and Experiment with Uncertainties for All Point Locations
def sim_exp_uncertainties_ALL(timeE,avg0,u0,l0,avg1,u1,l1,avg2,u2,l2,avg4,u4,l4,
                              tempS1,meanSamplesP0,stdSamplesP0,
                              meanSamplesP1,stdSamplesP1,
                              meanSamplesP2,stdSamplesP2,
                              meanSamplesP3,stdSamplesP3,
                              meanSamplesP4,stdSamplesP4,
                              save,plotexp,plotSradius):
    pyplot.figure(figsize=(6.4,4.8))
    pyplot.title('Simulations')
    if 0 in plotSradius:
        pyplot.plot(tempS1['Time [s]'],meanSamplesP0, label='r=0mm ', color='magenta',linewidth='2')
        pyplot.fill_between(tempS1['Time [s]'],meanSamplesP0+1*np.max(stdSamplesP0),meanSamplesP0-1*np.max(stdSamplesP0), color='purple',alpha=1.0)
    if 1 in plotSradius:
        pyplot.plot(tempS1['Time [s]'],meanSamplesP1, label='r=1mm', color='lightgreen',linewidth='2')
        pyplot.fill_between(tempS1['Time [s]'],meanSamplesP1+1*np.max(stdSamplesP1),meanSamplesP1-1*np.max(stdSamplesP1), color='green',alpha=0.6)
    if 2 in plotSradius:
        pyplot.plot(tempS1['Time [s]'],meanSamplesP2, label='r=2mm', color='yellow',linewidth='2')
        pyplot.fill_between(tempS1['Time [s]'],meanSamplesP2+1*np.max(stdSamplesP2),meanSamplesP2-1*np.max(stdSamplesP2), color='orange',alpha=0.6)
    if 3 in plotSradius:
        pyplot.plot(tempS1['Time [s]'],meanSamplesP3, label='r=3mm', color='white',linewidth='2')
        pyplot.fill_between(tempS1['Time [s]'],meanSamplesP3+1*np.max(stdSamplesP3),meanSamplesP3-1*np.max(stdSamplesP3), color='black',alpha=0.6)
    if 4 in plotSradius:
        pyplot.plot(tempS1['Time [s]'],meanSamplesP4, label='r=4mm', color='cyan',linewidth='2')
        pyplot.fill_between(tempS1['Time [s]'],meanSamplesP4+1*np.max(stdSamplesP4),meanSamplesP4-1*np.max(stdSamplesP4), color='cadetblue')

    if plotexp == 'YES':
        if 0 in plotSradius:
            pyplot.plot(timeE[75:150]-19.65, avg0,linestyle='-',label="Point0-Exp",color="darkblue",linewidth='3')
            pyplot.fill_between(timeE[75:150]-19.65,u0, l0, color="lightblue",alpha=0.5)
        if 1 in plotSradius:
            pyplot.plot(timeE[75:150]-19.65, avg1,linestyle='-',label="Point1-Exp",color="black",linewidth='3')
            pyplot.fill_between(timeE[75:150]-19.65,u1, l1, color="lightgray",alpha=0.75)
        if 2 in plotSradius:
            pyplot.plot(timeE[75:150]-19.65, avg2,linestyle='-',label="Point2-Exp",color="red",linewidth='3')
            pyplot.fill_between(timeE[75:150]-19.65,u2, l2, color="pink",alpha=0.6)
        if 4 in plotSradius:
            pyplot.plot(timeE[75:150]-19.65, avg4,linestyle='-',label="Point4-Exp",color="indigo",linewidth='3')
            pyplot.fill_between(timeE[75:150]-19.65,u4, l4, color="blueviolet",alpha=0.25)
    pyplot.grid()
    pyplot.xlabel('Time [s]')
    pyplot.ylabel('Temperature Rise, $\Delta T$')
    pyplot.legend(loc='upper left');
    pyplot.xlim(-0.1,12);
    if save == 'YES' :     
        pyplot.savefig('figures/sim_exp_uncert_points0-4.png', dpi=300,bbox_inches='tight')

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
##--------------------------------------------
############ Area Metric
def area_metricPlots(tbypoint,timeEdiff,expS,timeE):
    radii = [0,1,2,4];
    for r in radii:
        # Plots
        print('r',r)
        timesteps = [0.0,2.0,5.0,10.0,11.0,12.0]
        sim = [tbypoint[r][0],tbypoint[r][1],tbypoint[r][2],tbypoint[r][3],tbypoint[r][4],tbypoint[r][5]]
        exp = [expS[r][0],expS[r][1],expS[r][2],expS[r][3],expS[r][4],expS[r][5]]
        time_E = list(timeE[75:150]-20)
        s = 0 
        for t in timesteps:   
            print('t',t)
            index = time_E.index(t) # find index at time = 5.0[s]    
            timestep=timeE[index+75]-20
        #     pyplot.figure(figsize=(15,10)) 
            pyplot.title('Time = {}, Radius = %i'.format(timestep)%r)
            pyplot.xlabel('Temperature') 
            pyplot.ylabel('Probability (CDF)') 
            ecdfExp = ECDF(exp[s])
#             ecdfExpNorm = ECDF(normExp[s])
            ecdfExp.y[0] = 0
            ecdfExp.x[0] = ecdfExp.x[1]
#             ecdfSimNorm = ECDF(normSim[s]) 
            ecdfSim = ECDF(sim[s])
            pyplot.step(ecdfExp.x,ecdfExp.y,'--o', where='post',label='Experiment')
            pyplot.step(ecdfSim.x,ecdfSim.y,'--o', where='post',label='Simulation')
            pyplot.legend()
            pyplot.xlim(0, 3.1)
    #         pyplot.savefig('figures/ecdf_t%i_r%i.png'%t%r, dpi=300);
            times = int(timestep)
            file_name = 'figures/AreaMetricPoint{}_Time{}.png'.format(r, times)
            pyplot.savefig(file_name, dpi=300, bbox_inches='tight')
            pyplot.show()
            s+=1
        