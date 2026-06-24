from matplotlib import pyplot
import numpy as np

# Power Profiles and SRQ locations
def powerprof(dataPowerS1,dataPowerS2,dataPowerS3,dataPowerS4,dataPowerS5,dataPowerS6,dataPowerS7,dataPowerS8,dataPowerS9,dataPowerS10, save):
    pyplot.figure(figsize=(8,5))
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
    pyplot.figure(figsize=(6.4,4.8))
    pyplot.xlabel('Time [s]')
    pyplot.ylabel(r'Temperature Rise, $\Delta T$')
    sampleplotslabels(point,tempS1,tempS2,tempS3,tempS4,tempS5,tempS6,tempS7,tempS8,tempS9,tempS10,tempSLab)
    if plotexp == 'YES':
        pyplot.plot(timeE[75:150]-19.650, avg,linestyle='-',label="Experiment",color="black")
        pyplot.fill_between(timeE[75:150]-19.650,u,l, color="lightgray")
    pyplot.grid()
    pyplot.xlim([-0.5,12])
    pyplot.title('Radius='+str(r)+'mm')
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
    pyplot.xlabel('Time [s]') 
    pyplot.ylabel(r'Temperature Rise, $\Delta T$') 
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
    pyplot.ylabel(r'Temperature Rise, $\Delta T$')
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
        
# All Experimental Data
def exp_ALL_1plot(timeE,avg0,avg1,avg2,avg4,
                   u0,u1,u2,u4,
                   l0,l1,l2,l4,
                   tempET1,tempET2,tempET3,tempET4,save):
#     pyplot.figure(figsize=(20,7))
    pyplot.title('Experiments')
    pyplot.xlabel('Time [s]')
    pyplot.ylabel(r'Temperature Rise, $\Delta T$')
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
    if save == 'YES' :
        pyplot.savefig('figures/experiments_radiusALL_uq.png', dpi=300, bbox_inches='tight');
        
# Plot Simulation and Experiment with Uncertainties for All Point Locations
def sim_exp_uncertainties_ALL(timeE,avg0,u0,l0,avg1,u1,l1,avg2,u2,l2,avg4,u4,l4,
                              tempS1,meanSamplesP0,stdSamplesP0,
                              meanSamplesP1,stdSamplesP1,
                              meanSamplesP2,stdSamplesP2,
                              meanSamplesP3,stdSamplesP3,
                              meanSamplesP4,stdSamplesP4,
                              save,plotexp,plotSradius):
    pyplot.figure(figsize=(12,8))
    pyplot.title('Points 0-4: Uncertainties')
    if 0 in plotSradius:
        pyplot.plot(tempS1['Time [s]'],meanSamplesP0, label='Point0-Sim ', color='magenta', linewidth='3')
        pyplot.fill_between(tempS1['Time [s]'],meanSamplesP0+1*np.max(stdSamplesP0),meanSamplesP0-1*np.max(stdSamplesP0), color='purple',alpha=1.0)
    if 1 in plotSradius:
        pyplot.plot(tempS1['Time [s]'],meanSamplesP1, label='Point1-Sim', color='lightgreen', linewidth='3')
        pyplot.fill_between(tempS1['Time [s]'],meanSamplesP1+1*np.max(stdSamplesP1),meanSamplesP1-1*np.max(stdSamplesP1), color='green',alpha=0.6)
    if 2 in plotSradius:
        pyplot.plot(tempS1['Time [s]'],meanSamplesP2, label='Point2-Sim', color='yellow', linewidth='3')
        pyplot.fill_between(tempS1['Time [s]'],meanSamplesP2+1*np.max(stdSamplesP2),meanSamplesP2-1*np.max(stdSamplesP2), color='orange',alpha=0.6)
    if 3 in plotSradius:
        pyplot.plot(tempS1['Time [s]'],meanSamplesP3, label='Point3-Sim', color='white', linewidth='3')
        pyplot.fill_between(tempS1['Time [s]'],meanSamplesP3+1*np.max(stdSamplesP3),meanSamplesP3-1*np.max(stdSamplesP3), color='black',alpha=0.6)
    if 4 in plotSradius:
        pyplot.plot(tempS1['Time [s]'],meanSamplesP4, label='Point4-Sim', color='cyan', linewidth='3')
        pyplot.fill_between(tempS1['Time [s]'],meanSamplesP4+1*np.max(stdSamplesP4),meanSamplesP4-1*np.max(stdSamplesP4), color='cadetblue')

    if plotexp == 'YES':
        if 0 in plotSradius:
            pyplot.plot(timeE[75:150]-19.65, avg0,linestyle='-',label="Point0-Exp",color="darkblue", linewidth='3')
            pyplot.fill_between(timeE[75:150]-19.65,u0, l0, color="lightblue",alpha=0.5)
        if 1 in plotSradius:
            pyplot.plot(timeE[75:150]-19.65, avg1,linestyle='-',label="Point1-Exp",color="black", linewidth='3')
            pyplot.fill_between(timeE[75:150]-19.65,u1, l1, color="lightgray",alpha=0.75)
        if 2 in plotSradius:
            pyplot.plot(timeE[75:150]-19.65, avg2,linestyle='-',label="Point2-Exp",color="red", linewidth='3')
            pyplot.fill_between(timeE[75:150]-19.65,u2, l2, color="pink",alpha=0.6)
        if 4 in plotSradius:
            pyplot.plot(timeE[75:150]-19.65, avg4,linestyle='-',label="Point4-Exp",color="indigo", linewidth='3')
            pyplot.fill_between(timeE[75:150]-19.65,u4, l4, color="blueviolet",alpha=0.25)
    pyplot.grid()
    pyplot.xlabel('Time [s]')
    pyplot.ylabel(r'Temperature Rise, $\Delta T$')
    pyplot.legend(loc='upper left');
    pyplot.xlim(-0.1,12);
    if save == 'YES' :     
        pyplot.savefig('figures/sim_exp_uncert_points0-4.png', dpi=300)

        
# Get the mean and standard deviation for all samples at a Point. 
# For each timestep: 15 quantities around a single point location & 10 samples each = 150 samples total. 
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
####
 # Plots Simulation vs. Experiments - by Point Location - All Uncertainties
def sim_exp_point(timeE,avg,u,l,tempS1,tempS2,tempS3,tempS4,tempS5,
                  tempS6,tempS7,tempS8,tempS9,tempS10,points,
                  tempSLab,r,save,plotexp):
    pyplot.figure(figsize=(12,8))
    pyplot.title('Temperature Rise - Point'+ str(r));
    pyplot.xlabel('Time [s]')
    pyplot.ylabel(r'Temperature Rise, $\Delta T$')
    i, plotQTY = 0, 0
    for point in points: 
        if i == 0:
            sampleplotslabels(point,tempS1,tempS2,tempS3,tempS4,tempS5,tempS6,tempS7,tempS8,tempS9,tempS10,tempSLab)
        else:
            sampleplots(point,tempS1,tempS2,tempS3,tempS4,tempS5,tempS6,tempS7,tempS8,tempS9,tempS10,tempSLab)
        i+=1
    if plotexp == 'YES':
        pyplot.plot(timeE[75:150]-19.650, avg,linestyle='-',label="Experiment",color="black")
        pyplot.fill_between(timeE[75:150]-19.650,u,l, color="lightgray")
    pyplot.grid()
    pyplot.xlim([-0.5,12])
    pyplot.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0);
    if save == 'YES' :
            pyplot.savefig('figures/sim_exp_samplin_curves_point%i.png'%r, dpi=300)

#########

def plots_simMeans():
    #Point 0
    points = ['Monitor Point: Mouthpiece1mm0mmTempXdownYZ (Temperature) [K]','Monitor Point: Mouthpiece1mm0mmTempXupYZ (Temperature) [K]','Monitor Point: Mouthpiece1mm0mmTempX0YZ (Temperature) [K]','Monitor Point: Mouthpiece1mm0mmTempX0YupZ (Temperature) [K]','Monitor Point: Mouthpiece1mm0mmTempX0YdownZ (Temperature) [K]','Monitor Point: MouthpieceUp0mmTempzXdownYZ (Temperature) [K]','Monitor Point: MouthpieceUp0mmTempzXupYZup (Temperature) [K]','Monitor Point: MouthpieceUp0mmTempzX0YZup (Temperature) [K]','Monitor Point: MouthpieceUp0mmTempzX0YupZup (Temperature) [K]','Monitor Point: MouthpieceUp0mmTempzX0YdownZup (Temperature) [K]','Monitor Point: MouthpieceDown0mmTempzXdownYZ (Temperature) [K]','Monitor Point: MouthpieceDown0mmTempzXupYZdown (Temperature) [K]','Monitor Point: MouthpieceDown0mmTempzX0YZdown (Temperature) [K]','Monitor Point: MouthpieceDown0mmTempzX0YupZdown (Temperature) [K]','Monitor Point: MouthpieceDown0mmTempzX0YdownZdown (Temperature) [K]']
    r = 0# radius for label
    save = 'NO' # 'YES'
    plotexp = 'YES' # 'YES'
    sim_exp_point(timeE,avg0,u0,l0,tempS1,tempS2,tempS3,tempS4,tempS5,
                      tempS6,tempS7,tempS8,tempS9,tempS10,points,
                      tempSLab,r,save,plotexp)
    #Point 1
    points = ['Monitor Point: Mouthpiece1mm1mmTempXdownYZ (Temperature) [K]','Monitor Point: Mouthpiece1mm1mmTempXupYZ (Temperature) [K]','Monitor Point: Mouthpiece1mm1mmTempX0YZ (Temperature) [K]','Monitor Point: Mouthpiece1mm1mmTempX0YupZ (Temperature) [K]','Monitor Point: Mouthpiece1mm1mmTempX0YdownZ (Temperature) [K]','Monitor Point: MouthpieceUp1mmTempzXdownZ (Temperature) [K]','Monitor Point: MouthpieceUp1mmTempzXupYZup (Temperature) [K]','Monitor Point: MouthpieceUp1mmTempzX0YZup (Temperature) [K]','Monitor Point: MouthpieceUp1mmTempzX0YupZup (Temperature) [K]','Monitor Point: MouthpieceUp1mmTempzX0YdownZup (Temperature) [K]','Monitor Point: MouthpieceDown1mmTempzXdownYZ (Temperature) [K]','Monitor Point: MouthpieceDown1mmTempzXupYZdown (Temperature) [K]','Monitor Point: MouthpieceDown1mmTempzX0YZdown (Temperature) [K]','Monitor Point: MouthpieceDown1mmTempzX0YupZdown (Temperature) [K]','Monitor Point: MouthpieceDown1mmTempzX0YdownZdown (Temperature) [K]']
    r = 1 # radius for label
    save = 'NO' # 'YES'
    plotexp = 'YES' # 'YES'
    sim_exp_point(timeE,avg1,u1,l1,tempS1,tempS2,tempS3,tempS4,tempS5,
                      tempS6,tempS7,tempS8,tempS9,tempS10,points,
                      tempSLab,r,save,plotexp)
    #Point 2
    points = ['Monitor Point: Mouthpiece1mm2mmTempXdownYZ (Temperature) [K]','Monitor Point: Mouthpiece1mm2mmTempXupYZ (Temperature) [K]','Monitor Point: Mouthpiece1mm2mmTempX0YZ (Temperature) [K]','Monitor Point: Mouthpiece1mm2mmTempX0YupZ (Temperature) [K]','Monitor Point: Mouthpiece1mm2mmTempX0YdownZ (Temperature) [K]','Monitor Point: MouthpieceUp2mmTempzXdownYZ (Temperature) [K]','Monitor Point: MouthpieceUp2mmTempzXupYZup (Temperature) [K]','Monitor Point: MouthpieceUp2mmTempzX0YZup (Temperature) [K]','Monitor Point: MouthpieceUp2mmTempzX0YupZup (Temperature) [K]','Monitor Point: MouthpieceUp2mmTempzX0YdownZup (Temperature) [K]','Monitor Point: MouthpieceDown2mmTempzXdownYZ (Temperature) [K]','Monitor Point: MouthpieceDown2mmTempzXupYZdown (Temperature) [K]','Monitor Point: MouthpieceDown2mmTempzX0YZdown (Temperature) [K]','Monitor Point: MouthpieceDown2mmTempzX0YupZdown (Temperature) [K]','Monitor Point: MouthpieceDown2mmTempzX0YdownZdpwm (Temperature) [K]']
    r = 2 # radius for label
    save = 'NO' # 'YES'
    plotexp = 'YES' # 'YES'
    sim_exp_point(timeE,avg2,u2,l2,tempS1,tempS2,tempS3,tempS4,tempS5,
                      tempS6,tempS7,tempS8,tempS9,tempS10,points,
                      tempSLab,r,save,plotexp)
    #Point 3
    points = ['Monitor Point: Mouthpiece1mm3mmTempXdownYZ (Temperature) [K]','Monitor Point: Mouthpiece1mm3mmTempXupYZ (Temperature) [K]','Monitor Point: Mouthpiece1mm3mmTempX0YZ (Temperature) [K]','Monitor Point: Mouthpiece1mm3mmTempX0YupZ (Temperature) [K]','Monitor Point: Mouthpiece1mm3mmTempX0YdownZ (Temperature) [K]','Monitor Point: MouthpieceUp3mmTempzXdownYZ (Temperature) [K]','Monitor Point: MouthpieceUp3mmTempzXupYZup (Temperature) [K]','Monitor Point: MouthpieceUp3mmTempzX0YZup (Temperature) [K]','Monitor Point: MouthpieceUp3mmTempzX0YupZup (Temperature) [K]','Monitor Point: MouthpieceUp3mmTempzX0YdownZup (Temperature) [K]','Monitor Point: MouthpieceDown3mmTempzXdownYZ (Temperature) [K]','Monitor Point: MouthpieceDown3mmTempzXupYZdown (Temperature) [K]','Monitor Point: MouthpieceDown3mmTempzX0YZdown (Temperature) [K]','Monitor Point: MouthpieceDown3mmTempzX0YupZdown (Temperature) [K]','Monitor Point: MouthpieceDown3mmTempzX0YdownZdown (Temperature) [K]']
    r = 3 # radius for label
    save = 'NO' # 'YES'
    plotexp = 'NO' # 'This data doesn't exist 
    sim_exp_point(timeE,avg1,u1,l1,tempS1,tempS2,tempS3,tempS4,tempS5,
                      tempS6,tempS7,tempS8,tempS9,tempS10,points,
                      tempSLab,r,save,plotexp)
    #Point 4
    points = ['Monitor Point: Mouthpiece1mm4mmTempXdownYZ (Temperature) [K]','Monitor Point: Mouthpiece1mm4mmTempXupYZ (Temperature) [K]','Monitor Point: Mouthpiece1mm4mmTempX0YZ (Temperature) [K]','Monitor Point: Mouthpiece1mm4mmTempX0YupZ (Temperature) [K]','Monitor Point: Mouthpiece1mm4mmTempX0YdownZ (Temperature) [K]','Monitor Point: MouthpieceUp4mmTempXdownYZ (Temperature) [K]','Monitor Point: MouthpieceUp4mmTempXupYZup (Temperature) [K]','Monitor Point: MouthpieceUp4mmTempX0YZup (Temperature) [K]','Monitor Point: MouthpieceUp4mmTempX0YupZup (Temperature) [K]','Monitor Point: MouthpieceUp4mmTempX0YdownZup (Temperature) [K]','Monitor Point: MouthpieceDown4mmTempzXdownYZ (Temperature) [K]','Monitor Point: MouthpieceDown4mmTempzXupYZdown (Temperature) [K]','Monitor Point: MouthpieceDown4mmTempzX0YZdown (Temperature) [K]','Monitor Point: MouthpieceDown4mmTempzX0YupZdown (Temperature) [K]','Monitor Point: MouthpieceDown4mmTempzX0YdownZdown (Temperature) [K]']
    r = 4 # radius for label
    save = 'NO' # 'YES'
    plotexp = 'YES' # 'YES'
    sim_exp_point(timeE,avg4,u4,l4,tempS1,tempS2,tempS3,tempS4,tempS5,
                      tempS6,tempS7,tempS8,tempS9,tempS10,points,
                      tempSLab,r,save,plotexp)
    