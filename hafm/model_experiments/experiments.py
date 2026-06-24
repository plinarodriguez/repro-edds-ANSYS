import pandas as pd
import numpy as np

### Experiments: Raw Data - Temp Diff
# t=sample,T=radial position,T1:2mm,T2:r=0mm (Center),T3:r=1mm,T4:r=4mm 
tempEdiff = pd.read_csv('data/data_exp/Temp_Diff_Data/exp_1W_0.5LPM.csv')  # Power=1 Watt, Volumetric Airflow=0.5 L/min
# tempEdiff = pd.read_csv('data/Temp_Diff_Data/Exp_1W_1.65LPM.csv') # Power 1 Watt, VFR=1.65 L/min
timeEdiff   = tempEdiff['Time']
tempET1diff = [tempEdiff['T1_t1'],tempEdiff['T1_t2'],tempEdiff['T1_t3']]
tempET2diff = [tempEdiff['T2_t1'],tempEdiff['T2_t2'],tempEdiff['T2_t3']]
tempET3diff = [tempEdiff['T3_t1'],tempEdiff['T3_t2'],tempEdiff['T3_t3']]
tempET4diff = [tempEdiff['T4_t1'],tempEdiff['T4_t2'],tempEdiff['T4_t3']]
### Experiments: Raw Data 
# T(Sample) T(radial position) 
tempE = pd.read_csv('data/data_exp/Raw_Temp_Data/Air_Temperature_Measurements_1W_0.5LPM.csv')
timeE = tempE['Time']
tempET1 = [tempE['T1T1'],tempE['T2T1'],tempE['T3T1']]
tempET2 = [tempE['T1T2'],tempE['T2T2'],tempE['T3T2']]
tempET3 = [tempE['T1T3'],tempE['T2T3'],tempE['T3T3']]
tempET4 = [tempE['T1T4'],tempE['T2T4'],tempE['T3T4']]
tempE.tail()
#---------------------------------------------------------------
### Experiment Metrics: mean & standard deviation over 3 samples
i = 75 # Start here because there are a lot of power off values. 
avg0,avg1,avg2,avg4 = [],[],[],[]
sd0,sd1,sd2,sd4 = [],[],[],[]
while i < 150:
    a4 = [tempET4diff[0][i],tempET4diff[1][i],tempET4diff[2][i]] #4mm
    a2 = [tempET1diff[0][i],tempET1diff[1][i],tempET1diff[2][i]] #2mm
    a1 = [tempET3diff[0][i],tempET3diff[1][i],tempET3diff[2][i]] #1mm     
    a0 = [tempET2diff[0][i],tempET2diff[1][i],tempET2diff[2][i]] #center
    avg0.append(np.average(a0))
    avg1.append(np.average(a1))
    avg2.append(np.average(a2))
    avg4.append(np.average(a4))
    sd0.append(np.std(a0))
    sd1.append(np.std(a1))
    sd2.append(np.std(a2))
    sd4.append(np.std(a4))
    i +=1
# Take the largest standard deviation and apply that across all means 
# Key: u=upper bound, l=lower bound
u0=avg0 + 1*max(sd0) #/len(a0) # 3 samples
u1=avg1 + 1*max(sd1) #/len(a0) # 3 samples
u2=avg2 + 1*max(sd2) #/len(a0) # 3 samples
u4=avg4 + 1*max(sd4) #/len(a0) # 3 samples
l0=avg0 - 1*max(sd0) #/len(a0) # 3 samples
l1=avg1 - 1*max(sd1) #/len(a0) # 3 samples
l2=avg2 - 1*max(sd2) #/len(a0) # 3 samples
l4=avg4 - 1*max(sd4) #/len(a0) # 3 samples

exp_avg = {'time': timeE, 'avg_r0': avg0, 'avg_r1': avg1,'avg_r2': avg2,'avg_r4': avg4}   

#---------------------------------------------------------------
# Experiments Data - by timestep
# Radial = 0 mm, Temporal = 0,2,5,10,11,12 [s] 
# indexN = time_E.index(0.0) 
def exp_bytime(avg0,avg4,sd0,sd4,tempET1,tempET2,tempET3,tempET4,exp_t):
    # Identifying timestep indecies
    time_E = list(timeEdiff[75:150]-20)
    indexN = time_E.index(0.0) 
    normExpR0T0 = np.random.normal(avg0[indexN], sd0[indexN],100000)  
    normExpR4T0 = np.random.normal(avg4[indexN], sd4[indexN],100000)  
    expR0T0  = [tempET2[0][indexN+75],tempET2[1][indexN+75],tempET2[2][indexN+75]] #center
    expR1T0  = [tempET3[0][indexN+75],tempET3[1][indexN+75],tempET3[2][indexN+75]] #1mm
    expR2T0  = [tempET1[0][indexN+75],tempET1[1][indexN+75],tempET1[2][indexN+75]] #2mm
    expR4T0  = [tempET4[0][indexN+75],tempET4[1][indexN+75],tempET4[2][indexN+75]] #4mm
    expRallT0 = [expR0T0,expR1T0,expR2T0,expR4T0]

    indexN = time_E.index(2.0) 
    normExpR0T2 = np.random.normal(avg0[indexN], sd0[indexN],100000)  
    normExpR4T2 = np.random.normal(avg4[indexN], sd4[indexN],100000)  
    expR0T2  = [tempET2[0][indexN+75],tempET2[1][indexN+75],tempET2[2][indexN+75]] #center
    expR1T2  = [tempET3[0][indexN+75],tempET3[1][indexN+75],tempET3[2][indexN+75]] #1mm
    expR2T2  = [tempET1[0][indexN+75],tempET1[1][indexN+75],tempET1[2][indexN+75]] #2mm
    expR4T2  = [tempET4[0][indexN+75],tempET4[1][indexN+75],tempET4[2][indexN+75]] #4mm
    expRallT2 = [expR0T2,expR1T2,expR2T2,expR4T2]

    indexN = time_E.index(5.0)  
    normExpR0T5 = np.random.normal(avg0[indexN], sd0[indexN],100000)  
    normExpR4T5 = np.random.normal(avg4[indexN], sd4[indexN],100000)  
    expR0T5  = [tempET2[0][indexN+75],tempET2[1][indexN+75],tempET2[2][indexN+75]] #center
    expR1T5  = [tempET3[0][indexN+75],tempET3[1][indexN+75],tempET3[2][indexN+75]] #1mm
    expR2T5  = [tempET1[0][indexN+75],tempET1[1][indexN+75],tempET1[2][indexN+75]] #2mm
    expR4T5  = [tempET4[0][indexN+75],tempET4[1][indexN+75],tempET4[2][indexN+75]] #4mm
    expRallT5 = [expR0T5,expR1T5,expR2T5,expR4T5]

    indexN = time_E.index(10.0) 
    normExpR0T10 = np.random.normal(avg0[indexN], sd0[indexN],100000)  
    normExpR4T10 = np.random.normal(avg4[indexN], sd4[indexN],100000)  
    expR0T10 = [tempET2[0][indexN+75],tempET2[1][indexN+75],tempET2[2][indexN+75]] #center
    expR1T10 = [tempET3[0][indexN+75],tempET3[1][indexN+75],tempET3[2][indexN+75]] #1mm
    expR2T10 = [tempET1[0][indexN+75],tempET1[1][indexN+75],tempET1[2][indexN+75]] #2mm
    expR4T10 = [tempET4[0][indexN+75],tempET4[1][indexN+75],tempET4[2][indexN+75]] #4mm
    expRallT10 = [expR0T10,expR1T10,expR2T10,expR4T10]

    indexN = time_E.index(11.0) 
    normExpR0T11 = np.random.normal(avg0[indexN], sd0[indexN],100000)  
    normExpR4T11 = np.random.normal(avg4[indexN], sd4[indexN],100000)  
    expR0T11 = [tempET2[0][indexN+75],tempET2[1][indexN+75],tempET2[2][indexN+75]] #center
    expR1T11 = [tempET3[0][indexN+75],tempET3[1][indexN+75],tempET3[2][indexN+75]] #1mm
    expR2T11 = [tempET1[0][indexN+75],tempET1[1][indexN+75],tempET1[2][indexN+75]] #2mm
    expR4T11 = [tempET4[0][indexN+75],tempET4[1][indexN+75],tempET4[2][indexN+75]] #4mm
    expRallT11 = [expR0T11,expR1T11,expR2T11,expR4T11]

    indexN = time_E.index(12.0) 
    normExpR0T12 = np.random.normal(avg0[indexN], sd0[indexN], 100000) 
    normExpR4T12 = np.random.normal(avg4[indexN], sd4[indexN], 100000) 
    expR0T12 = [tempET2[0][indexN+75],tempET2[1][indexN+75],tempET2[2][indexN+75]] #center
    expR1T12 = [tempET3[0][indexN+75],tempET3[1][indexN+75],tempET3[2][indexN+75]] #1mm
    expR2T12 = [tempET1[0][indexN+75],tempET1[1][indexN+75],tempET1[2][indexN+75]] #2mm
    expR4T12 = [tempET4[0][indexN+75],tempET4[1][indexN+75],tempET4[2][indexN+75]] #4mm
    expRallT12 = [expR0T12,expR1T12,expR2T12,expR4T12]
    expRallTall = [expRallT0,expRallT2,expRallT5,expRallT10,expRallT11,expRallT12]
    
    if(exp_t == 'R0'):
        exp  = [expR0T0,expR0T2,expR0T5,expR0T10,expR0T11,expR0T12];
    if(exp_t == 'R1'):
        exp  = [expR1T0,expR1T2,expR1T5,expR1T10,expR1T11,expR1T12];
    if(exp_t == 'R2'):
        exp  = [expR2T0,expR2T2,expR2T5,expR2T10,expR2T11,expR2T12];
    if(exp_t == 'R4'):
        exp  = [expR4T0,expR4T2,expR4T5,expR4T10,expR4T11,expR4T12];
    return(exp)