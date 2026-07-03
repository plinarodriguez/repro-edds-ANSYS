from matplotlib import pyplot as plt
import pandas as pd


modelansys = pd.read_csv('data/data_sim/outputSample1.csv' , skiprows=4)
modelopenfoam = pd.read_csv('data/OFmodel_coarse0.75_20260623_ddt/qois_radial.csv')
experiment = pd.read_csv('data/data_exp/Raw_Temp_Data/Air_Temperature_Measurements_1W_0.5LPM.csv')

T_m_ansys = modelansys["Monitor Point: Mouthpiece1mm0mmTempX0YZ (Temperature) [K]"]
t_m_ansys = modelansys["Time [s]"]

T_e = experiment["T1T2"]
t_e = experiment["Time"]

T_m_openfoam = modelansys["Monitor Point: Mouthpiece1mm0mmTempX0YZ (Temperature) [K]"] = modelopenfoam["0"]
t_m_openfoam = modelopenfoam["time"]


plt.figure()
plt.plot(t_m_ansys,T_m_ansys-T_m_ansys[0], label='ANSYS')
plt.plot(t_m_openfoam,T_m_openfoam-T_m_openfoam[0], label='OpenFOAM')
plt.plot(t_e-20,T_e-T_e[0], label='Experiment')
plt.xlim(-0.1,13)
plt.legend()
plt.savefig('figures/ansysvsopenfoamvsexperiment.png', dpi=300, bbox_inches='tight')