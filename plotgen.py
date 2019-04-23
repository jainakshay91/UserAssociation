#!/usr/bin/env python 

# =============================
# Import the Necessary Binaries
# =============================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scenario_var import scenario_var 
import copy 

# =======================
# Generate Class Variable
# =======================

scn = scenario_var(); # Getting the class object

# ========================
# Generate Necessary Plots
# ========================

# ================================
# Load the Data from the Optimizer

Baseline_dat = np.load('/home/akshayjain/Desktop/Simulation/dat_Baseline.npz')
Dat_DC = np.load('/home/akshayjain/Desktop/Simulation/dat_DC.npz')
Dat_DC_MRT = np.load('/home/akshayjain/Desktop/Simulation/dat_DC_MRT.npz')
Data = Baseline_dat['arr_0'];
Data_DC = Dat_DC['arr_0'];
Data_DC_MRT = Dat_DC_MRT['arr_0'];
num_iter = ((scn.num_users_max - scn.num_users_min)/scn.user_steps_siml); 
Net_Throughput = np.empty((num_iter,1));
Net_Throughput_DC = copy.copy(Net_Throughput);
Net_Throughput_DC_MRT = copy.copy(Net_Throughput);
for k in range(0,num_iter):
	Net_Throughput[k] = Data.item()['Net_Throughput'+str(k)];
	Net_Throughput_DC[k] = Data_DC.item()['Net_Throughput'+str(k)];
	Net_Throughput_DC_MRT[k] = Data_DC_MRT.item()['Net_Throughput'+str(k)];
# ===============
# Throughput Plot
#print Net_Throughput_DC_MRT
#print Net_Throughput_DC

x_axis = np.arange(scn.num_users_min, scn.num_users_max, scn.user_steps_siml);
y_min = np.amin([np.amin(Net_Throughput), np.amin(Net_Throughput_DC), np.amin(Net_Throughput_DC_MRT)]);
y_max = np.max([np.amax(Net_Throughput), np.amax(Net_Throughput_DC), np.amax(Net_Throughput_DC_MRT)]);
#plotter.plotter('dashline',np.arange(scn.num_users_min, scn.num_users_max, scn.user_steps_siml),Net_Throughput,5,10,1,45,0,0,1,'major','both', 'yes', 'Total Network Throughput', np)
plt.plot(x_axis, Net_Throughput, 'r--*', x_axis, Net_Throughput_DC, 'b--*', x_axis, Net_Throughput_DC_MRT, 'g-.^');
plt.xticks(np.arange(scn.num_users_min, scn.num_users_max, scn.user_steps_siml));
plt.yticks(np.arange(y_min,y_max,5e10));
plt.legend(['Single Association (SA)','Dual Association (DA)', 'DA + Minimum Rate'])
plt.grid(which= 'major',axis= 'both');
plt.title('Network Wide Throughput')
plt.show()