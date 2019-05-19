#!/usr/bin/env python 

# =============================
# Import the Necessary Binaries
# =============================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scenario_var import scenario_var 
import copy 
import os, sys
# =======================
# Generate Class Variable
# =======================

scn = scenario_var(); # Getting the class object

# ==================================
# Initialize the Necessary Variables
# ==================================

MCMC_iter = 16; # Number of Iterations to be analyzed
simdata_path = os.getcwd() + '/Data/Process/'
constraint_fp = {'Baseline':'0000', 'DC':'1000', 'DC_MRT':'1100','DC_BHCAP':'1010', 'DC_BHCAP_LAT':'1011', 'DC_LAT':'1001', 'SA_MRT':'0100','SA_BHCAP':'0010','SA_BHCAP_LAT':'0011','SA_LAT':'0001'}

# ==============
# Data Extractor
# ==============

for i in range(0,MCMC_iter):

	# ================================
	# Load the Data from the Optimizer

	Baseline_dat = np.load(simdata_path +'_'+ str(i) +'dat_' + constraint_fp['Baseline'] + '.npz')
	Dat_DC = np.load(simdata_path +'_'+ str(i) +'dat_' + constraint_fp['DC'] + '.npz')
	Dat_DC_MRT = np.load(simdata_path +'_'+ str(i) +'dat_' + constraint_fp['DC_MRT'] + '.npz')
	Dat_DC_BHCAP = np.load(simdata_path +'_'+ str(i) +'dat_' + constraint_fp['DC_BHCAP'] + '.npz')
	Dat_DC_BHCAP_Lat = np.load(simdata_path +'_'+ str(i) +'dat_' + constraint_fp['DC_BHCAP_LAT'] + '.npz')
	Dat_DC_Lat = np.load(simdata_path +'_'+ str(i) +'dat_' + constraint_fp['DC_LAT'] + '.npz')
	Dat_SA_MRT = np.load(simdata_path +'_'+ str(i) +'dat_' + constraint_fp['SA_MRT'] + '.npz')
	Dat_SA_LAT = np.load(simdata_path +'_'+ str(i) +'dat_' + constraint_fp['SA_LAT'] + '.npz')
	Dat_SA_BHCAP = np.load(simdata_path +'_'+ str(i) +'dat_' + constraint_fp['SA_BHCAP'] + '.npz')
	Dat_SA_BHCAP_LAT = np.load(simdata_path +'_'+ str(i) +'dat_' + constraint_fp['SA_BHCAP_LAT'] + '.npz')


	Data = Baseline_dat['arr_0'];
	Data_DC = Dat_DC['arr_0'];
	Data_DC_MRT = Dat_DC_MRT['arr_0'];
	Data_DC_BHCAP = Dat_DC_BHCAP['arr_0'];
	Data_DC_BHCAP_LAT = Dat_DC_BHCAP_Lat['arr_0'];
	Data_DC_LAT = Dat_DC_Lat['arr_0'];
	Data_SA_MRT = Dat_SA_MRT['arr_0'];
	Data_SA_LAT = Dat_SA_LAT['arr_0'];
	Data_SA_BHCAP = Dat_SA_BHCAP['arr_0'];
	Data_SA_BHCAP_LAT = Dat_SA_BHCAP_LAT['arr_0'];

	num_iter = ((scn.num_users_max - scn.num_users_min)/scn.user_steps_siml); 
	Net_Throughput = np.empty((MCMC_iter, num_iter));
	Net_Throughput_DC = copy.copy(Net_Throughput);
	Net_Throughput_DC_MRT = copy.copy(Net_Throughput);
	Net_Throughput_DC_BHCAP = copy.copy(Net_Throughput);
	Net_Throughput_DC_BHCAP_LAT = copy.copy(Net_Throughput);
	Net_Throughput_DC_LAT = copy.copy(Net_Throughput);
	Net_Throughput_SA_MRT = copy.copy(Net_Throughput);
	Net_Throughput_SA_LAT = copy.copy(Net_Throughput);
	Net_Throughput_SA_BHCAP = copy.copy(Net_Throughput);
	Net_Throughput_SA_BHCAP_LAT = copy.copy(Net_Throughput);

	for k in range(0,num_iter):
		Net_Throughput[i,k] = Data.item()['Net_Throughput'+str(k)];
		#print Net_Throughput
		Net_Throughput_DC[i,k] = Data_DC.item()['Net_Throughput'+str(k)];
		#print Net_Throughput_DC
		Net_Throughput_DC_MRT[i,k] = Data_DC_MRT.item()['Net_Throughput'+str(k)];
		Net_Throughput_DC_BHCAP[i,k] = Data_DC_BHCAP.item()['Net_Throughput'+str(k)];
		Net_Throughput_DC_BHCAP_LAT[i,k] = Data_DC_BHCAP_LAT.item()['Net_Throughput'+str(k)];
		Net_Throughput_DC_LAT[i,k] = Data_DC_LAT.item()['Net_Throughput'+str(k)];
		Net_Throughput_SA_MRT[i,k] = Data_SA_MRT.item()['Net_Throughput'+str(k)];
		Net_Throughput_SA_LAT[i,k] = Data_SA_LAT.item()['Net_Throughput'+str(k)];
		Net_Throughput_SA_BHCAP[i,k] = Data_SA_BHCAP.item()['Net_Throughput'+str(k)];
		Net_Throughput_SA_BHCAP_LAT[i,k] = Data_SA_BHCAP_LAT.item()['Net_Throughput'+str(k)];

	#print "=================="
	#print Net_Throughput
	#print "=================="
	#print Net_Throughput_DC
	# ================
	# User Throughputs

	X_Optimal = np.empty((Data.item()['X_optimal_data0'].shape[0], Data.item()['X_optimal_data0'].shape[1]));
	X_Optimal_DC = copy.copy(X_Optimal);
	X_Optimal_DC_MRT = copy.copy(X_Optimal);
	X_Optimal_DC_BHCAP = copy.copy(X_Optimal);
	X_Optimal_DC_BHCAP_LAT = copy.copy(X_Optimal);
	X_Optimal_DC_LAT = copy.copy(X_Optimal);
	X_Optimal_SA_MRT = copy.copy(X_Optimal);
	X_Optimal_SA_LAT = copy.copy(X_Optimal);
	X_Optimal_SA_BHCAP = copy.copy(X_Optimal);
	X_Optimal_SA_BHCAP_LAT = copy.copy(X_Optimal);

	Rate = np.empty((Data.item()['X_optimal_data0'].shape[0], Data.item()['X_optimal_data0'].shape[1]));
	Rate_DC = copy.copy(Rate);
	Rate_DC_MRT = copy.copy(Rate);
	Rate_DC_BHCAP = copy.copy(Rate);
	Rate_DC_BHCAP_LAT = copy.copy(Rate);
	Rate_DC_LAT = copy.copy(Rate);
	Rate_SA_MRT = copy.copy(Rate);
	Rate_SA_LAT = copy.copy(Rate);
	Rate_SA_BHCAP = copy.copy(Rate);
	Rate_SA_BHCAP_LAT = copy.copy(Rate);


	X_Optimal = Data.item()['X_optimal_data'+str(10)];
	X_Optimal_DC = Data_DC.item()['X_optimal_data'+str(10)];
	X_Optimal_DC_MRT = Data_DC_MRT.item()['X_optimal_data'+str(10)];
	X_Optimal_DC_BHCAP = Data_DC_BHCAP.item()['X_optimal_data'+str(10)];
	X_Optimal_DC_BHCAP_LAT = Data_DC_BHCAP_LAT.item()['X_optimal_data'+str(10)];
	X_Optimal_DC_LAT = Data_DC_LAT.item()['X_optimal_data'+str(10)];
	X_Optimal_SA_MRT = Data_SA_MRT.item()['X_optimal_data'+str(10)];
	X_Optimal_SA_LAT = Data_SA_LAT.item()['X_optimal_data'+str(10)];
	X_Optimal_SA_BHCAP = Data_SA_BHCAP.item()['X_optimal_data'+str(10)];
	X_Optimal_SA_BHCAP_LAT = Data_SA_BHCAP_LAT.item()['X_optimal_data'+str(10)];

	Rate = Data.item()['Rates'+str(10)];
	Rate_DC = Data_DC.item()['Rates'+str(10)];
	Rate_DC_MRT = Data_DC_MRT.item()['Rates'+str(10)];
	Rate_DC_BHCAP = Data_DC_BHCAP.item()['Rates'+str(10)];
	Rate_DC_BHCAP_LAT = Data_DC_BHCAP_LAT.item()['Rates'+str(10)];
	Rate_DC_LAT = Data_DC_LAT.item()['Rates'+str(10)];
	Rate_SA_MRT = Data_SA_MRT.item()['Rates'+str(10)];
	Rate_SA_LAT = Data_SA_LAT.item()['Rates'+str(10)];
	Rate_SA_BHCAP = Data_SA_BHCAP.item()['Rates'+str(10)];
	Rate_SA_BHCAP_LAT = Data_SA_BHCAP_LAT.item()['Rates'+str(10)];


	Base_DR = []
	application_DR = []
	application_DR_DC = [];
	application_DR_DC_MRT = [];
	application_DR_DC_BHCAP = [];
	application_DR_DC_BHCAP_LAT = [];
	application_DR_DC_LAT = [];
	application_DR_SA_MRT = [];
	application_DR_SA_LAT = [];
	application_DR_SA_BHCAP = [];
	application_DR_SA_BHCAP_LAT = [];

	for i in range(0,X_Optimal.shape[0]):
		Base_DR.append(scn.eMBB_minrate); 
		application_DR.append(sum(Rate[i,:]*X_Optimal[i,:]));
		application_DR_DC.append(sum(Rate_DC[i,:]*X_Optimal_DC[i,:]));
		application_DR_DC_MRT.append(sum(Rate_DC_MRT[i,:]*X_Optimal_DC_MRT[i,:]));
		application_DR_DC_BHCAP.append(sum(Rate_DC_BHCAP[i,:]*X_Optimal_DC_BHCAP[i,:]));
		application_DR_DC_BHCAP_LAT.append(sum(Rate_DC_BHCAP_LAT[i,:]*X_Optimal_DC_BHCAP_LAT[i,:]));
		application_DR_DC_LAT.append(sum(Rate_DC_LAT[i,:]*X_Optimal_DC_LAT[i,:]));
		application_DR_SA_MRT.append(sum(Rate_SA_MRT[i,:]*X_Optimal_SA_MRT[i,:]));
		application_DR_SA_LAT.append(sum(Rate_SA_LAT[i,:]*X_Optimal_SA_LAT[i,:]));
		application_DR_SA_BHCAP.append(sum(Rate_SA_BHCAP[i,:]*X_Optimal_SA_BHCAP[i,:]));
		application_DR_SA_BHCAP_LAT.append(sum(Rate_SA_BHCAP_LAT[i,:]*X_Optimal_SA_BHCAP_LAT[i,:]));


# =======================
# Average Value Generator
# =======================


Net_Throughput_avg = np.sum(Net_Throughput, axis = 0); # We get the average throughput over MCMC Iteratios
Net_Throughput_DC_avg = np.sum(Net_Throughput_DC, axis = 0); # Average throughput


#Net_Throughput_avg = Net_Throughput[1,:]; # We get the average throughput over MCMC Iteratios
#Net_Throughput_DC_avg = Net_Throughput_DC[1,:]; # Average throughput


Net_Throughput_DC_MRT_avg = np.sum(Net_Throughput_DC_MRT, axis = 0); # DC + MRT Average throughput
Net_Throughput_DC_BHCAP_avg = np.sum(Net_Throughput_DC_BHCAP, axis = 0); # DC + BHCAP Average throughput
Net_Throughput_DC_LAT_avg = np.sum(Net_Throughput_DC_LAT, axis = 0); # DC + LAT Average throughput
Net_Throughput_DC_BHCAP_LAT_avg = np.sum(Net_Throughput_DC_BHCAP_LAT, axis = 0); # DC + BHCAP + LAT Average throughput
Net_Throughput_SA_MRT_avg = np.sum(Net_Throughput_SA_MRT, axis = 0); # SA + MRT average 
Net_Throughput_SA_LAT_avg = np.sum(Net_Throughput_SA_LAT, axis = 0); # SA + LAT average
Net_Throughput_SA_BHCAP_avg = np.sum(Net_Throughput_SA_BHCAP, axis = 0); # SA + BHCAP average
Net_Throughput_SA_BHCAP_LAT_avg = np.sum(Net_Throughput_SA_BHCAP_LAT, axis = 0); # SA + BHCAP + LAT average
	

	# ===============
	# Throughput Plot
	#print Net_Throughput_DC_MRT
	#print Net_Throughput_DC

x_axis = np.arange(scn.num_users_min, scn.num_users_max, scn.user_steps_siml);
y_min = np.amin([np.amin(Net_Throughput_avg), np.amin(Net_Throughput_DC_avg), np.amin(Net_Throughput_DC_MRT_avg), np.amin(Net_Throughput_DC_BHCAP_avg), np.amin(Net_Throughput_DC_BHCAP_LAT_avg), np.amin(Net_Throughput_DC_LAT_avg), np.amin(Net_Throughput_SA_MRT_avg), np.amin(Net_Throughput_SA_LAT_avg), np.amin(Net_Throughput_SA_BHCAP_avg), np.amin(Net_Throughput_SA_BHCAP_LAT_avg)]);
y_max = np.max([np.amax(Net_Throughput_avg), np.amax(Net_Throughput_DC_avg), np.amax(Net_Throughput_DC_MRT_avg), np.amax(Net_Throughput_DC_BHCAP_avg), np.amax(Net_Throughput_DC_BHCAP_LAT_avg), np.amax(Net_Throughput_DC_LAT_avg), np.amax(Net_Throughput_SA_MRT_avg), np.amax(Net_Throughput_SA_LAT_avg), np.amax(Net_Throughput_SA_BHCAP_avg), np.amax(Net_Throughput_SA_BHCAP_LAT_avg)]);
#plotter.plotter('dashline',np.arange(scn.num_users_min, scn.num_users_max, scn.user_steps_siml),Net_Throughput_avg,5,10,1,45,0,0,1,'major','both', 'yes', 'Total Network Throughput', np)
plt.plot(x_axis, Net_Throughput_avg, 'r--*', x_axis, Net_Throughput_DC_avg, 'b--*' , x_axis, Net_Throughput_DC_MRT_avg, 'g-.', x_axis, Net_Throughput_DC_BHCAP_avg, 'k--s', x_axis, Net_Throughput_DC_BHCAP_LAT_avg, 'm--d', x_axis , Net_Throughput_DC_LAT_avg, 'c--p',x_axis, Net_Throughput_SA_MRT_avg, 'k-.', x_axis, Net_Throughput_SA_LAT_avg, 'b:', x_axis, Net_Throughput_SA_BHCAP_avg, 'g--D', x_axis, Net_Throughput_SA_BHCAP_LAT_avg, 'r:');
plt.xticks(np.arange(scn.num_users_min, scn.num_users_max, scn.user_steps_siml));
plt.yticks(np.arange(y_min,y_max,5e10));
plt.legend(['Single Association (SA)','Dual Association (DA)', 'DA + Minimum Rate', 'DA + Constrained Backhaul (CB) [1% Bound Gap]', 'DA + CB + Constrained Path Latency (CPL) [1% Bound Gap]', 'DA + Minimum Rate + CPL', 'SA + Minimum Rate', 'SA + Minimum Rate + CPL', 'SA + CB [1% Bound Gap]', 'SA + CB + CPL [1% Bound Gap]'])
plt.grid(which= 'major',axis= 'both');
plt.title('Network Wide Throughput')
plt.show()

	#print application_DR.tolist()
	#plt.bar(np.arange(1,Rate.shape[0]+1),application_DR_SA_BHCAP_LAT)
	#plt.plot(np.arange(1,Rate.shape[0]+1), Base_DR, 'r--')
	#plt.xticks(np.arange(1, Rate.shape[0] + 1, 25), rotation=45);
	#plt.yticks(np.arange(min(application_DR),max(application_DR),1e9));
	#plt.legend(['Single Association (SA)','Dual Association (DA)', 'DA + Minimum Rate', 'DA + Constrained Backhaul (CB) [1% Bound Gap]', 'DA + CB + Constrained Path Latency (CPL) [1% Bound Gap]', 'DA + Minimum Rate + CPL', 'SA + Minimum Rate', 'SA + Minimum Rate + CPL', 'SA + CB [1% Bound Gap]', 'SA + CB + CPL [1% Bound Gap]'])
	#plt.grid(which= 'major',axis= 'both');
	#plt.title('Per Application Data Rate (SA + CB + CPL)')
	#plt.show()