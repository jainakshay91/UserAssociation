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


MCMC_iter = scn.MCMC_iter; # Number of Iterations to be analyzed
simdata_path = os.getcwd() + '/Data/Process/'
constraint_fp = {'Baseline':'0000', 'DC':'1000', 'DC_MRT':'1100','DC_BHCAP':'1010', 'DC_BHCAP_LAT':'1011', 'DC_LAT':'1001', 'DC_MRT_LAT':'1101', 'SA_MRT':'0100','SA_BHCAP':'0010','SA_BHCAP_LAT':'0011','SA_LAT':'0001','SA_MRT_LAT':'0101'}
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
Net_Throughput_SA_MRT_LAT = copy.copy(Net_Throughput);
Net_Throughput_DC_MRT_LAT = copy.copy(Net_Throughput);

iters_infeas = [0]*num_iter; # Infeasible iteration numbers 
iters_infeas_DC = [0]*num_iter; 
iters_infeas_DC_MRT = [0]*num_iter;
iters_infeas_DC_BHCAP = [0]*num_iter;
iters_infeas_DC_BHCAP_LAT = [0]*num_iter;
iters_infeas_DC_LAT = [0]*num_iter;
iters_infeas_SA_MRT = [0]*num_iter;
iters_infeas_SA_LAT = [0]*num_iter;
iters_infeas_SA_BHCAP = [0]*num_iter;
iters_infeas_SA_BHCAP_LAT = [0]*num_iter;
iters_infeas_SA_MRT_LAT = [0]*num_iter;
iters_infeas_DC_MRT_LAT = [0]*num_iter;

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
application_DR_SA_MRT_LAT = [];
application_DR_DC_MRT_LAT = [];

avg_idx = []; # This is for calculating the average application throughput 


# ========================
# Jain's Fairness Function
# ========================

def jains_fairness(Data, idx_vec):
	
	mean_vec = []; # List to hold the mean values of all iterations
	x2_mean_vec = []; # List to hold the variance of all iterations
	jfr_vec = []; # Jain's Fairness Index
	idx_begin = 0; # Starting index 
	for z in range(0,len(idx_vec)):
		D_Base = Data[idx_begin:idx_begin+idx_vec[z]];
		x2_mean_vec.append(np.mean(np.power(D_Base,2)));
		mean_vec.append(np.mean(D_Base));
		jfr_vec.append((mean_vec[z]**2)/x2_mean_vec[z]);
		idx_begin = idx_begin + idx_vec[z]; # Increasing the index	

	return jfr_vec


# =====================
# Zero Division Checker
# =====================

def zero_div(num, denom):
	output = np.empty((num.shape[0],1))
	for i in range(0, denom.shape[0]):
		if denom[i] == 0:
			output[i] = 0
		else:
			output[i] = num[i]/denom[i]
	return output 

# ==============
# Data Extractor
# ==============

for i in range(0,MCMC_iter):

	# ================================
	# Load the Data from the Optimizer


	Baseline_dat = np.load(simdata_path +'_'+ str(i) +'dat_' + constraint_fp['Baseline'] + '.npz', allow_pickle='True')
	Dat_DC = np.load(simdata_path +'_'+ str(i) +'dat_' + constraint_fp['DC'] + '.npz',  allow_pickle='True')
	Dat_DC_MRT = np.load(simdata_path +'_'+ str(i) +'dat_' + constraint_fp['DC_MRT'] + '.npz',  allow_pickle='True')
	Dat_DC_BHCAP = np.load(simdata_path +'_'+ str(i) +'dat_' + constraint_fp['DC_BHCAP'] + '.npz',  allow_pickle='True')
	Dat_DC_BHCAP_Lat = np.load(simdata_path +'_'+ str(i) +'dat_' + constraint_fp['DC_BHCAP_LAT'] + '.npz',  allow_pickle='True')
	Dat_DC_Lat = np.load(simdata_path +'_'+ str(i) +'dat_' + constraint_fp['DC_LAT'] + '.npz',  allow_pickle='True')
	Dat_SA_MRT = np.load(simdata_path +'_'+ str(i) +'dat_' + constraint_fp['SA_MRT'] + '.npz',  allow_pickle='True')
	Dat_SA_LAT = np.load(simdata_path +'_'+ str(i) +'dat_' + constraint_fp['SA_LAT'] + '.npz',  allow_pickle='True')
	Dat_SA_BHCAP = np.load(simdata_path +'_'+ str(i) +'dat_' + constraint_fp['SA_BHCAP'] + '.npz',  allow_pickle='True')
	Dat_SA_BHCAP_LAT = np.load(simdata_path +'_'+ str(i) +'dat_' + constraint_fp['SA_BHCAP_LAT'] + '.npz',  allow_pickle='True')
	Dat_SA_MRT_LAT = np.load(simdata_path +'_'+ str(i) +'dat_' + constraint_fp['SA_MRT_LAT'] + '.npz',  allow_pickle='True')
	Dat_DC_MRT_LAT = np.load(simdata_path +'_'+ str(i) +'dat_' + constraint_fp['DC_MRT_LAT'] + '.npz',  allow_pickle='True')


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
	Data_SA_MRT_LAT = Dat_SA_MRT_LAT['arr_0'];
	Data_DC_MRT_LAT = Dat_DC_MRT_LAT['arr_0'];

	for k in range(0,num_iter):
		if Data.item()['Status'+str(k)] == 2:
			Net_Throughput[i,k] = Data.item()['Net_Throughput'+str(k)];
		else:
			Net_Throughput[i,k] = 0; # Zero if its an infeasible solution
			#iters_infeas.append(str(i)+str(k)); # Inserting the iteration number for infeasible solution
			iters_infeas[k] = iters_infeas[k] + 1; # Increment the number of Infeasible solution sets
		if Data_DC.item()['Status'+str(k)] == 2:
			Net_Throughput_DC[i,k] = Data_DC.item()['Net_Throughput'+str(k)];
		else:
			Net_Throughput_DC[i,k] = 0; # Zero if its an infeasible solution
			#iters_infeas_DC.append(str(i)+str(k)); # Inserting the iteration number for infeasible solution
			iters_infeas_DC[k] = iters_infeas_DC[k] + 1; # Increment the number of Infeasible solution sets
		if Data_DC_MRT.item()['Status'+str(k)] == 2:
			#print Data_DC_MRT.item()['Status'+str(k)]
			Net_Throughput_DC_MRT[i,k] = Data_DC_MRT.item()['Net_Throughput'+str(k)];
		else:
			#print Data_DC_MRT.item()['Status'+str(k)]
			Net_Throughput_DC_MRT[i,k] = 0; # Zero if its an infeasible solution
			#iters_infeas_DC_MRT.append(str(i)+str(k)); # Inserting the iteration number for infeasible solution
			iters_infeas_DC_MRT[k] = iters_infeas_DC_MRT[k] + 1; # Increment the number of infeasible solutions
		if Data_DC_BHCAP.item()['Status'+str(k)] == 2:
			Net_Throughput_DC_BHCAP[i,k] = Data_DC_BHCAP.item()['Net_Throughput'+str(k)];
		else:
			Net_Throughput_DC_BHCAP[i,k] = 0; # Zero if its an infeasible solution
			#iters_infeas_DC_BHCAP.append(str(i)+str(k)); # Inserting the iteration number for infeasible solution
			iters_infeas_DC_BHCAP[k] = iters_infeas_DC_BHCAP[k] + 1; # Increment the number of infeasible solutions
		if Data_DC_BHCAP_LAT.item()['Status'+str(k)] == 2:
			Net_Throughput_DC_BHCAP_LAT[i,k] = Data_DC_BHCAP_LAT.item()['Net_Throughput'+str(k)];
		else:
			Net_Throughput_DC_BHCAP_LAT[i,k] = 0; # Zero if its an infeasible solution
			#iters_infeas_DC_BHCAP_LAT.append(str(i)+str(k)); # Inserting the iteration number for infeasible solution
			iters_infeas_DC_BHCAP_LAT[k] = iters_infeas_DC_BHCAP_LAT[k] + 1; # Increment the number of infeasible solution
		if Data_DC_LAT.item()['Status'+str(k)] == 2:
			Net_Throughput_DC_LAT[i,k] = Data_DC_LAT.item()['Net_Throughput'+str(k)];
		else:
			Net_Throughput_DC_LAT[i,k] = 0; # Zero if its an infeasible solution
			#iters_infeas_DC_LAT.append(str(i)+str(k)); # Inserting the iteration number for infeasible solution
			iters_infeas_DC_LAT[k] = iters_infeas_DC_LAT[k] + 1; # Increment the number of infeasible solution
		if Data_SA_MRT.item()['Status'+str(k)] == 2:
			Net_Throughput_SA_MRT[i,k] = Data_SA_MRT.item()['Net_Throughput'+str(k)];
		else:
			Net_Throughput_SA_MRT[i,k] = 0; # Zero if its an infeasible solution
			#iters_infeas_SA_MRT.append(str(i)+str(k)); # Inserting the iteration number for infeasible solution
			iters_infeas_SA_MRT[k] = iters_infeas_SA_MRT[k] + 1; # Increment the number of infeasible solution
		if Data_SA_LAT.item()['Status'+str(k)] == 2:
			Net_Throughput_SA_LAT[i,k] = Data_SA_LAT.item()['Net_Throughput'+str(k)];
		else:
			Net_Throughput_SA_LAT[i,k] = 0; # Zero if its an infeasible solution
			#iters_infeas_SA_LAT.append(str(i)+str(k)); # Inserting the iteration number for infeasible solution
			iters_infeas_SA_LAT[k] = iters_infeas_SA_LAT[k] + 1; # Increment the number of infeasible solution
		if Data_SA_BHCAP.item()['Status'+str(k)] == 2:
			Net_Throughput_SA_BHCAP[i,k] = Data_SA_BHCAP.item()['Net_Throughput'+str(k)];
		else: 
			Net_Throughput_SA_BHCAP[i,k] = 0; #Zero if its an infeasible solution
			#iters_infeas_SA_BHCAP.append(str(i)+str(k)); # Inserting the iteration number for infeasible solution
			iters_infeas_SA_BHCAP[k] = iters_infeas_SA_BHCAP[k] + 1; # Increment the number of infeasible solution
		if Data_SA_BHCAP_LAT.item()['Status'+str(k)] == 2:
			Net_Throughput_SA_BHCAP_LAT[i,k] = Data_SA_BHCAP_LAT.item()['Net_Throughput'+str(k)];
		else:
			Net_Throughput_SA_BHCAP_LAT[i,k] = 0; #Zero if its an infeasible solution
			#iters_infeas_SA_BHCAP_LAT.append(str(i)+str(k)); # Inserting the iteration number for infeasible solution
			iters_infeas_SA_BHCAP_LAT[k] = iters_infeas_SA_BHCAP_LAT[k] + 1; # Increment the number of infeasible solution
		if Data_SA_MRT_LAT.item()['Status'+str(k)] == 2:
			Net_Throughput_SA_MRT_LAT[i,k] = Data_SA_MRT_LAT.item()['Net_Throughput'+str(k)];
		else:
			Net_Throughput_SA_MRT_LAT[i,k] = 0; #Zero if its an infeasible solution
			#iters_infeas_SA_BHCAP_LAT.append(str(i)+str(k)); # Inserting the iteration number for infeasible solution
			iters_infeas_SA_MRT_LAT[k] = iters_infeas_SA_MRT_LAT[k] + 1; # Increment the number of infeasible solution
		if Data_DC_MRT_LAT.item()['Status'+str(k)] == 2:
			Net_Throughput_DC_MRT_LAT[i,k] = Data_DC_MRT_LAT.item()['Net_Throughput'+str(k)];
		else:
			Net_Throughput_DC_MRT_LAT[i,k] = 0; #Zero if its an infeasible solution
			#iters_infeas_SA_BHCAP_LAT.append(str(i)+str(k)); # Inserting the iteration number for infeasible solution
			iters_infeas_DC_MRT_LAT[k] = iters_infeas_DC_MRT_LAT[k] + 1; # Increment the number of infeasible solution
	#print "=================="
	#print Net_Throughput
	#print "=================="
	#print Net_Throughput_DC_MRT
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
	X_Optimal_SA_MRT_LAT = copy.copy(X_Optimal);
	X_Optimal_DC_MRT_LAT = copy.copy(X_Optimal);

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
	Rate_SA_MRT_LAT = copy.copy(Rate);
	Rate_DC_MRT_LAT = copy.copy(Rate);


	if Data.item()['Status'+str(10)] == 2:
		X_Optimal = Data.item()['X_optimal_data'+str(10)];
		Rate = Data.item()['Rates'+str(10)];
	else:
		pass
	if Data_DC.item()['Status'+str(10)] == 2:
		X_Optimal_DC = Data_DC.item()['X_optimal_data'+str(10)];
		Rate_DC = Data_DC.item()['Rates'+str(10)];
	else:
		pass
	if Data_DC_MRT.item()['Status'+str(10)] == 2:
		X_Optimal_DC_MRT = Data_DC_MRT.item()['X_optimal_data'+str(10)];
		Rate_DC_MRT = Data_DC_MRT.item()['Rates'+str(10)];
	else:
		pass
	if Data_DC_BHCAP.item()['Status'+str(10)] == 2:
		X_Optimal_DC_BHCAP = Data_DC_BHCAP.item()['X_optimal_data'+str(10)];
		Rate_DC_BHCAP = Data_DC_BHCAP.item()['Rates'+str(10)];
	else:
		pass
	if Data_DC_BHCAP_LAT.item()['Status'+str(10)] == 2:
		X_Optimal_DC_BHCAP_LAT = Data_DC_BHCAP_LAT.item()['X_optimal_data'+str(10)];
		Rate_DC_BHCAP_LAT = Data_DC_BHCAP_LAT.item()['Rates'+str(10)];
	else:
		pass
	if Data_DC_LAT.item()['Status'+str(10)] == 2:
		X_Optimal_DC_LAT = Data_DC_LAT.item()['X_optimal_data'+str(10)];
		Rate_DC_LAT = Data_DC_LAT.item()['Rates'+str(10)];
	else:
		pass
	if Data_SA_MRT.item()['Status'+str(10)] == 2:
		X_Optimal_SA_MRT = Data_SA_MRT.item()['X_optimal_data'+str(10)];
		Rate_SA_MRT = Data_SA_MRT.item()['Rates'+str(10)];
	else:
		pass
	if Data_SA_LAT.item()['Status'+str(10)] == 2:
		X_Optimal_SA_LAT = Data_SA_LAT.item()['X_optimal_data'+str(10)];
		Rate_SA_LAT = Data_SA_LAT.item()['Rates'+str(10)];
	else:
		pass
	if Data_SA_BHCAP.item()['Status'+str(10)] == 2:
		X_Optimal_SA_BHCAP = Data_SA_BHCAP.item()['X_optimal_data'+str(10)];
		Rate_SA_BHCAP = Data_SA_BHCAP.item()['Rates'+str(10)];
	else:
		pass
	if Data_SA_BHCAP_LAT.item()['Status'+str(10)] == 2:
		X_Optimal_SA_BHCAP_LAT = Data_SA_BHCAP_LAT.item()['X_optimal_data'+str(10)];
		Rate_SA_BHCAP_LAT = Data_SA_BHCAP_LAT.item()['Rates'+str(10)];
	else:
		pass
	if Data_SA_MRT_LAT.item()['Status'+str(10)] == 2:
		X_Optimal_SA_MRT_LAT = Data_SA_MRT_LAT.item()['X_optimal_data'+str(10)];
		Rate_SA_MRT_LAT = Data_SA_MRT_LAT.item()['Rates'+str(10)];
	else:
		pass
	if Data_DC_MRT_LAT.item()['Status'+str(10)] == 2:
		X_Optimal_DC_MRT_LAT = Data_DC_MRT_LAT.item()['X_optimal_data'+str(10)];
		Rate_DC_MRT_LAT = Data_DC_MRT_LAT.item()['Rates'+str(10)];
	else:
		pass
	
	
	avg_idx.append(X_Optimal.shape[0])	

	for j in range(0,X_Optimal.shape[0]):
		Base_DR.append(scn.eMBB_minrate); 
		if Data.item()['Status'+str(10)] == 2:
			application_DR.append(sum(Rate[j,:]*X_Optimal[j,:]));
		else:
			pass
		if Data_DC.item()['Status'+str(10)] == 2:
			application_DR_DC.append(sum(Rate_DC[j,:]*X_Optimal_DC[j,:]));
		else:
			pass
		if Data_DC_MRT.item()['Status'+str(10)] == 2:
			application_DR_DC_MRT.append(sum(Rate_DC_MRT[j,:]*X_Optimal_DC_MRT[j,:]));
		else:
			pass
		if Data_DC_BHCAP.item()['Status'+str(10)] == 2:
			application_DR_DC_BHCAP.append(sum(Rate_DC_BHCAP[j,:]*X_Optimal_DC_BHCAP[j,:]));
		else:
			pass
		if Data_DC_BHCAP_LAT.item()['Status'+str(10)] == 2:
			application_DR_DC_BHCAP_LAT.append(sum(Rate_DC_BHCAP_LAT[j,:]*X_Optimal_DC_BHCAP_LAT[j,:]));
		else:
			pass
		if Data_DC_LAT.item()['Status'+str(10)] == 2:
			application_DR_DC_LAT.append(sum(Rate_DC_LAT[j,:]*X_Optimal_DC_LAT[j,:]));
		else:
			pass
		if Data_SA_MRT.item()['Status'+str(10)] == 2:
			application_DR_SA_MRT.append(sum(Rate_SA_MRT[j,:]*X_Optimal_SA_MRT[j,:]));
		else:
			pass
		if Data_SA_LAT.item()['Status'+str(10)] == 2:
			application_DR_SA_LAT.append(sum(Rate_SA_LAT[j,:]*X_Optimal_SA_LAT[j,:]));
		else: 
			pass
		if Data_SA_BHCAP.item()['Status'+str(10)] == 2:
			application_DR_SA_BHCAP.append(sum(Rate_SA_BHCAP[j,:]*X_Optimal_SA_BHCAP[j,:]));
		else:
			pass
		if Data_SA_BHCAP_LAT.item()['Status'+str(10)] == 2:
			application_DR_SA_BHCAP_LAT.append(sum(Rate_SA_BHCAP_LAT[j,:]*X_Optimal_SA_BHCAP_LAT[j,:]));
		else:
			pass
		if Data_SA_MRT_LAT.item()['Status'+str(10)] == 2:
			application_DR_SA_MRT_LAT.append(sum(Rate_SA_MRT_LAT[j,:]*X_Optimal_SA_MRT_LAT[j,:]));
		else:
			pass
		if Data_DC_MRT_LAT.item()['Status'+str(10)] == 2:
			application_DR_DC_MRT_LAT.append(sum(Rate_DC_MRT_LAT[j,:]*X_Optimal_DC_MRT_LAT[j,:]));
		else:
			pass


# ===============
# Analysis Values
# ===============

# ==============
# Net Throughput


Net_Throughput_avg = zero_div(np.sum(Net_Throughput, axis = 0),(MCMC_iter - np.array(iters_infeas))) ; # We get the average throughput over MCMC Iteratios
Net_Throughput_DC_avg = zero_div(np.sum(Net_Throughput_DC, axis = 0),(MCMC_iter - np.array(iters_infeas_DC))); # Average throughput
Net_Throughput_DC_MRT_avg = zero_div(np.sum(Net_Throughput_DC_MRT, axis = 0),(MCMC_iter - np.array(iters_infeas_DC_MRT))); # DC + MRT Average throughput
Net_Throughput_DC_BHCAP_avg = zero_div(np.sum(Net_Throughput_DC_BHCAP, axis = 0),(MCMC_iter - np.array(iters_infeas_DC_BHCAP))); # DC + BHCAP Average throughput
Net_Throughput_DC_LAT_avg = zero_div(np.sum(Net_Throughput_DC_LAT, axis = 0),(MCMC_iter - np.array(iters_infeas_DC_LAT))); # DC + LAT Average throughput
Net_Throughput_DC_BHCAP_LAT_avg = zero_div(np.sum(Net_Throughput_DC_BHCAP_LAT, axis = 0),(MCMC_iter - np.array(iters_infeas_DC_BHCAP_LAT))); # DC + BHCAP + LAT Average throughput
Net_Throughput_SA_MRT_avg = zero_div(np.sum(Net_Throughput_SA_MRT, axis = 0),(MCMC_iter - np.array(iters_infeas_SA_MRT))); # SA + MRT average 
Net_Throughput_SA_LAT_avg = zero_div(np.sum(Net_Throughput_SA_LAT, axis = 0),(MCMC_iter - np.array(iters_infeas_SA_LAT))); # SA + LAT average
Net_Throughput_SA_BHCAP_avg = zero_div(np.sum(Net_Throughput_SA_BHCAP, axis = 0),(MCMC_iter - np.array(iters_infeas_SA_BHCAP))); # SA + BHCAP average
Net_Throughput_SA_BHCAP_LAT_avg = zero_div(np.sum(Net_Throughput_SA_BHCAP_LAT, axis = 0),(MCMC_iter - np.array(iters_infeas_SA_BHCAP_LAT))); # SA + BHCAP + LAT average
Net_Throughput_SA_MRT_LAT_avg = zero_div(np.sum(Net_Throughput_SA_MRT_LAT, axis = 0),(MCMC_iter - np.array(iters_infeas_SA_MRT_LAT))); # SA + LAT average
Net_Throughput_DC_MRT_LAT_avg = zero_div(np.sum(Net_Throughput_DC_MRT_LAT, axis = 0),(MCMC_iter - np.array(iters_infeas_DC_MRT_LAT))); # SA + LAT average
#print iters_infeas_DC_MRT
# ========================================
# Jain's Fairness Index and t-student test

jfr_SA = jains_fairness(application_DR, avg_idx);
jfr_DC = jains_fairness(application_DR_DC, avg_idx);
jfr_DC_MRT = jains_fairness(application_DR_DC_MRT, avg_idx); 
jfr_DC_BHCAP = jains_fairness(application_DR_DC_BHCAP, avg_idx);
jfr_DC_BHCAP_LAT = jains_fairness(application_DR_DC_BHCAP_LAT, avg_idx);
jfr_DC_LAT = jains_fairness(application_DR_DC_LAT, avg_idx);
jfr_SA_MRT = jains_fairness(application_DR_SA_MRT, avg_idx);
jfr_SA_LAT = jains_fairness(application_DR_SA_LAT, avg_idx);
jfr_SA_BHCAP = jains_fairness(application_DR_SA_BHCAP, avg_idx);
jfr_SA_BHCAP_LAT = jains_fairness(application_DR_SA_BHCAP_LAT, avg_idx);
jfr_SA_MRT_LAT = jains_fairness(application_DR_SA_MRT_LAT, avg_idx);
jfr_DC_MRT_LAT = jains_fairness(application_DR_DC_MRT_LAT, avg_idx);





# ===============
# Throughput Plot

x_axis = np.arange(scn.num_users_min, scn.num_users_max, scn.user_steps_siml);
y_min = np.amin([np.amin(Net_Throughput_avg), np.amin(Net_Throughput_DC_avg), np.amin(Net_Throughput_DC_MRT_avg), np.amin(Net_Throughput_DC_BHCAP_avg), np.amin(Net_Throughput_DC_BHCAP_LAT_avg), np.amin(Net_Throughput_DC_LAT_avg), np.amin(Net_Throughput_SA_MRT_avg), np.amin(Net_Throughput_SA_LAT_avg), np.amin(Net_Throughput_SA_BHCAP_avg), np.amin(Net_Throughput_SA_BHCAP_LAT_avg), np.amin(Net_Throughput_SA_MRT_LAT_avg), np.amin(Net_Throughput_DC_MRT_LAT_avg)]);
y_max = np.max([np.amax(Net_Throughput_avg), np.amax(Net_Throughput_DC_avg), np.amax(Net_Throughput_DC_MRT_avg), np.amax(Net_Throughput_DC_BHCAP_avg), np.amax(Net_Throughput_DC_BHCAP_LAT_avg), np.amax(Net_Throughput_DC_LAT_avg), np.amax(Net_Throughput_SA_MRT_avg), np.amax(Net_Throughput_SA_LAT_avg), np.amax(Net_Throughput_SA_BHCAP_avg), np.amax(Net_Throughput_SA_BHCAP_LAT_avg), np.amax(Net_Throughput_SA_MRT_LAT_avg), np.amax(Net_Throughput_DC_MRT_LAT_avg)]);
#plotter.plotter('dashline',np.arange(scn.num_users_min, scn.num_users_max, scn.user_steps_siml),Net_Throughput_avg,5,10,1,45,0,0,1,'major','both', 'yes', 'Total Network Throughput', np)
plt.plot(x_axis, Net_Throughput_avg, 'r--*', x_axis, Net_Throughput_DC_avg, 'b--*' , x_axis, Net_Throughput_DC_MRT_avg, 'g-.', x_axis, Net_Throughput_DC_BHCAP_avg, 'k--s', x_axis, Net_Throughput_DC_BHCAP_LAT_avg, 'm--d', x_axis , Net_Throughput_DC_LAT_avg, 'c--p',x_axis, Net_Throughput_SA_MRT_avg, 'k-.', x_axis, Net_Throughput_SA_LAT_avg, 'b:', x_axis, Net_Throughput_SA_BHCAP_avg, 'g--D', x_axis, Net_Throughput_SA_BHCAP_LAT_avg, 'r:', x_axis, Net_Throughput_SA_MRT_LAT_avg, 'r-o', x_axis, Net_Throughput_DC_MRT_LAT_avg, 'k-o');
plt.xticks(np.arange(scn.num_users_min, scn.num_users_max, scn.user_steps_siml));
plt.yticks(np.arange(y_min,y_max,5e10));
plt.legend(['Single Association (SA)','Dual Connectivity (DC)', 'DC + Minimum Rate', 'DC + Constrained Backhaul (CB) [1% Bound Gap]', 'DC + CB + Constrained Path Latency (CPL) [1% Bound Gap]', 'DC + CPL', 'SA + Minimum Rate', 'SA + CPL', 'SA + CB [1% Bound Gap]', 'SA + CB + CPL [1% Bound Gap]','SA + Minimum Rate + CPL', 'DC + Minimum Rate + CPL'], loc='upper left', bbox_to_anchor=(0., 0.5, 0.5, 0.5), prop={'size': 6})
plt.grid(which= 'major',axis= 'both');
plt.title('Network Wide Throughput')
plt.savefig('NetThrough', dpi=1200, facecolor='w', edgecolor='w',
        orientation='landscape', papertype='letter', format='png',
        transparent=False, bbox_inches='tight', pad_inches=0.1,
        frameon=None, metadata=None)


# ================
# Fairness BoxPlot

box_data = [jfr_SA, jfr_DC, jfr_DC_MRT, jfr_SA_MRT, jfr_SA_LAT, jfr_DC_LAT, jfr_DC_BHCAP, jfr_SA_BHCAP, jfr_SA_BHCAP_LAT, jfr_DC_BHCAP_LAT, jfr_DC_MRT_LAT, jfr_SA_MRT_LAT] 
fig, ax = plt.subplots()
plt.title('Jain\'s Fairness Index Deviation')
plt.boxplot(box_data)
plt.xticks(range(1,13), ['SA', 'DC', 'DC+MRT', 'SA+MRT', 'SA+LAT', 'DC+LAT', 'DC+BHCAP', 'SA+BHCAP', 'SA+BHCAP+LAT', 'DC+BHCAP+LAT', 'DC+MRT+LAT', 'SA+MRT+LAT'], fontsize = 8, rotation = '45')
plt.savefig('Boxplot', dpi=1200, facecolor='w', edgecolor='w',
        orientation='landscape', papertype='letter', format='png',
        transparent=False, bbox_inches='tight', pad_inches=0.1,
        frameon=None, metadata=None)

#print application_DR.tolist()
#plt.bar(np.arange(1,Rate.shape[0]+1),application_DR_SA_BHCAP_LAT)
#plt.plot(np.arange(1,Rate.shape[0]+1), Base_DR, 'r--')
#plt.xticks(np.arange(1, Rate.shape[0] + 1, 25), rotation=45);
#plt.yticks(np.arange(min(application_DR),max(application_DR),1e9));
#plt.legend(['Single Association (SA)','Dual Association (DA)', 'DA + Minimum Rate', 'DA + Constrained Backhaul (CB) [1% Bound Gap]', 'DA + CB + Constrained Path Latency (CPL) [1% Bound Gap]', 'DA + Minimum Rate + CPL', 'SA + Minimum Rate', 'SA + Minimum Rate + CPL', 'SA + CB [1% Bound Gap]', 'SA + CB + CPL [1% Bound Gap]'])
#plt.grid(which= 'major',axis= 'both');
#plt.title('Per Application Data Rate (SA + CB + CPL)')
#plt.show()
