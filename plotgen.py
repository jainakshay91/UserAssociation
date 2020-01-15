#!/usr/bin/env python 

# =============================
# Import the Necessary Binaries
# =============================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scenario_var import scenario_var 
import copy 
import os, sys
import plotter
import zipfile
import csvsaver

# =======================
# Generate Class Variable
# =======================

scn = scenario_var(); # Getting the class object

# ==================================
# Initialize the Necessary Variables
# ==================================


MCMC_iter = scn.MCMC_iter; # Number of Iterations to be analyzed
#simdata_path = '/home/akshayjain/Desktop/Results/mMTC/CircularDeploy_AnyDC/BF/Process/'
#simdata_path = '/home/akshayjain/Desktop/Results/eMBB/Result_files/SquareDeploy_AnyDC/IF/MRT1/Process/'
simdata_path = '/home/akshayjain/Desktop/Results/Relaxed_Constraints/mMTC/Sq_BF_AnyDC/Plots/BH_MRT_Relax/Process5/'

constraint_fp = {'Baseline':'0000', 'DC':'1000', 'DC_MRT':'1100','DC_BHCAP':'1010', 'DC_BHCAP_LAT':'1011', 'DC_LAT':'1001', 'DC_MRT_LAT':'1101', 'SA_MRT':'0100','SA_BHCAP':'0010','SA_BHCAP_LAT':'0011','SA_LAT':'0001','SA_MRT_LAT':'0101', 'DC_MRT_BHCAP':'1110', 'DC_MRT_BHCAP_LAT':'1111', 'SA_MRT_BHCAP':'0110','SA_MRT_BHCAP_LAT':'0111'}
num_iter = ((scn.num_users_max - scn.num_users_min)/scn.user_steps_siml); 
print num_iter-1

rate_matrix_DC = []
rate_matrix_DC_MRT = []
rate_matrix_DC_MRT_LAT = []
rate_matrix_DC_BHCAP_LAT = []
rate_matrix_DC_BHCAP = []
rate_matrix_DC_LAT = []
rate_matrix_SA_MRT = []
rate_matrix_SA_LAT = []
rate_matrix_SA_BHCAP = []
rate_matrix_SA_BHCAP_LAT = []
rate_matrix_SA_MRT_LAT = []
rate_matrix_SA = []
rate_matrix_DC_MRT_BHCAP = []
rate_matrix_DC_MRT_BHCAP_LAT = []
rate_matrix_SA_MRT_BHCAP = []
rate_matrix_SA_MRT_BHCAP_LAT = []


bhutil_val_DC = []
bhutil_val_DC_BHCAP = []
bhutil_val_DC_BHCAP_LAT = []
bhutil_val_DC_MRT_BHCAP = []
bhutil_val_DC_MRT_BHCAP_LAT = []

latprov_DC_MRT_LAT = []
latprov_DC = []
latprov_DC_BHCAP_LAT = []
latprov_DC_MRT_BHCAP_LAT = []
latprov_DC_LAT = []
avail_bh = []

Net_Throughput = np.empty((MCMC_iter, num_iter));
Net_Throughput_DC = copy.deepcopy(Net_Throughput);
Net_Throughput_DC_MRT = copy.deepcopy(Net_Throughput);
Net_Throughput_DC_BHCAP = copy.deepcopy(Net_Throughput);
Net_Throughput_DC_BHCAP_LAT = copy.deepcopy(Net_Throughput);
Net_Throughput_DC_LAT = copy.deepcopy(Net_Throughput);
Net_Throughput_SA_MRT = copy.deepcopy(Net_Throughput);
Net_Throughput_SA_LAT = copy.deepcopy(Net_Throughput);
Net_Throughput_SA_BHCAP = copy.deepcopy(Net_Throughput);
Net_Throughput_SA_BHCAP_LAT = copy.deepcopy(Net_Throughput);
Net_Throughput_SA_MRT_LAT = copy.deepcopy(Net_Throughput);
Net_Throughput_DC_MRT_LAT = copy.deepcopy(Net_Throughput);
Net_Throughput_DC_MRT_BHCAP = copy.deepcopy(Net_Throughput);
Net_Throughput_DC_MRT_BHCAP_LAT = copy.deepcopy(Net_Throughput);
Net_Throughput_SA_MRT_BHCAP = copy.deepcopy(Net_Throughput);
Net_Throughput_SA_MRT_BHCAP_LAT = copy.deepcopy(Net_Throughput);


B_Dat_DR = copy.deepcopy(Net_Throughput);
B_Dat_DR_fs = copy.deepcopy(Net_Throughput);
B_Dat_DR_sn = copy.deepcopy(Net_Throughput);
B_Dat_DR_sn_fs = copy.deepcopy(Net_Throughput);

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
iters_infeas_DC_MRT_BHCAP = [0]*num_iter;
iters_infeas_DC_MRT_BHCAP_LAT = [0]*num_iter;
iters_infeas_SA_MRT_BHCAP = [0]*num_iter;
iters_infeas_SA_MRT_BHCAP_LAT = [0]*num_iter;



iters_timeout = [0]*num_iter; # Infeasible iteration numbers 
iters_timeout_DC = [0]*num_iter; 
iters_timeout_DC_MRT = [0]*num_iter;
iters_timeout_DC_BHCAP = [0]*num_iter;
iters_timeout_DC_BHCAP_LAT = [0]*num_iter;
iters_timeout_DC_LAT = [0]*num_iter;
iters_timeout_SA_MRT = [0]*num_iter;
iters_timeout_SA_LAT = [0]*num_iter;
iters_timeout_SA_BHCAP = [0]*num_iter;
iters_timeout_SA_BHCAP_LAT = [0]*num_iter;
iters_timeout_SA_MRT_LAT = [0]*num_iter;
iters_timeout_DC_MRT_LAT = [0]*num_iter;
iters_timeout_DC_MRT_BHCAP = [0]*num_iter;
iters_timeout_DC_MRT_BHCAP_LAT = [0]*num_iter;
iters_timeout_SA_MRT_BHCAP = [0]*num_iter;
iters_timeout_SA_MRT_BHCAP_LAT = [0]*num_iter;


Base_DR = [];
Base_DR_fs = [];
application_DR = [];
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
application_DR_DC_MRT_BHCAP = [];
application_DR_DC_MRT_BHCAP_LAT = [];
application_DR_SA_MRT_BHCAP = [];
application_DR_SA_MRT_BHCAP_LAT = [];


AU_Base_DR = np.zeros((MCMC_iter, num_iter))
AU_Base_DR_fs = np.zeros((MCMC_iter, num_iter))
AU_DR = np.zeros((MCMC_iter, num_iter))
AU_DR_DC = np.zeros((MCMC_iter, num_iter))
AU_DR_DC_MRT = np.zeros((MCMC_iter, num_iter))
AU_DR_DC_BHCAP = np.zeros((MCMC_iter, num_iter))
AU_DR_DC_BHCAP_LAT = np.zeros((MCMC_iter, num_iter))
AU_DR_DC_LAT = np.zeros((MCMC_iter, num_iter))
AU_DR_SA_MRT = np.zeros((MCMC_iter, num_iter))
AU_DR_SA_LAT = np.zeros((MCMC_iter, num_iter))
AU_DR_SA_BHCAP = np.zeros((MCMC_iter, num_iter))
AU_DR_SA_BHCAP_LAT = np.zeros((MCMC_iter, num_iter))
AU_DR_SA_MRT_LAT = np.zeros((MCMC_iter, num_iter)) 
AU_DR_DC_MRT_LAT = np.zeros((MCMC_iter, num_iter))
AU_DR_DC_MRT_BHCAP = np.zeros((MCMC_iter, num_iter))
AU_DR_DC_MRT_BHCAP_LAT = np.zeros((MCMC_iter, num_iter))
AU_DR_SA_MRT_BHCAP = np.zeros((MCMC_iter, num_iter))
AU_DR_SA_MRT_BHCAP_LAT = np.zeros((MCMC_iter, num_iter))

time_DC = np.zeros((MCMC_iter, num_iter))
time_DC_MRT = np.zeros((MCMC_iter, num_iter))
time_SA_MRT = np.zeros((MCMC_iter, num_iter))
time_DC_MRT_BHCAP = np.zeros((MCMC_iter, num_iter))
time_DC_MRT_BHCAP_LAT = np.zeros((MCMC_iter, num_iter))
time_DC_MRT_LAT = np.zeros((MCMC_iter, num_iter))
time_DC_LAT = np.zeros((MCMC_iter, num_iter))
time_DC_BHCAP_LAT = np.zeros((MCMC_iter, num_iter))
time_DC_BHCAP = np.zeros((MCMC_iter, num_iter))
time_SA_MRT_BHCAP = np.zeros((MCMC_iter, num_iter))
time_SA_MRT_BHCAP_LAT = np.zeros((MCMC_iter, num_iter))
time_SA_MRT_LAT = np.zeros((MCMC_iter, num_iter))
time_SA = np.zeros((MCMC_iter, num_iter))
time_SA_LAT = np.zeros((MCMC_iter, num_iter))
time_SA_BHCAP = np.zeros((MCMC_iter, num_iter))
time_SA_BHCAP_LAT = np.zeros((MCMC_iter, num_iter))




DC_avg_rt = []
DC_MRT_avg_rt = []
DC_MRT_BHCAP_avg_rt = []
SA_avg_rt = []

avg_idx = []; # This is for calculating the average application throughput 


DC_BW = []
DC_MRT_BW_MC = []
DC_MRT_BW_SC = []
DC_MRT_BW_TOT = []
SINR_DC_BW = []
SINR_DC_MRT_BW_MC = []
SINR_DC_MRT_BW_SC = []
SINR_SA_BW = []
idx_MRT_BW_SC = []
idx_MRT_BW_MC = []
# ========================
# Jain's Fairness Function
# ========================

def jains_fairness(Data, idx_vec):
	#print idx_vec
	#mean_vec = []; # List to hold the mean values of all iterations
	#x2_mean_vec = []; # List to hold the variance of all iterations
	jfr_vec = []; # Jain's Fairness Index
	idx_begin = 0; # Starting index 
	# print idx_vec
	# print len(idx_vec)
	# print len(Data)
	for z in range(0,len(idx_vec)):
		if idx_vec[z] != 0:
			#print idx_vec[z]
			D_Base = Data[idx_begin:idx_begin+int(idx_vec[z])];
			denom = np.mean(np.power(D_Base,2));
			num = np.mean(D_Base);
			jfr_vec.append((num**2)/denom);
			idx_begin = idx_begin + int(idx_vec[z]); # Increasing the index	

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

def baseline_cal(i,k,simdata_path):
	filename = simdata_path + 'Baseline' + str(i) + str(k) + '.npz';

	if zipfile.ZipFile(filename, mode='r',).testzip() == None:
		#print simdata_path + 'Baseline' + str(i) + str(k) + '.npz'
		B_Dat = np.load(simdata_path + 'Baseline' + str(i) + str(k) + '.npz')
		#B_Dat_fs = np.load(simdata_path + 'Baseline_minrate' + str(i) + str(k) + '.npz')
		B_Dat_DR = B_Dat['arr_0']
		B_DAT_user = B_Dat['arr_1']
		#B_DAT_DR_fs = B_Dat_fs['arr_0']
		#B_DAT_DR_user = B_Dat_fs['arr_1']
		return B_Dat_DR, B_DAT_user
	else:
		print ("Erroneous CRC-32 for file:", 'Baseline' + str(i) + str(k) + '.npz')
		return 0,0

def user_count(Optim_mat):
	Assoc_users = 0; # Initialize the number of associated users
	for i in range(Optim_mat.shape[0]):
		for j in range(Optim_mat.shape[1]):
			if Optim_mat[i,j] == 1:
				Assoc_users = Assoc_users + 1;
				break;
	return Assoc_users

# ==============
# Data Extractor
# ==============

for i in range(0,MCMC_iter):

	# ================================
	# Load the Data from the Optimizer

	Baseline_dat = np.load(simdata_path +'_'+ str(i) +'dat_' + constraint_fp['Baseline'] + '.npz', allow_pickle='True') # Single Association
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
	Dat_DC_MRT_BHCAP = np.load(simdata_path +'_'+ str(i) +'dat_' + constraint_fp['DC_MRT_BHCAP'] + '.npz',  allow_pickle='True')
	Dat_DC_MRT_BHCAP_LAT = np.load(simdata_path +'_'+ str(i) +'dat_' + constraint_fp['DC_MRT_BHCAP_LAT'] + '.npz',  allow_pickle='True')
	Dat_SA_MRT_BHCAP = np.load(simdata_path +'_'+ str(i) +'dat_' + constraint_fp['SA_MRT_BHCAP'] + '.npz',  allow_pickle='True')
	Dat_SA_MRT_BHCAP_LAT = np.load(simdata_path +'_'+ str(i) +'dat_' + constraint_fp['SA_MRT_BHCAP_LAT'] + '.npz',  allow_pickle='True')


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
	Data_DC_MRT_BHCAP = Dat_DC_MRT_BHCAP['arr_0'];
	Data_DC_MRT_BHCAP_LAT = Dat_DC_MRT_BHCAP_LAT['arr_0'];
	Data_SA_MRT_BHCAP = Dat_SA_MRT_BHCAP['arr_0'];
	Data_SA_MRT_BHCAP_LAT = Dat_SA_MRT_BHCAP_LAT['arr_0'];

	if Data_DC_BHCAP.item()['Status'+str(num_iter-1)] == 2:
		if Data_DC.item()['Status'+str(num_iter-1)] == 2:
			bhutil_val_DC = (Data_DC.item()['BHUTIL'+str(num_iter-1)].tolist())
			latprov_DC = (Data_DC.item()['LatenOff'+str(num_iter-1)].tolist())
		else:
			bhutil_val_DC = (np.zeros(Data_DC.item()['APs'+str(num_iter-1)]).tolist())
			latprov_DC = (np.zeros(Data_DC.item()['APs'+str(num_iter-1)]).tolist())
		
		if Data_DC_LAT.item()['Status'+str(num_iter-1)] == 2:
			latprov_DC_LAT = (Data_DC_LAT.item()['LatenOff'+str(num_iter-1)])
			#print latprov_DC_LAT
		else:
			latprov_DC_LAT = (np.zeros(Data_DC_LAT.item()['APs'+str(num_iter-1)]).tolist())

		if Data_DC_BHCAP_LAT.item()['Status'+str(num_iter-1)] == 2: 
			bhutil_val_DC_BHCAP_LAT = (Data_DC_BHCAP_LAT.item()['BHUTIL'+str(num_iter-1)].tolist())
			latprov_DC_BHCAP_LAT = (Data_DC_BHCAP_LAT.item()['LatenOff'+str(num_iter-1)].tolist())
		else: 
			bhutil_val_DC_BHCAP_LAT = (np.zeros(Data_DC_BHCAP_LAT.item()['APs'+str(num_iter-1)]).tolist())
			latprov_DC_BHCAP_LAT = (np.zeros(Data_DC_BHCAP_LAT.item()['APs'+str(num_iter-1)]).tolist())	
		
		if Data_DC_BHCAP.item()['Status'+str(num_iter-1)] == 2: 
			bhutil_val_DC_BHCAP = (Data_DC_BHCAP.item()['BHUTIL'+str(num_iter-1)].tolist())
		else:
			bhutil_val_DC_BHCAP = (np.zeros(Data_DC_BHCAP.item()['APs'+str(num_iter-1)]).tolist())

		if Data_DC_MRT_BHCAP_LAT.item()['Status'+str(num_iter-1)] == 2: 
			bhutil_val_DC_MRT_BHCAP_LAT = (Data_DC_MRT_BHCAP_LAT.item()['BHUTIL'+str(num_iter-1)].tolist())
			latprov_DC_MRT_BHCAP_LAT = (Data_DC_MRT_BHCAP_LAT.item()['LatenOff'+str(num_iter-1)].tolist())
		else:
			bhutil_val_DC_MRT_BHCAP_LAT = (np.zeros(Data_DC_MRT_BHCAP_LAT.item()['APs'+str(num_iter-1)]).tolist())
			latprov_DC_MRT_BHCAP_LAT = (np.zeros(Data_DC_MRT_BHCAP_LAT.item()['APs'+str(num_iter-1)]).tolist())

		if Data_DC_MRT_LAT.item()['Status'+str(num_iter-1)] == 2: 
			latprov_DC_MRT_LAT = (Data_DC_MRT_LAT.item()['LatenOff'+str(num_iter-1)].tolist())
		else:
			latprov_DC_MRT_LAT = (np.zeros(Data_DC_MRT_LAT.item()['APs'+str(num_iter-1)]).tolist())	

		avail_bh = Data_DC.item()['AvailBHUtil_SC'+str(num_iter-1)]

	if Data_DC_MRT_BHCAP.item()['Status'+str(num_iter-1)] == 2: 
			bhutil_val_DC_MRT_BHCAP = (Data_DC_MRT_BHCAP.item()['BHUTIL'+str(num_iter-1)].tolist())
	else:
			bhutil_val_DC_MRT_BHCAP = (np.zeros(Data_DC_MRT_BHCAP.item()['APs'+str(num_iter-1)]).tolist())

		#print Data_DC_LAT.item()['X_optimal_data'+str(1)]
		
		#print avail_bh

	#if i == 1:
	if Data_DC.item()['Status'+str(0)] == 2:
		#temp = Data_DC.item()['Rates'+str(1)]
		if i == 1:
			rate_matrix_DC = np.sum(Data_DC.item()['Rates'+str(0)], axis = 1).tolist()
			#print len(rate_matrix_DC)
		DC_avg_rt = DC_avg_rt + (np.sum(Data_DC.item()['Rates'+str(0)], axis = 1).tolist())
		#print len(DC_avg_rt)

		# if i == 0:
		# 	DC_avg_rt = rate_matrix_DC
		# else:
		# 	DC_avg_rt = [x + y for x, y in zip(rate_matrix_DC, DC_avg_rt)]
	if Data.item()['Status'+str(0)] == 2:
		#temp1 = Data.item()['Rates'+str(1)]
		if i == 1:
			rate_matrix_SA = np.sum(Data.item()['Rates'+str(0)], axis = 1).tolist()
		SA_avg_rt = SA_avg_rt + (np.sum(Data.item()['Rates'+str(0)], axis = 1).tolist())
		# if i == 0:
		# 	SA_avg_rt = rate_matrix_SA
		# else:
		# 	SA_avg_rt = [x + y for x, y in zip(rate_matrix_SA, SA_avg_rt)]

	if Data_DC_MRT.item()['Status'+str(num_iter-1)] == 2:
		global optim_val
		global iter_num

		iter_num = str(num_iter-1) + str(i); 
		optim_val = Data_DC_MRT.item()['X_optimal_data'+str(num_iter-1)]
	
	if Data_DC_MRT.item()['Status'+str(0)] == 2:
		
		rate_matrix_DC_MRT = np.sum(Data_DC_MRT.item()['Rates'+str(0)], axis=1).tolist()


		#print Data_DC_MRT.item()['Optimal_BW'+str(0)].shape
		#print len(Data_DC.item()['AvailBHUtil_SC'+str(0)])
		# SINR_DC_MRT_BW_TOT = np.ones((Data_DC_MRT.item()['SINR'+str(num_iter-1)].shape[0],Data_DC_MRT.item()['SINR'+str(num_iter-1)].shape[1]))
		# SINR_DC_MRT_BW_MC = np.ones((Data_DC_MRT.item()['SINR'+str(num_iter-1)].shape[0],1))*120
		# SINR_DC_MRT_BW_SC = np.ones((Data_DC_MRT.item()['SINR'+str(num_iter-1)].shape[0],1))*120
		
		# DC_MRT_BW_SC = np.sum(Data_DC_MRT.item()['Optimal_BW'+str(num_iter-1)][:,:len(Data_DC.item()['AvailBHUtil_SC'+str(num_iter-1)])], axis = 1).tolist()
		# DC_MRT_BW_MC = np.sum(Data_DC_MRT.item()['Optimal_BW'+str(num_iter-1)][:,len(Data_DC.item()['AvailBHUtil_SC'+str(num_iter-1)]):], axis = 1).tolist()
		# DC_MRT_BW_TOT = np.sum(Data_DC_MRT.item()['Optimal_BW'+str(num_iter-1)], axis = 1).tolist()

		
		# for ii in range(Data_DC_MRT.item()['SINR'+str(num_iter-1)].shape[0]):
		# 	#count = 0;
		# 	for jj in range(Data_DC_MRT.item()['SINR'+str(num_iter-1)].shape[1]):
		# 		# if (Data_DC_MRT.item()['X_optimal_data'+str(num_iter-1)])[ii,jj] == 1 and count < 2 :
		# 		#if Data_DC_MRT.item()['SINR'+str(num_iter-1)][ii,jj] != 350:
		# 		SINR_DC_MRT_BW_TOT[ii,jj] = Data_DC_MRT.item()['SINR'+str(num_iter-1)][ii,jj]
		# 		#else:
		# 		#	SINR_DC_MRT_BW_TOT[ii,jj] = np.amin(SINR_DC_MRT_BW_TOT[ii,jj])-10
		# 		# 	count = count + 1
		# 		if jj < len(Data_DC.item()['AvailBHUtil_SC'+str(0)]):
		# 			if (Data_DC_MRT.item()['X_optimal_data'+str(num_iter-1)])[ii,jj] == 1:
		# 				SINR_DC_MRT_BW_SC[ii,0] = Data_DC_MRT.item()['SINR'+str(num_iter-1)][ii,jj]
						
		# 		else:
		# 			if (Data_DC_MRT.item()['X_optimal_data'+str(num_iter-1)])[ii,jj] == 1:
		# 				SINR_DC_MRT_BW_MC[ii,0] = Data_DC_MRT.item()['SINR'+str(num_iter-1)][ii,jj]
						
		#plt.bar(np.arange(len(DC_MRT_BW_MC)), DC_MRT_BW_MC)
		#plt.show()
		DC_MRT_avg_rt = DC_MRT_avg_rt + (np.sum(Data_DC_MRT.item()['Rates'+str(0)], axis=1).tolist())
		# if i == 0:
		# 	DC_MRT_avg_rt = rate_matrix_DC_MRT
		# else:
		# 	DC_MRT_avg_rt = [x + y for x, y in zip(rate_matrix_DC_MRT, DC_MRT_avg_rt)]

	if Data_DC_BHCAP_LAT.item()['Status'+str(0)] == 2:
		rate_matrix_DC_BHCAP_LAT = np.sum(Data_DC_BHCAP_LAT.item()['Rates'+str(0)], axis=1).tolist()

	if Data_DC_MRT_BHCAP.item()['Status'+str(0)] == 2:
		rate_matrix_DC_MRT_BHCAP = np.sum(Data_DC_MRT_BHCAP.item()['Rates'+str(0)], axis=1).tolist()

		DC_MRT_BHCAP_avg_rt = DC_MRT_BHCAP_avg_rt + (np.sum(Data_DC_MRT_BHCAP.item()['Rates'+str(0)], axis=1).tolist())

	if Data_DC_LAT.item()['Status'+str(0)] == 2:
		rate_matrix_DC_LAT = np.sum(Data_DC_LAT.item()['Rates'+str(0)], axis=1).tolist()
	
	if Data_DC_MRT_LAT.item()['Status'+str(0)] == 2:
		rate_matrix_DC_MRT_LAT = np.sum(Data_DC_MRT_LAT.item()['Rates'+str(0)], axis=1).tolist()

	if Data_DC_BHCAP.item()['Status'+str(0)] == 2:
		rate_matrix_DC_BHCAP = np.sum(Data_DC_BHCAP.item()['Rates'+str(0)], axis=1).tolist()

	if Data_SA_MRT.item()['Status'+str(0)] == 2:
		rate_matrix_SA_MRT = np.sum(Data_SA_MRT.item()['Rates'+str(0)], axis=1).tolist()

	if Data_SA_LAT.item()['Status'+str(0)] == 2:
		rate_matrix_SA_LAT = np.sum(Data_SA_LAT.item()['Rates'+str(0)], axis=1).tolist()

	if Data_SA_BHCAP.item()['Status'+str(0)] == 2:
		rate_matrix_SA_BHCAP = np.sum(Data_SA_BHCAP.item()['Rates'+str(0)], axis=1).tolist()
	
	if Data_SA_BHCAP_LAT.item()['Status'+str(0)] == 2:			
		rate_matrix_SA_BHCAP_LAT = np.sum(Data_SA_BHCAP_LAT.item()['Rates'+str(0)], axis=1).tolist()
	
	if Data_SA_MRT_LAT.item()['Status'+str(0)] == 2:			
		rate_matrix_SA_MRT_LAT = np.sum(Data_SA_MRT_LAT.item()['Rates'+str(0)], axis=1).tolist()
				

	for k in range(0,num_iter):
		if Data.item()['Status' + str(k)] == 2:
			Net_Throughput[i,k] = Data.item()['Net_Throughput'+str(k)];
			time_SA[i,k] = Data.item()['Time'+str(k)];
		else:
			Net_Throughput[i,k] = 0; # Zero if its an infeasible or timed out solution
			#iters_infeas.append(str(i)+str(k)); # Inserting the iteration number for infeasible solution
			if Data.item()['Status' + str(k)] == 3:
				iters_infeas[k] = iters_infeas[k] + 1; # Increment the number of Infeasible solution sets
				time_SA[i,k] = 700;
			elif Data.item()['Status' + str(k)] == 9:
				iters_timeout[k] = iters_timeout[k] + 1; # Increment the number of Timedout solution sets
				time_SA[i,k] = 700

		if Data_DC.item()['Status' + str(k)] == 2:
			Net_Throughput_DC[i,k] = Data_DC.item()['Net_Throughput'+str(k)];
			time_DC[i,k] = Data_DC.item()['Time'+str(k)];
		else:
			Net_Throughput_DC[i,k] = 0; # Zero if its an infeasible or timed out solution
			#iters_infeas_DC.append(str(i)+str(k)); # Inserting the iteration number for infeasible solution
			if Data_DC.item()['Status' + str(k)] == 3:	
				iters_infeas_DC[k] = iters_infeas_DC[k] + 1; # Increment the number of Infeasible solution sets
				time_DC[i,k] = 700
			elif Data_DC.item()['Status' + str(k)] == 9:
				iters_timeout_DC[k] = iters_timeout_DC[k] + 1; # Increment the number of Timedout solution sets
				time_DC[i,k] = 700	
				

		if Data_DC_MRT.item()['Status' + str(k)] == 2:
			#print Data_DC_MRT.item()['Status'+str(k)]
			Net_Throughput_DC_MRT[i,k] = Data_DC_MRT.item()['Net_Throughput'+str(k)];
			time_DC_MRT[i,k] = Data_DC_MRT.item()['Time'+str(k)];
		else:
			#print Data_DC_MRT.item()['Status'+str(k)]
			Net_Throughput_DC_MRT[i,k] = 0; # Zero if its an infeasible or timed out solution
			#iters_infeas_DC_MRT.append(str(i)+str(k)); # Inserting the iteration number for infeasible solution
			if Data_DC_MRT.item()['Status' + str(k)] == 3:
				iters_infeas_DC_MRT[k] = iters_infeas_DC_MRT[k] + 1; # Increment the number of infeasible solutions
				time_DC_MRT[i,k] = 700
			elif Data_DC_MRT.item()['Status' + str(k)] == 9:
				iters_timeout_DC_MRT[k] = iters_timeout_DC_MRT[k] + 1; # Increment the number of Timedout solution sets
				time_DC_MRT[i,k] = 700

		if Data_DC_BHCAP.item()['Status' + str(k)] == 2:
			Net_Throughput_DC_BHCAP[i,k] = Data_DC_BHCAP.item()['Net_Throughput'+str(k)];
			time_DC_BHCAP[i,k] = Data_DC_BHCAP.item()['Time'+str(k)];
		else:
			Net_Throughput_DC_BHCAP[i,k] = 0; # Zero if its an infeasible or timed out solution
			#iters_infeas_DC_BHCAP.append(str(i)+str(k)); # Inserting the iteration number for infeasible solution
			if Data_DC_BHCAP.item()['Status' + str(k)] == 3:
				iters_infeas_DC_BHCAP[k] = iters_infeas_DC_BHCAP[k] + 1; # Increment the number of infeasible solutions
				time_DC_BHCAP[i,k] = 700
			elif Data_DC_BHCAP.item()['Status' + str(k)] == 9:
				iters_timeout_DC_BHCAP[k] = iters_timeout_DC_BHCAP[k] + 1; # Increment the number of Timedout solution sets
				time_DC_BHCAP[i,k] = 700

		if Data_DC_BHCAP_LAT.item()['Status' + str(k)] == 2:
			Net_Throughput_DC_BHCAP_LAT[i,k] = Data_DC_BHCAP_LAT.item()['Net_Throughput'+str(k)];
			time_DC_BHCAP_LAT[i,k] = Data_DC_BHCAP_LAT.item()['Time'+str(k)];
		else:
			Net_Throughput_DC_BHCAP_LAT[i,k] = 0; # Zero if its an infeasible or timed out solution
			#iters_infeas_DC_BHCAP_LAT.append(str(i)+str(k)); # Inserting the iteration number for infeasible solution
			if Data_DC_BHCAP_LAT.item()['Status' + str(k)] == 3:
				iters_infeas_DC_BHCAP_LAT[k] = iters_infeas_DC_BHCAP_LAT[k] + 1; # Increment the number of infeasible solution
				time_DC_BHCAP_LAT[i,k] = 700
			elif Data_DC_BHCAP_LAT.item()['Status' + str(k)] == 9:
				iters_timeout_DC_BHCAP_LAT[k] = iters_timeout_DC_BHCAP_LAT[k] + 1; # Increment the number of Timedout solution sets
				time_DC_BHCAP_LAT[i,k] = 700

		if Data_DC_LAT.item()['Status' + str(k)] == 2:
			Net_Throughput_DC_LAT[i,k] = Data_DC_LAT.item()['Net_Throughput'+str(k)];
			time_DC_LAT[i,k] = Data_DC_LAT.item()['Time'+str(k)];
		else:
			Net_Throughput_DC_LAT[i,k] = 0; # Zero if its an infeasible or timed out solution
			#iters_infeas_DC_LAT.append(str(i)+str(k)); # Inserting the iteration number for infeasible solution
			if Data_DC_LAT.item()['Status' + str(k)] == 3:
				iters_infeas_DC_LAT[k] = iters_infeas_DC_LAT[k] + 1; # Increment the number of infeasible solution
				time_DC_LAT[i,k] = 700
			elif Data_DC_LAT.item()['Status' + str(k)] == 9:
				iters_timeout_DC_LAT[k] = iters_timeout_DC_LAT[k] + 1; # Increment the number of Timedout solution sets
				time_DC_LAT[i,k] = 700

		if Data_SA_MRT.item()['Status' + str(k)] == 2:
			Net_Throughput_SA_MRT[i,k] = Data_SA_MRT.item()['Net_Throughput'+str(k)];
			time_SA_MRT[i,k] = Data_SA_MRT.item()['Time'+str(k)];
		else:
			Net_Throughput_SA_MRT[i,k] = 0; # Zero if its an infeasible or timed out solution
			#iters_infeas_SA_MRT.append(str(i)+str(k)); # Inserting the iteration number for infeasible solution
			if Data_SA_MRT.item()['Status' + str(k)] == 3:
				iters_infeas_SA_MRT[k] = iters_infeas_SA_MRT[k] + 1; # Increment the number of infeasible solution
				time_SA_MRT[i,k] = 700
			elif Data_SA_MRT.item()['Status' + str(k)] == 9:
				iters_timeout_SA_MRT[k] = iters_timeout_SA_MRT[k] + 1; # Increment the number of Timedout solution sets
				time_SA_MRT[i,k] = 700


		if Data_SA_LAT.item()['Status' + str(k)] == 2:
			Net_Throughput_SA_LAT[i,k] = Data_SA_LAT.item()['Net_Throughput'+str(k)];
			time_SA_LAT[i,k] = Data_SA_LAT.item()['Time'+str(k)];
		else:
			Net_Throughput_SA_LAT[i,k] = 0; # Zero if its an infeasible or timed out solution
			#iters_infeas_SA_LAT.append(str(i)+str(k)); # Inserting the iteration number for infeasible solution
			if Data_SA_LAT.item()['Status' + str(k)] == 3:
				iters_infeas_SA_LAT[k] = iters_infeas_SA_LAT[k] + 1; # Increment the number of infeasible solution
				time_SA_LAT[i,k] = 700

			elif Data_SA_LAT.item()['Status' + str(k)] == 9:
				iters_timeout_SA_LAT[k] = iters_timeout_SA_LAT[k] + 1; # Increment the number of Timedout solution sets
				time_SA_LAT[i,k] = 700

		if Data_SA_BHCAP.item()['Status' + str(k)] == 2:
			Net_Throughput_SA_BHCAP[i,k] = Data_SA_BHCAP.item()['Net_Throughput'+str(k)];
			time_SA_BHCAP[i,k] = Data_SA_BHCAP.item()['Time'+str(k)];
		else: 
			Net_Throughput_SA_BHCAP[i,k] = 0; #Zero if its an infeasible or timed out solution
			#iters_infeas_SA_BHCAP.append(str(i)+str(k)); # Inserting the iteration number for infeasible solution
			if Data_SA_BHCAP.item()['Status' + str(k)] == 3:
				iters_infeas_SA_BHCAP[k] = iters_infeas_SA_BHCAP[k] + 1; # Increment the number of infeasible solution
				time_SA_BHCAP[i,k] = 700
			elif Data_SA_BHCAP.item()['Status' + str(k)] == 9:
				iters_timeout_SA_BHCAP[k] = iters_timeout_SA_BHCAP[k] + 1; # Increment the number of Timedout solution sets
				time_SA_BHCAP[i,k] = 700

		if Data_SA_BHCAP_LAT.item()['Status' + str(k)] == 2:
			Net_Throughput_SA_BHCAP_LAT[i,k] = Data_SA_BHCAP_LAT.item()['Net_Throughput'+str(k)];
			time_SA_BHCAP_LAT[i,k] = Data_SA_BHCAP_LAT.item()['Time'+str(k)];
		else:
			Net_Throughput_SA_BHCAP_LAT[i,k] = 0; #Zero if its an infeasible or timed out solution
			#iters_infeas_SA_BHCAP_LAT.append(str(i)+str(k)); # Inserting the iteration number for infeasible solution
			if Data_SA_BHCAP_LAT.item()['Status' + str(k)] == 3:
				iters_infeas_SA_BHCAP_LAT[k] = iters_infeas_SA_BHCAP_LAT[k] + 1; # Increment the number of infeasible solution
				time_SA_BHCAP_LAT[i,k] = 700
			elif Data_SA_BHCAP_LAT.item()['Status' + str(k)] == 9:
				iters_timeout_SA_BHCAP_LAT[k] = iters_timeout_SA_BHCAP_LAT[k] + 1; # Increment the number of Timedout solution sets
				time_SA_BHCAP_LAT[i,k] = 700

		if Data_SA_MRT_LAT.item()['Status' + str(k)] == 2:
			Net_Throughput_SA_MRT_LAT[i,k] = Data_SA_MRT_LAT.item()['Net_Throughput'+str(k)];
			time_SA_MRT_LAT[i,k] = Data_SA_MRT_LAT.item()['Time'+str(k)];
		else:
			Net_Throughput_SA_MRT_LAT[i,k] = 0; #Zero if its an infeasible or timed out solution
			#iters_infeas_SA_BHCAP_LAT.append(str(i)+str(k)); # Inserting the iteration number for infeasible solution
			if Data_SA_MRT_LAT.item()['Status' + str(k)] == 3:
				iters_infeas_SA_MRT_LAT[k] = iters_infeas_SA_MRT_LAT[k] + 1; # Increment the number of infeasible solution
				time_SA_MRT_LAT[i,k] = 700
			if Data_SA_MRT_LAT.item()['Status' + str(k)] == 9:
				iters_timeout_SA_MRT_LAT[k] = iters_timeout_SA_MRT_LAT[k] + 1; # Increment the number of Timedout solution sets
				time_SA_MRT_LAT[i,k] = 700

		if Data_DC_MRT_LAT.item()['Status' + str(k)] == 2:
			Net_Throughput_DC_MRT_LAT[i,k] = Data_DC_MRT_LAT.item()['Net_Throughput'+str(k)];
			time_DC_MRT_LAT[i,k] = Data_DC_MRT_LAT.item()['Time'+str(k)];
		else:
			Net_Throughput_DC_MRT_LAT[i,k] = 0; #Zero if its an infeasible or timed out solution
			#iters_infeas_SA_BHCAP_LAT.append(str(i)+str(k)); # Inserting the iteration number for infeasible solution
			if Data_DC_MRT_LAT.item()['Status' + str(k)] == 3:
				iters_infeas_DC_MRT_LAT[k] = iters_infeas_DC_MRT_LAT[k] + 1; # Increment the number of infeasible solution
				time_DC_MRT_LAT[i,k] = 700
			elif Data_DC_MRT_LAT.item()['Status' + str(k)] == 9:
				iters_timeout_DC_MRT_LAT[k] = iters_timeout_DC_MRT_LAT[k] + 1; # Increment the number of Timedout solution sets
				time_DC_MRT_LAT[i,k] = 700


		if Data_DC_MRT_BHCAP.item()['Status' + str(k)] == 2:
			Net_Throughput_DC_MRT_BHCAP[i,k] = Data_DC_MRT_BHCAP.item()['Net_Throughput'+str(k)];
			time_DC_MRT_BHCAP[i,k] = Data_DC_MRT_BHCAP.item()['Time'+str(k)];
		else:
			Net_Throughput_DC_MRT_BHCAP[i,k] = 0; #Zero if its an infeasible or timed out solution
			#iters_infeas_SA_BHCAP_LAT.append(str(i)+str(k)); # Inserting the iteration number for infeasible solution
			if Data_DC_MRT_BHCAP.item()['Status' + str(k)] == 3:
				iters_infeas_DC_MRT_BHCAP[k] = iters_infeas_DC_MRT_BHCAP[k] + 1; # Increment the number of infeasible solution
				time_DC_MRT_BHCAP[i,k] = 700
			elif Data_DC_MRT_BHCAP.item()['Status' + str(k)] == 9:
				iters_timeout_DC_MRT_BHCAP[k] = iters_timeout_DC_MRT_BHCAP[k] + 1; # Increment the number of Timedout solution sets
				time_DC_MRT_BHCAP[i,k] = 700

		if Data_DC_MRT_BHCAP_LAT.item()['Status' + str(k)] == 2:
			Net_Throughput_DC_MRT_BHCAP_LAT[i,k] = Data_DC_MRT_BHCAP_LAT.item()['Net_Throughput'+str(k)];
			time_DC_MRT_BHCAP_LAT[i,k] = Data_DC_MRT_BHCAP_LAT.item()['Time'+str(k)];
		else:
			Net_Throughput_DC_MRT_BHCAP_LAT[i,k] = 0; #Zero if its an infeasible or timed out solution
			#iters_infeas_SA_BHCAP_LAT.append(str(i)+str(k)); # Inserting the iteration number for infeasible solution
			if Data_DC_MRT_BHCAP_LAT.item()['Status' + str(k)] == 3:
				iters_infeas_DC_MRT_BHCAP_LAT[k] = iters_infeas_DC_MRT_BHCAP_LAT[k] + 1; # Increment the number of infeasible solution
				time_DC_MRT_BHCAP_LAT[i,k] = 700
			elif Data_DC_MRT_BHCAP_LAT.item()['Status' + str(k)] == 9:
				iters_timeout_DC_MRT_BHCAP_LAT[k] = iters_timeout_DC_MRT_BHCAP_LAT[k] + 1; # Increment the number of Timedout solution sets
				time_DC_MRT_BHCAP_LAT[i,k] = 700
		
		if Data_SA_MRT_BHCAP.item()['Status' + str(k)] == 2:
			Net_Throughput_SA_MRT_BHCAP[i,k] = Data_SA_MRT_BHCAP.item()['Net_Throughput'+str(k)];
			time_SA_MRT_BHCAP[i,k] = Data_SA_MRT_BHCAP.item()['Time'+str(k)];
		else:
			Net_Throughput_SA_MRT_BHCAP[i,k] = 0; #Zero if its an infeasible or timed out solution
			#iters_infeas_SA_BHCAP_LAT.append(str(i)+str(k)); # Inserting the iteration number for infeasible solution
			if Data_SA_MRT_BHCAP.item()['Status' + str(k)] == 3:
				iters_infeas_SA_MRT_BHCAP[k] = iters_infeas_SA_MRT_BHCAP[k] + 1; # Increment the number of infeasible solution
				time_SA_MRT_BHCAP[i,k] = 700
			elif Data_SA_MRT_BHCAP.item()['Status' + str(k)] == 9:
				iters_timeout_SA_MRT_BHCAP[k] = iters_timeout_SA_MRT_BHCAP[k] + 1; # Increment the number of Timedout solution sets
				time_SA_MRT_BHCAP[i,k] = 700

		if Data_SA_MRT_BHCAP_LAT.item()['Status' + str(k)] == 2:
			Net_Throughput_SA_MRT_BHCAP_LAT[i,k] = Data_SA_MRT_BHCAP_LAT.item()['Net_Throughput'+str(k)];
			time_SA_MRT_BHCAP_LAT[i,k] = Data_SA_MRT_BHCAP_LAT.item()['Time'+str(k)];
		else:
			Net_Throughput_SA_MRT_BHCAP_LAT[i,k] = 0; #Zero if its an infeasible or timed out solution
			#iters_infeas_SA_BHCAP_LAT.append(str(i)+str(k)); # Inserting the iteration number for infeasible solution
			if Data_SA_MRT_BHCAP_LAT.item()['Status' + str(k)] == 3:
				iters_infeas_SA_MRT_BHCAP_LAT[k] = iters_infeas_SA_MRT_BHCAP_LAT[k] + 1; # Increment the number of infeasible solution
				time_SA_MRT_BHCAP_LAT[i,k] = 700
			elif Data_SA_MRT_BHCAP_LAT.item()['Status' + str(k)] == 9:
				iters_timeout_SA_MRT_BHCAP_LAT[k] = iters_timeout_SA_MRT_BHCAP_LAT[k] + 1; # Increment the number of Timedout solution sets
				time_SA_MRT_BHCAP_LAT[i,k] = 700

		B_Dat_DR[i,k], AU_Base_DR[i,k] = baseline_cal(i,k,simdata_path)
	
	#print "=================="
	#print Net_Throughput
	#print "=================="
	#print Net_Throughput_DC_MRT
	# ================
	# User Throughputs

	X_Optimal_jfr = np.zeros((Data.item()['Apps'+str(k)], Data.item()['APs'+str(k)]));
	X_Optimal_DC_jfr = copy.deepcopy(X_Optimal_jfr);
	X_Optimal_DC_MRT_jfr = copy.deepcopy(X_Optimal_jfr);
	X_Optimal_DC_BHCAP_jfr = copy.deepcopy(X_Optimal_jfr);
	X_Optimal_DC_BHCAP_LAT_jfr = copy.deepcopy(X_Optimal_jfr);
	X_Optimal_DC_LAT_jfr = copy.deepcopy(X_Optimal_jfr);
	X_Optimal_SA_MRT_jfr = copy.deepcopy(X_Optimal_jfr);
	X_Optimal_SA_LAT_jfr = copy.deepcopy(X_Optimal_jfr);
	X_Optimal_SA_BHCAP_jfr = copy.deepcopy(X_Optimal_jfr);
	X_Optimal_SA_BHCAP_LAT_jfr = copy.deepcopy(X_Optimal_jfr);
	X_Optimal_SA_MRT_LAT_jfr = copy.deepcopy(X_Optimal_jfr);
	X_Optimal_DC_MRT_LAT_jfr = copy.deepcopy(X_Optimal_jfr);
	X_Optimal_DC_MRT_BHCAP_jfr = copy.deepcopy(X_Optimal_jfr);
	X_Optimal_DC_MRT_BHCAP_LAT_jfr = copy.deepcopy(X_Optimal_jfr);
	X_Optimal_SA_MRT_BHCAP_jfr = copy.deepcopy(X_Optimal_jfr);
	X_Optimal_SA_MRT_BHCAP_LAT_jfr = copy.deepcopy(X_Optimal_jfr);

	Rate = np.zeros((Data.item()['Apps'+str(k)], Data.item()['APs'+str(k)]));
	Rate_DC = copy.deepcopy(Rate);
	Rate_DC_MRT = copy.deepcopy(Rate);
	Rate_DC_BHCAP = copy.deepcopy(Rate);
	Rate_DC_BHCAP_LAT = copy.deepcopy(Rate);
	Rate_DC_LAT = copy.deepcopy(Rate);
	Rate_SA_MRT = copy.deepcopy(Rate);
	Rate_SA_LAT = copy.deepcopy(Rate);
	Rate_SA_BHCAP = copy.deepcopy(Rate);
	Rate_SA_BHCAP_LAT = copy.deepcopy(Rate);
	Rate_SA_MRT_LAT = copy.deepcopy(Rate);
	Rate_DC_MRT_LAT = copy.deepcopy(Rate);
	Rate_DC_MRT_BHCAP = copy.deepcopy(Rate);
	Rate_DC_MRT_BHCAP_LAT = copy.deepcopy(Rate);
	Rate_SA_MRT_BHCAP = copy.deepcopy(Rate);
	Rate_SA_MRT_BHCAP_LAT = copy.deepcopy(Rate);


	for k in range(0,num_iter):
		X_Optimal = np.empty((Data.item()['Apps'+str(k)], Data.item()['APs'+str(k)]));
		X_Optimal_DC = copy.deepcopy(X_Optimal);
		X_Optimal_DC_MRT = copy.deepcopy(X_Optimal);
		X_Optimal_DC_BHCAP = copy.deepcopy(X_Optimal);
		X_Optimal_DC_BHCAP_LAT = copy.deepcopy(X_Optimal);
		X_Optimal_DC_LAT = copy.deepcopy(X_Optimal);
		X_Optimal_SA_MRT = copy.deepcopy(X_Optimal);
		X_Optimal_SA_LAT = copy.deepcopy(X_Optimal);
		X_Optimal_SA_BHCAP = copy.deepcopy(X_Optimal);
		X_Optimal_SA_BHCAP_LAT = copy.deepcopy(X_Optimal);
		X_Optimal_SA_MRT_LAT = copy.deepcopy(X_Optimal);
		X_Optimal_DC_MRT_LAT = copy.deepcopy(X_Optimal);
		X_Optimal_DC_MRT_BHCAP = copy.deepcopy(X_Optimal);
		X_Optimal_DC_MRT_BHCAP_LAT = copy.deepcopy(X_Optimal);
		X_Optimal_SA_MRT_BHCAP = copy.deepcopy(X_Optimal);
		X_Optimal_SA_MRT_BHCAP_LAT = copy.deepcopy(X_Optimal);


		Rate_jfr = np.zeros((Data.item()['Apps'+str(k)], Data.item()['APs'+str(k)]));
		Rate_DC_jfr = copy.deepcopy(Rate);
		Rate_DC_MRT_jfr = copy.deepcopy(Rate);
		Rate_DC_BHCAP_jfr = copy.deepcopy(Rate);
		Rate_DC_BHCAP_LAT_jfr = copy.deepcopy(Rate);
		Rate_DC_LAT_jfr = copy.deepcopy(Rate);
		Rate_SA_MRT_jfr = copy.deepcopy(Rate);
		Rate_SA_LAT_jfr = copy.deepcopy(Rate);
		Rate_SA_BHCAP_jfr = copy.deepcopy(Rate);
		Rate_SA_BHCAP_LAT_jfr = copy.deepcopy(Rate);
		Rate_SA_MRT_LAT = copy.deepcopy(Rate);
		Rate_DC_MRT_LAT = copy.deepcopy(Rate);
		Rate_DC_MRT_BHCAP = copy.deepcopy(Rate);
		Rate_DC_MRT_BHCAP_LAT = copy.deepcopy(Rate);
		Rate_SA_MRT_BHCAP = copy.deepcopy(Rate);
		Rate_SA_MRT_BHCAP_LAT = copy.deepcopy(Rate);
		
		
		if Data.item()['Status'+str(k)] == 2:
			X_Optimal = Data.item()['X_optimal_data'+str(k)];
			Rate = Data.item()['Rates'+str(k)];
			AU_DR[i,k] = user_count(X_Optimal)
		else:
			pass
		if Data_DC.item()['Status'+str(k)] == 2:
			X_Optimal_DC = Data_DC.item()['X_optimal_data'+str(k)];
			Rate_DC = Data_DC.item()['Rates'+str(k)];
			AU_DR_DC[i,k] = user_count(X_Optimal_DC)
		else:
			pass
		if Data_DC_MRT.item()['Status'+str(k)] == 2:
			X_Optimal_DC_MRT = Data_DC_MRT.item()['X_optimal_data'+str(k)];
			Rate_DC_MRT = Data_DC_MRT.item()['Rates'+str(k)];
			AU_DR_DC_MRT[i,k] = user_count(X_Optimal_DC_MRT)
		else:
			pass
		if Data_DC_BHCAP.item()['Status'+str(k)] == 2:
			X_Optimal_DC_BHCAP = Data_DC_BHCAP.item()['X_optimal_data'+str(k)];
			Rate_DC_BHCAP = Data_DC_BHCAP.item()['Rates'+str(k)];
			AU_DR_DC_BHCAP[i,k] = user_count(X_Optimal_DC_BHCAP)
		else:
			pass
		if Data_DC_BHCAP_LAT.item()['Status'+str(k)] == 2:
			X_Optimal_DC_BHCAP_LAT = Data_DC_BHCAP_LAT.item()['X_optimal_data'+str(k)];
			Rate_DC_BHCAP_LAT = Data_DC_BHCAP_LAT.item()['Rates'+str(k)];
			AU_DR_DC_BHCAP_LAT[i,k] = user_count(X_Optimal_DC_BHCAP_LAT)
		else:
			pass
		if Data_DC_LAT.item()['Status'+str(k)] == 2:
			X_Optimal_DC_LAT = Data_DC_LAT.item()['X_optimal_data'+str(k)];
			Rate_DC_LAT = Data_DC_LAT.item()['Rates'+str(k)];
			AU_DR_DC_LAT[i,k] = user_count(X_Optimal_DC_LAT)
		else:
			pass
		if Data_SA_MRT.item()['Status'+str(k)] == 2:
			X_Optimal_SA_MRT = Data_SA_MRT.item()['X_optimal_data'+str(k)];
			Rate_SA_MRT = Data_SA_MRT.item()['Rates'+str(k)];
			AU_DR_SA_MRT[i,k] = user_count(X_Optimal_SA_MRT)
		else:
			pass
		if Data_SA_LAT.item()['Status'+str(k)] == 2:
			X_Optimal_SA_LAT = Data_SA_LAT.item()['X_optimal_data'+str(k)];
			Rate_SA_LAT = Data_SA_LAT.item()['Rates'+str(k)];
			AU_DR_SA_LAT[i,k] = user_count(X_Optimal_SA_LAT)
		else:
			pass
		if Data_SA_BHCAP.item()['Status'+str(k)] == 2:
			X_Optimal_SA_BHCAP = Data_SA_BHCAP.item()['X_optimal_data'+str(k)];
			Rate_SA_BHCAP = Data_SA_BHCAP.item()['Rates'+str(k)];
			AU_DR_SA_BHCAP[i,k] = user_count(X_Optimal_SA_BHCAP)
		else:
			pass
		if Data_SA_BHCAP_LAT.item()['Status'+str(k)] == 2:
			X_Optimal_SA_BHCAP_LAT = Data_SA_BHCAP_LAT.item()['X_optimal_data'+str(k)];
			Rate_SA_BHCAP_LAT = Data_SA_BHCAP_LAT.item()['Rates'+str(k)];
			AU_DR_SA_BHCAP_LAT[i,k] = user_count(X_Optimal_SA_BHCAP_LAT)
		else:
			pass
		if Data_SA_MRT_LAT.item()['Status'+str(k)] == 2:
			X_Optimal_SA_MRT_LAT = Data_SA_MRT_LAT.item()['X_optimal_data'+str(k)];
			Rate_SA_MRT_LAT = Data_SA_MRT_LAT.item()['Rates'+str(k)];
			AU_DR_SA_MRT_LAT[i,k] = user_count(X_Optimal_SA_MRT_LAT)
		else:
			pass
		if Data_DC_MRT_LAT.item()['Status'+str(k)] == 2:
			X_Optimal_DC_MRT_LAT = Data_DC_MRT_LAT.item()['X_optimal_data'+str(k)];
			Rate_DC_MRT_LAT = Data_DC_MRT_LAT.item()['Rates'+str(k)];
			AU_DR_DC_MRT_LAT[i,k] = user_count(X_Optimal_DC_MRT_LAT)
		else:
			pass
		if Data_DC_MRT_BHCAP.item()['Status'+str(k)] == 2:
			X_Optimal_DC_MRT_BHCAP = Data_DC_MRT_BHCAP.item()['X_optimal_data'+str(k)];
			Rate_DC_MRT_BHCAP = Data_DC_MRT_BHCAP.item()['Rates'+str(k)];
			AU_DR_DC_MRT_BHCAP[i,k] = user_count(X_Optimal_DC_MRT_BHCAP)
		else:
			pass
		if Data_DC_MRT_BHCAP_LAT.item()['Status'+str(k)] == 2:
			X_Optimal_DC_MRT_BHCAP_LAT = Data_DC_MRT_BHCAP_LAT.item()['X_optimal_data'+str(k)];
			Rate_DC_MRT_BHCAP_LAT = Data_DC_MRT_BHCAP_LAT.item()['Rates'+str(k)];
			AU_DR_DC_MRT_BHCAP_LAT[i,k] = user_count(X_Optimal_DC_MRT_BHCAP_LAT)
		else:
			pass
		if Data_SA_MRT_BHCAP.item()['Status'+str(k)] == 2:
			X_Optimal_SA_MRT_BHCAP = Data_SA_MRT_BHCAP.item()['X_optimal_data'+str(k)];
			Rate_SA_MRT_BHCAP = Data_SA_MRT_BHCAP.item()['Rates'+str(k)];
			AU_DR_SA_MRT_BHCAP[i,k] = user_count(X_Optimal_SA_MRT_BHCAP)
		else:
			pass
		if Data_SA_MRT_BHCAP_LAT.item()['Status'+str(k)] == 2:
			X_Optimal_SA_MRT_BHCAP_LAT = Data_SA_MRT_BHCAP_LAT.item()['X_optimal_data'+str(k)];
			Rate_SA_MRT_BHCAP_LAT = Data_SA_MRT_BHCAP_LAT.item()['Rates'+str(k)];
			AU_DR_SA_MRT_BHCAP_LAT[i,k] = user_count(X_Optimal_SA_MRT_BHCAP_LAT)
		else:
			pass
	
	avg_idx.append(X_Optimal.shape[0])
	#print avg_idx	

	for j in range(0,X_Optimal.shape[0]):
		Base_DR.append(scn.eMBB_minrate); 
		if Data.item()['Status'+str(num_iter-1)] == 2:
			# X_Optimal_jfr = Data.item()['Optimal_BW'+str(num_iter-1)];
			#Rate_jfr = Data.item()['Rates'+str(0)]; 
			application_DR.append(sum(Rate[j,:]));
		else:
			pass
		if Data_DC.item()['Status'+str(num_iter-1)] == 2:
			# X_Optimal_DC_jfr = Data_DC.item()['Optimal_BW'+str(num_iter-1)];
			#Rate_DC_jfr = Data_DC.item()['Rates'+str(1)];
			application_DR_DC.append(sum(Rate_DC[j,:]));
		else:
			pass
		if Data_DC_MRT.item()['Status'+str(num_iter-1)] == 2:
			# X_Optimal_DC_MRT_jfr = Data_DC_MRT.item()['Optimal_BW'+str(num_iter-1)];
			# print X_Optimal_DC_MRT_jfr[j,:]
			# print Rate_DC_MRT[j,:]
			# #Rate_DC_MRT_jfr = Data_DC_MRT.item()['Rates'+str(1)];
			application_DR_DC_MRT.append(sum(Rate_DC_MRT[j,:]));
			#print application_DR_DC_MRT
		else:
			pass
		if Data_DC_BHCAP.item()['Status'+str(num_iter-1)] == 2:
			#X_Optimal_DC_BHCAP_jfr = Data_DC_BHCAP.item()['X_optimal_data'+str(1)];
			#Rate_DC_BHCAP_jfr = Data_DC_BHCAP.item()['Rates'+str(1)];
			application_DR_DC_BHCAP.append(sum(Rate_DC_BHCAP[j,:]*X_Optimal_DC_BHCAP[j,:]));
		else:
			pass
		if Data_DC_BHCAP_LAT.item()['Status'+str(num_iter-1)] == 2:
			#X_Optimal_DC_BHCAP_LAT_jfr = Data_DC_BHCAP_LAT.item()['X_optimal_data'+str(1)];
			#Rate_DC_BHCAP_LAT_jfr = Data_DC_BHCAP_LAT.item()['Rates'+str(1)];
			application_DR_DC_BHCAP_LAT.append(sum(Rate_DC_BHCAP_LAT[j,:]*X_Optimal_DC_BHCAP_LAT[j,:]));
		else:
			pass
		if Data_DC_LAT.item()['Status'+str(num_iter-1)] == 2:
			#X_Optimal_DC_LAT_jfr = Data_DC_LAT.item()['X_optimal_data'+str(1)];
			#Rate_DC_LAT_jfr = Data_DC_LAT.item()['Rates'+str(1)];
			application_DR_DC_LAT.append(sum(Rate_DC_LAT[j,:]*X_Optimal_DC_LAT[j,:]));
		else:
			pass
		if Data_SA_MRT.item()['Status'+str(num_iter-1)] == 2:
			#X_Optimal_SA_MRT_jfr = Data_SA_MRT.item()['X_optimal_data'+str(1)];
			#Rate_SA_MRT_jfr = Data_SA_MRT.item()['Rates'+str(1)];
			application_DR_SA_MRT.append(sum(Rate_SA_MRT[j,:]*X_Optimal_SA_MRT[j,:]));
		else:
			pass
		if Data_SA_LAT.item()['Status'+str(num_iter-1)] == 2:
			#X_Optimal_SA_LAT_jfr = Data_SA_LAT.item()['X_optimal_data'+str(1)];
			#Rate_SA_LAT_jfr = Data_SA_LAT.item()['Rates'+str(1)];
			application_DR_SA_LAT.append(sum(Rate_SA_LAT[j,:]*X_Optimal_SA_LAT[j,:]));
		else: 
			pass
		if Data_SA_BHCAP.item()['Status'+str(num_iter-1)] == 2:
			#X_Optimal_SA_BHCAP_jfr = Data_SA_BHCAP.item()['X_optimal_data'+str(1)];
			#Rate_SA_BHCAP_jfr = Data_SA_BHCAP.item()['Rates'+str(1)];
			application_DR_SA_BHCAP.append(sum(Rate_SA_BHCAP[j,:]*X_Optimal_SA_BHCAP[j,:]));
		else:
			pass
		if Data_SA_BHCAP_LAT.item()['Status'+str(num_iter-1)] == 2:
			#X_Optimal_SA_BHCAP_LAT_jfr = Data_SA_BHCAP_LAT.item()['X_optimal_data'+str(1)];
			#Rate_SA_BHCAP_LAT_jfr = Data_SA_BHCAP_LAT.item()['Rates'+str(1)];
			application_DR_SA_BHCAP_LAT.append(sum(Rate_SA_BHCAP_LAT[j,:]*X_Optimal_SA_BHCAP_LAT[j,:]));
		else:
			pass
		if Data_SA_MRT_LAT.item()['Status'+str(num_iter-1)] == 2:
			#X_Optimal_SA_MRT_MAT_jfr = Data_SA_MRT_LAT.item()['X_optimal_data'+str(1)];
			#Rate_SA_MRT_LAT_jfr = Data_SA_MRT_LAT.item()['Rates'+str(1)];
			application_DR_SA_MRT_LAT.append(sum(Rate_SA_MRT_LAT[j,:]*X_Optimal_SA_MRT_LAT[j,:]));
		else:
			pass
		if Data_DC_MRT_LAT.item()['Status'+str(num_iter-1)] == 2:
			#X_Optimal_DC_MRT_LAT_jfr = Data_DC_MRT_LAT_.item()['X_optimal_data'+str(1)];
			#Rate_DC_MRT_LAT_jfr = Data_DC_MRT_LAT.item()['Rates'+str(1)];
			application_DR_DC_MRT_LAT.append(sum(Rate_DC_MRT_LAT[j,:]*X_Optimal_DC_MRT_LAT[j,:]));
		else:
			pass

		if Data_DC_MRT_BHCAP.item()['Status'+str(num_iter-1)] == 2:
			#X_Optimal_DC_MRT_LAT_jfr = Data_DC_MRT_LAT_.item()['X_optimal_data'+str(1)];
			#Rate_DC_MRT_LAT_jfr = Data_DC_MRT_LAT.item()['Rates'+str(1)];
			application_DR_DC_MRT_BHCAP.append(sum(Rate_DC_MRT_BHCAP[j,:]*X_Optimal_DC_MRT_BHCAP[j,:]));
		else:
			pass
		if Data_DC_MRT_BHCAP_LAT.item()['Status'+str(num_iter-1)] == 2:
			#X_Optimal_DC_MRT_LAT_jfr = Data_DC_MRT_LAT_.item()['X_optimal_data'+str(1)];
			#Rate_DC_MRT_LAT_jfr = Data_DC_MRT_LAT.item()['Rates'+str(1)];
			application_DR_DC_MRT_BHCAP_LAT.append(sum(Rate_DC_MRT_BHCAP_LAT[j,:]*X_Optimal_DC_MRT_BHCAP_LAT[j,:]));
		else:
			pass
		if Data_SA_MRT_BHCAP.item()['Status'+str(num_iter-1)] == 2:
			#X_Optimal_DC_MRT_LAT_jfr = Data_DC_MRT_LAT_.item()['X_optimal_data'+str(1)];
			#Rate_DC_MRT_LAT_jfr = Data_DC_MRT_LAT.item()['Rates'+str(1)];
			application_DR_SA_MRT_BHCAP.append(sum(Rate_SA_MRT_BHCAP[j,:]*X_Optimal_SA_MRT_BHCAP[j,:]));
		else:
			pass
		if Data_SA_MRT_BHCAP_LAT.item()['Status'+str(num_iter-1)] == 2:
			#X_Optimal_DC_MRT_LAT_jfr = Data_DC_MRT_LAT_.item()['X_optimal_data'+str(1)];
			#Rate_DC_MRT_LAT_jfr = Data_DC_MRT_LAT.item()['Rates'+str(1)];
			application_DR_SA_MRT_BHCAP_LAT.append(sum(Rate_SA_MRT_BHCAP_LAT[j,:]*X_Optimal_SA_MRT_BHCAP_LAT[j,:]));
		else:
			pass


# ===============
# Analysis Values
# ===============

# ==============
# Net Throughput

#print iters_infeas 
#print iters_timeout
Net_Throughput_avg = zero_div(np.sum(Net_Throughput, axis = 0),(MCMC_iter - np.array(iters_infeas) - np.array(iters_timeout))) ; # We get the average throughput over MCMC Iteratios
Net_Throughput_DC_avg = zero_div(np.sum(Net_Throughput_DC, axis = 0),(MCMC_iter - np.array(iters_infeas_DC) - np.array(iters_timeout_DC))); # Average throughput
Net_Throughput_DC_MRT_avg = zero_div(np.sum(Net_Throughput_DC_MRT, axis = 0),(MCMC_iter - np.array(iters_infeas_DC_MRT) - np.array(iters_timeout_DC_MRT))); # DC + MRT Average throughput
Net_Throughput_DC_BHCAP_avg = zero_div(np.sum(Net_Throughput_DC_BHCAP, axis = 0),(MCMC_iter - np.array(iters_infeas_DC_BHCAP) - np.array(iters_timeout_DC_BHCAP))); # DC + BHCAP Average throughput
Net_Throughput_DC_LAT_avg = zero_div(np.sum(Net_Throughput_DC_LAT, axis = 0),(MCMC_iter - np.array(iters_infeas_DC_LAT) - np.array(iters_timeout_DC_LAT))); # DC + LAT Average throughput
Net_Throughput_DC_BHCAP_LAT_avg = zero_div(np.sum(Net_Throughput_DC_BHCAP_LAT, axis = 0),(MCMC_iter - np.array(iters_infeas_DC_BHCAP_LAT) - np.array(iters_timeout_DC_BHCAP_LAT))); # DC + BHCAP + LAT Average throughput
Net_Throughput_SA_MRT_avg = zero_div(np.sum(Net_Throughput_SA_MRT, axis = 0),(MCMC_iter - np.array(iters_infeas_SA_MRT) - np.array(iters_timeout_SA_MRT))); # SA + MRT average 
Net_Throughput_SA_LAT_avg = zero_div(np.sum(Net_Throughput_SA_LAT, axis = 0),(MCMC_iter - np.array(iters_infeas_SA_LAT) - np.array(iters_timeout_SA_LAT))); # SA + LAT average
Net_Throughput_SA_BHCAP_avg = zero_div(np.sum(Net_Throughput_SA_BHCAP, axis = 0),(MCMC_iter - np.array(iters_infeas_SA_BHCAP) - np.array(iters_timeout_SA_BHCAP))); # SA + BHCAP average
Net_Throughput_SA_BHCAP_LAT_avg = zero_div(np.sum(Net_Throughput_SA_BHCAP_LAT, axis = 0),(MCMC_iter - np.array(iters_infeas_SA_BHCAP_LAT) - np.array(iters_timeout_SA_BHCAP_LAT))); # SA + BHCAP + LAT average
Net_Throughput_SA_MRT_LAT_avg = zero_div(np.sum(Net_Throughput_SA_MRT_LAT, axis = 0),(MCMC_iter - np.array(iters_infeas_SA_MRT_LAT) - np.array(iters_timeout_SA_MRT_LAT))); # SA + LAT average
Net_Throughput_DC_MRT_LAT_avg = zero_div(np.sum(Net_Throughput_DC_MRT_LAT, axis = 0),(MCMC_iter - np.array(iters_infeas_DC_MRT_LAT) - np.array(iters_timeout_DC_MRT_LAT))); # SA + LAT average
Net_Throughput_DC_MRT_BHCAP_avg = zero_div(np.sum(Net_Throughput_DC_MRT_BHCAP, axis = 0),(MCMC_iter - np.array(iters_infeas_DC_MRT_BHCAP) - np.array(iters_timeout_DC_MRT_BHCAP))); # SA + LAT average
Net_Throughput_DC_MRT_BHCAP_LAT_avg = zero_div(np.sum(Net_Throughput_DC_MRT_BHCAP_LAT, axis = 0),(MCMC_iter - np.array(iters_infeas_DC_MRT_BHCAP_LAT) - np.array(iters_timeout_DC_MRT_BHCAP_LAT))); # SA + LAT average
Net_Throughput_SA_MRT_BHCAP_avg = zero_div(np.sum(Net_Throughput_SA_MRT_BHCAP, axis = 0),(MCMC_iter - np.array(iters_infeas_SA_MRT_BHCAP) - np.array(iters_timeout_SA_MRT_BHCAP))); # SA + LAT average
Net_Throughput_SA_MRT_BHCAP_LAT_avg = zero_div(np.sum(Net_Throughput_SA_MRT_BHCAP_LAT, axis = 0),(MCMC_iter - np.array(iters_infeas_SA_MRT_BHCAP_LAT) - np.array(iters_timeout_SA_MRT_BHCAP_LAT))); # SA + LAT average

# Net_Throughput_avg = np.sum(Net_Throughput, axis = 0)/MCMC_iter ; # We get the average throughput over MCMC Iteratios
# Net_Throughput_DC_avg = np.sum(Net_Throughput_DC, axis = 0)/MCMC_iter; # Average throughput
# Net_Throughput_DC_MRT_avg = np.sum(Net_Throughput_DC_MRT, axis = 0)/MCMC_iter; # DC + MRT Average throughput
# Net_Throughput_DC_BHCAP_avg = np.sum(Net_Throughput_DC_BHCAP, axis = 0)/MCMC_iter; # DC + BHCAP Average throughput
# Net_Throughput_DC_LAT_avg = np.sum(Net_Throughput_DC_LAT, axis = 0)/MCMC_iter; # DC + LAT Average throughput
# Net_Throughput_DC_BHCAP_LAT_avg = np.sum(Net_Throughput_DC_BHCAP_LAT, axis = 0)/MCMC_iter; # DC + BHCAP + LAT Average throughput
# Net_Throughput_SA_MRT_avg = np.sum(Net_Throughput_SA_MRT, axis = 0)/MCMC_iter; # SA + MRT average 
# Net_Throughput_SA_LAT_avg = np.sum(Net_Throughput_SA_LAT, axis = 0)/MCMC_iter; # SA + LAT average
# Net_Throughput_SA_BHCAP_avg = np.sum(Net_Throughput_SA_BHCAP, axis = 0)/MCMC_iter; # SA + BHCAP average
# Net_Throughput_SA_BHCAP_LAT_avg = np.sum(Net_Throughput_SA_BHCAP_LAT, axis = 0)/MCMC_iter; # SA + BHCAP + LAT average
# Net_Throughput_SA_MRT_LAT_avg = np.sum(Net_Throughput_SA_MRT_LAT, axis = 0)/MCMC_iter; # SA + LAT average
# Net_Throughput_DC_MRT_LAT_avg = np.sum(Net_Throughput_DC_MRT_LAT, axis = 0)/MCMC_iter; # SA + LAT average
# Net_Throughput_DC_MRT_BHCAP_avg = np.sum(Net_Throughput_DC_MRT_BHCAP, axis = 0)/MCMC_iter; # SA + LAT average
# Net_Throughput_DC_MRT_BHCAP_LAT_avg = np.sum(Net_Throughput_DC_MRT_BHCAP_LAT, axis = 0)/MCMC_iter; # SA + LAT average
# Net_Throughput_SA_MRT_BHCAP_avg = np.sum(Net_Throughput_SA_MRT_BHCAP, axis = 0)/MCMC_iter; # SA + LAT average
# Net_Throughput_SA_MRT_BHCAP_LAT_avg = np.sum(Net_Throughput_SA_MRT_BHCAP_LAT, axis = 0)/MCMC_iter; # SA + LAT average

B_Dat_DR_avg = np.sum(B_Dat_DR, axis =0)/MCMC_iter; # Baseline with BW restriction
B_Dat_DR_fs_avg = np.sum(B_Dat_DR_fs, axis = 0)/MCMC_iter;
#B_Dat_DR_fs_avg = np.sum(B_Dat_DR_fs, axis = 0)/MCMC_iter; # Baseline with 1GHz BW
#B_Dat_DR_sn_avg = np.sum(B_Dat_DR_sn, axis = 0)/MCMC_iter; # Baseline with SINR and BW restriction
#B_Dat_DR_sn_fs_avg = np.sum(B_Dat_DR_sn_fs, axis = 0)/MCMC_iter; #Baseline with SINR and 1GHz BW
#print iters_infeas_DC_MRT


# ====================
# Satisfied User Count

#AU_Base_DR_fs_avg = np.floor(np.sum(AU_Base_DR_fs, axis = 0)/MCMC_iter);
AU_Base_DR_avg = np.floor(np.sum(AU_Base_DR, axis = 0)/(MCMC_iter)); 
AU_DR_avg = np.floor(zero_div(np.sum(AU_DR, axis = 0),(MCMC_iter- np.array(iters_infeas) - np.array(iters_timeout))));
AU_DR_DC_avg = np.floor(zero_div(np.sum(AU_DR_DC, axis = 0),(MCMC_iter- np.array(iters_infeas_DC) - np.array(iters_timeout_DC))));
AU_DR_DC_MRT_avg = np.floor(zero_div(np.sum(AU_DR_DC_MRT, axis = 0),(MCMC_iter- np.array(iters_infeas_DC_MRT) - np.array(iters_timeout_DC_MRT))));
AU_DR_DC_BHCAP_avg = np.floor(zero_div(np.sum(AU_DR_DC_BHCAP, axis = 0),(MCMC_iter - np.array(iters_infeas_DC_BHCAP) - np.array(iters_timeout_DC_BHCAP))));
AU_DR_DC_LAT_avg = np.floor(zero_div(np.sum(AU_DR_DC_LAT, axis = 0),(MCMC_iter- np.array(iters_infeas_DC_LAT) - np.array(iters_timeout_DC_LAT))));
AU_DR_DC_BHCAP_LAT_avg = np.floor(zero_div(np.sum(AU_DR_DC_BHCAP_LAT, axis = 0),(MCMC_iter- np.array(iters_infeas_DC_BHCAP_LAT) - np.array(iters_timeout_DC_BHCAP_LAT))));
AU_DR_SA_MRT_avg = np.floor(zero_div(np.sum(AU_DR_SA_MRT, axis = 0),(MCMC_iter- np.array(iters_infeas_SA_MRT) - np.array(iters_timeout_SA_MRT))));
AU_DR_SA_LAT_avg = np.floor(zero_div(np.sum(AU_DR_SA_LAT, axis = 0),(MCMC_iter- np.array(iters_infeas_SA_LAT) - np.array(iters_timeout_SA_LAT))));
AU_DR_SA_BHCAP_avg = np.floor(zero_div(np.sum(AU_DR_SA_BHCAP, axis = 0),(MCMC_iter- np.array(iters_infeas_SA_BHCAP) - np.array(iters_timeout_SA_BHCAP))));
AU_DR_SA_BHCAP_LAT_avg = np.floor(zero_div(np.sum(AU_DR_SA_BHCAP_LAT, axis = 0),(MCMC_iter- np.array(iters_infeas_SA_BHCAP_LAT) - np.array(iters_timeout_SA_BHCAP_LAT))));
AU_DR_SA_MRT_LAT_avg = np.floor(zero_div(np.sum(AU_DR_SA_MRT, axis = 0),(MCMC_iter- np.array(iters_infeas_SA_MRT) - np.array(iters_timeout_SA_MRT))));
AU_DR_DC_MRT_LAT_avg = np.floor(zero_div(np.sum(AU_DR_DC_MRT_LAT, axis = 0),(MCMC_iter- np.array(iters_infeas_DC_MRT_LAT) - np.array(iters_timeout_DC_MRT_LAT))));
AU_DR_DC_MRT_BHCAP_avg = np.floor(zero_div(np.sum(AU_DR_DC_MRT_BHCAP, axis = 0),(MCMC_iter- np.array(iters_infeas_DC_MRT_BHCAP) - np.array(iters_timeout_DC_MRT_BHCAP))));
AU_DR_DC_MRT_BHCAP_LAT_avg = np.floor(zero_div(np.sum(AU_DR_DC_MRT_BHCAP_LAT, axis = 0),(MCMC_iter- np.array(iters_infeas_DC_MRT_BHCAP_LAT) - np.array(iters_timeout_DC_MRT_BHCAP_LAT))));
AU_DR_SA_MRT_BHCAP_avg = np.floor(zero_div(np.sum(AU_DR_SA_MRT_BHCAP, axis = 0),(MCMC_iter- np.array(iters_infeas_SA_MRT_BHCAP) - np.array(iters_timeout_SA_MRT_BHCAP))));
AU_DR_SA_MRT_BHCAP_LAT_avg = np.floor(zero_div(np.sum(AU_DR_SA_MRT_BHCAP_LAT, axis = 0),(MCMC_iter- np.array(iters_infeas_SA_MRT_BHCAP_LAT) - np.array(iters_timeout_SA_MRT_BHCAP_LAT))));
# AU_Base_DR_avg = np.floor(np.sum(AU_Base_DR, axis = 0)/(MCMC_iter)); 
# AU_DR_avg = np.floor(np.sum(AU_DR, axis = 0)/(MCMC_iter));
# AU_DR_DC_avg = np.floor(np.sum(AU_DR_DC, axis = 0)/(MCMC_iter));
# AU_DR_DC_MRT_avg = np.floor(np.sum(AU_DR_DC_MRT, axis = 0)/(MCMC_iter));
# AU_DR_DC_BHCAP_avg = np.floor(np.sum(AU_DR_DC_BHCAP, axis = 0)/(MCMC_iter));
# AU_DR_DC_LAT_avg = np.floor(np.sum(AU_DR_DC_LAT, axis = 0)/(MCMC_iter));
# AU_DR_DC_BHCAP_LAT_avg = np.floor(np.sum(AU_DR_DC_BHCAP_LAT, axis = 0)/(MCMC_iter));
# AU_DR_SA_MRT_avg = np.floor(np.sum(AU_DR_SA_MRT, axis = 0)/(MCMC_iter));
# AU_DR_SA_LAT_avg = np.floor(np.sum(AU_DR_SA_LAT, axis = 0)/(MCMC_iter));
# AU_DR_SA_BHCAP_avg = np.floor(np.sum(AU_DR_SA_BHCAP, axis = 0)/(MCMC_iter));
# AU_DR_SA_BHCAP_LAT_avg = np.floor(np.sum(AU_DR_SA_BHCAP_LAT, axis = 0)/(MCMC_iter));
# AU_DR_SA_MRT_LAT_avg = np.floor(np.sum(AU_DR_SA_MRT, axis = 0)/(MCMC_iter));
# AU_DR_DC_MRT_LAT_avg = np.floor(np.sum(AU_DR_DC_MRT_LAT, axis = 0)/(MCMC_iter));
# AU_DR_DC_MRT_BHCAP_avg = np.floor(np.sum(AU_DR_DC_MRT_BHCAP, axis = 0)/(MCMC_iter));
# AU_DR_DC_MRT_BHCAP_LAT_avg = np.floor(np.sum(AU_DR_DC_MRT_BHCAP_LAT, axis = 0)/(MCMC_iter));
# AU_DR_SA_MRT_BHCAP_avg = np.floor(np.sum(AU_DR_SA_MRT_BHCAP, axis = 0)/(MCMC_iter));
# AU_DR_SA_MRT_BHCAP_LAT_avg = np.floor(np.sum(AU_DR_SA_MRT_BHCAP_LAT, axis = 0)/(MCMC_iter));

#np.savetxt("Accepted_USER.csv",AU_Base_DR_fs_avg, AU_Base_DR_avg, AU_DR_avg, AU_DR_DC_avg, AU_DR_DC_MRT_avg, AU_DR_DC_BHCAP_avg, AU_DR_DC_LAT_avg, AU_DR_DC_BHCAP_LAT_avg, AU_DR_SA_MRT_avg, AU_DR_SA_LAT_avg, AU_DR_SA_BHCAP_avg, AU_DR_SA_BHCAP_LAT_avg, AU_DR_SA_MRT_LAT_avg, AU_DR_DC_MRT_LAT_avg, delimiter=",")

# =================================
# Save multiple Numpy arrays to CSV

# df = pd.DataFrame({"Baseline": AU_Base_DR_avg, "Single Association": AU_DR_avg, "Dual Association": AU_DR_DC_avg, "Dual Association MinRate": AU_DR_DC_MRT_avg, "Dual Association BHaul": AU_DR_DC_BHCAP_avg, "Dual Association LAT": AU_DR_DC_LAT_avg, "Dual Association Bhaul LAT": AU_DR_DC_BHCAP_LAT_avg, "Single Association MRT": AU_DR_SA_MRT_avg, "Single Association LAT": AU_DR_SA_LAT_avg, "Single Association Bhaul": AU_DR_SA_BHCAP_avg, "Single Association BHCAP+LAT": AU_DR_SA_BHCAP_LAT_avg, "Single Association MRT+LAT": AU_DR_SA_MRT_LAT_avg, "Dual Association MRT+LAT": AU_DR_DC_MRT_LAT_avg, "Dual Association MRT+BHCAP": AU_DR_DC_MRT_BHCAP_avg, "Dual Association MRT+BHCAP+LAT": AU_DR_DC_MRT_BHCAP_LAT_avg, "Single Association MRT+BHCAP": AU_DR_SA_MRT_BHCAP_avg, "Single Association MRT+BHCAP+LAT": AU_DR_SA_MRT_BHCAP_LAT_avg})
# df.to_csv("AcceptedUsers.csv", index=False)

# ========================================
# Jain's Fairness Index and t-student test
#print application_DR_DC_MRT
#print X_Optimal_DC_MRT_jfr
jfr_SA = jains_fairness(application_DR, AU_DR[:, num_iter-1]);
jfr_DC = jains_fairness(application_DR_DC, AU_DR_DC[:, num_iter-1]);
jfr_DC_MRT = jains_fairness(application_DR_DC_MRT, AU_DR_DC_MRT[:, num_iter-1]); 
jfr_DC_BHCAP = jains_fairness(application_DR_DC_BHCAP, AU_DR_DC_BHCAP[:, num_iter-1]);
jfr_DC_BHCAP_LAT = jains_fairness(application_DR_DC_BHCAP_LAT, AU_DR_DC_BHCAP_LAT[:, num_iter-1]);
jfr_DC_LAT = jains_fairness(application_DR_DC_LAT, AU_DR_DC_LAT[:, num_iter-1]);
jfr_SA_MRT = jains_fairness(application_DR_SA_MRT, AU_DR_SA_MRT[:, num_iter-1]);
jfr_SA_LAT = jains_fairness(application_DR_SA_LAT, AU_DR_SA_LAT[:, num_iter-1]);
jfr_SA_BHCAP = jains_fairness(application_DR_SA_BHCAP, AU_DR_SA_BHCAP[:, num_iter-1]);
jfr_SA_BHCAP_LAT = jains_fairness(application_DR_SA_BHCAP_LAT, AU_DR_SA_BHCAP_LAT[:, num_iter-1]);
jfr_SA_MRT_LAT = jains_fairness(application_DR_SA_MRT_LAT, AU_DR_SA_MRT_LAT[:, num_iter-1]);
jfr_DC_MRT_LAT = jains_fairness(application_DR_DC_MRT_LAT, AU_DR_DC_MRT_LAT[:, num_iter-1]);
jfr_DC_MRT_BHCAP = jains_fairness(application_DR_DC_MRT_BHCAP, AU_DR_DC_MRT_BHCAP[:, num_iter-1]);
jfr_DC_MRT_BHCAP_LAT = jains_fairness(application_DR_DC_MRT_BHCAP_LAT, AU_DR_DC_MRT_BHCAP_LAT[:, num_iter-1]);
jfr_SA_MRT_BHCAP = jains_fairness(application_DR_SA_MRT_BHCAP, AU_DR_SA_MRT_BHCAP[:, num_iter-1]);
jfr_SA_MRT_BHCAP_LAT = jains_fairness(application_DR_SA_MRT_BHCAP_LAT, AU_DR_SA_MRT_BHCAP_LAT[:, num_iter-1]);


#jfr_Baseline = jains_fairness(B_Dat_DR_avg, AU_Base_DR[:,num_iter-1]);

#print jfr_SA
#print jfr_DC
#print jfr_DC_BHCAP
#print jfr_DC_BHCAP_LAT



# ===============
# Throughput Plot

x_axis = np.arange(scn.num_users_min, scn.num_users_max, scn.user_steps_siml);
# y_min_1 = np.amin([np.amin(Net_Throughput_avg),np.amin(Net_Throughput_DC_avg), np.amin(Net_Throughput_DC_MRT_avg), np.amin(Net_Throughput_DC_BHCAP_avg), np.amin(Net_Throughput_DC_BHCAP_LAT_avg), np.amin(Net_Throughput_DC_LAT_avg), np.amin(Net_Throughput_SA_MRT_avg), np.amin(Net_Throughput_SA_LAT_avg), np.amin(Net_Throughput_SA_BHCAP_avg), np.amin(Net_Throughput_SA_BHCAP_LAT_avg), np.amin(Net_Throughput_SA_MRT_LAT_avg), np.amin(B_Dat_DR_avg), np.amin(Net_Throughput_DC_MRT_LAT_avg), np.amin(Net_Throughput_DC_MRT_BHCAP_avg), np.amin(Net_Throughput_DC_MRT_BHCAP_LAT_avg), np.amin(Net_Throughput_SA_MRT_BHCAP_avg), np.amin(Net_Throughput_SA_MRT_BHCAP_LAT_avg)]); #np.amin(B_Dat_DR_avg), , np.amin(B_Dat_DR_sn_avg)
# y_max_1 = np.max([np.amax(Net_Throughput_avg), np.amax(Net_Throughput_DC_avg), np.amax(Net_Throughput_DC_MRT_avg), np.amax(Net_Throughput_DC_BHCAP_avg), np.amax(Net_Throughput_DC_BHCAP_LAT_avg), np.amax(Net_Throughput_DC_LAT_avg), np.amax(Net_Throughput_SA_MRT_avg), np.amax(Net_Throughput_SA_LAT_avg), np.amax(Net_Throughput_SA_BHCAP_avg), np.amax(Net_Throughput_SA_BHCAP_LAT_avg), np.amax(Net_Throughput_SA_MRT_LAT_avg), np.amax(B_Dat_DR_avg), np.amax(Net_Throughput_DC_MRT_LAT_avg), np.amax(Net_Throughput_DC_MRT_BHCAP_avg), np.amax(Net_Throughput_DC_MRT_BHCAP_LAT_avg), np.amax(Net_Throughput_SA_MRT_BHCAP_avg), np.amax(Net_Throughput_SA_MRT_BHCAP_LAT_avg)]); #np.amax(B_Dat_DR_avg),, np.amax(B_Dat_DR_sn_avg)
y_min_1 = np.amin([np.amin(Net_Throughput_avg),np.amin(Net_Throughput_DC_avg), np.amin(Net_Throughput_DC_MRT_avg), np.amin(Net_Throughput_DC_BHCAP_avg), np.amin(Net_Throughput_DC_BHCAP_LAT_avg), np.amin(Net_Throughput_DC_LAT_avg), np.amin(Net_Throughput_SA_LAT_avg), np.amin(Net_Throughput_SA_BHCAP_avg), np.amin(Net_Throughput_SA_BHCAP_LAT_avg), np.amin(B_Dat_DR_avg), np.amin(Net_Throughput_DC_MRT_LAT_avg), np.amin(Net_Throughput_DC_MRT_BHCAP_avg), np.amin(Net_Throughput_DC_MRT_BHCAP_LAT_avg)]); #np.amin(B_Dat_DR_avg), , np.amin(B_Dat_DR_sn_avg)
y_max_1 = np.max([np.amax(Net_Throughput_avg), np.amax(Net_Throughput_DC_avg), np.amax(Net_Throughput_DC_MRT_avg), np.amax(Net_Throughput_DC_BHCAP_avg), np.amax(Net_Throughput_DC_BHCAP_LAT_avg), np.amax(Net_Throughput_DC_LAT_avg), np.amax(Net_Throughput_SA_LAT_avg), np.amax(Net_Throughput_SA_BHCAP_avg), np.amax(Net_Throughput_SA_BHCAP_LAT_avg), np.amax(B_Dat_DR_avg), np.amax(Net_Throughput_DC_MRT_LAT_avg), np.amax(Net_Throughput_DC_MRT_BHCAP_avg), np.amax(Net_Throughput_DC_MRT_BHCAP_LAT_avg)]); #np.amax(B_Dat_DR_avg),, np.amax(B_Dat_DR_sn_avg)
#y_min_1 = np.amin([np.amin(Net_Throughput_DC_BHCAP_avg), np.amin(Net_Throughput_DC_BHCAP_LAT_avg), np.amin(Net_Throughput_SA_BHCAP_avg), np.amin(Net_Throughput_SA_BHCAP_LAT_avg), np.amin(Net_Throughput_DC_MRT_BHCAP_avg), np.amin(Net_Throughput_DC_MRT_BHCAP_LAT_avg)]); #np.amin(B_Dat_DR_avg), , np.amin(B_Dat_DR_sn_avg)
#y_max_1 = np.max([np.amax(Net_Throughput_DC_BHCAP_avg), np.amax(Net_Throughput_DC_BHCAP_LAT_avg), np.amax(Net_Throughput_SA_BHCAP_avg), np.amax(Net_Throughput_SA_BHCAP_LAT_avg), np.amax(Net_Throughput_DC_MRT_BHCAP_avg), np.amax(Net_Throughput_DC_MRT_BHCAP_LAT_avg)]); #np.amax(B_Dat_DR_avg),, np.amax(B_Dat_DR_sn_avg)

y_min_2 = np.amin([np.amin(B_Dat_DR_avg)]);
y_max_2 = np.max([np.amax(B_Dat_DR_avg)]);
#plotter.plotter('dashline',np.arange(scn.num_users_min, scn.num_users_max, scn.user_steps_siml),Net_Throughput_avg,5,10,1,45,0,0,1,'major','both', 'yes', 'Total Network Throughput', np)
#print Net_Throughput_SA_LAT_avg
# plt.plot(x_axis, Net_Throughput_avg, 'r-o', x_axis, Net_Throughput_DC_avg, 'b-o' , x_axis, Net_Throughput_DC_MRT_avg, 'g-.', x_axis, Net_Throughput_DC_BHCAP_avg, 'k--s',  x_axis, Net_Throughput_DC_BHCAP_LAT_avg, 'm--d',  x_axis , Net_Throughput_DC_LAT_avg, 'c--p', x_axis, Net_Throughput_SA_MRT_avg, 'k-.', x_axis, Net_Throughput_SA_LAT_avg, 'b:', x_axis, Net_Throughput_SA_BHCAP_avg, 'g--D', x_axis, Net_Throughput_SA_BHCAP_LAT_avg, 'r:', x_axis, Net_Throughput_SA_MRT_LAT_avg, 'b--*', x_axis, Net_Throughput_DC_MRT_LAT_avg, 'k--*',  x_axis, Net_Throughput_DC_MRT_BHCAP_avg, 'c--o', x_axis, Net_Throughput_DC_MRT_BHCAP_LAT_avg, 'm-^', x_axis, Net_Throughput_SA_MRT_BHCAP_avg, 'g-^', x_axis, Net_Throughput_SA_MRT_BHCAP_LAT_avg, 'b-s'); 
#plt.plot(x_axis, Net_Throughput_avg/1e9, 'r-x', x_axis, Net_Throughput_DC_avg/1e9, 'b-o' , x_axis, Net_Throughput_DC_MRT_avg/1e9, 'g-.', x_axis, Net_Throughput_SA_MRT_avg/1e9, 'm-.', x_axis, Net_Throughput_DC_BHCAP_avg/1e9, 'k--s',  x_axis, Net_Throughput_DC_BHCAP_LAT_avg/1e9, 'm--d',  x_axis , Net_Throughput_DC_LAT_avg/1e9, 'c--p', x_axis, Net_Throughput_SA_LAT_avg/1e9, 'b-x', x_axis, Net_Throughput_SA_BHCAP_avg/1e9, 'g--D', x_axis, Net_Throughput_SA_BHCAP_LAT_avg/1e9, 'r:', fillstyle = 'none'); 
plt.plot(x_axis, Net_Throughput_DC_BHCAP_avg/1e9, 'k--s',  x_axis, Net_Throughput_DC_BHCAP_LAT_avg/1e9, 'm--d', x_axis, Net_Throughput_SA_BHCAP_avg/1e9, 'g--D', x_axis, Net_Throughput_SA_BHCAP_LAT_avg/1e9, 'r:', fillstyle = 'none'); 
plt.xticks(np.arange(scn.num_users_min, scn.num_users_max, scn.user_steps_siml));
#plt.yticks(np.arange(y_min_1,y_max_1,5e10));
plt.legend(['DC + CB', 'DC + CB + CPL', 'SA + CB', 'SA + CB + CPL'], loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol = 4) #'Baseline (RSSI)',
plt.grid(which= 'major',axis= 'both');
#plt.title('Network Wide Throughput')
plt.ylim(0,120)
plt.xlabel('Number of eMBB users')
plt.ylabel('Throughput (Gbps)')
plt.savefig('NetThrough', dpi=1200, facecolor='w', edgecolor='w',
        orientation='landscape', papertype='letter', format='png',
        transparent=False, bbox_inches='tight', pad_inches=0.1,
        frameon=None, metadata=None)


# ===================
# Accepted Users Plot 

#ind = np.arange(1,14); # The x locations for the grouped plots
#width = 0.20; # Width of the bars

#fig, ax = plt.subplots()
#rects1 = ax.bar(ind - 13*width/13, AU_Base_DR, width, label='Baseline')
#                label='Women')

# Add some text for labels, title and custom x-axis tick labels, etc.
#ax.set_ylabel('Scores')
#ax.set_title('Scores by group and gender')
#ax.set_xticks(ind)
#ax.set_xticklabels(('G1', 'G2', 'G3', 'G4', 'G5'))
#ax.legend()

print ('Baseline Accepted Users:', AU_Base_DR_avg)
#print ('Baseline with Minimum rate Accepted Users:', AU_Base_DR_fs_avg)
print ('SA Accepted Users:', AU_DR_avg)
print ('DC Accepted Users:', AU_DR_DC_avg)
print ('DC+MRT Accepted Users:', AU_DR_DC_MRT_avg)
print ('DC+BHCAP Accepted Users:', AU_DR_DC_BHCAP_avg)
print ('DC+LAT Accepted Users:', AU_DR_DC_LAT_avg)
print ('DC+BHCAP+LAT Accepted Users:',AU_DR_DC_BHCAP_LAT_avg)
print ('SA+MRT Accepted Users:',AU_DR_SA_MRT_avg)
print ('SA+LAT Accepted Users:',AU_DR_SA_LAT_avg)
print ('SA+BHCAP Accepted Users:',AU_DR_SA_BHCAP_avg)
print ('SA+BHCAP+LAT Accepted Users:',AU_DR_SA_BHCAP_LAT_avg)
#print ('SA+MRT+LAT Accepted Users:',AU_DR_SA_MRT_LAT_avg)
print ('DC+MRT+LAT Accepted Users:',AU_DR_DC_MRT_LAT_avg)
print ('DC+MRT+BHCAP Accepted Users:',AU_DR_DC_MRT_BHCAP_avg)
print ('DC+MRT+BHCAP+LAT Accepted Users:',AU_DR_DC_MRT_BHCAP_LAT_avg)
#print ('SA+MRT+BHCAP Accepted Users:',AU_DR_SA_MRT_BHCAP_avg)
#print ('SA+MRT+BHCAP+LAT Accepted Users:',AU_DR_SA_MRT_BHCAP_LAT_avg)

# ===> Setting up a Broken plot to depict Values with distinct ranges

f, (ax, ax2) = plt.subplots(2, 1, sharex = True, gridspec_kw={'height_ratios': [3, 1]})

# plot the same data on both axes
#ax.plot(x_axis, Net_Throughput_avg, 'r-o', x_axis, Net_Throughput_DC_avg, 'b-o' , x_axis, Net_Throughput_DC_MRT_avg, 'g-.', x_axis, Net_Throughput_DC_BHCAP_avg, 'm--', x_axis, Net_Throughput_DC_BHCAP_LAT_avg, 'm-.', x_axis , Net_Throughput_DC_LAT_avg, 'c--x',x_axis, Net_Throughput_SA_MRT_avg, 'k-.', x_axis, Net_Throughput_SA_LAT_avg, 'b:', x_axis, Net_Throughput_SA_BHCAP_avg, 'g--x', x_axis, Net_Throughput_SA_BHCAP_LAT_avg, 'r:', x_axis, Net_Throughput_SA_MRT_LAT_avg, 'g:', x_axis, Net_Throughput_DC_MRT_LAT_avg, 'k:')
#ax.plot(x_axis, Net_Throughput_avg, 'r-o', x_axis, Net_Throughput_DC_avg, 'b-o' , x_axis, Net_Throughput_DC_MRT_avg, 'g-.', x_axis, Net_Throughput_DC_BHCAP_avg, 'k--s',  x_axis, Net_Throughput_DC_BHCAP_LAT_avg, 'm--d',  x_axis , Net_Throughput_DC_LAT_avg, 'c--p', x_axis, Net_Throughput_SA_MRT_avg, 'k-.', x_axis, Net_Throughput_SA_LAT_avg, 'b:', x_axis, Net_Throughput_SA_BHCAP_avg, 'g--D', x_axis, Net_Throughput_SA_BHCAP_LAT_avg, 'r:', x_axis, Net_Throughput_SA_MRT_LAT_avg, 'b--*', x_axis, Net_Throughput_DC_MRT_LAT_avg, 'k--*',  x_axis, Net_Throughput_DC_MRT_BHCAP_avg, 'c--o', x_axis, Net_Throughput_DC_MRT_BHCAP_LAT_avg, 'm-^', x_axis, Net_Throughput_SA_MRT_BHCAP_avg, 'g-^', x_axis, Net_Throughput_SA_MRT_BHCAP_LAT_avg, 'b-s'); 
ax.plot(x_axis, Net_Throughput_avg/1e9, 'r-o', x_axis, Net_Throughput_DC_avg/1e9, 'b-o' , x_axis, Net_Throughput_DC_MRT_avg/1e9, 'g-.', x_axis, Net_Throughput_SA_MRT_avg/1e9, 'm-.', x_axis, Net_Throughput_DC_BHCAP_avg/1e9, 'k--s',  x_axis, Net_Throughput_DC_BHCAP_LAT_avg/1e9, 'm--d',  x_axis , Net_Throughput_DC_LAT_avg/1e9, 'c--p',  x_axis, Net_Throughput_SA_LAT_avg/1e9, 'b:', x_axis, Net_Throughput_SA_BHCAP_avg/1e9, 'g--D', x_axis, Net_Throughput_SA_BHCAP_LAT_avg/1e9, 'r:'); 
ax2.plot(x_axis, B_Dat_DR_avg/1e9, 'k--x') #x_axis, B_Dat_DR_sn_avg, 'b--x', 

ax2.set_ylim(0.8*(y_min_2/1e9),1.3*(y_max_2/1e9))
#ax2.set_yticks((0,1.5*y_max_2,0.5*1e8))
#ax.set_ylim(30,225)
ax.set_ylim(30,1000)

#ax.set_yticks((50,1000,50))

ax.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax.xaxis.tick_top()
ax.tick_params(labeltop=False)
ax2.xaxis.tick_bottom()

f.legend(['SA','DC', 'DC + MRT','SA + MRT', 'DC + CB', 'DC + CB + CPL', 'DC + CPL', 'SA + CPL', 'SA + CB', 'SA + CB + CPL', 'Baseline'], bbox_to_anchor=(0.5, 0.17), loc='lower left', ncol=2, prop={'size': 6.5})#, prop={'size': 7.5}) #'Baseline (RSSI)',

d1 = 0.025
d = .015  # how big to make the diagonal lines in axes coordinates
# arguments to pass to plot, just so we don't keep repeating them
kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

#plt.plot(x_axis, B_Dat_DR_avg, 'b-', x_axis, B_Dat_DR_sn_avg, 'r-')
ax.grid(which= 'both',axis= 'both')
ax2.grid(which= 'both',axis= 'both')

#f.suptitle('Total Network Throughput -- Baseline Comparison ')
f.text(0.04, 0.5, 'Throughput (Gbps)', va='center', rotation='vertical')
f.text(0.5, 0.04, 'Number of eMBB users', ha='center')
#ax2.xlabel('Number of eMBB users')

plt.savefig('NetThrough_Split', dpi=1200, facecolor='w', edgecolor='w',
        orientation='landscape', papertype='letter', format='png',
        transparent=False, bbox_inches='tight', pad_inches=0.1,
        frameon=None, metadata=None)

# ================
# Fairness BoxPlot

#box_data = [jfr_DC, jfr_SA, jfr_DC_MRT, jfr_SA_MRT, jfr_DC_LAT, jfr_SA_LAT, jfr_DC_BHCAP, jfr_SA_BHCAP, jfr_DC_BHCAP_LAT, jfr_SA_BHCAP_LAT, jfr_DC_MRT_LAT, jfr_SA_MRT_LAT, jfr_DC_MRT_BHCAP, jfr_SA_MRT_BHCAP, jfr_DC_MRT_BHCAP_LAT, jfr_SA_MRT_BHCAP_LAT] 
box_data = [jfr_DC, jfr_SA, jfr_DC_MRT, jfr_SA_MRT, jfr_DC_LAT, jfr_SA_LAT, jfr_DC_BHCAP, jfr_SA_BHCAP, jfr_DC_BHCAP_LAT, jfr_SA_BHCAP_LAT] 

fig, ax = plt.subplots()
#plt.title('Jain\'s Fairness Index Deviation')
plt.ylabel("Fairness index measure")

plt.boxplot(box_data)
ax.set_ylim(0,1,0.1)
#plt.xticks(range(1,17), ['DC', 'SA', 'DC+MRT', 'SA+MRT', 'DC+LAT', 'SA+LAT', 'DC+BHCAP', 'SA+BHCAP', 'DC+BHCAP+LAT', 'SA+BHCAP+LAT', 'DC+MRT+LAT', 'SA+MRT+LAT', 'DC+MRT+BHCAP', 'SA+MRT+BHCAP', 'DC+MRT+BHCAP+LAT', 'SA+MRT+BHCAP+LAT'], fontsize = 8, rotation = '90')
plt.xticks(range(1,11), ['DC', 'SA', 'DC+MRT', 'SA+MRT', 'DC+CPL', 'SA+CPL', 'DC+CB', 'SA+CB', 'DC+CB+CPL', 'SA+CB+CPL'], fontsize = 8, rotation = '90')

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


# ===============
# Heatmap Plotter

plt.close("all")
hmap_data = np.load('/home/akshayjain/Desktop/Results/mMTC/SquareDeploy_MCSC/BF/Temp/hmap_'+iter_num+'.npz', allow_pickle='True')
usr_locs = hmap_data['arr_0']
mc_locs = hmap_data['arr_2']
sc_locs = hmap_data['arr_1']
#print num_iter
# plt.close('all')
# f, ax1 = plt.subplots()
# ax1.bar(np.arange(len(DC_MRT_BW_SC)), DC_MRT_BW_SC)
# ax1.bar(np.arange(len(DC_MRT_BW_MC)), DC_MRT_BW_MC)
# #ax1.bar(np.arange(len(DC_MRT_BW_TOT)), DC_MRT_BW_TOT)
# ax2 = ax1.twinx()
# ax2.plot(np.arange(len(DC_MRT_BW_SC)), SINR_DC_MRT_BW_SC, 'wo', markersize = 12)
# ax2.plot(np.arange(len(DC_MRT_BW_MC)), SINR_DC_MRT_BW_MC, 'k^', markersize = 12)

# f.tight_layout()
# plt.show()
plotter.hmap_creator(usr_locs, mc_locs, sc_locs, rate_matrix_DC, optim_val, np, scn)

# SINR File
#SINR_DC_MRT_BW_TOT[np.where(SINR_DC_MRT_BW_TOT==350)] = float('Nan')
#csvsaver.csvsaver(SINR_DC_MRT_BW_TOT,[], "SINRIFMCSC9users.csv")
#csvsaver.csvsaver(Data_DC_MRT.item()['X_optimal_data'+str(num_iter-1)], [], "OptBFMCSC9users.csv")

#plotter.hist_plotter(DC_avg_rt, SA_avg_rt, DC_MRT_BHCAP_avg_rt ,rate_matrix_DC_BHCAP, rate_matrix_SA_BHCAP, rate_matrix_SA_LAT, rate_matrix_SA_MRT, DC_MRT_avg_rt, rate_matrix_DC_LAT, rate_matrix_SA_MRT_LAT, rate_matrix_DC_MRT_LAT, rate_matrix_SA_BHCAP_LAT, rate_matrix_DC_BHCAP_LAT, np, scn)
# #plotter.scatter_plotter(rate_matrix_DC, rate_matrix_DC_MRT,np,scn)
# #plotter.accepted_user_plotter(AU_Base_DR_avg,AU_DR_avg,AU_DR_DC_avg,AU_DR_DC_MRT_avg,AU_DR_DC_BHCAP_avg,AU_DR_DC_LAT_avg,AU_DR_DC_BHCAP_LAT_avg,AU_DR_SA_MRT_avg,AU_DR_SA_LAT_avg,AU_DR_SA_BHCAP_avg,AU_DR_SA_BHCAP_LAT_avg,AU_DR_SA_MRT_LAT_avg,AU_DR_DC_MRT_LAT_avg,np,scn)
#plotter.bhutil_latprov_plotter(bhutil_val_DC,bhutil_val_DC_MRT_BHCAP, bhutil_val_DC_BHCAP, bhutil_val_DC_BHCAP_LAT, avail_bh, latprov_DC, latprov_DC_LAT, latprov_DC_MRT_LAT, latprov_DC_BHCAP_LAT, np, scn)
# #plt.close('all') # Close all existing figures
for idx in range(1,num_iter+1):
	plotter.infeasible_iter_counter(iters_infeas[num_iter-idx], iters_infeas_DC[num_iter-idx], iters_infeas_DC_MRT[num_iter-idx], iters_infeas_SA_MRT_LAT[num_iter-idx], 
		iters_infeas_SA_MRT_BHCAP[num_iter-idx], iters_infeas_DC_MRT_BHCAP[num_iter-idx], iters_infeas_DC_MRT_BHCAP_LAT[num_iter-idx], iters_infeas_SA_MRT[num_iter-idx] , 
		iters_timeout[num_iter-idx], iters_timeout_DC[num_iter-idx], iters_timeout_DC_MRT[num_iter-idx], iters_timeout_SA_MRT_LAT[num_iter-idx], iters_timeout_SA_MRT_BHCAP[num_iter-idx], 
		iters_timeout_DC_MRT_BHCAP[num_iter-idx], iters_timeout_DC_MRT_BHCAP_LAT[num_iter-idx], iters_timeout_SA_MRT[num_iter-idx], iters_infeas_SA_MRT_BHCAP_LAT[num_iter-idx], iters_timeout_SA_MRT_BHCAP_LAT[num_iter-idx] , 
		iters_infeas_DC_MRT_LAT[num_iter-idx], iters_timeout_DC_MRT_LAT[num_iter-idx], iters_infeas_SA_BHCAP[num_iter-idx], iters_timeout_SA_BHCAP[num_iter-idx], iters_infeas_SA_LAT[num_iter-idx], iters_timeout_SA_LAT[num_iter-idx],
		iters_infeas_SA_BHCAP_LAT[num_iter-idx], iters_timeout_SA_BHCAP_LAT[num_iter-idx], iters_infeas_DC_BHCAP[num_iter-idx], iters_timeout_DC_BHCAP[num_iter-idx], iters_infeas_DC_LAT[num_iter-idx], iters_timeout_DC_LAT[num_iter-idx],
		iters_infeas_DC_BHCAP_LAT[num_iter-idx], iters_timeout_DC_BHCAP_LAT[num_iter-idx], np,scn, idx)
#plotter.timecdf(time_DC, time_DC_MRT , time_SA_MRT , time_DC_BHCAP , time_DC_BHCAP_LAT , time_DC_LAT , time_SA_BHCAP , time_SA_BHCAP_LAT , time_SA_LAT ,time_SA, np, scn )
