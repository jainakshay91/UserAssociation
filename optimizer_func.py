#!/usr/bin/env python

# =============================
# Import the necessary binaries
# =============================

from gurobipy import *
import numpy as np 
import plotter
import os
from scenario_var import scenario_var 
from argparse import ArgumentParser
from rssi_assoc import baseline_assoc
import time
import csv
import csvsaver

# =======================
# Optimizer Configuration
# =======================

parser = ArgumentParser(description = 'The Optimizer Function for User Association'); # Initializing the class variable

# =========================================================
# Add the Command Line Arguments to Configure the Optimizer

parser.add_argument('-iter', type = int, help = 'Iteration Number of the Simulation');
parser.add_argument('-minRate', type = int, help = 'Minimum Rate Constraint Flag');
parser.add_argument('-dual', type = int, help = 'Dual Connectivity Flag');
parser.add_argument('-bhaul', type = int, help = 'Backhaul Capacity Constraint Flag');
parser.add_argument('-latency', type = int, help = 'Path Latency Constraint Flag');
parser.add_argument('-mipGP', type = int, help = 'Optimizer bound Interval'); 

args = parser.parse_args(); # Parse the Arguments

#print vars(args)['iter']

# =====================================
# Check Presence of Storage Directories
# =====================================

path = os.getcwd() + '/Data'; # This is the path we have to check for
subpath = os.getcwd() + '/Data/Process'; # This is the subdirectory to store data  
if os.path.isdir(path):
	print "Directory to save data found"
	print "----------------------------"
	print ""
	if os.path.isdir(subpath):
		print "Subdirectory found"
		print "------------------"
		print ""
	else: 
		os.mkdir(subpath)
		print "Subdirectory Created"
		print "--------------------"
		print ""
else:
	os.mkdir(path); # Create this directory 
	os.mkdir(subpath); # Created the Subdirectory 
	print "Created the Directory to save data"
	print "----------------------------------"
	print ""

# ==============================
# Create the Model and Variables
# ==============================
scn = scenario_var(); # Initializing the class variables

Data = {}; # Dictionary that holds the data 
num_iter = ((scn.num_users_max - scn.num_users_min)/scn.user_steps_siml); 

# ==============================
# Load Massive Machine Type Data
# ==============================

optim_data_mMTC = np.load(os.getcwd() + '/Data/Temp/optim_var_mMTC'+ str(vars(args)['iter']) +'.npz', allow_pickle = True); # Extracting the mMTC data 
#Rx_power_mMTC = optim_data_mMTC['arr_11']; # Received Power from Small cells for all the mMTC devices
num_AP_mMTC = optim_data_mMTC['arr_0']; # Number of mMTC devices per AP (MC first and then SC)
data_rate_consum_BH_mMTC = np.empty((num_AP_mMTC.shape[0],1)) # Consumed Data Rate 

#print np.sum(num_AP_mMTC)
# ====> The bandwidth consumption at each cell by the mMTC devices (In the Backhaul)
for i in range(num_AP_mMTC.shape[0]):
	data_rate_consum_BH_mMTC[i] = np.sum(np.random.random_sample((num_AP_mMTC[i],1))*(scn.mMTC_maxrate[1] - scn.mMTC_maxrate[0]) + scn.mMTC_maxrate[0])

#RX_power_mc_mMTC = optim_data_mMTC['arr_12']; # Received Power from Macro cells for all the mMTC devices
#RX_power_mMTC = np.hstack((Rx_power_sc_mMTC, RX_power_mc_mMTC)); # Stack all the received powers for the mMTC users

for N in range(0,num_iter):
	#k = 1
	print "=============================================="
	print "Dataset # " + str(N) + ": Collecting the Stored Variables"

	optim_data = np.load(os.getcwd() + '/Data/Temp/optim_var_'+ str(N) + str(vars(args)['iter']) +'.npz', allow_pickle = True)
	sinr_eMBB = optim_data['arr_0']; # Load the SINR data to be used for the optimization
	#user_AP_assoc = optim_data['arr_1'].item()['user_app' + str(N)]; # Load the User-Applications Association data to be used for the optimization
	#sinr_eMBB = np.empty([np.sum(user_AP_assoc[:,1]),sinr_APs.shape[1]],dtype=float); # Array that holds the Application SINR values
	sinr_pad_val = optim_data['arr_3']; # In the small cell calculation we use an sinr pad value for ease of computation
	num_scbs = optim_data['arr_4']; # Number of Small cells
	num_mcbs = optim_data['arr_5']; # Number of Macro cells
	mat_wlbh_sc = optim_data['arr_6']; # Wireless backhaul matrix for Small Cells
	mat_wrdbh_sc = optim_data['arr_7']; # Wired backhaul matrix for Macro cells
	Hops_MC = optim_data['arr_8']; # Number of hops to the IMS core from Macro cells
	Hops_SC = optim_data['arr_9']; # Number of hops to the IMS core from Small cells
	BH_Capacity_SC = optim_data['arr_10']; # Backhaul capacity for Small cells
	BH_Capacity_MC = scn.fib_BH_MC_capacity + 10*scn.avg_fib_BH_capacity; # Backhaul capacity for Macro cells
	SNR_eMBB = optim_data['arr_11']; # Small Cell Received Power 
	SCBS_per_MCBS = optim_data['arr_12']; # Number of small cells per macro cell
	
	#RX_power_mc = optim_data['arr_13']; # Macro Cell Received Power 
	#RX_power = np.hstack((RX_power_mc,RX_power_sc)); # Stack all the received powers for the eMBB users
	#SNR_eMBB = np.empty([np.sum(user_AP_assoc[:,1]),sinr_APs.shape[1]],dtype=float); # Array that holds the Application SINR values
	# ==================================
	# Print to Understand Matrix Formats
	#print num_scbs
	#print num_mcbs

	#print Hops_MC.shape
	#print "==========="
	#print Hops_SC.shape
	#print "==========="
	#print Hops_MC
	#print "==========="
	#print Hops_SC
	#print BH_Capacity_SC.shape 
	#print mat_wlbh_sc
	#print mat_wlbh_sc.shape 
	#print "========="
	#print mat_wrdbh_sc
	#print mat_wrdbh_sc.shape
	#print sinr_APs.shape

	# ================================================================
	# Update the Available Backhaul Capacity based on mMTC Consumption
	#print num_scbs
	BH_Capacity_SC = BH_Capacity_SC - data_rate_consum_BH_mMTC[0:num_scbs]; # We reduce the available backhaul capacity for small cells
	BH_Capacity_MC = BH_Capacity_MC - data_rate_consum_BH_mMTC[num_scbs:num_scbs+num_mcbs]; # We reduce the available backhaul capacity for macro cells

	# ===> Add the consumption from mMTC devices associated to small cell, which is in turn associated with a macro cell

	ini_idx = 0; # Initial index
	nex_idx = SCBS_per_MCBS[0]; # The next index
	for i in range(SCBS_per_MCBS.shape[0]):
		BH_Capacity_MC[i] = BH_Capacity_MC[i] - np.sum(data_rate_consum_BH_mMTC[ini_idx:nex_idx]) # Reduce the Available MC backhaul capacity further due the mMTC devices on the SCs associated to it
		if i < (SCBS_per_MCBS.shape[0] - 1):
			ini_idx = nex_idx # Update the indices
			nex_idx = nex_idx + SCBS_per_MCBS[i+1] # Update the indices
		else:
			break

	#print "Creating the Application to Access Point SINR association matrix"
	#iter = 0; # Application number tracking
	# for i in range(0,sinr_APs.shape[0]):
	# 	#sinr_eMBB [iter:iter + np.sum(user_AP_assoc[i,1]), :] = np.delete(np.outer(user_AP_assoc[i,1],sinr_APs[i,:]), np.where(user_AP_assoc[i,1] == 0), 0);# Application to Base Station SINR matrix 
	#  	SNR_eMBB[iter:iter + np.sum(user_AP_assoc[i,1]), :] = np.delete(np.outer(user_AP_assoc[i,1],SNR_iter[i,:]), np.where(user_AP_assoc[i,1] == 0), 0);# Application to Base Station SINR matrix 
	#  	iter = iter + np.sum(user_AP_assoc[i,1]); # Incrementing the iterator for the next user-application sets
	
	#sinr_eMBB = sinr_APs
	#SNR_eMBB = SNR_iter
	#np.savetxt('SINR.csv',sinr_eMBB, delimiter=",")
	#print sinr_eMBB
	csvsaver.csvsaver(sinr_eMBB, [], "SINR"+str(N)+".csv")
	#print sinr_eMBB
	#print sinr_applications[0]

	print "Calculating the Rate Matrix"
	rate = np.empty((sinr_eMBB.shape[0], sinr_eMBB.shape[1])); # Initializing the received data rate matrix

	for i in range(0, sinr_eMBB.shape[1]):
		#if i <= num_scbs:
			#rate[:,i] = np.where(sinr_eMBB[:,i] == sinr_pad_val, 0, scn.sc_bw*np.log2(1 + 10**(sinr_eMBB[:,i]/10))); # Rate calculation for SC
			#rate[:,i] = np.where(sinr_eMBB[:,i] == sinr_pad_val, 0, scn.usr_scbw*np.log2(1 + 10**(sinr_eMBB[:,i]/10))); # Rate calculation for SC
			#rate[:,i] = np.where(sinr_eMBB[:,i] == sinr_pad_val, 0, np.log2(1 + 10**(sinr_eMBB[:,i]/10))); # Rate calculation for SC				
		#else:
			#rate[:,i] = np.where(sinr_eMBB[:,i] == sinr_pad_val, 0, scn.mc_bw*np.log2(1 + 10**(sinr_eMBB[:,i]/10))); # Rate calculation for MC  
			rate[:,i] = np.where(sinr_eMBB[:,i] == sinr_pad_val, 0, np.log2(1 + 10**(sinr_eMBB[:,i]/10))); # Rate calculation for MC  

	var_row_num = sinr_eMBB.shape[0];
	var_col_num = sinr_eMBB.shape[1];

	#print var_row_num

	print "Calculating the AP path latencies"
	#print Hops_SC
	bh_paths = np.empty((num_scbs+num_mcbs)); # We consider just a single path per AP
	Hops_sc = (np.sum((Hops_SC - 1), axis = 1)).reshape(Hops_SC.shape[0],1); # Reshaping the Small cells hops count matrix
	#print mat_wlbh_sc
	#print mat_wrdbh_sc
	#print Hops_sc
	for i in range(0,Hops_sc.shape[0]):
		if np.nonzero(mat_wlbh_sc[i,:]):
			bh_paths[i] = Hops_sc[i,:]*scn.wrd_link_delay + scn.wl_link_delay; # Small cell path latency with wireless backhaul
		elif np.nonzero(mat_wrdbh_sc[i,:]):
			bh_paths[i] = (Hops_sc[i,:] + 1)*scn.wrd_link_delay; # Small cell path latency with wired backhaul
	#bh_paths[:Hops_sc.shape[0],:] = Hops_sc*scn.wrd_link_delay + scn.wl_link_delay; # Small cell path latency
	bh_paths[Hops_sc.shape[0]:bh_paths.shape[0]] = Hops_MC*scn.wrd_link_delay; # Macro cell path latency  
	#print bh_paths
	#print var_col_num

	#print rate

	# =================================
	# Baseline Cell Selection Algorithm
	# =================================

	
	#DR_eMBB_scbw, DR_eMBB_fscbw, DR_mMTC, DR_eMBB_sinr_scbw, DR_eMBB_sinr_fscbw, DR_mMTC_sinr = baseline_assoc(RX_power_eMBB, RX_power_mMTC, sinr_eMBB, sinr_mMTC, np, scn); # Baseline association function 
	#np.savez_compressed(os.getcwd()+'/Data/Temp/Baseline'+ str(vars(args)['iter']) + str(k), DR_eMBB_scbw, DR_eMBB_fscbw, DR_mMTC, DR_eMBB_sinr_scbw, DR_eMBB_sinr_fscbw, DR_mMTC_sinr, allow_pickle = True); # Save these variables to be utilized by the optimizer

	
	Tot_Data_Rate, Associated_users = baseline_assoc(SNR_eMBB, 0, sinr_eMBB, 0, BH_Capacity_SC, BH_Capacity_MC, num_scbs, num_mcbs, np, scn, 0); # Baseline association function 
	np.savez_compressed(os.getcwd()+'/Data/Process/Baseline'+ str(vars(args)['iter']) + str(N), Tot_Data_Rate, Associated_users, allow_pickle = True); # Save these variables to be utilized by the optimizer
	
	Tot_Data_Rate_min, Associated_users_min = baseline_assoc(SNR_eMBB, 0, sinr_eMBB, 0, BH_Capacity_SC, BH_Capacity_MC, num_scbs, num_mcbs, np, scn, 1); # Baseline association function with minimum rate
	np.savez_compressed(os.getcwd()+'/Data/Process/Baseline_minrate'+str(vars(args)['iter'])+str(N), Tot_Data_Rate_min, Associated_users_min, allow_pickle = True); # Save these variables to plot the baseline with min rate also  
	
	# =========
	# Optimizer
	# =========
	#print rate 
	print "Entering the Optimizer"

	try:
		m = Model("mip1") # Creates the MIP model 
		X = m.addVars(var_row_num, var_col_num , vtype = GRB.BINARY, name = "X"); # We create the X matrix that has to be found 
		BW_MC = m.addVars(var_row_num, 5, vtype = GRB.BINARY, name = "bwmc"); # We create the MC bandwidth matrix 
		BW_SC = m.addVars(var_row_num, 3, vtype = GRB.BINARY, name = "bwsc"); # We create the SC bandwidth matrix
		G_SC = m.addVars(int(var_row_num), int(num_scbs), 3, vtype = GRB.BINARY, name = "GSC"); # Linearizing variable for small cells
		G_MC = m.addVars(int(var_row_num), int(num_mcbs), 5, vtype = GRB.BINARY, name = "GMC"); # Linearizing variable for macro cells

		# ===> Establish the Objective Function

		obj_func = quicksum(G_SC[i,j,k]*scn.BW_SC[k]*rate[i,j] for i in range(var_row_num) for j in range(num_scbs) for k in range(3)) + quicksum(G_MC[i,j - num_scbs,k]*scn.BW_MC[k]*rate[i,j] for i in range(var_row_num) for j in range(num_scbs, num_scbs + num_mcbs) for k in range(5)); 
		#obj_func = LinExpr(); # Initialize the objective function
		#obj_func.addTerms(rate,X);
		# for i in range(0,var_row_num):
		# 	for j in range(0, var_col_num):
		# 			if j < num_scbs:
		# 				for k in range(3):
		# 					obj_func = obj_func + (G_SC[i,j,k]*scn.BW_SC[k]*rate[i,j]); # Small cell contribution  
		# 			elif j >= num_scbs:
		# 				for f in range(5):
		# 					obj_func = obj_func + (G_MC[i,j - num_scbs,f]*scn.BW_MC[k]*rate[i,j]); # Macro cell contribution
		# 			#print obj_func

		#print obj_func

		# ===================================================
		# Set up the Dual and Single Connectivity Constraints

		DC = m.addVars(var_row_num,1, name = "DC"); # Initializing the Constraint Variables
		for i in range(0,var_row_num):
			#DC[i,0] = X.sum(i,'*'); # Constraint Expression
			DC[i,0] = quicksum(X[i,j] for j in range(var_col_num))
		
		MC = m.addVars(var_row_num,1, name = "K"); # Initializing the Constraint Variables with MC 
		for i in range(0,var_row_num):
			#MC[i,0] = X.sum(i,np.arange(num_scbs,num_scbs+num_mcbs).tolist()); # Macro Cell Constraint Expression
			MC[i,0] = quicksum(X[i,j] for j in range(num_scbs, num_mcbs+num_scbs)) # Macro Cell Constraint Expression

		SC = m.addVars(var_row_num,1, name = "J"); # Initializing the Constraint Variable with SC
		for i in range(0,var_row_num):
			#SC[i,0] = X.sum(i,np.arange(0,num_scbs).tolist()); # Small Cell Constraint Expression
			SC[i,0] = quicksum(X[i,j] for j in range(num_scbs)) # Small Cell Constraint Expression
		# ======================================================= 
		# Set up the Minimum Rate Constraint for the Applications


		min_RATE = m.addVars(var_row_num, 1, name = "min_RATE"); # Initializing the Constraint variable
		for i in range(0,var_row_num):
			min_RATE[i,0] = quicksum(rate[i,j]*scn.BW_SC[k]*G_SC[i,j,k] for j in range(num_scbs) for k in range(3)) + quicksum(rate[i,j]*scn.BW_MC[k]*G_MC[i,j - num_scbs,k] for j in range(num_scbs, num_scbs+num_mcbs) for k in range(5))
			# for j in range(0, var_col_num):
			# 	if j < num_scbs:
			# 		for k in range(3):
			# 			min_RATE[i,0] = min_RATE[i,0] + rate[i,j]*scn.BW_SC[k]*G_SC[i,j,k]; # Constraint expression

			# 	elif j >= num_scbs:
			# 		for k in range(5):
			# 			min_RATE[i,0] = min_RATE[i,0] + rate[i,j]*scn.BW_MC[k]*G_MC[i,j - num_scbs,k]; # Constraint expression


		# ===> Set up the Resource Allocation Constraint for an AP

		RB = m.addVars(var_col_num, 1, name = "Subcarriers"); # Allocated Subcarriers
		for j in range(0, var_col_num):
			if j < num_scbs:
				RB[j,0] = quicksum(G_SC[i,j,k]*scn.BW_SC[k] for i in range(var_row_num) for k in range(3))
			elif j >= num_scbs:
				RB[j,0] = quicksum(G_MC[i,j - num_scbs,k]*scn.BW_MC[k] for i in range(var_row_num) for k in range(5))
			# for i in range(0, var_row_num):
			# 	if j < num_scbs:
			# 		#RB[i,0] = LinExpr([scn.usr_scbw]*var_row_num,X.select('*',i)); # Constraint Expression for SCBS
			# 		#print LinExpr(G_SC.select(i,j,'*'), scn.BW_SC)
			# 		for k in range(3):
			# 			#RB[j,0] = RB[j,0] + LinExpr(G_SC.select(i,j,'*'), scn.BW_SC)
			# 			RB[j,0] = RB[j,0] + G_SC[i,j,k]*scn.BW_SC[k]
			# 	elif j >= num_scbs:
			# 		#RB[j,0] = RB[j,0] + LinExpr(G_MC.select(i,j - num_scbs,'*'), scn.BW_MC)
			# 		for k in range(5):
			# 			RB[j,0] = RB[j,0] + G_MC[i,j - num_scbs,k]*scn.BW_MC[k]
			# 	#RB[i,0] = LinExpr([scn.mc_bw]*var_row_num, X.select('*',i)); # Constraint Expression for MCBS
		#max_BW = m.addVars(var_col_num,1, name="Max_BW"); # Initializing the Constraint Variable
		#for i in range(0,)

		# =======================================
		# Set up the Backhaul Capacity constraint 

		BH_CAP_RES = m.addVars(var_col_num, 1, name = "BH_CAP_RES"); # Initializing the Constraint variable
		for j in range(0, num_scbs):
			# for i in range(0, var_row_num):
			# 	for k in range(3):
			# BH_CAP_RES[j,0] = BH_CAP_RES[j,0] + rate[i,j]*G_SC[i,j,k]*scn.BW_SC[k]; # Constraint Expression
			BH_CAP_RES[j,0] = quicksum(rate[i,j]*G_SC[i,j,k]*scn.BW_SC[k] for i in range(var_row_num) for k in range(3)); # Constraint Expression

		count_scbs = 0; # Counter to keep track of the SCBS for a given MCBS
		for j in range(num_scbs, num_scbs + num_mcbs):
			ini_idx = count_scbs; # Initial index
			out_idx = count_scbs + SCBS_per_MCBS[j - num_scbs];
			BH_CAP_RES[j,0] = quicksum(rate[i,j]*G_MC[i,j - num_scbs,k]*scn.BW_MC[k] for i in range(var_row_num) for k in range(5)) + quicksum(rate[i,l]*G_SC[i,l,k]*scn.BW_SC[k] for i in range(var_row_num) for l in range(ini_idx,out_idx) for k in range(3))
			# for i in range(var_row_num):
			# 	for k in range(5):
			# 		BH_CAP_RES[j,0] = BH_CAP_RES[j,0] + rate[i,j]*G_MC[i,j - num_scbs,k]*scn.BW_MC[k]; # Macro cell backhaul capacity computation for constraint expression
			# for l in range(ini_idx, out_idx):
			#  	BH_CAP_RES[j,0] = BH_CAP_RES[j,0] + BH_CAP_RES[l,0];
			count_scbs = out_idx; # Updated the counter for the next round  

			#print BH_CAP_RES[i,0]

		# ================================== 
		# Set up the Path Latency constraint

		AP_latency = m.addVars(var_row_num, var_col_num, name="AP_latency"); # Initializing the Path Latency constraint
		for i in range(0, var_row_num):
			for j in  range(0, var_col_num):
				AP_latency[i,j] = bh_paths[j]*X[i,j]; # Constraint Expression
		

		# ============================
		# Unity Assignment Constraints

		U_SC = m.addVars(var_row_num, int(num_scbs), name="USC");
		
		for i in range(var_row_num):
			#U_SC[i,j] = LinExpr([1]*3,G_SC.select(i,j,'*'))
			for j in range(num_scbs):
				U_SC[i,j] = quicksum(G_SC[i,j,k] for k in range(3))

		U_MC = m.addVars(var_row_num, int(num_mcbs), name="UMC")
		for i in range(var_row_num):
			#U_MC[i,j] = LinExpr([1]*5,G_MC.select(i,j,'*'))
			for j in range(num_mcbs):
				U_MC[i,j] = quicksum(G_MC[i,j,k] for k in range(5))


		# ==========================
		# Set up the mMTC Constraint

		#mMTC_BW = m.addVars()


		# ======================
		# Solve the MILP problem 

		m.setObjective(obj_func, GRB.MAXIMIZE); # This is the objective function that we aim to maximize
		
		# We add a Compulsory Resource allocation Constraint 

		m.addConstrs((RB[i,0] <= scn.sc_bw for i in range(num_scbs)), name = 'c0'); # Small cells have their bandwidth distributed 
		m.addConstrs((RB[i,0] <= scn.eNB_bw for i in range(num_scbs, num_scbs+num_mcbs)), name = 'c10')
		m.addConstrs((G_SC[i,j,k] <= BW_SC[i,k] for i in range(var_row_num) for j in range(num_scbs) for k in range(3)), name = 'l1'); # Linearization constraint 1
		m.addConstrs((G_SC[i,j,k] <= X[i,j] for i in range(var_row_num) for j in range(num_scbs) for k in range(3)), name = 'l2'); # Linearization constraint 2
		m.addConstrs((G_SC[i,j,k] >= (BW_SC[i,k] + X[i,j] -1) for i in range(var_row_num) for j in range(num_scbs) for k in range(3)), name = 'l3'); # Linearization constraint 3
		m.addConstrs((G_MC[i,j - num_scbs,k] <= BW_MC[i,k] for i in range(var_row_num) for j in range(num_scbs, num_scbs+num_mcbs) for k in range(5)), name = 'l4'); # Linearization constraint 4
		m.addConstrs((G_MC[i,j - num_scbs,k] <= X[i,j] for i in range(var_row_num) for j in range(num_scbs, num_scbs+num_mcbs) for k in range(5)), name = 'l5'); # Linearization constraint 5
		m.addConstrs((G_MC[i,j - num_scbs,k] >= (BW_MC[i,k] + X[i,j] -1) for i in range(var_row_num) for j in range(num_scbs, num_scbs+num_mcbs) for k in range(5)), name = 'l6'); # Linearization constraint 6
		m.addConstrs((U_SC[i,j] <= 1 for i in range(var_row_num) for j in range(num_scbs)),name = 'U1')
		m.addConstrs((U_MC[i,j] <= 1 for i in range(var_row_num) for j in range(num_mcbs)),name = 'U2')

		if vars(args)['dual'] == 0:
			print "==================="
			print "Single Connectivity"
			print "==================="
			#m.addConstrs((U_SC[i,j] == 0 for i in range(var_row_num) for j in range(num_scbs)))
			#m.addConstrs((U_MC[i,j] == 1 for i in range(var_row_num) for j in range(num_mcbs)))
			m.addConstrs((DC[i,0] <= 1 for i in range(var_row_num)), name ='c'); # Adding the Single Connectivity constraint 
			#m.addConstrs((MC[i,0] == 1 for i in range(var_row_num)), name ='c'); # Adding the Single Connectivity constraint 
			#m.addConstrs((SC[i,0] == 0 for i in range(var_row_num)), name ='c14'); # Adding the Single Connectivity constraint 
			
			if vars(args)['minRate'] == 1:
				m.addConstrs((min_RATE[i,0] >= scn.eMBB_minrate for i in range(var_row_num)), name ='c1'); # Adding the minimum rate constraint
			else:
				m.addConstrs((min_RATE[i,0] >= 1 for i in range(var_row_num)), name ='c15'); # Adding the minimum rate constraint for other strategies, which is to provide some data rate atleast (1 bps)
			
			if vars(args)['bhaul'] == 1:
				m.addConstrs((BH_CAP_RES[i,0] <= BH_Capacity_SC[i,0] for i in range(num_scbs)), name = 'c2'); # Adding the Backhaul capacity constraint
				m.addConstrs((BH_CAP_RES[i,0] <= BH_Capacity_MC[i-num_scbs] for i in range(num_scbs,num_scbs + num_mcbs)), name = 'c3'); # Adding the Backhaul capacity constraint
				
			if vars(args)['latency'] == 1:
				m.addConstrs((AP_latency[i,j] <= scn.eMBB_latency_req for i in range(0, var_row_num) for j in range(0, var_col_num)), name = 'c4'); # Path latency constraint 
			
		elif vars(args)['dual'] == 1:	
			print "================="	
			print "Dual Connectivity"
			print "================="
			m.addConstrs((DC[i,0] == 2 for i in range(var_row_num)), name ='c'); # Adding the Dual Connectivity constraint 
			#m.addConstrs((U_SC[i,j] <= 1 for i in range(var_row_num) for j in range(num_scbs)))
			#m.addConstrs((U_MC[i,j] == 1 for i in range(var_row_num) for j in range(num_mcbs)))
			#m.addConstrs((MC[i,0] == 1 for i in range(var_row_num)), name ='c'); # Adding the Dual Connectivity constraint 
			#m.addConstrs((quicksum(X[i,j] for j in range(num_scbs, num_scbs+num_mcbs)) == 1 for i in range(var_row_num)), name ='c8'); # Adding the Dual Connectivity constraint 
			#m.addConstrs((SC[i,0] <= 1 for i in range(var_row_num)), name ='c5'); # Adding the Dual Connectivity constraint 
			

			#m.addConstrs((DC[i,0] >= 1 for i in range(var_row_num)), name ='c5'); # Adding the Dual Connectivity constraint 
			if vars(args)['minRate'] == 1:
				m.addConstrs((min_RATE[i,0] >= scn.eMBB_minrate for i in range(var_row_num)), name ='c1'); # Adding the minimum rate constraint
			else:
				m.addConstrs((min_RATE[i,0] >= 1 for i in range(var_row_num)), name ='c15'); # Adding the minimum rate constraint for other strategies, which is to provide some data rate atleast (1 bps)
				
			if vars(args)['bhaul'] == 1:
				#print "here"
				m.addConstrs((BH_CAP_RES[i,0] <= BH_Capacity_SC[i,0] for i in range(num_scbs)), name = 'c2'); # Adding the Backhaul capacity constraint
				m.addConstrs((BH_CAP_RES[i,0] <= BH_Capacity_MC[i-num_scbs] for i in range(num_scbs,num_scbs + num_mcbs)), name = 'c3'); # Adding the Backhaul capacity constraint
				#print "Applied Constraints"

			if vars(args)['latency'] == 1:
				#print "LATHere"
				m.addConstrs((AP_latency[i,j] <= scn.eMBB_latency_req for i in range(0, var_row_num) for j in range(0, var_col_num)), name = 'c4'); # Path latency constraint 
				#print "Applied Constraints"
					
		#m.addConstrs((min_RATE[i,0] >= scn.eMBB_minrate for i in range(var_row_num)), name ='c1'); # Adding the minimum rate constraint 
		#m.addConstrs((BH_CAP_RES[i,0] <= BH_Capacity_SC[i,0] for i in range(num_scbs)), name = 'c2'); # Adding the Backhaul capacity constraint
		#m.addConstrs((BH_CAP_RES[i,0] <= BH_Capacity_MC for i in range(num_scbs,num_scbs + num_mcbs)), name = 'c3'); # Adding the Backhaul capacity constraint
		#m.addConstrs((AP_latency[i,0] <= scn.eMBB_latency_req for i in range(var_row_num)), name = 'c4'); # Path latency constraint 
		
		#if vars(args)['mipGP'] == 1:
		m.Params.MIPGap = 0.05; # Set the Upper and Lower Bound Gap to 5%
		m.Params.TIME_LIMIT = 600; # Set a timelimit of 10 seconds
		if vars(args)['dual'] == 1 and vars(args)['minRate'] == 1: 
				m.Params.MIPFocus = 1; # To aid in convergence of the DC-MRT solution
				m.Params.Cuts = 3; # Aggressive Cuts for faster convergence
		else:
			pass

		m.update()
		m.write("debug.lp");

		m.optimize()
		print "Optimized"
		if m.status == 2:
			# ============================ 
			# Print the Optimized Solution 

			print "Plotting the Optimized Association"

			X_optimal = []; # Initializing the Variable to store the optimal solution

			for v in m.getVars():
				#print ('%s %g' % (v.varName, v.X))
				X_optimal.append(v.x); 
				#print X_optimal
				if len(X_optimal) >= var_row_num*var_col_num:
					break
			#plotter.optimizer_plotter(np.asarray(X_optimal).reshape((var_row_num,var_col_num)));
			print('Obj:', m.objVal)
			
			G_plt = []
			M_plt = []
			SCbw = []
			MCbw = []
			MC_sel = []

			for v in m.getVars():
				if "GSC" in v.varName:
					G_plt.append(v.x)
				if "GMC" in v.varName:
					M_plt.append(v.x)
				#if "" in v.varName:
				#	print v.x
			#	if "bwmc" in v.varName:
			#		MCbw.append(v.x)
			#	if "bwsc" in v.varName:
			#		SCbw.append(v.x)

			# =============================
			# Visualization and Computation			

			
			#fin_rate = np.zeros((var_row_num,1))
			#for i in range(var_row_num):
			#	for j in range(var_col_num):
			#		if j < num_scbs:
			#			fin_rate[i] = fin_rate[i] + X[i,j]*scn.BW_SC[np.where(G_plt[i,:] == 1)]*rate[i,j]
			#		elif j >= num_scbs:
			#			fin_rate[i] = fin_rate[i] + X[i,j]*scn.BW_MC[np.where(M_plt[i,:] == 1)]*rate[i,j]

			#plotter.optimizer_plotter(np.asarray(G_plt).reshape((var_row_num,3)))
			#plotter.optimizer_plotter(np.asarray(M_plt).reshape((var_row_num,5)))
			#plotter.optimizer_plotter(fin_rate)
			#print np.sum(G_plt)			
			#print np.sum(M_plt)
			G_plt_idx = np.asarray(G_plt).reshape((var_row_num,num_scbs,3))
			M_plt_idx = np.asarray(M_plt).reshape((var_row_num,num_mcbs,5))
			new_rate = np.empty((rate.shape[0], rate.shape[1])); # The actual rate matrix
			
			#test = G_plt_idx[1,1,:]*scn.BW_SC
			GSC_compute = np.empty((var_row_num, num_scbs)); # This holds the bandwidth contribution from each small cell
			GMC_compute = np.empty((var_row_num, num_mcbs)); # This holds the bandwidth contribution from each macro cell

			for i in range(var_row_num):
				for j in range(num_scbs):
					GSC_compute[i,j] = sum(G_plt_idx[i,j,:]*np.asarray(scn.BW_SC[:]))

			for i in range(var_row_num):
				for j in range(num_mcbs):
					GMC_compute[i,j] = sum(M_plt_idx[i,j,:]*np.asarray(scn.BW_MC[:]))

			G_total_compute = np.concatenate((GSC_compute, GMC_compute), axis = 1) # Bandwidth Contribution matrix
			new_rate = rate*G_total_compute; # New rate matrix
			#print new_rate

			# ==> Data rate each user gets from the SC and MC for MCSC - DC

			Rate_data = np.empty((var_row_num,4))
			G_sum = np.empty((num_scbs,1))
			M_sum = np.empty((num_mcbs,1))
			for i in range(var_row_num):
				# print "=============="
				# print ("For user #",i)
				# #print new_rate[i,num_scbs + np.nonzero(new_rate[i,num_scbs:num_scbs+num_mcbs])]
				# print("Small Cell Data Rate:", new_rate[i, np.nonzero(new_rate[i,:num_scbs])])
				# print("Macro Cell Data Rate:", new_rate[i, num_scbs + np.nonzero(new_rate[i,num_scbs:num_scbs+num_mcbs])])
				if new_rate[i, np.nonzero(new_rate[i,:num_scbs])].size == 0:
					Rate_data[i,0] = 0
				else:
					Rate_data[i,0] = np.sum(new_rate[i, np.nonzero(new_rate[i,:num_scbs])])

				if new_rate[i, num_scbs + np.nonzero(new_rate[i,num_scbs:num_scbs+num_mcbs])].size == 0:
					Rate_data[i,1] = 0
				else:
					Rate_data[i,1] = np.sum(new_rate[i, num_scbs + np.nonzero(new_rate[i,num_scbs:num_scbs+num_mcbs])])
				#print np.where(new_rate[i, num_scbs + np.nonzero(new_rate[i,num_scbs:num_scbs+num_mcbs])].size == 0 , "Size is 0", new_rate[i, num_scbs + np.nonzero(new_rate[i,num_scbs:num_scbs+num_mcbs])]) 

			#csvsaver.csvsaver(Rate_data,["SC Rate","MC Rate"], "OptimizedDataRate_user_MCSC.csv")
			#print np.sum(new_rate, axis = 1)
			print np.amin(np.sum(new_rate,axis  = 1))
			print np.amax(np.sum(new_rate,axis  = 1))
			print "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
			print ("MINRATE Constraint satisfied?:", np.all(np.sum(new_rate,axis  = 1)>=scn.eMBB_minrate))
			print "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
			# #print "============================"
			#print "BHCAP check"
			#bhval = 0; # Initialize the Utilized Bhaul capacity value
			
			#		new_rate[i,j] = (np.sum(G_plt_idx[i,j,:]*np.asarray(scn.BW_SC),axis = 0) + np.sum(M_plt_idx[i,j,:]*np.asarray(scn.BW_MC),axis = 0))*rate[i,j]			
	
			G_sum[:,0] = np.sum(G_plt_idx[:,:,0] + G_plt_idx[:,:,1] + G_plt_idx[:,:,2], axis = 0)
			M_sum[:,0] = np.sum(M_plt_idx[:,:,0] + M_plt_idx[:,:,1] + M_plt_idx[:,:,2] + M_plt_idx[:,:,3] + M_plt_idx[:,:,4], axis = 0)
			#print G_sum.shape
			#print M_sum.shape
			#print ("SC:", G_sum)
			#print ("MC:", M_sum)
			csvsaver.csvsaver(G_sum,["Accepted Users per SC"], "GS.csv")
			csvsaver.csvsaver(M_sum,["Accepted Users per MC"], "MC.csv")			

			# if N == (num_iter-1) and (vars(args)['dual'] == 1 or vars(args)['bhaul'] == 1 or vars(args)['minRate'] == 1 or vars(args)['latency'] == 1):
			# 	#plotter.optimizer_plotter(new_rate) # We get the plot for the rates with maximum number of users
			# 	with open("Rate" + str(vars(args)['iter']) + str(vars(args)['dual']) + str(vars(args)['bhaul']) + str(vars(args)['minRate']) + str(vars(args)['latency']) + ".csv", "w+") as my_csv:
			# 		csvWriter = csv.writer(my_csv,delimiter=',')
			# 		csvWriter.writerows(new_rate) # We write the rate matrix to the csv file for visualization
			# 	with open("OptAssignment" + str(vars(args)['iter']) + str(vars(args)['dual']) + str(vars(args)['bhaul']) + str(vars(args)['minRate']) + str(vars(args)['latency']) + ".csv", "w+") as my_csv2:
			# 		csvWriter = csv.writer(my_csv2,delimiter=',')
			# 		csvWriter.writerows(np.asarray(X_optimal).reshape((var_row_num,var_col_num))) # We write the optimal association matrix to csv files for analysis purposes
			#plotter.optimizer_plotter(M_plt_idx[:,:,0] + M_plt_idx[:,:,1] + M_plt_idx[:,:,2] + M_plt_idx[:,:,3] + M_plt_idx[:,:,4])0
			#plotter.optimizer_plotter(G_plt_idx[:,:,0] + G_plt_idx[:,:,1] + G_plt_idx[:,:,2])	

			# =========================
			# Store Optimized Variables

			print "Saving Data"

			#Data['X_optimal_data' + str(N)] = np.asarray(X_optimal).reshape((var_row_num,var_col_num)); # Optimal Association Matrix
			Data['X_optimal_data' + str(N)] = (G_total_compute>0)*1; # Optimal Association Matrix
			Data['Net_Throughput' + str(N)] = m.objVal; # Network wide throughput
			Data['Optimal_BW' + str(N)] = G_total_compute; # Stores the optimal Bandwidth 
			Data['Rates' + str(N)] = new_rate; # Data rate matrix  
			Data['Status' + str(N)] = m.status; # Insert the status
			Data['Apps'+str(N)] = var_row_num;
			Data['APs'+str(N)] = var_col_num;
			Data['Time'+str(N)] = m.Runtime;
			Data['SINR'+str(N)] = sinr_eMBB;

			#print np.sum((G_total_compute>0)*1, axis = 1) 
			#print np.sum((np.asarray(X_optimal).reshape((var_row_num,var_col_num))>0)*1, axis =1)
			#print np.nonzero((np.asarray(X_optimal).reshape((var_row_num,var_col_num))>0)*1)
			#print np.array_equal((G_total_compute>0)*1, (np.asarray(X_optimal).reshape((var_row_num,var_col_num))>0)*1)
			# ========================
			# Validity of the Solution

			# ===> Backhaul Capacity Utilization

			bhutil = np.empty((var_col_num,1)) # This variable will hold the utilized Backhaul capacity
			ct = 0; # Counting the number of small cells within a macro cell
			for j in range(var_col_num):
				if j < num_scbs:
					bhutil[j] = np.sum(new_rate[:,j]) # We sum the data rate of the users getting access from a small cell
					#print ("Available Backhaul Capacity for SC#"+str(j),BH_Capacity_SC[j,0])
				elif j >= num_scbs:
					idx_in = ct
					idx_ot = ct + SCBS_per_MCBS[j - num_scbs]
					bhutil[j] = np.sum(new_rate[:,j]) + np.sum(bhutil[idx_in:idx_ot]) # Total BH utilization for the BH on macro cell
					#print ("Available Backhaul Capacity for MC#"+str(j-num_scbs),BH_Capacity_MC)
					ct = idx_ot # Update the count variable for the next Macro BS

				#print ("Backhaul Utilization for AP#" + str(j), bhutil[j])
			Data['BHUTIL'+str(N)] = bhutil; # Save the Backhaul utilization values for visualization
			Data['AvailBHUtil_SC'+str(N)] = BH_Capacity_SC; # Save the Available Small Cell BH Capacity. For MC it is constant and can be extracted from the scenario_file 
			# ===> Latency Provisioning 

			lat_prov = np.matmul(np.asarray(X_optimal).reshape((var_row_num,var_col_num)),np.diag(bh_paths)); # This variable will hold the actual latency offered
			# print np.asarray(X_optimal).reshape((var_row_num,var_col_num))
			# print lat_prov
			# print np.diag(bh_paths)
			# #print var_row_num
			# for i in range(var_row_num):
			# 	print ("Latency offered for element #"+str(i),lat_prov[i,np.nonzero(lat_prov[i,:])])

			Data['LatenOff'+str(N)] = lat_prov; # Save the latency provisioned by the algorithm 

		else:
			Data['Status' + str(N)] = m.status; # Add the status for detecting infeasible solution
			print ("Status_Flags:" + str(N), str(m.status))
			Data['Apps' + str(N)] = var_row_num;
			Data['APs' + str(N)] = var_col_num;
			continue
	except GurobiError:
		print('Error Reported')
np.savez_compressed(os.getcwd() +'/Data/Process/_' + str(vars(args)['iter']) + 'dat_' + str(vars(args)['dual']) + str(vars(args)['minRate']) + str(vars(args)['bhaul']) + str(vars(args)['latency']), Data, allow_pickle = True); # Saving the necessary data to generate plots later 
