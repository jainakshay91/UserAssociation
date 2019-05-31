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

# =========================================
# Baseline Scenario: RSSI based association
# =========================================


optim_data_mMTC = np.load(os.getcwd() + '/Data/Temp/optim_var_mMTC'+ str(vars(args)['iter']) +'.npz', allow_pickle = True); # Extracting the mMTC data 


for k in range(0,num_iter):
	#k = 1
	print "=============================================="
	print "Dataset # " + str(k) + ": Collecting the Stored Variables"

	optim_data = np.load(os.getcwd() + '/Data/Temp/optim_var_'+ str(k) + str(vars(args)['iter']) +'.npz', allow_pickle = True)
	sinr_APs = optim_data['arr_0']; # Load the SINR data to be used for the optimization
	user_AP_assoc = optim_data['arr_1'].item()['user_app' + str(k)]; # Load the User-Applications Association data to be used for the optimization
	sinr_eMBB = np.empty([np.sum(user_AP_assoc[:,1]),sinr_APs.shape[1]],dtype=float); # Array that holds the Application SINR values
	sinr_pad_val = optim_data['arr_4']; # In the small cell calculation we use an sinr pad value for ease of computation
	num_scbs = optim_data['arr_5']; # Number of Small cells
	num_mcbs = optim_data['arr_6']; # Number of Macro cells
	mat_wlbh_sc = optim_data['arr_7']; # Wireless backhaul matrix for Small Cells
	mat_wrdbh_sc = optim_data['arr_8']; # Wired backhaul matrix for Macro cells
	Hops_MC = optim_data['arr_9']; # Number of hops to the IMS core from Macro cells
	Hops_SC = optim_data['arr_10']; # Number of hops to the IMS core from Small cells
	BH_Capacity_SC = optim_data['arr_11']; # Backhaul capacity for Small cells
	BH_Capacity_MC = scn.fib_BH_MC_capacity; # Backhaul capacity for Macro cells

	# ==================================
	# Print to Understand Matrix Formats

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


	print "Creating the Application to Access Point SINR association matrix"
	iter = 0; # Application number tracking
	for i in range(0,sinr_APs.shape[0]):
		sinr_eMBB [iter:iter + np.sum(user_AP_assoc[i,1]), :] = np.delete(np.outer(user_AP_assoc[i,1],sinr_APs[i,:]), np.where(user_AP_assoc[i,1] == 0), 0);# Application to Base Station SINR matrix 
	 	iter = iter + np.sum(user_AP_assoc[i,1]); # Incrementing the iterator for the next user-application sets


	#print sinr_applications[0]

	print "Calculating the Rate Matrix"
	rate = np.empty((sinr_eMBB.shape[0], sinr_eMBB.shape[1])); # Initializing the received data rate matrix

	for i in range(0, sinr_eMBB.shape[1]):
		if i <= num_scbs:
			#rate[:,i] = np.where(sinr_eMBB[:,i] == sinr_pad_val, 0, scn.sc_bw*np.log2(1 + 10**(sinr_eMBB[:,i]/10))); # Rate calculation for SC
			rate[:,i] = np.where(sinr_eMBB[:,i] == sinr_pad_val, 0, scn.usr_scbw*np.log2(1 + 10**(sinr_eMBB[:,i]/10))); # Rate calculation for SC

			#rate[:,i] = np.where(sinr_eMBB[:,i] == sinr_pad_val, 0, np.log2(1 + 10**(sinr_eMBB[:,i]/10))); # Rate calculation for SC				
		else:
			rate[:,i] = np.where(sinr_eMBB[:,i] == sinr_pad_val, 0, scn.mc_bw*np.log2(1 + 10**(sinr_eMBB[:,i]/10))); # Rate calculation for MC  
			#rate[:,i] = np.where(sinr_eMBB[:,i] == sinr_pad_val, 0, np.log2(1 + 10**(sinr_eMBB[:,i]/10))); # Rate calculation for MC  

	var_row_num = sinr_eMBB.shape[0];
	var_col_num = sinr_APs.shape[1];

	print "Calculating the AP path latencies"

	bh_paths = np.empty((num_scbs+num_mcbs,1)); # We consider just a single path per AP
	Hops_sc = (np.sum((Hops_SC - 1), axis = 1)).reshape(Hops_SC.shape[0],1); # Reshaping the Small cells hops count matrix
	bh_paths[:Hops_sc.shape[0],:] = Hops_sc*scn.wrd_link_delay + scn.wl_link_delay; # Small cell path latency
	bh_paths[Hops_sc.shape[0]:,0] = Hops_MC*scn.wrd_link_delay; # Macro cell path latency  
	#print bh_paths
	#print var_col_num

	#print rate

	# =========
	# Optimizer
	# =========

	print "Entering the Optimizer"

	try:
		m = Model("mip1") # Creates the MIP model 
		X = m.addVars(var_row_num, var_col_num , vtype = GRB.BINARY, name = "X"); # We create the X matrix that has to be found 
		
		# ===> Establish the Objective Function

		obj_func = 0; 
		#obj_func = LinExpr(); # Initialize the objective function
		#obj_func.addTerms(rate,X);
		for i in range(0,var_row_num):
			for j in range(0, var_col_num):
					obj_func = obj_func + X[i,j]*rate[i,j]; #np.where( log_pow[i,j] == 0, X[i,j]*0, X[i,j]*log_pow[i,j] ); # Objective function 
					#print obj_func

		#print obj_func

		# ===================================================
		# Set up the Dual and Single Connectivity Constraints

		DC = m.addVars(var_row_num,1, name = "DC"); # Initializing the Constraint Variables
		for i in range(0,var_row_num):
			DC[i,0] = X.sum(i,'*'); # Constraint Expression
		
		# ======================================================= 
		# Set up the Minimum Rate Constraint for the Applications

		min_RATE = m.addVars(var_row_num, 1, name = "min_RATE"); # Initializing the Constraint variable
		for i in range(0,var_row_num):
			min_RATE[i,0] = LinExpr(rate[i,:],X.select(i,'*')); # Constraint expression

		# ===> Set up the Resource Allocation Constraint for an AP

		RB = m.addVars(var_col_num, 1, name = "Subcarriers"); # Allocated Subcarriers
		for i in range(0, var_col_num):
			RB[i,0] = LinExpr([scn.usr_scbw]*var_row_num,X.select('*',i)); # Constraint Expression

		#max_BW = m.addVars(var_col_num,1, name="Max_BW"); # Initializing the Constraint Variable
		#for i in range(0,)

		# =======================================
		# Set up the Backhaul Capacity constraint 

		BH_CAP_RES = m.addVars(var_col_num, 1, name = "BH_CAP_RES"); # Initializing the Constraint variable
		for i in range(0, var_col_num):
			BH_CAP_RES[i,0] = LinExpr(rate[:,i],X.select('*',i)); # Constraint Expression
			#print BH_CAP_RES[i,0]

		# ================================== 
		# Set up the Path Latency constraint

		AP_latency = m.addVars(var_row_num, 1, name="AP_latency"); # Initializing the Path Latency constraint
		for i in range(0, var_col_num):
			AP_latency[i,0] = LinExpr(bh_paths,X.select(i,'*')); # Constraint Expression

		# ======================
		# Solve the MILP problem 

		m.setObjective(obj_func, GRB.MAXIMIZE); # This is the objective function that we aim to maximize
		
		# We add a Compulsory Resource allocation Constraint 

		m.addConstrs((RB[i,0] <= scn.sc_bw for i in range(num_scbs)), name = 'c0'); # Small cells have their bandwidth distributed 

		if vars(args)['dual'] == 0:
			print "==================="
			print "Single Connectivity"
			print "==================="
			m.addConstrs((DC[i,0] == 1 for i in range(var_row_num)), name ='c'); # Adding the Single Connectivity constraint 
			if vars(args)['minRate'] == 1:
				m.addConstrs((min_RATE[i,0] >= scn.eMBB_minrate for i in range(var_row_num)), name ='c1'); # Adding the minimum rate constraint
			if vars(args)['bhaul'] == 1:
				m.addConstrs((BH_CAP_RES[i,0] <= BH_Capacity_SC[i,0] for i in range(num_scbs)), name = 'c2'); # Adding the Backhaul capacity constraint
				m.addConstrs((BH_CAP_RES[i,0] <= BH_Capacity_MC for i in range(num_scbs,num_scbs + num_mcbs)), name = 'c3'); # Adding the Backhaul capacity constraint
			if vars(args)['latency'] == 1:
				m.addConstrs((AP_latency[i,0] <= scn.eMBB_latency_req for i in range(var_row_num)), name = 'c4'); # Path latency constraint 
		elif vars(args)['dual'] == 1:	
			print "================="	
			print "Dual Connectivity"
			print "================="
			m.addConstrs((DC[i,0] <= 2 for i in range(var_row_num)), name ='c'); # Adding the Dual Connectivity constraint 
			m.addConstrs((DC[i,0] >= 1 for i in range(var_row_num)), name ='c5'); # Adding the Dual Connectivity constraint 
			if vars(args)['minRate'] == 1:
				m.addConstrs((min_RATE[i,0] >= scn.eMBB_minrate for i in range(var_row_num)), name ='c1'); # Adding the minimum rate constraint
			if vars(args)['bhaul'] == 1:
				m.addConstrs((BH_CAP_RES[i,0] <= BH_Capacity_SC[i,0] for i in range(num_scbs)), name = 'c2'); # Adding the Backhaul capacity constraint
				m.addConstrs((BH_CAP_RES[i,0] <= BH_Capacity_MC for i in range(num_scbs,num_scbs + num_mcbs)), name = 'c3'); # Adding the Backhaul capacity constraint
			if vars(args)['latency'] == 1:
				m.addConstrs((AP_latency[i,0] <= scn.eMBB_latency_req for i in range(var_row_num)), name = 'c4'); # Path latency constraint 
		

		#m.addConstrs((min_RATE[i,0] >= scn.eMBB_minrate for i in range(var_row_num)), name ='c1'); # Adding the minimum rate constraint 
		#m.addConstrs((BH_CAP_RES[i,0] <= BH_Capacity_SC[i,0] for i in range(num_scbs)), name = 'c2'); # Adding the Backhaul capacity constraint
		#m.addConstrs((BH_CAP_RES[i,0] <= BH_Capacity_MC for i in range(num_scbs,num_scbs + num_mcbs)), name = 'c3'); # Adding the Backhaul capacity constraint
		#m.addConstrs((AP_latency[i,0] <= scn.eMBB_latency_req for i in range(var_row_num)), name = 'c4'); # Path latency constraint 
		
		if vars(args)['mipGP'] == 1:
			m.Params.MIPGap = 0.01; # Set the Upper and Lower Bound Gap to 0.1%
		else:
			pass

		m.optimize()

		if m.status == 2:
			# ============================ 
			# Print the Optimized Solution 

			print "Plotting the Optimized Association"

			X_optimal = []; # Initializing the Variable to store the optimal solution

			for v in m.getVars():
				X_optimal.append(v.x); 
				if len(X_optimal) >= var_row_num*var_col_num:
					break
			#plotter.optimizer_plotter(np.asarray(X_optimal).reshape((var_row_num,var_col_num)));
			print('Obj:', m.objVal)

			# =========================
			# Store Optimized Variables

			print "Saving Data"

			Data['X_optimal_data' + str(k)] = np.asarray(X_optimal).reshape((var_row_num,var_col_num)); # Optimal Association Matrix
			Data['Net_Throughput' + str(k)] = m.objVal; # Network wide throughput
			Data['Rates' + str(k)] = rate; # Data rate matrix  
			Data['Status' + str(k)] = m.status; # Insert the status
		else:
			Data['Status' + str(k)] = m.status; # Add the status for detecting infeasible solution
			continue
	except GurobiError:
		print('Error Reported')
np.savez_compressed(os.getcwd() +'/Data/Process/_' + str(vars(args)['iter']) + 'dat_' + str(vars(args)['dual']) + str(vars(args)['minRate']) + str(vars(args)['bhaul']) + str(vars(args)['latency']), Data, allow_pickle = True); # Saving the necessary data to generate plots later 
