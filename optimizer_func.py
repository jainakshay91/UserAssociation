#!/usr/bin/env python

# =============================
# Import the necessary binaries
# =============================

from gurobipy import *
import numpy as np 
import plotter
from scenario_var import scenario_var 

# ==============================
# Create the Model and Variables
# ==============================
scn = scenario_var(); # Initializing the class variables

print "Collecting the Stored Variables"

optim_data = np.load('/home/akshayjain/Desktop/Simulation/optim_var.npz')
sinr_APs = optim_data['arr_0']; # Load the SINR data to be used for the optimization
user_AP_assoc = optim_data['arr_1'].item()['user_app0']; # Load the User-Applications Association data to be used for the optimization
sinr_eMBB = np.empty([np.sum(user_AP_assoc[:,1]),sinr_APs.shape[1]],dtype=float); # Array that holds the Application SINR values
sinr_pad_val = optim_data['arr_4']; # In the small cell calculation we use an sinr pad value for ease of computation
num_scbs = optim_data['arr_5']; # Number of Small cells
num_mcbs = optim_data['arr_6']; # Number of Macro cells

print "Creating the Application to Access Point SINR association matrix"
iter = 0; # Application number tracking
for i in range(0,sinr_APs.shape[0]):
	sinr_eMBB [iter:iter + np.sum(user_AP_assoc[i,1]), :] = np.delete(np.outer(user_AP_assoc[i,1],sinr_APs[i,:]), np.where(user_AP_assoc[i,1] == 0), 0);# Application to Base Station SINR matrix 
 	iter = iter + np.sum(user_AP_assoc[i,1]); # Incrementing the iterator for the next user-application sets


#print sinr_applications[0]
rate = np.empty((sinr_eMBB.shape[0], sinr_eMBB.shape[1])); # Initializing the received data rate matrix

for i in range(0, sinr_eMBB.shape[1]):
	if i <= num_scbs:
		rate[:,i] = np.where(sinr_eMBB[:,i] == sinr_pad_val, 0, scn.sc_bw*np.log2(1 + 10**(sinr_eMBB[:,i]/10))); # Log Power calculation part of the Shannon-Hartley theorem
	else:
		rate[:,i] = scn.mc_bw*np.log2(1 + 10**(sinr_eMBB[:,i]/10)); 

var_row_num = sinr_eMBB.shape[0];
var_col_num = sinr_APs.shape[1];

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

	# ===> Set up the Dual and Single Connectivity Constraints

	DC = m.addVars(var_row_num,1, name = "DC"); # Initializing the Constraint Variables
	for i in range(0,var_row_num):
		DC[i,0] = X.sum(i,'*'); # Constraint Expression
	
	# ===> Set up the Minimum Rate Constraint for the Applications

	min_RATE = m.addVars(var_row_num, 1, name = "min_RATE"); # Initializing the Constraint variable
	for i in range(0,var_row_num):
		min_RATE[i,0] = LinExpr(rate[i,:],X.select(i,'*')); # Constraint expression

	# ===> Set up the Maximum Bandwidth Constraint for an AP

	max_BW = m.addVars(var_col_num,1, name="Max_BW"); # Initializing the Constraint Variable
	

	# ===> Solve the MILP problem 

	m.setObjective(obj_func, GRB.MAXIMIZE); # This is the objective function that we aim to maximize
	#m.addConstrs((DC[i,0] == 1 for i in range(var_row_num)), name ='c'); # Adding the Single Connectivity constraint 
	m.addConstrs((DC[i,0] <= 2 for i in range(var_row_num)), name ='c'); # Adding the Dual Connectivity constraint 
	m.addConstrs((min_RATE[i,0] >= scn.eMBB_minrate for i in range(var_row_num)), name ='c1'); # Adding the minimum rate constraint 

	m.optimize()

	# ===> Print the Optimized Solution 

	print "Plotting the Optimized Association"

	X_optimal = []; # Initializing the Variable to store the optimal solution

	for v in m.getVars():
		X_optimal.append(v.x); 
		if len(X_optimal) >= var_row_num*var_col_num:
			break
	plotter.optimizer_plotter(np.asarray(X_optimal).reshape((var_row_num,var_col_num)));
	print('Obj:', m.objVal)

except GurobiError:
	print('Error Reported')