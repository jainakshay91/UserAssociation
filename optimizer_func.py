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
user_AP_assoc = optim_data['arr_1'].item()['user_app0']; # Load the User Association data to be used for the optimization
sinr_applications = np.empty([np.sum(user_AP_assoc),sinr_APs.shape[1]],dtype=float); # Array that holds the Application SINR values
sinr_pad_val = optim_data['arr_4'];

print "Creating the Application to Access Point SINR association matrix"
iter = 0; # Application number tracking
for i in range(0,sinr_APs.shape[0]):
	sinr_applications [iter:iter + np.sum(user_AP_assoc[i,:]), :] = np.delete(np.outer(user_AP_assoc[i,:],sinr_APs[i,:]), np.where(user_AP_assoc[i,:] == 0), 0);# Application to Base Station SINR matrix 
 	iter = iter + np.sum(user_AP_assoc[i,:]); # Incrementing the iterator for the next user-application sets


#print sinr_applications[0]

log_pow = np.where(sinr_applications == sinr_pad_val, 0, scn.sc_bw*np.log2(1 + 10**(sinr_applications/10))); # Log Power calculation part of the Shannon-Hartley theorem
var_row_num = sinr_applications.shape[0];
var_col_num = sinr_APs.shape[1];

print log_pow
# =========
# Optimizer
# =========

print "Entering the Optimizer"

try:
	m = Model("mip1") # Creates the MIP model 
	X = m.addVars(var_row_num, var_col_num , vtype = GRB.BINARY, name = "X"); # We create the X matrix that has to be found 
	
	# ===> Establish the Objective Function

	obj_func = 0; # Initialize the objective function

	for i in range(0,var_row_num):
		for j in range(0, var_col_num):
				obj_func = obj_func + X[i,j]*log_pow[i,j]; #np.where( log_pow[i,j] == 0, X[i,j]*0, X[i,j]*log_pow[i,j] ); # Objective function 
				#print obj_func

	#print obj_func
	# ===> Set up the Constraints

	# print "Setting up the Constraints"

	m.setObjective(obj_func, GRB.MAXIMIZE); # This is the objective function that we aim to maximize

	m.optimize()

	# ===> Print the Optimized Solution 

	print "Plotting the Optimized Association"

	X_optimal = []; # Initializing the Variable to store the optimal solution

	for v in m.getVars():
		X_optimal.append(v.x); 

	plotter.optimizer_plotter(np.asarray(X_optimal).reshape((var_row_num,var_col_num)));
	print('Obj:', m.objVal)

except GurobiError:
	print('Error Reported')