#!/usr/bin/env python

# =============================
# Import the necessary binaries
# =============================

from gurobipy import *
import numpy as np 
from scenario_var import scenario_var 

# ==============================
# Create the Model and Variables
# ==============================
scn = scenario_var(); # Initializing the class variables

optim_data = np.load('/home/akshayjain/Desktop/Simulation/optim_var.npz')
sinr_APs = optim_data['arr_0']; # Load the SINR data to be used for the optimization
user_AP_assoc = optim_data['arr_1'].item()['user_app0']; # Load the User Association data to be used for the optimization
sinr_applications = np.empty([np.sum(user_AP_assoc),sinr_APs.shape[1]]); # Array that holds the Application SINR values

iter = 0; # Application number tracking
for i in range(0,sinr_APs.shape[0]):
	sinr_applications [iter:iter + np.sum(user_AP_assoc[i,:]), :] = np.delete(np.outer(user_AP_assoc[i,:],sinr_APs[i,:]), np.where(user_AP_assoc[i,:] == 0), 0);# Application to Base Station SINR matrix 
 	iter = iter + np.sum(user_AP_assoc[i,:]); # Incrementing the iterator for the next user-application sets

log_pow = scn.sc_bw*np.log2(1 + 10**(sinr_applications/10)); # Log Power calculation part of the Shannon-Hartley theorem
var_row_num = sinr_applications.shape[0];
var_col_num = sinr_APs.shape[1];


# =========
# Optimizer
# =========

try:
	m = Model("mip1") # Creates the MIP model 
	X = m.addVars(var_row_num, var_col_num , vtype = GRB.BINARY, name = "X"); # We create the X matrix that has to be found 
	#X = m.addVars(3,5, vtype=GRB.BINARY, name = "X"); # We create the X matrix that has to be found 
	print X
	m.setObjective(X * log_pow, GRB.MAXIMIZE); # This is the objective function that we aim to maximize

	m.optimize()

except GurobiError:
	print('Error Reported')