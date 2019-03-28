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

optim_data = np.load('/home/akshayjain/Desktop/Simulations/optim_var.npz'); # Load the data to be used for the optimization  

try:
	m = Model("mip1") # Creates the MIP model 
	x = model.addVars(optim_data['sinr_sc_embb'].shape[0],optim_data)

