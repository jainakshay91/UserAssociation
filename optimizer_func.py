#!/usr/bin/env python

# =============================
# Import the necessary binaries
# =============================

from gurobipy import *
import cPickle as pickle
import numpy as np 
from scenario_var import scenario_var 
# ==============================
# Create the Model and Variables
# ==============================
scn = scenario_var(); # Initializing the class variables

# Load the data to be used for the optimization

optim_data = np.load('/home/akshayjain/Desktop/Simulation/optim_var.npz')
#test = optim_data['arr_1'].item() # Selecting the eMBB users

#try:
m = Model("mip1") # Creates the MIP model 
X = m.addVars(np.sum(optim_data['usr_maps_assoc']),optim_data['sinr_sc'].shape[1], vtype = GRB.binary, name = "X"); # We create the X matrix that has to be found 
