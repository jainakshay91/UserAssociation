#!/usr/bin/env python

# =============================
# Import the necessary binaries
# =============================

import numpy as np
import scenario_gen
import dist_check as dsc 
import plotter
# from gurobipy import *
from scenario_var import scenario_var 


# ==============================
# Initialize the class variables
# ==============================
scn = scenario_var(); # Getting the class object

# ====================
# Macro cell placement 
# ====================

macro_cell_locations = scenario_gen.macro_cell(scn.simulation_area, scn.MCBS_intersite, np, dsc); # Get the macro cell locations

#print macro_cell_locations
#print "===================

# ===================================================================
# Small cell placement and MCBS-SCBS Additional attributes generation
# ===================================================================

SCBS_per_MCBS = np.random.randint(3,10,size=macro_cell_locations.shape[0]); # Randomly choosing number of SCBS within an MCBS domain in the range 3 to 1
SCBS_MCBS_assoc = np.zeros((sum(SCBS_per_MCBS),macro_cell_locations.shape[0]), dtype=int); # Create a MCBS and SCBS association matrix (distance based)
#print sum(SCBS_per_MCBS)
locs_SCBS = np.empty([sum(SCBS_per_MCBS),2]); # We create an empty list of numpy arrays
l_idx = 0; # lower index for the association matrix 
u_idx = SCBS_per_MCBS[0]; # upper index for the association matrix
for i in range(0,macro_cell_locations.shape[0]):
    small_cell_locations = scenario_gen.small_cell(i, macro_cell_locations, scn.SCBS_intersite, SCBS_per_MCBS[i], scn.MCBS_intersite, np, dsc); #Get the small cell locations for each macro cell domain 
    locs_SCBS[l_idx:u_idx,:] = small_cell_locations; # Store the small cell locations in the list of numpy arrays
    SCBS_MCBS_assoc[l_idx:u_idx,i] = dsc.dist_calc(small_cell_locations,macro_cell_locations[i], 0, 0, '2d', np); # Insert ones in these indexes for the association matrix
    #print SCBS_MCBS_assoc[l_idx:u_idx,i]
    l_idx = l_idx + SCBS_per_MCBS[i]; # Update the lower index 
    if i < (macro_cell_locations.shape[0]-1):
        u_idx = u_idx + SCBS_per_MCBS[i+1]; # Update the upper index
    #print locs_SCBS[:,:,i]
    #print "==================="
#print SCBS_MCBS_assoc

# ========================================================
# Create the AP-Backhaul association for the scenario dump
# ========================================================

SC_wl_bh, SC_wrd_bh, MC_hops, SC_hops = scenario_gen.backhaul_dump(scn, SCBS_per_MCBS, macro_cell_locations, SCBS_MCBS_assoc, np); # We drop the backhaul into the scenario

# ====================================
# Dump the Users onto the scenario map 
# ====================================

usr_locs,usr_apps_assoc = scenario_gen.user_dump(scn, SCBS_per_MCBS, macro_cell_locations.shape[0], np); # We also retrieve the user and applications association matrix
print "User and AP Dump completed"
# ======================================
# Generate the SINR values for the users
# ======================================

#sinr_sc_embb,locs_sc_ret, usr_lcs = scenario_gen.pathloss_tester(scn, np, dsc); # Testing the Pathloss function implementation

sinr_sc, locs_sc_ret, usr_lcs, idx_sc = scenario_gen.sinr_gen (scn, sum(SCBS_per_MCBS), macro_cell_locations, np.asarray(locs_SCBS), usr_locs['user_locations0'], dsc, np)

# ================================
# Create Compressed Variable Files
# ================================

np.savez_compressed('/home/akshayjain/Desktop/Simulation/optim_var',sinr_sc, usr_apps_assoc, usr_lcs, idx_sc); # Save these variables to be utilized by the optimizer

# ===========================
# Plotting and Proof Checking

#plotter.plotter('dashline',locs_sc_ret,sinr_sc_embb,5,10,1,45,0,0,1,'major','both', 'yes', 'SNR profile of Small Cell', np)
#plotter.plotter('heatmap',sinr_sc_embb,locs_sc_ret,5,10,1,45,0,0,1,'major','both', 'yes', 'SNR profile of Small Cell', np)