#!/usr/bin/env python

# =============================
# Import the necessary binaries
# =============================

import numpy as np
import matplotlib.pyplot as plt
import scenario_gen
import dist_check as dsc 
# from gurobipy import *
from scenario_var import *

# ====================
# Macro cell placement 
# ====================

macro_cell_locations = scenario_gen.macro_cell(simulation_area, MCBS_intersite, np,dsc); # Get the macro cell locations

#print macro_cell_locations
#print "===================

# ===================================================================
# Small cell placement and MCBS-SCBS Additional attributes generation
# ===================================================================

SCBS_per_MCBS = np.random.randint(3,10,size=macro_cell_locations.shape[0]); # Randomly choosing number of SCBS within an MCBS domain in the range 3 to 1
SCBS_MCBS_assoc = np.zeros((sum(SCBS_per_MCBS),macro_cell_locations.shape[0]), dtype=int); # Create a MCBS and SCBS association matrix (distance based)
#print sum(SCBS_per_MCBS)
locs_SCBS = []; # We create an empty list of numpy arrays
l_idx = 0; # lower index for the association matrix 
u_idx = SCBS_per_MCBS[0]; # upper index for the association matrix
for i in range(0,macro_cell_locations.shape[0]):
    small_cell_locations = scenario_gen.small_cell(i, macro_cell_locations, SCBS_intersite, SCBS_per_MCBS[i], MCBS_intersite, np, dsc); #Get the small cell locations for each macro cell domain 
    locs_SCBS.append(small_cell_locations); # Store the small cell locations in the list of numpy arrays
    SCBS_MCBS_assoc[l_idx:u_idx,i] = dsc.dist_calc(small_cell_locations,macro_cell_locations[i],np); # Insert ones in these indexes for the association matrix
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

SC_wl_bh, SC_wrd_bh, MC_hops, SC_hops = scenario_gen.backhaul_dump(min_num_hops, max_num_hops, SCBS_per_MCBS, macro_cell_locations, SCBS_MCBS_assoc, np, wl_bh_bp); # We drop the backhaul into the scenario

# ====================================
# Dump the Users onto the scenario map 
# ====================================

usr_loc_eMBB, usr_loc_URLLC, usr_loc_mMTC = scenario_gen.user_dump(UE_density_eMBB, UE_density_URLLC, UE_density_mMTC, MCBS_intersite, SCBS_per_MCBS, macro_cell_locations.shape[0], simulation_area,np); 



# Plots

#plt.plot(macro_cell_locations[:,0], macro_cell_locations[:,1],'rs'); # Plot the macro cells
#for j in range(0,macro_cell_locations.shape[0]):
#    print_element = locs_SCBS[j]; #Accessing the numpy array of SC locations corresponding to the Macro Cell    
#   plt.plot(print_element[:,0], print_element[:,1], 'b*'); # Plot the small cells
# plt.plot(usr_loc_eMBB[:,0],usr_loc_eMBB[:,1],'k+')
# plt.plot(usr_loc_URLLC[:,0],usr_loc_URLLC[:,1],'cs')
# #plt.plot(usr_loc_mMTC[:,0],usr_loc_mMTC[:,1],'go')
#plt.show() # Show the small cells
