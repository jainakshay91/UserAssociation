#!/usr/bin/env python

# =============================
# Import the necessary binaries
# =============================

import numpy as np
import scenario_gen
import dist_check as dsc 
import plotter
import os, sys
from scenario_var import scenario_var 
from argparse import ArgumentParser


# ==================================
# Capture the Command Line Arguments
# ==================================

parser = ArgumentParser(description = 'Scenario Generator Main Function'); # Initializing the class variable
parser.add_argument('-iter', type = int, help = 'Iteration Number of the Simulation');
args = parser.parse_args(); # Parse the Arguments

# =====================================
# Check Presence of Storage Directories
# =====================================

path = os.getcwd() + '/Data'; # This is the path we have to check for
subpath = os.getcwd() + '/Data/Temp'; # This is the subdirectory to store data  
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
# Initialize the class variables
# ==============================
scn = scenario_var(); # Getting the class object

# ====================
# Macro cell placement 
# ====================
try:
	macro_cell_locations = scenario_gen.macro_cell(scn.simulation_area, scn.MCBS_intersite, np, dsc); # Get the macro cell locations

	#print macro_cell_locations
	#print "===================

	# ===================================================================
	# Small cell placement and MCBS-SCBS Additional attributes generation
	# ===================================================================

	SCBS_per_MCBS = np.random.randint(3,10,size=macro_cell_locations.shape[0]); # Randomly choosing number of SCBS within an MCBS domain in the range 3 to 1
	SCBS_MCBS_assoc = np.zeros((sum(SCBS_per_MCBS),macro_cell_locations.shape[0]), dtype=int); # Create a MCBS and SCBS association matrix (distance based)
	#print sum(SCBS_per_MCBS)
	locs_SCBS = np.empty([sum(SCBS_per_MCBS),2], dtype = int); # We create an empty list of numpy arrays
	l_idx = 0; # lower index for the association matrix 
	u_idx = SCBS_per_MCBS[0]; # upper index for the association matrix
	for i in range(0,macro_cell_locations.shape[0]):
	    small_cell_locations = scenario_gen.small_cell(i, macro_cell_locations[i,:], scn.SCBS_intersite, SCBS_per_MCBS[i], scn.MCBS_intersite, np, dsc); #Get the small cell locations for each macro cell domain 
	    #print small_cell_locations
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
	BH_capacity_SC = scenario_gen.backhaul_tput(SCBS_MCBS_assoc, SCBS_per_MCBS, SC_wl_bh, np, scn, dsc); # Also Calculate the# BH capacity vector
	#print BH_capacity_SC 

	# ====================================
	# Dump the Users onto the scenario map 
	# ====================================

	AP_locs = np.vstack((macro_cell_locations, locs_SCBS)); # All AP locations
	usr_locs,usr_apps_assoc = scenario_gen.user_dump(scn, SCBS_per_MCBS, macro_cell_locations.shape[0], AP_locs, np, dsc); # We also retrieve the user and applications association matrix
	mMTC_locs = scenario_gen.mMTC_user_dump(scn,SCBS_per_MCBS,macro_cell_locations.shape[0],np); # Massive  Machine Type User locations
	print "User and AP Dump completed"
	# ======================================
	# Generate the SINR values for the users
	# ======================================

	#sinr_sc_embb,locs_sc_ret, usr_lcs = scenario_gen.pathloss_tester(scn, np, dsc); # Testing the Pathloss function implementation
	for i in range(0,len(usr_locs.keys())):
		print "Iteration #" + str(i)
		print "====================="
		sinr_sorted, locs_sc_ret, usr_lcs, idx_sc, idx_mc, sinr_pad, num_SCBS, num_MCBS, num_MCBS_tot, RX_eMBB = scenario_gen.sinr_gen (scn, sum(SCBS_per_MCBS), macro_cell_locations, np.asarray(locs_SCBS), usr_locs['user_locations'+str(i)], dsc, np)
		sinr = dsc.reorganizer(sinr_sorted, idx_sc, idx_mc, num_SCBS, num_MCBS_tot, sinr_pad, np, scn); # We reorganize the SINR matrix for the optimization framework
		RX = dsc.reorganizer(RX_eMBB, idx_sc, idx_mc, num_SCBS, num_MCBS_tot, float('nan'), np, scn); # We reorganize the RX Power matrix for the Baseline framework
		# ================================
		# Create Compressed Variable Files
		
		np.savez_compressed(os.getcwd()+'/Data/Temp/optim_var_'+ str(i) + str(vars(args)['iter']),sinr, usr_apps_assoc, usr_lcs, idx_sc, sinr_pad, num_SCBS, num_MCBS_tot, SC_wl_bh, SC_wrd_bh, MC_hops, SC_hops, BH_capacity_SC, RX, SCBS_per_MCBS, allow_pickle = True); # Save these variables to be utilized by the optimizer
		np.savez_compressed(os.getcwd()+'/Data/Temp/hmap_' + str(i) + str(vars(args)['iter']), usr_lcs, locs_SCBS, macro_cell_locations, SCBS_per_MCBS, SCBS_MCBS_assoc) # Data necessary for heatmap is saved here
		#np.savez_compressed('/home/akshayjain/Desktop/Simulation/optim_var_1',sinr_sorted, usr_apps_assoc, usr_lcs, idx_sc, sinr_pad, num_SCBS, num_MCBS, SC_wl_bh, SC_wrd_bh, MC_hops, SC_hops, BH_capacity_SC); # Save these variables to be utilized by the optimizer

		# ===========================
		# Plotting and Proof Checking

		#plotter.plotter('dashline',locs_sc_ret,sinr_sc_embb,5,10,1,45,0,0,1,'major','both', 'yes', 'SNR profile of Small Cell', np)
		#plotter.plotter('heatmap',sinr,locs_sc_ret,5,10,1,45,0,0,1,'major','both', 'yes', 'SNR profile of Small Cell', np)

	#sinr_sorted_mMTC, locs_sc_ret_mMTC, usr_lcs_mMTC, idx_sc_mMTC, idx_mc_mMTC, sinr_pad_mMTC, num_SCBS_mMTC, num_MCBS_mMTC, num_MCBS_tot_mMTC, RX_mMTC = scenario_gen.sinr_gen (scn, sum(SCBS_per_MCBS), macro_cell_locations, np.asarray(locs_SCBS), mMTC_locs['user_locations'], dsc, np)
	#sinr_mMTC = dsc.reorganizer(sinr_sorted_mMTC, idx_sc_mMTC, idx_mc_mMTC, num_SCBS_mMTC, num_MCBS_tot_mMTC, sinr_pad_mMTC, np, scn); # We reorganize the SINR matrix for the optimization framework
	#RX_mMTC_reorg = dsc.reorganizer(RX_mMTC, idx_sc_mMTC, idx_mc_mMTC, num_SCBS_mMTC, num_MCBS_tot_mMTC, float('nan'), np, scn); # We reorganize the RX Power matrix for the Baseline framework
	#np.savez_compressed(os.getcwd()+'/Data/Temp/optim_var_mMTC'+ str(vars(args)['iter']),sinr_mMTC, usr_lcs_mMTC, idx_sc_mMTC, sinr_pad_mMTC, num_SCBS_mMTC, num_MCBS_tot_mMTC, SC_wl_bh, SC_wrd_bh, MC_hops, SC_hops, BH_capacity_SC, RX_mMTC_reorg, allow_pickle = True); # Save these variables to be utilized by the optimizer
	
except KeyboardInterrupt:
	sys.exit("Exiting this process with Iteration Number" + str(vars(args)['iter']))
