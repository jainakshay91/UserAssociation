# ==> This file enables the scenario generation process for the network to be analyzed.

# ==============================
# Import the necessary libraries
# ==============================

import os.path
import dist_check

# ===================================================
# Load/Generate the Macro Cell base station locations
# ===================================================

def macro_cell(simulation_area,MCBS_intersite,np,dsc):
    # We distribute the Macro BSs as a grid
#    num_macro_BS = simulation_area/MCBS_intersite; # Number of BSs on the grid
    offset = MCBS_intersite/2; # Offset param
    locs_interim = np.arange(offset, np.sqrt(simulation_area).astype(int), MCBS_intersite); # Range of numbers from 0 to the end of the grid area, with intersite distance spacing 
    print locs_interim
    locs_MCBS = dsc.gridder(locs_interim,MCBS_intersite,np); # Calling a permutation function that generates the grid
    return locs_MCBS

# ===================================================
# Load/Generate the Small Cell base station locations
# ===================================================

def small_cell(num, MCBS_locs, SCBS_intersite,SCBS_per_MCBS,MCBS_intersite,np,dsc):
    offset = MCBS_intersite/2; # Offset param
    while True:	
	   locs_SCBS_x = np.random.uniform(MCBS_locs[num,0] - offset,MCBS_locs[num,0] + offset,(SCBS_per_MCBS,1)); # Generating the X coordinate of the small cells for a given macro cell
	   locs_SCBS_y = np.random.uniform(MCBS_locs[num,1] - offset,MCBS_locs[num,1] + offset,(SCBS_per_MCBS,1)); # Generating the Y coordinate of the small cells for a given macro cell
	   locs_SCBS = np.concatenate((locs_SCBS_x, locs_SCBS_y), axis=1); 
	   if dsc.checker(locs_SCBS,SCBS_intersite,np)==1:
		  break
    return locs_SCBS

# ================================
# Load/Generate the User locations
# ================================

def user_dump(UE_density_eMBB, UE_density_URLLC, UE_density_mMTC, MCBS_intersite, SCBS_per_MCBS, num_MCBS, simulation_area, np):

    tot_dev_eMBB = (sum(SCBS_per_MCBS)+num_MCBS)*UE_density_eMBB; # Total eMBB devices in the scenario
    tot_dev_URLLC = UE_density_URLLC*simulation_area; # Total URLLC devices in the scenario
    tot_dev_mMTC = UE_density_mMTC*num_MCBS; # Total mMTC devices in the scenario
    usr_locs_eMBB = np.random.uniform(0,np.sqrt(simulation_area),(tot_dev_eMBB,2)); # We obtain a set of eMBB locations
    usr_locs_URLLC = np.random.uniform(0,np.sqrt(simulation_area),(int(tot_dev_URLLC),2)); # We obtain a set of URLLC locations
    usr_locs_mMTC = np.random.uniform(0,np.sqrt(simulation_area),(tot_dev_mMTC,2)); # We obtain a set of mMTC locations
    return usr_locs_eMBB, usr_locs_URLLC, usr_locs_mMTC; # Return the locations of these applications/users with these applications

# =============================
# Generate the backhaul network
# =============================

def backhaul_dump(min_hops, max_hops, SCBS_per_MCBS, MCBS_locs, assoc_mat, np, wl_bh_bp):

    # We create the backhaul matrix specifying the number of wired/wireless links and the total number of hops

    mat_backhaul_sc = np.zeros((sum(SCBS_per_MCBS),3)); # For the Small Cells 

    for 
    mat_backhaul_sc[:,0] = np.random.randint(1,2,size = mat_backhaul_sc.shape[0]); # The number of backhaul links for the small cells
    mat_backhaul_sc[:,1] = np.random.randint(0,2,size = mat_backhaul_sc.shape[0]); # The number of wireless backhaul links for the small cells
    mat_backhaul_sc[:,2] = np.random.randint(min_hops,max_hops,size = mat_backhaul_sc.shape[0]) + np.ones(mat_backhaul_sc.shape[0],dtype = int); # The total backhaul hops for small cells
    #print mat_backhaul_sc    
    mat_backhaul_mc = np.random.randint(min_hops,max_hops,size = MCBS_locs.shape[0]) - np.ones(MCBS_locs.shape[0],dtype = int); # The macro cells always have wired backhaul and they may have a local breakout
    
    return 0, 0, 0; # Bogus returs so far

        

