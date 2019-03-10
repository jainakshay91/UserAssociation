# ==> This file enables the scenario generation process for the network to be analyzed.

# ==============================
# Import the necessary libraries
# ==============================

#import os.path
#import dist_check
from pathloss import *

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

    # We create the wired and wireless backhaul matrix (Restricting it to just one backhaul link currently)

    mat_wlbh_sc = (assoc_mat <= wl_bh_bp)*1; # Wireless backhaul enabled small cells
    mat_wrdbh_sc = (assoc_mat > wl_bh_bp)*1; # Wired backhaul enabled small cells
    MC_hops = np.random.randint(min_hops,max_hops,size = MCBS_locs.shape[0]); # The macro cells always have wired backhaul (Local breakouts can be added later)
    SC_hops = ((assoc_mat > 0)*1)*np.transpose(MC_hops) + 1; # The number of hops for each small cells to the IMS core

    return mat_wlbh_sc, mat_wrdbh_sc, MC_hops, SC_hops # Return the hops and wired/wireless backhaul configuration 

# ===============================
# SINR Calculator per Application
# ===============================

def sinr_gen (mc_locs, sc_locs, usr_locs_eMBB, usr_locs_URLLC, usr_locs_mMTC, dsc): # Generates the SINR per application      
    
    # First the distances to the serving and interfering base stations is calculated

    for i in range(0,mc_locs.shape[0]): # Distance to all MC cells
        dist_serv_cell_eMBB[:,i] = np.sort(dsc.dist_calc(usr_locs_eMBB, mc_locs[i], np),type = 'mergesort'); # Calculate the distance of each eMBB application location with each MC and sort them
        dist_serv_cell_URLLC[:,i] = np.sort(dsc.dist_calc(usr_locs_URLLC, mc_locs[i], np),type = 'mergesort'); # Calculate the distance of each URLLC application location with each MC and sort them
        dist_serv_cell_mMTC[:,i] = np.sort(dsc.dist_calc(usr_locs_mMTC, mc_locs[i], np), type = 'mergesort'); # Calculate the distance of each mMTC application location with each MC and sort them
    
    # For small cells we consider only the 4 closest Macro Cell domains
    sc_domain = 