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
    
    # =====================================
    # We distribute the Macro BSs as a grid
    
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

def user_dump(scn, SCBS_per_MCBS, num_MCBS, np):

    tot_dev_eMBB = (sum(SCBS_per_MCBS)+num_MCBS)*scn.UE_density_eMBB; # Total eMBB devices in the scenario
    tot_dev_URLLC = scn.UE_density_URLLC*scn.simulation_area; # Total URLLC devices in the scenario
    tot_dev_mMTC = scn.UE_density_mMTC*num_MCBS; # Total mMTC devices in the scenario
    usr_locs_eMBB = np.random.uniform(0,np.sqrt(scn.simulation_area),(tot_dev_eMBB,2)); # We obtain a set of eMBB locations
    usr_locs_URLLC = np.random.uniform(0,np.sqrt(scn.simulation_area),(int(tot_dev_URLLC),2)); # We obtain a set of URLLC locations
    usr_locs_mMTC = np.random.uniform(0,np.sqrt(scn.simulation_area),(tot_dev_mMTC,2)); # We obtain a set of mMTC locations
    return usr_locs_eMBB, usr_locs_URLLC, usr_locs_mMTC; # Return the locations of these applications/users with these applications

# =============================
# Generate the backhaul network
# =============================

def backhaul_dump(scn, SCBS_per_MCBS, MCBS_locs, assoc_mat, np):

    # =====================================================================================================
    # We create the wired and wireless backhaul matrix (Restricting it to just one backhaul link currently)

    mat_wlbh_sc = (assoc_mat <= scn.wl_bh_bp)*1; # Wireless backhaul enabled small cells
    mat_wrdbh_sc = (assoc_mat > scn.wl_bh_bp)*1; # Wired backhaul enabled small cells
    MC_hops = np.random.randint(scn.min_hops, scn.max_hops,size = MCBS_locs.shape[0]); # The macro cells always have wired backhaul (Local breakouts can be added later)
    SC_hops = ((assoc_mat > 0)*1)*np.transpose(MC_hops) + 1; # The number of hops for each small cells to the IMS core

    return mat_wlbh_sc, mat_wrdbh_sc, MC_hops, SC_hops # Return the hops and wired/wireless backhaul configuration 

# ===============================
# SINR Calculator per Application
# ===============================

def sinr_gen (scn, mc_locs, sc_locs, usr_locs_eMBB, usr_locs_URLLC, usr_locs_mMTC, dsc, np): # Generates the SINR per application      
    
    # ======================================================
    # First the distances
     to the base stations is calculated

    for i in range(0,mc_locs.shape[0]): # Distance to all MC cells

        # ==> 2D distance calculation 
        dist_serv_cell_eMBB[:,i] = dsc.dist_calc(usr_locs_eMBB, mc_locs[i], 0, 0, '2d', np); # Calculate the distance of each eMBB application location with each MC and sort them
        dist_serv_cell_URLLC[:,i] = dsc.dist_calc(usr_locs_URLLC, mc_locs[i], 0, 0, '2d', np); # Calculate the distance of each URLLC application location with each MC and sort them
        dist_serv_cell_mMTC[:,i] = dsc.dist_calc(usr_locs_mMTC, mc_locs[i], 0, 0,'2d', np); # Calculate the distance of each mMTC application location with each MC and sort them
        
        # ==> 3D distance calculation
        dist_serv_cell_eMBB_3d[:,i] = dsc.dist_calc(usr_locs_eMBB, mc_locs[i], scn.bs_ht_mc, scn.usr_ht, '3d', np); # Calculate the distance of each eMBB application location with each MC and sort them
        dist_serv_cell_URLLC_3d[:,i] = dsc.dist_calc(usr_locs_URLLC, mc_locs[i], scn.bs_ht_mc, scn.usr_ht, '3d', np); # Calculate the distance of each URLLC application location with each MC and sort them
        dist_serv_cell_mMTC_3d[:,i] = dsc.dist_calc(usr_locs_mMTC, mc_locs[i], scn.bs_ht_mc, scn.usr_ht,'3d', np); # Calculate the distance of each mMTC application location with each MC and sort them


    for i in range(0,sc_locs.shape[0]): # Distance to all small cells
       
        # ==> 2D distance calulation
        dist_serv_sc_eMBB[:,i] = dsc.dist_calc(usr_locs_eMBB, sc_locs[i], 0, 0,'2d', np); # Distance of each eMBB application location with each SC
        dist_serv_sc_URLLC[:,i] = dsc.dist_calc(usr_locs_URLLC, sc_locs[i], 0, 0,'2d', np); # Distance of each URLLC application location with each SC
        dist_serv_sc_mMTC[:,i] = dsc.dist_calc(usr_locs_mMTC, sc_locs[i], 0, 0,'2d', np); # Distance of each mMTC application location with each SC

        # ==> 3D distance calculation
        dist_serv_sc_eMBB_3d[:,i] = dsc.dist_calc(usr_locs_eMBB, mc_locs[i], scn.bs_ht_sc, scn.usr_ht, '3d', np); # Calculate the distance of each eMBB application location with each MC and sort them
        dist_serv_sc_URLLC_3d[:,i] = dsc.dist_calc(usr_locs_URLLC, mc_locs[i], scn.bs_ht_sc, scn.usr_ht, '3d', np); # Calculate the distance of each URLLC application location with each MC and sort them
        dist_serv_sc_mMTC_3d[:,i] = dsc.dist_calc(usr_locs_mMTC, mc_locs[i], scn.bs_ht_sc, scn.usr_ht,'3d', np); # Calculate the distance of each mMTC application location with each MC and sort them


    # ======================================================
    # Limit the number of MC and SC for the SINR calculation

    
    # ==> eMBB users

    num_MCBS_SINR_eMBB = 4; # We choose the 4 closest MCs for the SINR calculation 
    dist_SCBS_SINR = 200; # We choose the range of the farthest SC that will impact SINR calculation for a user to be 200 meters
    sorted_MCBS_eMBB_mat, idx_MCBS_SINR_eMBB = dsc.idx_mat(dist_serv_cell_eMBB, num_MCBS_SINR,'minimum',np); # Distance based sorted matrix and index of the MCBS under consideration for the PL calculation
    sorted_SCBS_eMBB_mat, idx_SCBS_SINR_eMBB = dsc.idx_mat(dist_serv_sc_eMBB, dist_SCBS_SINR, 'distance', np); # Distance based sorted matrix and index of the SCBS under consideration for the PL calculation

    # ====================
    # Pathloss Calculation

    PL_sc = pathloss.pathloss_SC(scn, sorted_SCBS_eMBB_mat, np, dist_serv_sc_eMBB_3d, dsc); # Calculating the pathloss for Small cells

    # ================
    # SINR Calculation

    snr_sc = scn.transmit_power + scn.transmit_gain_sc + scn.receiver_gain - PL_sc + scn.N*scn.sc_bw; # This is the SNR from one Small cell 
    prx_sc_others = scn.transmit_power + scn.transmit_gain_sc + scn.receiver_gain - PL_sc; # This is the received power from other Small cells
    sinr_sc = snr_sc - prx_sc_others; # We subtract the received power from other small cells to obtain the sinr 

    # The above calculation has to be optimally calculated for N users and M small cells. 


