# ==> This file enables the scenario generation process for the network to be analyzed.

# ==============================
# Import the necessary libraries
# ==============================

#import dist_check
import pathloss
import copy
import csv
import csvsaver
import sectbeam
import pdb
#from multiprocessing import Process
# ===================================================
# Load/Generate the Macro Cell base station locations
# ===================================================

def macro_cell(simulation_area,MCBS_intersite,np,dsc):
    
    # =====================================
    # We distribute the Macro BSs as a grid
    
    offset = MCBS_intersite/2; # Offset param
    locs_interim = np.arange(offset, np.sqrt(simulation_area).astype(int), MCBS_intersite); # Range of numbers from 0 to the end of the grid area, with intersite distance spacing 
    #print locs_interim
    locs_MCBS = dsc.gridder(locs_interim,MCBS_intersite,np); # Calling a permutation function that generates the grid
    return locs_MCBS

# ===================================================
# Load/Generate the Small Cell base station locations
# ===================================================

def small_cell(num, MCBS_locs, SCBS_intersite,SCBS_per_MCBS,MCBS_intersite,np,dsc):
    offset = MCBS_intersite/2; # Offset param
    while True:	
        dist_from_MCBS = np.random.uniform(0,offset,(SCBS_per_MCBS,1))
        angular_disp = np.random.uniform(0,2*np.pi,(SCBS_per_MCBS,1))
        locs_SCBS_x = np.multiply(dist_from_MCBS,np.cos(angular_disp)) + MCBS_locs[0]
        locs_SCBS_y = np.multiply(dist_from_MCBS,np.sin(angular_disp)) + MCBS_locs[1]
        #print MCBS_locs
        #locs_SCBS_x = np.random.uniform(MCBS_locs[0] - offset,MCBS_locs[0] + offset,(SCBS_per_MCBS,1)); # Generating the X coordinate of the small cells for a given macro cell
        #locs_SCBS_y = np.random.uniform(MCBS_locs[1] - offset,MCBS_locs[1] + offset,(SCBS_per_MCBS,1)); # Generating the Y coordinate of the small cells for a given macro cell
        locs_SCBS = np.concatenate((locs_SCBS_x, locs_SCBS_y), axis=1); 
        #print locs_SCBS
        if dsc.checker(locs_SCBS,SCBS_intersite,np)==1 and dsc.locs_checker(locs_SCBS, MCBS_locs,np, 'sc')==1:
            break
    return locs_SCBS

# ================================
# Load/Generate the User locations
# ================================

def user_dump(scn, SCBS_per_MCBS, num_MCBS, AP_locs, np, dsc):

    # =============================================================
    # Compute total users and total applications in Simulation area
    while True:
        tot_users_scenario = np.arange(scn.num_users_min, scn.num_users_max, scn.user_steps_siml, dtype='int'); # Adding the users list for simulation 
        #print tot_users_scenario
        #tot_dev_eMBB = (sum(SCBS_per_MCBS)+num_MCBS)*scn.UE_density_eMBB; # Total eMBB devices in the scenario
        #tot_dev_URLLC = scn.UE_density_URLLC*scn.simulation_area; # Total URLLC devices in the scenario
        #tot_dev_mMTC = scn.UE_density_mMTC*num_MCBS; # Total mMTC devices in the scenario
        
        # =======================================================
        # Generate User locations and User-App Association Matrix

        usr_locs = {}; # We establish an empty dictionary
        assoc_usapp = {}; # We establish an empty dictionary for USER and APPs association
        attr_name_usr = 'user_locations'; # Attribute name
        attr_name_assoc = 'user_app'; # Attribute name for the USER-APPs association matrix (eMBB)
        for i in range(0,tot_users_scenario.shape[0]):
            usr_locs[attr_name_usr + str(i)] = np.random.uniform(0,np.sqrt(scn.simulation_area),(tot_users_scenario[i],2)); # Generate User locations
            if dsc.locs_checker(usr_locs[attr_name_usr + str(i)], AP_locs,np,'user')==0:
               i = i - 1; # We go back and start the for loop from the current instance
               continue
            #assoc_usapp[attr_name_assoc + str(i)] = np.random.randint(2, size = (tot_users_scenario[i], scn.max_num_appl_UE)); # Generate User-App Association 
            assoc_usapp[attr_name_assoc + str(i)] = np.ones((tot_users_scenario[i], scn.max_num_appl_UE), dtype = int); # Generate User-App Association 
        with open("ActualUsers.csv",'wb') as f:
    		w = csv.DictWriter(f,assoc_usapp.keys())
    		w.writeheader()
    		w.writerow(assoc_usapp)
        return usr_locs, assoc_usapp
        
	#usr_locs_eMBB = np.random.uniform(0,np.sqrt(scn.simulation_area),(tot_dev_eMBB,2)); # We obtain a set of eMBB locations
    #usr_locs_URLLC = np.random.uniform(0,np.sqrt(scn.simulation_area),(int(tot_dev_URLLC),2)); # We obtain a set of URLLC locations
    #usr_locs_mMTC = np.random.uniform(0,np.sqrt(scn.simulation_area),(tot_dev_mMTC,2)); # We obtain a set of mMTC locations
    #return usr_locs_eMBB, usr_locs_URLLC, usr_locs_mMTC; # Return the locations of these applications/users with these applications

def mMTC_user_dump(scn, SCBS_per_MCBS, num_MCBS, np):

    # =============================================================
    # Compute total users and total applications in Simulation area
    #print tot_users_scenario
    #tot_dev_eMBB = (sum(SCBS_per_MCBS)+num_MCBS)*scn.UE_density_eMBB; # Total eMBB devices in the scenario
    #tot_dev_URLLC = scn.UE_density_URLLC*scn.simulation_area; # Total URLLC devices in the scenario
    tot_dev_mMTC = scn.UE_density_mMTC*num_MCBS; # Total mMTC devices in the scenario
    
    # =======================================================
    # Generate User locations and User-App Association Matrix

    usr_locs = {}; # We establish an empty dictionary
    #assoc_usapp = {}; # We establish an empty dictionary for USER and APPs association
    attr_name_usr = 'user_locations'; # Attribute name
    #attr_name_assoc = 'user_app'; # Attribute name for the USER-APPs association matrix (eMBB)
    usr_locs[attr_name_usr] = np.random.uniform(0,np.sqrt(scn.simulation_area),(tot_dev_mMTC,2)); # Generate User locations
    #assoc_usapp[attr_name_assoc + str(i)] = np.random.randint(2, size = (tot_dev_mMTC, scn.max_num_appl_UE)); # Generate User-App Association 
    return usr_locs

# ===========================
# mMTC User Location Selector
# ===========================

def mMTC_user_selector(scn, np, mmtc_usr_lcs, AP_locs, gencase, dsc, percentage):

    if gencase == 0: # Gencase = 0 indicates that we use an uniform distribution to select the user locations

        #print ("Total mMTC devices:", mmtc_usr_lcs['user_locations'].shape)
        num_mMTC = np.random.randint(0,mmtc_usr_lcs['user_locations'].shape[0]); # First we select the number of users from the total possible users using the Uniform distribution
        #num_mMTC = 20000
        #print ("Number of mMTC:", num_mMTC)
        mMTC_selected_idx = np.random.randint(0,mmtc_usr_lcs['user_locations'].shape[0],(num_mMTC,1)); # We select the indices of the mMTC devices
        #print ("Selected Indices:", mMTC_selected_idx)
        #selected_mmtc = np.take(mmtc_usr_lcs['user_locations'] ,mMTC_selected_idx, axis = 1); # THis is the variable that will store the chosen mMTC device locations
        selected_mmtc = np.empty((num_mMTC,2)); # THis is the variable that will store the chosen mMTC device locations
        for i in range(num_mMTC):
            selected_mmtc[i,:] = mmtc_usr_lcs['user_locations'][mMTC_selected_idx[i],:]; # Select Locations -- this is the inefficient way. Study np.take to understand the efficient way of getting user locations

        mMTC_AP_dist_mat = np.empty((num_mMTC, AP_locs.shape[0])) # Matrix that holds the distance between mMTC devices and APs
        for i in range(AP_locs.shape[0]):
            mMTC_AP_dist_mat[:,i] = dsc.dist_calc(selected_mmtc, AP_locs[i,:], 0, 0, '2d', np) # Calculate the distance of each mMTC device to each of the APs in the scenario

        mmtc_AP_asso = np.argmin(mMTC_AP_dist_mat, axis = 1); # We get the sorted matrix
        num_mMTC_AP = np.bincount(mmtc_AP_asso); # We get the number of users for a given AP  
        print "mMTC dump done"

    elif gencase == 1: # Gencase = 1 indicates that we provision 5%-100% load per cell for mMTC devices

        mMTC_AP_dist_mat = np.empty((mmtc_usr_lcs.shape[0], AP_locs.shape[0])) # Matrix that holds the distance between mMTC devices and APs
        
        for i in range(AP_locs.shape[0]):
            mMTC_AP_dist_mat[:,i] = dsc.dist_calc(mmtc_usr_lcs, AP_locs[i,:], 0, 0, '2d', np) # Calculate the distance of each mMTC device to each of the APs in the scenario

        sorted_mMTC_arr = np.argmin(mMTC_AP_dist_mat, axis = 1)
        mMTC_assoc_dict = {}; # We have an empty dictionary to hold the AP and mMTC user association
        mMTC_assoc_dict['APasso'] = np.sort(sorted_mMTC_arr) # Store the sorted AP association array
        mMTC_assoc_dict['SortAsso'] = np.argsort(sorted_mMTC_arr) # Sort the association vector based on AP ID and store the indices of sorting  
        mMTC_assoc_dict['Count'] = np.bincount(sorted_mMTC_arr) # Store the bincount for each AP

        num_users_perbin = np.floor(mMTC_assoc_dict['Count']*percentage/100) # We get the vector for the number of mMTC devices active per AP
        #for i in range() ==> Continue this when the basic case (gencase 0) is done
    
    #return selected_mmtc, mmtc_AP_asso, num_mMTC_AP
    return num_mMTC_AP    

# =============================
# Generate the backhaul network
# =============================

def backhaul_dump(scn, SCBS_per_MCBS, MCBS_locs, assoc_mat, np):

    # =====================================================================================================
    # We create the wired and wireless backhaul matrix (Restricting it to just one backhaul link currently)

    mat_wlbh_sc = np.where(assoc_mat != 0, (assoc_mat <= scn.wl_bh_bp)*1, 0); # Wireless backhaul enabled small cells
    #print mat_wlbh_sc
    mat_wrdbh_sc = (assoc_mat > scn.wl_bh_bp)*1; # Wired backhaul enabled small cells
    MC_hops = np.random.randint(scn.min_num_hops, scn.max_num_hops,size = MCBS_locs.shape[0]); # The macro cells always have wired backhaul (Local breakouts can be added later)
    SC_hops = ((assoc_mat > 0)*1)*np.transpose(MC_hops) + 1; # The number of hops for each small cells to the IMS core
    return mat_wlbh_sc, mat_wrdbh_sc, MC_hops, SC_hops # Return the hops and wired/wireless backhaul configuration 

# ===============================
# SINR Calculator per Application
# ===============================

def sinr_gen (scn, num_SCBS, mc_locs, sc_locs, usr_lcs, dsc, np, inter_limit_flag): # Generates the SINR per application      
    #print tau_flag
    # ======================================================
    # First the distances to the base stations is calculated

    # ==> We declare a set of empty arrays
    dist_serv_cell = np.empty([usr_lcs.shape[0],mc_locs.shape[0]]);

    #dist_serv_cell_eMBB = np.empty([usr_locs_eMBB.shape[0],mc_locs.shape[0]]);
    #dist_serv_cell_URLLC = np.empty([usr_locs_URLLC.shape[0],mc_locs.shape[0]]);
    #dist_serv_cell_mMTC = np.empty([usr_locs_mMTC.shape[0],mc_locs.shape[0]]); 

    dist_serv_cell_3d = np.empty([usr_lcs.shape[0],mc_locs.shape[0]]);
   
    #dist_serv_cell_eMBB_3d = np.empty([usr_locs_eMBB.shape[0],mc_locs.shape[0]]);
    #dist_serv_cell_URLLC_3d = np.empty([usr_locs_URLLC.shape[0],mc_locs.shape[0]]);
    #dist_serv_cell_mMTC_3d = np.empty([usr_locs_mMTC.shape[0],mc_locs.shape[0]]); 

    for i in range(0,mc_locs.shape[0]): # Distance to all MC cells

        # ==> 2D distance calculation 
        
        dist_serv_cell[:,i] = dsc.dist_calc(usr_lcs, mc_locs[i,:], 0, 0, '2d', np); # Calculate the distance of each eMBB application location with each MC and sort them
        
        #dist_serv_cell_eMBB[:,i] = dsc.dist_calc(usr_locs_eMBB, mc_locs[i,:], 0, 0, '2d', np); # Calculate the distance of each eMBB application location with each MC and sort them
        #dist_serv_cell_URLLC[:,i] = dsc.dist_calc(usr_locs_URLLC, mc_locs[i,:], 0, 0, '2d', np); # Calculate the distance of each URLLC application location with each MC and sort them
        #dist_serv_cell_mMTC[:,i] = dsc.dist_calc(usr_locs_mMTC, mc_locs[i,:], 0, 0,'2d', np); # Calculate the distance of each mMTC application location with each MC and sort them
        
        # ==> 3D distance calculation
        
        dist_serv_cell_3d[:,i] = dsc.dist_calc(usr_lcs, mc_locs[i,:], scn.bs_ht_mc, scn.usr_ht, '3d', np); # Calculate the distance of each eMBB application location with each MC and sort them
       

        #dist_serv_cell_eMBB_3d[:,i] = dsc.dist_calc(usr_locs_eMBB, mc_locs[i,:], scn.bs_ht_mc, scn.usr_ht, '3d', np); # Calculate the distance of each eMBB application location with each MC and sort them
        #dist_serv_cell_URLLC_3d[:,i] = dsc.dist_calc(usr_locs_URLLC, mc_locs[i,:], scn.bs_ht_mc, scn.usr_ht, '3d', np); # Calculate the distance of each URLLC application location with each MC and sort them
        #dist_serv_cell_mMTC_3d[:,i] = dsc.dist_calc(usr_locs_mMTC, mc_locs[i,:], scn.bs_ht_mc, scn.usr_ht,'3d', np); # Calculate the distance of each mMTC application location with each MC and sort them


    # ==> We declare empty arrays first
    #print sc_locs.shape
    
    dist_serv_sc = np.empty([usr_lcs.shape[0],num_SCBS]);
    
    #dist_serv_sc_eMBB = np.empty([usr_locs_eMBB.shape[0],num_SCBS]);
    #dist_serv_sc_URLLC = np.empty([usr_locs_URLLC.shape[0],num_SCBS]);
    #dist_serv_sc_mMTC = np.empty([usr_locs_mMTC.shape[0],num_SCBS]); 

    dist_serv_sc_3d = np.empty([usr_lcs.shape[0],num_SCBS]);
    
    #dist_serv_sc_eMBB_3d = np.empty([usr_locs_eMBB.shape[0],num_SCBS]);
    #dist_serv_sc_URLLC_3d = np.empty([usr_locs_URLLC.shape[0],num_SCBS]);
    #dist_serv_sc_mMTC_3d = np.empty([usr_locs_mMTC.shape[0],num_SCBS]); 

    for i in range(0,num_SCBS): # Distance to all small cells
       
        # ==> 2D distance calulation
        dist_serv_sc[:,i] = dsc.dist_calc(usr_lcs, sc_locs[i,:], 0, 0,'2d', np); # Distance of each eMBB application location with each SC
        #print dist_serv_sc[:,i]
        #dist_serv_sc_eMBB[:,i] = dsc.dist_calc(usr_locs_eMBB, sc_locs[i,:], 0, 0,'2d', np); # Distance of each eMBB application location with each SC
        #dist_serv_sc_URLLC[:,i] = dsc.dist_calc(usr_locs_URLLC, sc_locs[i,:], 0, 0,'2d', np); # Distance of each URLLC application location with each SC
        #dist_serv_sc_mMTC[:,i] = dsc.dist_calc(usr_locs_mMTC, sc_locs[i,:], 0, 0,'2d', np); # Distance of each mMTC application location with each SC

        # ==> 3D distance calculation
        dist_serv_sc_3d[:,i] = dsc.dist_calc(usr_lcs, sc_locs[i,:], scn.bs_ht_sc, scn.usr_ht, '3d', np); # Calculate the distance of each eMBB application location with each SC and sort them
        
        #dist_serv_sc_eMBB_3d[:,i] = dsc.dist_calc(usr_locs_eMBB, sc_locs[i,:], scn.bs_ht_sc, scn.usr_ht, '3d', np); # Calculate the distance of each eMBB application location with each MC and sort them
        #dist_serv_sc_URLLC_3d[:,i] = dsc.dist_calc(usr_locs_URLLC, sc_locs[i,:], scn.bs_ht_sc, scn.usr_ht, '3d', np); # Calculate the distance of each URLLC application location with each MC and sort them
        #dist_serv_sc_mMTC_3d[:,i] = dsc.dist_calc(usr_locs_mMTC, sc_locs[i,:], scn.bs_ht_sc, scn.usr_ht,'3d', np); # Calculate the distance of each mMTC application location with each MC and sort them

    print "Finished Distance calculation"
    # ======================================================
    # Limit the number of MC and SC for the SINR calculation

    #pdb.set_trace()
    
    # ==> eMBB users
    if inter_limit_flag == 1: # Interference limited scenario with no sectoring employed
        print "================================================="
        print "==========Interference Limited Regime============"
        print "================================================="

        num_MCBS_SINR = 4; # We choose the 4 closest MCs for the SINR calculation 
        dist_SCBS_SINR = 100; # We choose the range of the farthest SC that will impact SINR calculation for a user to be 200 meters
        sorted_MCBS_mat, idx_MCBS_SINR = dsc.idx_mat(dist_serv_cell, num_MCBS_SINR,'minimum',np); # Distance based sorted matrix and index of the MCBS under consideration for the PL calculation
        sorted_SCBS_mat, idx_SCBS_SINR = dsc.idx_mat(dist_serv_sc, dist_SCBS_SINR, 'distance', np); # Distance based sorted matrix and index of the SCBS under consideration for the PL calculation
        #print sorted_MCBS_mat.shape
        #print sorted_SCBS_mat.shape
        #print idx_MCBS_SINR
        #print "============"
        #print idx_SCBS_SINR
        
        # ====================
        # Pathloss Calculation

        # Note: This part can be optimized even more -- Potential Compute time reduction

        # ==> For Small Cell 

        print "Initiating Pathloss Calculation for Small Cells"

        PL_sc = np.empty((sorted_SCBS_mat.shape[0],sorted_SCBS_mat.shape[1])); # Initializing the Pathloss matrix 
        l_nl = np.zeros((usr_lcs.shape[0], num_SCBS + mc_locs.shape[0])); # This variable will hold the LOS-NLOS values for the user. 
        for i in range(0,sorted_SCBS_mat.shape[0]):
            for j in range(0,sorted_SCBS_mat.shape[1]):
                #print dist_serv_sc_3d[i][j]
                if sorted_SCBS_mat[i][j] != 0:
                    PL_sc[i,j], l_nl[i,j]  = pathloss.pathloss_CI(scn, sorted_SCBS_mat[i][j], np, dist_serv_sc_3d[i][int(idx_SCBS_SINR[i,j])], dsc, 1); # Calculating the pathloss for Small cells
                    #snr_sc[i][j] = scn.transmit_power + scn.transmit_gain_sc + scn.receiver_gain - PL_sc - (scn.N + 10*np.log10(scn.sc_bw)); # This is the SNR from one Small cell 
                else:
                    PL_sc[i,j] = float('nan'); # Nan for no PL calc

        # ==> For Macro Cell

        print "Initiating Pathloss Calculation for Macro Cells"

        PL_mc = np.empty((sorted_MCBS_mat.shape[0], sorted_MCBS_mat.shape[1])); # Initializing the Pathloss matrix
        for i in range(0, sorted_MCBS_mat.shape[0]):
            for j in range(0, sorted_MCBS_mat.shape[1]):
                PL_mc[i,j], l_nl[i, j+ num_SCBS] = pathloss.pathloss_CI(scn, sorted_MCBS_mat[i][j], np, dist_serv_cell_3d[i][int(idx_MCBS_SINR[i,j])], dsc, 0); # Calculating the pathloss for Macro cells
        
        csvsaver.csvsaver(l_nl,[],"LosNlos.csv")

        #print l_nl.shape 
        #print np.sum(l_nl)
        # ========================
        # Interference Calculation

        
        print "Performing Interference Calculation"
        interf_sc = dsc.interf(PL_sc, scn, np, scn.transmit_power, scn.transmit_gain_sc, scn.receiver_gain); # Calculate the interference matrix for small cells
        interf_mc = dsc.interf(PL_mc, scn, np, scn.max_tnsmtpow_MCBS, scn.ant_gain_MCBS, scn.rx_mc_gain); # Calculate the interference matrix for macro cells. MCs and SCs work on different frequency bands and hence do not interfere with each other
        csvsaver.csvsaver(interf_sc,[],"InterferenceSC.csv")
        #print interf_sc[1,:]

        # ====================
        # Rx Power Computation

        print "Performing Received Power Calculation"

        RX_sc = np.empty((PL_sc.shape[0], PL_sc.shape[1]));
        RX_mc = np.empty((PL_mc.shape[0], PL_mc.shape[1]));

        for i in range(0, RX_sc.shape[0]): # Small cell Received Power
            for j in range(0, RX_sc.shape[1]):
                RX_sc[i,j] = np.where(np.isnan(PL_sc[i,j]) != True, 10*np.log10((10**(scn.transmit_power/10)*(10**(scn.transmit_gain_sc/10))*(10**(scn.receiver_gain/10)*10**(-3))/(10**(PL_sc[i,j]/10)))/(10**(scn.N/10)*scn.usr_scbw*10**(-3))), float('nan'));

        for i in range(0, RX_mc.shape[0]): # Macro cell Received Power
            for j in range(0, RX_mc.shape[1]):
                RX_mc[i,j] = 10*np.log10((10**(scn.max_tnsmtpow_MCBS/10)*(10**(scn.ant_gain_MCBS/10))*(10**(scn.rx_mc_gain/10)*10**(-3))/(10**(PL_mc[i,j]/10)))/(10**(scn.N/10)*scn.mc_bw*10**(-3)))

        # ================
        # SINR Calculation

        print "Performing SINR Calculation for Small cells"

        sinr_sc = np.empty((sorted_SCBS_mat.shape[0],sorted_SCBS_mat.shape[1])); # Initialize SINR array
        sinr_pad_value = 350; # This is a pad value to be padded at the end of the vectors 
        #nz_idx = np.nonzero(PL_sc); # We store the non zero indices to extract the right SINR values for each user-AP pair
        
        for i in range(0,PL_sc.shape[0]):
            for j in range(0,PL_sc.shape[1]):
                sinr_sc[i,j] = np.where(np.isnan(PL_sc[i,j]) != True, 10*np.log10((10**(scn.transmit_power/10)*(10**(scn.transmit_gain_sc/10))*(10**(scn.receiver_gain/10)*10**(-3))/(10**(PL_sc[i,j]/10)))/(interf_sc[i,j] + 10**(scn.N/10)*scn.usr_scbw*10**(-3))), float('nan')); # We subtract the received power from other small cells to obtain the sinr 
            # print sinr_sc[i,:] 10*np.log10((10**(scn.transmit_power/10)*(10**(scn.transmit_gain_sc/10))*(10**(scn.receiver_gain/10)*10**(-3))/(10**(PL_sc[i,j]/10)))/(interf_sc[i,j] + 10**(scn.N/10)*scn.sc_bw*10**(-3)))
            # (scn.transmit_power - 30 + scn.transmit_gain_sc + scn.receiver_gain - PL_sc[i,j] - 10*np.log10(interf_sc[i,j] + 10**(scn.N/10)*scn.sc_bw*10**(-3)))
            sinr_sc[i,:] = np.where(np.isnan(sinr_sc[i,:]), sinr_pad_value, sinr_sc[i,:]);
            #sinr_sc[i, np.where(np.isnan(sinr_sc[i,:]) == True )] = np.amin(np.where(np.isnan(sinr_sc[i,:]) != True )); # Replace the None values with the minimum of that row 
            #print sinr_sc[i,:]
      
        csvsaver.csvsaver(sinr_sc,[],"SINR_SC.csv")
        print "Performing SINR Calculation for Macro cells"

        sinr_mc = np.empty((sorted_MCBS_mat.shape[0], sorted_MCBS_mat.shape[1])); # Initialize SINR matrix
        for i in range(0,PL_mc.shape[0]):
            for j in range(0, PL_mc.shape[1]):
                sinr_mc[i,j] = 10*np.log10((10**(scn.max_tnsmtpow_MCBS/10)*(10**(scn.ant_gain_MCBS/10))*(10**(scn.rx_mc_gain/10)*10**(-3))/(10**(PL_mc[i,j]/10)))/(interf_mc[i,j] + 10**(scn.N/10)*scn.mc_bw*10**(-3)))
            #print sinr_mc[i,:]
        print "Finished All Calculations and Returning to main Function"
        return np.hstack((sinr_sc,sinr_mc)), sorted_SCBS_mat, usr_lcs, idx_SCBS_SINR, idx_MCBS_SINR, sinr_pad_value, PL_sc.shape[1], PL_mc.shape[1], mc_locs.shape[0], np.hstack((RX_sc, RX_mc)), l_nl
        # The above calculation has to be optimally calculated for N users and M small cells. 

        
    
    else:
        print "==================================================="
        print "=========Sectorized and Beamformed Regime=========="
        print "==================================================="
        
        
        num_MCBS_SINR = 4; # We choose the 4 closest MCs for the SINR calculation 
        dist_SCBS_SINR = 100; # We choose the range of the farthest SC that will impact SINR calculation for a user to be 200 meters
        sorted_MCBS_mat, idx_MCBS_SINR = dsc.idx_mat(dist_serv_cell, num_MCBS_SINR,'minimum',np); # Distance based sorted matrix and index of the MCBS under consideration for the PL calculation
        sorted_SCBS_mat, idx_SCBS_SINR = dsc.idx_mat(dist_serv_sc, dist_SCBS_SINR, 'distance', np); # Distance based sorted matrix and index of the SCBS under consideration for the PL calculation

        #print idx_SCBS_SINR
        #print sorted_MCBS_mat.shape
        # ==> For Macro Cell

        print "\n Sectorizing Macro Cells and Computing Interferers"
        
        sector_UEMCBS = sectbeam.MCBS_sectorizer(np, scn, mc_locs.shape[0], mc_locs, usr_lcs) # Computing the Sectorization Matrix to compute the Interference
        l_nl = np.zeros((usr_lcs.shape[0], num_SCBS + mc_locs.shape[0])); # This variable will hold the LOS-NLOS values for the user. 
        
        print "Initiating Pathloss and Interference Calculation for Macro Cells"
        
        PL_mc = np.empty((sorted_MCBS_mat.shape[0], sorted_MCBS_mat.shape[1])); # Initializing the Pathloss matrix
        interf_mc = np.zeros((sorted_MCBS_mat.shape[0], sorted_MCBS_mat.shape[1])); # This is the matrix that will hold the interference values for each UE and significant TXs    
        #interf_sect_data = np.zeros((sorted_MCBS_mat.shape[0], sorted_MCBS_mat.shape[1]), dtype = object)
        for i in range(sorted_MCBS_mat.shape[0]):
            interf_sect = [] # An empty list of indices which will store the list of other interfering cells
            for j in range(sorted_MCBS_mat.shape[1]):
                interf_sect = np.where(sector_UEMCBS[i,:] == sector_UEMCBS[i,idx_MCBS_SINR[i,j]])[0] # The interfering cells
                #print ("SOmething Else:", np.where(sector_UEMCBS[i,:] == sector_UEMCBS[i,idx_MCBS_SINR[i,j]]))
                # print ("MCBS of Interest:", idx_MCBS_SINR[i,j])
                # print ("Sector of MCBS of Interest:", sector_UEMCBS[i,idx_MCBS_SINR[i,j]])
                # print ("Interfering APs:", interf_sect)
                #interf_sect_data[i,j] = interf_sect
                #PL_mc[i,j], l_nl[i, j+ num_SCBS] = pathloss.pathloss_CI(scn, sorted_MCBS_mat[i][j], np, dist_serv_cell_3d[i][j], dsc, 0); # Calculating the pathloss for Macro cells
                PL_mc[i,j], l_nl[i, j+ num_SCBS] = pathloss.pathloss_CI(scn, sorted_MCBS_mat[i][j], np, dist_serv_cell_3d[i][int(idx_MCBS_SINR[i,j])], dsc, 0); # Calculating the pathloss for Macro cells
                temp = np.empty((len(interf_sect)-1)); # An empty numpy array to hold the pathloss of interfereing cells
                #print len(interf_sect)
                #print interf_sect[0].shape[0]
                idx_temp = 0;
                for k in range(len(interf_sect)):
                    #print interf_sect[k]
                    if interf_sect[k] != idx_MCBS_SINR[i,j]:
                        #print ("Interference Calculation using indexes:", interf_sect[k])
                        temp[idx_temp], dummy = pathloss.pathloss_CI(scn, dist_serv_cell[i][interf_sect[k]], np, dist_serv_cell_3d[i][interf_sect[k]], dsc, 0); # Calculate the pathloss from the similar sector antennas to the UE
                        idx_temp = idx_temp + 1; # Increment the temp vector index
                        #print temp
                interf_mc[i,j] =  np.sum((10**(scn.max_tnsmtpow_MCBS/10)*(10**(scn.ant_gain_MCBS/10))*(10**(scn.rx_mc_gain/10)*10**(-3)))/(10**(temp/10))); # Interference for User i and AP j
        
        print "Performing SINR Calculation for Macro cells"
        #csvsaver.csvsaver(interf_sect_data,[],'Interfering Sectors Data')
        sinr_mc = np.empty((sorted_MCBS_mat.shape[0], sorted_MCBS_mat.shape[1])); # Initialize SINR matrix
        for i in range(0,PL_mc.shape[0]):
            for j in range(0, PL_mc.shape[1]):
                sinr_mc[i,j] = 10*np.log10((10**(scn.max_tnsmtpow_MCBS/10)*(10**(scn.ant_gain_MCBS/10))*(10**(scn.rx_mc_gain/10)*10**(-3))/(10**(PL_mc[i,j]/10)))/(interf_mc[i,j] + 10**(scn.N/10)*scn.mc_bw*10**(-3)))
                    
        #print interf_mc
        #print sinr_mc
        #print "==============="


        # ==> For Small Cell 

        print "Initiating Pathloss Calculation for Small Cells"

        PL_sc = np.empty((sorted_SCBS_mat.shape[0],sorted_SCBS_mat.shape[1])); # Initializing the Pathloss matrix 
        beam_sc = np.empty((sorted_SCBS_mat.shape[0], sorted_SCBS_mat.shape[1])); # Intializing the Beam ID matrix
        
        l_nl = np.zeros((usr_lcs.shape[0], num_SCBS + mc_locs.shape[0])); # This variable will hold the LOS-NLOS values for the user. 
        for i in range(0,sorted_SCBS_mat.shape[0]):
            for j in range(0,sorted_SCBS_mat.shape[1]):
                #print dist_serv_sc_3d[i][j]
                if sorted_SCBS_mat[i][j] != 0:
                    PL_sc[i,j], l_nl[i,j]  = pathloss.pathloss_CI(scn, sorted_SCBS_mat[i][j], np, dist_serv_sc_3d[i][int(idx_SCBS_SINR[i,j])], dsc, 1); # Calculating the pathloss for Small cells
                    #snr_sc[i][j] = scn.transmit_power + scn.transmit_gain_sc + scn.receiver_gain - PL_sc - (scn.N + 10*np.log10(scn.sc_bw)); # This is the SNR from one Small cell 
                else:
                    PL_sc[i,j] = float('nan'); # Nan for no PL calc

                
        # ===> Computing the Interference

        #print "Computing Receive and Transmit beamforming based angles"
        print "Entering the Approximate Small Cell Interference Computation"

        interf_sc = np.zeros((sorted_SCBS_mat.shape[0], sorted_SCBS_mat.shape[1])) # We consider 0 interference to analyze our results

        glob_angle_sc_rx = np.zeros((sorted_SCBS_mat.shape[0], sorted_SCBS_mat.shape[1])); # Initializing the matrix to hold UE to AP angles
        #glob_angle_sc_tx = np.empty((sorted_SCBS_mat.shape[1], sorted_SCBS_mat.shape[0])); # THis is the TX angle matrix
        
        #print usr_lcs.shape
        #print sc_locs.shape
        print "Computing the UE based sector and APs located in it"

        for i in range(sorted_SCBS_mat.shape[0]):
            for j in range(sorted_SCBS_mat.shape[1]):
                #print idx_SCBS_SINR[i,j]
                #rint usr_lcs[i,:]
                if idx_SCBS_SINR[i,j] != 'None':
                    #print "Here"
                    glob_angle_sc_rx[i,j] = dsc.angsc(usr_lcs[i,:],sc_locs[int(idx_SCBS_SINR[i,j]),:],np,scn) # Angle calculator to determine if 
                else:
                    glob_angle_sc_rx[i,j] = float('Nan') # Nan for the APs beyond 200m radius
        #print glob_angle_sc_rx
        csvsaver.csvsaver(usr_lcs,[],"UELOCS.csv")
        csvsaver.csvsaver(glob_angle_sc_rx,[],"SCAngles.csv")
        csvsaver.csvsaver(sc_locs,[],"SCLocs.csv")
        csvsaver.csvsaver(idx_SCBS_SINR,[],"SelectSCIDX.csv")
        csvsaver.csvsaver(PL_sc,[],"PL_sc.csv")
        print "Common Sector and Average Interference Computation"

        for i in range(sorted_SCBS_mat.shape[0]):
            for j in range(sorted_SCBS_mat.shape[1]):
                ap_int_idx = j; # This is our AP of interest
                interf_ap_idx = np.where(glob_angle_sc_rx[i,:] == glob_angle_sc_rx[i,ap_int_idx])[0] # These are the indexes of the APs that will be interfering with the AP of interest
                #print interf_ap_idx
                #interf_sc[i,ap_int_idx] = np.sum((scn.beam_hpbw_tx/(360))*PL_sc[i,interf_ap_idx]) 
                # (10**(tx_power/10)*(10**(gain/10))*(10**(scn.receiver_gain/10)*10**(-3)))/(10**(PL_temp/10))
                #print PL_sc[i,interf_ap_idx]
                interf_sc[i,ap_int_idx] = np.sum((10**(scn.transmit_power/10)*(10**(scn.transmit_gain_sc/10))*(10**(scn.receiver_gain/10)*10**(-3)))/(10**(PL_sc[i,interf_ap_idx]/10))) - (10**(scn.transmit_power/10)*(10**(scn.transmit_gain_sc/10))*(10**(scn.receiver_gain/10)*10**(-3)))/(10**(PL_sc[i,ap_int_idx]/10)) # We just use the calculated PL
        csvsaver.csvsaver(interf_sc,[],"InterferenceSC.csv")

        # ===> We try the SNR regime (Best Case solution with extreme directivity)

        #interf_sc = np.zeros((sorted_SCBS_mat.shape[0], sorted_SCBS_mat.shape[1])) # We consider 0 interference to analyze our results

        sinr_sc = np.empty((sorted_SCBS_mat.shape[0],sorted_SCBS_mat.shape[1])); # Initialize SINR array
        sinr_pad_value = 350; # This is a pad value to be padded at the end of the vectors 
        #nz_idx = np.nonzero(PL_sc); # We store the non zero indices to extract the right SINR values for each user-AP pair
        
        for i in range(0,PL_sc.shape[0]):
            for j in range(0,PL_sc.shape[1]):
                sinr_sc[i,j] = np.where(np.isnan(PL_sc[i,j]) != True, 10*np.log10((10**(scn.transmit_power/10)*(10**(scn.transmit_gain_sc/10))*(10**(scn.receiver_gain/10)*10**(-3))/(10**(PL_sc[i,j]/10)))/(interf_sc[i,j] + 10**(scn.N/10)*scn.usr_scbw*10**(-3))), float('nan')); # We subtract the received power from other small cells to obtain the sinr 
            # print sinr_sc[i,:] 10*np.log10((10**(scn.transmit_power/10)*(10**(scn.transmit_gain_sc/10))*(10**(scn.receiver_gain/10)*10**(-3))/(10**(PL_sc[i,j]/10)))/(interf_sc[i,j] + 10**(scn.N/10)*scn.sc_bw*10**(-3)))
            # (scn.transmit_power - 30 + scn.transmit_gain_sc + scn.receiver_gain - PL_sc[i,j] - 10*np.log10(interf_sc[i,j] + 10**(scn.N/10)*scn.sc_bw*10**(-3)))
            sinr_sc[i,:] = np.where(np.isnan(sinr_sc[i,:]), sinr_pad_value, sinr_sc[i,:]);
            #sinr_sc[i, np.where(np.isnan(sinr_sc[i,:]) == True )] = np.amin(np.where(np.isnan(sinr_sc[i,:]) != True )); # Replace the None values with the minimum of that row 
            #print sinr_sc[i,:] 

        csvsaver.csvsaver(sinr_sc,[],"SINR_SC.csv")
        #print sinr_sc.shape 
        #print sinr_mc.shape 
        # ====================
        # Rx Power Computation

        print "Performing Received Power Calculation"

        RX_sc = np.empty((PL_sc.shape[0], PL_sc.shape[1]));
        RX_mc = np.empty((PL_mc.shape[0], PL_mc.shape[1]));

        for i in range(0, RX_sc.shape[0]): # Small cell Received Power
            for j in range(0, RX_sc.shape[1]):
                RX_sc[i,j] = np.where(np.isnan(PL_sc[i,j]) != True, 10*np.log10((10**(scn.transmit_power/10)*(10**(scn.transmit_gain_sc/10))*(10**(scn.receiver_gain/10)*10**(-3))/(10**(PL_sc[i,j]/10)))/(10**(scn.N/10)*scn.usr_scbw*10**(-3))), float('nan'));

        for i in range(0, RX_mc.shape[0]): # Macro cell Received Power
            for j in range(0, RX_mc.shape[1]):
                RX_mc[i,j] = 10*np.log10((10**(scn.max_tnsmtpow_MCBS/10)*(10**(scn.ant_gain_MCBS/10))*(10**(scn.rx_mc_gain/10)*10**(-3))/(10**(PL_mc[i,j]/10)))/(10**(scn.N/10)*scn.mc_bw*10**(-3)))
                    
        print "Finished All Calculations and Returning to main Function"
        return np.hstack((sinr_sc,sinr_mc)), sorted_SCBS_mat, usr_lcs, idx_SCBS_SINR, idx_MCBS_SINR, sinr_pad_value, PL_sc.shape[1], PL_mc.shape[1], mc_locs.shape[0], np.hstack((RX_sc, RX_mc)), l_nl




# =========================
# Pathloss function Checker
# =========================

def pathloss_tester(scn,np,dsc): # This function helps to test the pathloss model implementation 
    
    # ======================================
    # Generate the test UE and eNB locations

    ue_sim_x = np.arange(10,300,1).reshape(((300-10)/1,1)); # Generates the location of a single UE along the x axis
    ue_sim_y = np.zeros((1,ue_sim_x.shape[0]),dtype='int').reshape(((300-10)/1,1)); # The UE is moving along the x axis only
    eNB_loc =  [min(ue_sim_x),min(ue_sim_y)]; # We place the eNB at the start point of the UE trajectory

    # ================================
    # Calculate the 2D and 3D distance

    test_dist_2d = dsc.dist_calc(np.concatenate((ue_sim_x,ue_sim_y),axis=1),eNB_loc, 0, 0,'2d',np); # Calculate the 2d distance
    test_dist_3d = dsc.dist_calc(np.concatenate((ue_sim_x,ue_sim_y),axis=1),eNB_loc, scn.usr_ht , scn.bs_ht_sc ,'3d',np); 

    # ======================
    # Calculate the Pathloss

    PL_sc = np.empty((ue_sim_x.shape[0],1)); # Empty array
    for i in range(0,ue_sim_x.shape[0]):
        PL_sc[i] = pathloss.pathloss_CI(scn, test_dist_2d[i], np, test_dist_3d[i], dsc, 1); # Calculating the pathloss for Small cells
    
    # ================
    # SINR Calculation
    snr_sc = 10*np.log10((10**(scn.transmit_power/10)*(10**(scn.transmit_gain_sc/10))*(10**(scn.receiver_gain/10)*10**(-3))/(10**(PL_sc/10)))/(10**(scn.N/10)*scn.sc_bw*10**(-3))); # This is the SNR from one Small cell 
    #snr_sc_1 = 10*np.log10((10**(55/10)*(10**(scn.transmit_gain_sc/10))*(10**(scn.receiver_gain/10)*10**(-3))/(10**(PL_sc/10)))/(10**(scn.N/10)*scn.sc_bw*10**(-3))); # This is the SNR from one Small cell 
    prx_sc_others = 0; # This is the received power from other Small cells
    sinr_sc = snr_sc - prx_sc_others; # We subtract the received power from other small cells to obtain the sinr 
    return sinr_sc, ue_sim_x, eNB_loc

# =============================
# Backhaul Reliability Estimate
# =============================

def bh_reliability(scn, np, critical_time):

    # ==============================================
    # Establishing the Outage probability Parameters

    fade_margin = [-20,-15,-10,-5,0,5]; # We randomly choose a fade margin between -20 dB and 5dB for a BH link. It indicates the ratio of sensitivity to received power in dB.  
    fmgn_selector = np.random.randint(0,6,1); # Fade Margin value selector
    K = 10; # We choose the rician shape parameter to be 10 based on Mona Jaber's paper 
    f = (3e9/(38*1e9))*(73*1e9/(3*1e8)); # We compute the doppler shift at 73 GHz  due to scattering objects between T-R pair using 10 Hz from Mona Jaber's paper as a baseline
    
    # ==================================
    # Compute the Expected Fade Duration

    rho = 10**(fade_margin[fmgn_selector]/10); # Convert from dB to real power
    exp_fade_dur_numr = np.array([5.0187*1e-8, 5.244*1e-7, 7.709*1e-6, 7.387*1e-4, 5.4309*1e-1, 1]); # Values from Octave for the Marcum Q Function
    fad_dur_bess_func = np.array([1.011,1.1131,2.4421, 1.2016*1e2, 1.1286*1e8, 3.1529*1e27]); # Values from Octave for the Bessel Function 
    exp_fade_dur_deno = np.sqrt(2*np.pi*(K+1.0))*f*rho*np.exp(-K-(K+1.0)*np.power(rho,2))*fad_dur_bess_func[fmgn_selector]; 
    exp_fade_dur = exp_fade_dur_numr[fmgn_selector]/exp_fade_dur_deno; # Expected value of the Fade duration 
    outage_prob = exp((-1*critical_time)/exp_fade_dur); # Given the critical time for a given application we can compute the outage probability for the BS. 

# ===================
# Backhaul Throughput
# ===================

def backhaul_tput(assoc_mat, SCBS_per_MCBS, wl_mat, np, scn, dsc):

    # ==========================================================
    # We compute the throughput for the backhaul link of each SC

    #print wl_mat
    PL_SC_MC = np.empty((wl_mat.shape[0],1)); # Initialize the Pathloss matrix
    tput_SC = copy.copy(PL_SC_MC); # Initialize the Throughput matrix
    dist_SC_MC = copy.copy(PL_SC_MC); # Initialize the 3D distance matrix
    #print ("Matrix Shape:",dist_SC_MC.shape)
    #print ("Association Matrix:", assoc_mat.shape)
    #print assoc_mat

    # ===> Computing the 3D distance 
    for k in range(0,assoc_mat.shape[0]):
        #print ("K:",k) 
        #print ("Wireless Matrix:", wl_mat[k,:])
        #print ("Association Matrix Values:", assoc_mat[k,next((i for i, x in enumerate(wl_mat[k,:].tolist()) if x), None)])
        #print ("Distance:", np.sqrt(assoc_mat[k,next((i for i, x in enumerate(wl_mat[k,:].tolist()) if x), None)]**2 + (scn.bs_ht_mc-scn.bs_ht_sc)**2))
        if next((i for i, x in enumerate(wl_mat[k,:].tolist()) if x), None) != None:
            dist_SC_MC[k] = np.sqrt(assoc_mat[k,next((i for i, x in enumerate(wl_mat[k,:].tolist()) if x), None)]**2 + (scn.bs_ht_mc-scn.bs_ht_sc)**2); # The 3D distance from the MC for a given SC

    # ===> Computing the Pathloss for the Small Cells to the Macro cells

    for l in range(0, tput_SC.shape[0]):
        if next((i for i, x in enumerate(wl_mat[l,:].tolist()) if x), None) != None:
            #print assoc_mat[l,next((i for i, x in enumerate(assoc_mat[l,:].tolist()) if x), None)]
            PL_SC_MC[l], flg = pathloss.pathloss_CI(scn, assoc_mat[l,next((i for i, x in enumerate(wl_mat[l,:].tolist()) if x), None)], np, dist_SC_MC[l], dsc, 2); # Calculating the pathloss for Small cells to Macro Cells
        else:
            idx_BH_chx = np.random.randint(0,3); # BH Relaxation choice is added here
            PL_SC_MC[l] = 0; # This is the Fiber based backhaul
            tput_SC[l] = scn.fib_BH_capacity + scn.perct_incr[idx_BH_chx]*scn.avg_fib_BH_capacity; # Fiber backhaul capacity

    #print PL_SC_MC
    # ===> Computing the Throughput for the Small Cells to Macro Cells

    #interf_sc_mc = dsc.interf(PL_SC_MC, scn, np); # Calculate the interference matrix for small cells
    l_idx = 0; 
    u_idx = SCBS_per_MCBS[0];
    #print SCBS_per_MCBS
    for j in range(0,tput_SC.shape[0]):
        if j < u_idx: 
            tput_SC[j] = np.where(PL_SC_MC[j] != 0, (scn.sc_bw/SCBS_per_MCBS[l_idx])*np.log2(1+(10**(scn.transmit_power/10)*(10**(scn.transmit_gain_sc/10))*(10**(scn.ant_gain_MCBS/10)*10**(-3))/(10**(PL_SC_MC[j]/10)))/(10**(scn.N/10)*(scn.sc_bw/SCBS_per_MCBS[l_idx])*10**(-3))), tput_SC[j]); # We subtract the received power from other small cells to obtain the sinr 
        else:
            l_idx = l_idx + 1; # Increment the lower index
            u_idx = u_idx + SCBS_per_MCBS[l_idx]; # Increment the 
            tput_SC[j] = np.where(PL_SC_MC[j] != 0, (scn.sc_bw/SCBS_per_MCBS[l_idx])*np.log2(1+(10**(scn.transmit_power/10)*(10**(scn.transmit_gain_sc/10))*(10**(scn.ant_gain_MCBS/10)*10**(-3))/(10**(PL_SC_MC[j]/10)))/(10**(scn.N/10)*(scn.sc_bw/SCBS_per_MCBS[l_idx])*10**(-3))), tput_SC[j]); # We subtract the received power from other small cells to obtain the sinr 
    return tput_SC

