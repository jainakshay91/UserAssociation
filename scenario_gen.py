# ==> This file enables the scenario generation process for the network to be analyzed.

# ==============================
# Import the necessary libraries
# ==============================

#import dist_check
import pathloss
#from multiprocessing import Process
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

    # =============================================================
    # Compute total users and total applications in Simulation area

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
        assoc_usapp[attr_name_assoc + str(i)] = np.random.randint(2, size = (tot_users_scenario[i], scn.max_num_appl_UE)); # Generate User-App Association 

    return usr_locs, assoc_usapp
    #usr_locs_eMBB = np.random.uniform(0,np.sqrt(scn.simulation_area),(tot_dev_eMBB,2)); # We obtain a set of eMBB locations
    #usr_locs_URLLC = np.random.uniform(0,np.sqrt(scn.simulation_area),(int(tot_dev_URLLC),2)); # We obtain a set of URLLC locations
    #usr_locs_mMTC = np.random.uniform(0,np.sqrt(scn.simulation_area),(tot_dev_mMTC,2)); # We obtain a set of mMTC locations
    #return usr_locs_eMBB, usr_locs_URLLC, usr_locs_mMTC; # Return the locations of these applications/users with these applications


# =============================
# Generate the backhaul network
# =============================

def backhaul_dump(scn, SCBS_per_MCBS, MCBS_locs, assoc_mat, np):

    # =====================================================================================================
    # We create the wired and wireless backhaul matrix (Restricting it to just one backhaul link currently)

    mat_wlbh_sc = (assoc_mat <= scn.wl_bh_bp)*1; # Wireless backhaul enabled small cells
    mat_wrdbh_sc = (assoc_mat > scn.wl_bh_bp)*1; # Wired backhaul enabled small cells
    MC_hops = np.random.randint(scn.min_num_hops, scn.max_num_hops,size = MCBS_locs.shape[0]); # The macro cells always have wired backhaul (Local breakouts can be added later)
    SC_hops = ((assoc_mat > 0)*1)*np.transpose(MC_hops) + 1; # The number of hops for each small cells to the IMS core

    return mat_wlbh_sc, mat_wrdbh_sc, MC_hops, SC_hops # Return the hops and wired/wireless backhaul configuration 

# ===============================
# SINR Calculator per Application
# ===============================

def sinr_gen (scn, num_SCBS, mc_locs, sc_locs, usr_lcs, dsc, np): # Generates the SINR per application      
    
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
        dist_serv_sc_3d[:,i] = dsc.dist_calc(usr_lcs, sc_locs[i,:], scn.bs_ht_sc, scn.usr_ht, '3d', np); # Calculate the distance of each eMBB application location with each MC and sort them
        
        #dist_serv_sc_eMBB_3d[:,i] = dsc.dist_calc(usr_locs_eMBB, sc_locs[i,:], scn.bs_ht_sc, scn.usr_ht, '3d', np); # Calculate the distance of each eMBB application location with each MC and sort them
        #dist_serv_sc_URLLC_3d[:,i] = dsc.dist_calc(usr_locs_URLLC, sc_locs[i,:], scn.bs_ht_sc, scn.usr_ht, '3d', np); # Calculate the distance of each URLLC application location with each MC and sort them
        #dist_serv_sc_mMTC_3d[:,i] = dsc.dist_calc(usr_locs_mMTC, sc_locs[i,:], scn.bs_ht_sc, scn.usr_ht,'3d', np); # Calculate the distance of each mMTC application location with each MC and sort them

    print "Finished Distance calculation"
    # ======================================================
    # Limit the number of MC and SC for the SINR calculation

    
    # ==> eMBB users
    
    num_MCBS_SINR = 4; # We choose the 4 closest MCs for the SINR calculation 
    dist_SCBS_SINR = 200; # We choose the range of the farthest SC that will impact SINR calculation for a user to be 200 meters
    sorted_MCBS_mat, idx_MCBS_SINR = dsc.idx_mat(dist_serv_cell, num_MCBS_SINR,'minimum',np); # Distance based sorted matrix and index of the MCBS under consideration for the PL calculation
    sorted_SCBS_mat, idx_SCBS_SINR = dsc.idx_mat(dist_serv_sc, dist_SCBS_SINR, 'distance', np); # Distance based sorted matrix and index of the SCBS under consideration for the PL calculation
    #print sorted_MCBS_mat
    # ====================
    # Pathloss Calculation

    # Note: This part can be optimized even more -- Potential Compute time reduction

    # ==> For Small Cell 

    print "Initiating Pathloss Calculation for Small Cells"

    PL_sc = np.empty((sorted_SCBS_mat.shape[0],sorted_SCBS_mat.shape[1])); # Initializing the Pathloss matrix 
    for i in range(0,sorted_SCBS_mat.shape[0]):
        for j in range(0,sorted_SCBS_mat.shape[1]):
            if sorted_SCBS_mat[i][j] != 0:
                PL_sc[i,j] = pathloss.pathloss_CI(scn, sorted_SCBS_mat[i][j], np, dist_serv_sc_3d[i][j], dsc, 1); # Calculating the pathloss for Small cells
                #snr_sc[i][j] = scn.transmit_power + scn.transmit_gain_sc + scn.receiver_gain - PL_sc - (scn.N + 10*np.log10(scn.sc_bw)); # This is the SNR from one Small cell 
            else:
                PL_sc[i,j] = float('nan'); # Nan for no PL calc

    # ==> For Macro Cell

    print "Initiating Pathloss Calculation for Small Cells"

    #PL_mc = np.empty((sorted_MCBS_mat.shape[0], sorted_MCBS_mat.shape[1])); # Initializing the Pathloss matrix

    # ========================
    # Interference Calculation
    
    print "Performing Interference Calculation"
    interf_sc = dsc.interf(PL_sc, scn, np); # Calulate the interference matrix
    #print interf_sc[1,:]

    # ================
    # SINR Calculation

    print "Performing SINR Calculation for Small cells"

    sinr_sc = np.empty((sorted_SCBS_mat.shape[0],sorted_SCBS_mat.shape[1])); # Initialize SINR array
    sinr_pad_value = 350; # This is a pad value to be padded at the end of the vectors 
    #nz_idx = np.nonzero(PL_sc); # We store the non zero indices to extract the right SINR values for each user-AP pair
    
    for i in range(0,PL_sc.shape[0]):
        for j in range(0,PL_sc.shape[1]):
            sinr_sc[i,j] = np.where(np.isnan(PL_sc[i,j]) != True, (scn.transmit_power - 30 + scn.transmit_gain_sc + scn.receiver_gain - PL_sc[i,j] - 10*np.log10(interf_sc[i,j] + 10**(scn.N/10)*scn.sc_bw*10**(-3))), float('nan')); # We subtract the received power from other small cells to obtain the sinr 
        #print sinr_sc[i,:] 10*np.log10((10**(scn.transmit_power/10)*(10**(scn.transmit_gain_sc/10))*(10**(scn.receiver_gain/10)*10**(-3))/(10**(PL_sc[i,j]/10)))/(interf_sc[i,j] + 10**(scn.N/10)*scn.sc_bw*10**(-3)))
        sinr_sc[i,:] = np.where(np.isnan(sinr_sc[i,:]), sinr_pad_value, sinr_sc[i,:]);
        #sinr_sc[i, np.where(np.isnan(sinr_sc[i,:]) == True )] = np.amin(np.where(np.isnan(sinr_sc[i,:]) != True )); # Replace the None values with the minimum of that row 
    #print sinr_sc[1,:]
    print "Finished All Calculations and Returning to main Function"
    return sinr_sc, sorted_SCBS_mat, usr_lcs, idx_SCBS_SINR, sinr_pad_value
    # The above calculation has to be optimally calculated for N users and M small cells. 


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

