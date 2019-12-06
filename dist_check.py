# ==================================================================================================================================
# This is a function file enables the program to check if the ISD rule, according to the considered specifications, is satisfied
# This function file also consists of a grid creator function to place the Macro BS
# This function file also consists of other miscellaneous functions
# ==================================================================================================================================

# ======================
# Import Necessary Files
# ======================

from bp_assister import bp_assist
import copy
import csvsaver

# ===========================
# Inter-site distance checker
# ===========================

def checker(locs,isd,np): # Check the ISD for the selected 
    flag = [None]*locs.shape[0]; #Empty array for the flag list
    i = 0; # Initialize the iterator variable
    for i in range(0,(locs.shape[0])):
    	dist_X = locs[i,0]*np.ones((locs.shape[0],1), dtype=int) - np.reshape(locs[:,0],(locs[:,0].shape[0],1)); #X coordinate diff	
    	#print dist_X
    	dist_Y = locs[i,1]*np.ones((locs.shape[0],1), dtype=int) - np.reshape(locs[:,1],(locs[:,1].shape[0],1)); #Y coordinate diff
    	#print dist_Y
    	tot_dist = np.sqrt(dist_X**2 + dist_Y**2); # Distance from other base stations
    	#print tot_dist
    	tot_dist[i,:] = tot_dist[i,:] + isd; # Set the self distance to be the minimum to avoid the base stations own location from affecting the decision
    	#print tot_dist
    	flag[i] = (tot_dist>=isd).all(); # Generate the flag (False if ISD violated; True if not)
        #print flag
    if all(flag) == False:
    	#print all(flag)
	   return 0
    else:	
	   return 1 # Return 1 if none of the above checks return 0	

# =====================
# Same Location Checker
# =====================

def locs_checker(locs, other_locs, np, indix):
    if indix == 'sc':
        flag = [None]*locs.shape[0]; #Empty array for the flag list
        for i in range(locs.shape[0]):
            for j in range(other_locs.shape[0]):
                if locs[i,0] == other_locs[0] and locs[i,1] == other_locs[1]:
                    flag[i] = False # If the location matches any of the previous generated locations then regenerate the location
                else:
                    flag[i] = True # If not, then pass
        if all(flag) == False:
            return 0    
        else:
            return 1
    elif indix == 'user':
        flag = [None]*locs.shape[0]; #Empty array for the flag list
        for i in range(locs.shape[0]):
            if i != locs.shape[0]:
                comp_locs = np.vstack((other_locs,locs[i+1:,:]))
            else:
                comp_locs = other_locs
            for j in range(comp_locs.shape[0]):
                if locs[i,0] == comp_locs[j,0] and locs[i,1] == comp_locs[j,1]:
                    flag[i] = False # If the location matches any of the previous generated locations then regenerate the location
                else:
                    flag[i] = True # If not, then pass
        if all(flag) == False:
            return 0    
        else:
            return 1
# =======================
# Macro Cell Grid Creator
# =======================

def gridder(locs_interim,MCBS_intersite,np): # Create the grid through random permutation 
    #print locs_interim.shape[0]
    locs_MCBS = np.empty([locs_interim.shape[0]**2,2]); # Create the 2-D vector for holding the X-Y coordinates of the MCBS 
    idx = 0; # Iterator variable
    for i in range(0,locs_MCBS.shape[0]):
     #   print i
        if i%(locs_interim.shape[0]) == 0:
            locs_MCBS[i,0]=locs_interim[idx];
            locs_MCBS[i,1]=locs_interim[0];
            idx = idx + 1; #Iterator to go through the grid
        else: 
            locs_MCBS[i,0]=locs_interim[idx-1];
            locs_MCBS[i,1]=locs_MCBS[i-1,1]+MCBS_intersite; 
    
    return locs_MCBS

# =========================
# LOS Probability Generator
# =========================

def los_prob_var_gen(h): # Generates the variables needed for the los probability calculation
    if h <= 13:
        C = 0; 
    elif h > 13 and h <= 23:
        C = ((h-13)/10)**(3/2);
    return C

# ==============================
# Breakpoint distance calculator
# ==============================

def breakpt_dist (scn, dist, flag_sc, np): # Generates the breakpoint distance parameter
    # We first initiate some local parameters to calculate the breakpoint distance
    # Calculating the effective environmental height
    if flag_sc: # This is for small cells
        eff_ht = 1; # Effective environment height 
        bs_ht = scn.bs_ht_sc; # Small cell height
        fc = scn.fc_sc; #Small cell frequency
    else: # This is when we have an urban macro
        # We have to calculate the probability for the effective environmental height
        bs_ht = scn.bs_ht_mc; # Macro cell height
        fc = scn.fc_mc; # Macro cell frequency
        bp = bp_assist(); # breakpoint distance
        prob_1m = 1/(1+bp.bp_assister(dist,scn.usr_ht)); # Probability of effective environmental height being 1m
        if prob_1m > 0.5:
            eff_ht = 1; # Effective environment height
        else: 
            eff_ht = 12 + ((scn.usr_ht-1.5)-12)*np.random_integers(np.floor((scn.usr_ht-13.5)/3.)-1)/(np.floor((scn.usr_ht-13.5)/3.));
    
    # Final Breakpoint distance calculation
    bs_eff = bs_ht - eff_ht; 
    usr_eff = scn.usr_ht - eff_ht; 
    bp_dist = 4*bs_eff*usr_eff*fc/scn.c; # Breakpoint dist 
    return bp_dist

# ===========================
# Generic Distance Calculator
# ===========================

def dist_calc(locs_src, locs_tgt, usr_ht, bs_ht, dist_type, np):
    #print locs_src
    #print locs_tgt
    if dist_type == '2d':
        x_diff = locs_src[:,0] - locs_tgt[0]; # X coordinate difference
        y_diff = locs_src[:,1] - locs_tgt[1]; # Y coordinate difference
        return np.sqrt(np.power(x_diff,2) + np.power(y_diff,2)) # Returning the 2-D distance between two points

    elif dist_type == '3d':
        x_diff = locs_src[:,0] - locs_tgt[0]; # X coordinate difference
        y_diff = locs_src[:,1] - locs_tgt[1]; # Y coordinate difference
        z_diff = bs_ht - usr_ht; # Z coordinate difference
        return np.sqrt(np.power(x_diff,2) + np.power(y_diff,2) + np.power(z_diff,2)) # Returning the 3-D distance between two points

# =======================================
# Matrix Array Element Locator and Sorter
# =======================================

def idx_mat(src_mat, param_val, srch_type, np): # This function works as an element locator and distance based Sorter
   
    if srch_type == 'minimum':
        #print src_mat
        sorted_mat = np.sort(src_mat,kind='mergesort')[:,:param_val]; # Sort the matrix first
        #print sorted_mat
        sorted_idx = np.argsort(src_mat,kind='mergesort')[:,:param_val]; #Indices of the sorted matrix
        #print sorted_idx
        return sorted_mat,sorted_idx # Returning the sorted matrix and the index of the requested elements in the original matrix
   
    # This function can be extended further for maximum or non-maximal/minimal scenarios
   
    elif srch_type == 'distance':
        #print src_mat[1,:]
        sorted_mat = np.sort(src_mat,kind='mergesort'); # Sort the SC distance matrix
        #print sorted_mat[1,:]
        sorted_idx = np.where(sorted_mat<=100,np.argsort(src_mat,kind='mergesort'),'None'); # Indices of the SCs that are within 200m and can impact the UE through interference
        #print sorted_idx[1]
        return np.where(sorted_mat>100,0,sorted_mat),sorted_idx # Return Sorted Matrix and the indices 
    
# ============================================================
# Interference Limited Scenario Interference Matrix Calculator
# ============================================================

def interf(PL, scn, np, tx_power, gain, rx_gain): # This function returns the overall interference matrix given a Pathloss matrix
    interf = np.empty((PL.shape[0],PL.shape[1])); # Initialize the interference matrix
    PR_interf = interf; # This is a temporary matrix to hold Rx Power values due to all other APs other than AP of interest
    #print PL.shape 
    #print PL[1,:]
    #print "Next"
    for i in range(0, PL.shape[1]):
        PL_temp = copy.copy(PL); # This is a temporary array store
        PL_temp[:,i] = float('nan'); # So the array now has Nan where we have our AP of interest
        #print ("PL matrix is:", PL_temp)
        #csvsaver.csvsaver(PL_temp,[],"PL_temp"+str(i)+".csv")
        PR_interf = (10**(tx_power/10)*(10**(gain/10))*(10**(rx_gain/10)*10**(-3)))/(10**(PL_temp/10)); # Compute the received power on each UE-AP pair
        #print ("Received Interference is:", PR_interf)
        #csvsaver.csvsaver(PR_interf,[],"PR_interf"+str(i)+".csv")
        interf[:,i] = np.sum(np.where(np.isnan(PR_interf), 0, PR_interf), axis=1); #Interference from other APs for a given UE-AP pair
    return interf

# ===========================================================
# Small Cell Beamforming Based Interference Matrix Calculator
# ===========================================================
 
def angsc(usr_loc, sc_loc, np, scn): # This function will find the beam that points in the direction of the user
    
    beam_angles = np.arange(scn.beam_hpbw_rx, 360 + scn.beam_hpbw_rx , scn.beam_hpbw_rx) # UE RX beam angles
    #print beam_angles

    #csvsaver.csvsaver(usr_loc,[],"USERlocsFORSC.csv")
    coord_diff = usr_loc - sc_loc # Compute the Perpendicular
    theta_diff = np.degrees(np.arctan2(coord_diff[1],coord_diff[0])) # Computes the vector of angles each UE makes with a MCBS
    #print ("Calculated SC angles:", theta_diff)
    #csvsaver.csvsaver(theta_diff,[],"ComputedAngleForSC.csv")

    theta_diff = np.where(theta_diff >= 0 , theta_diff, 360 + theta_diff)
    #print theta_diff



    angle_diff = beam_angles - theta_diff; # Subtract the angle of the current AP of interest with the beam sectors
    #print angle_diff

    sect_ang = np.where(angle_diff>=0)[0][0]; # Stores the index where a given AP is located within the UE beam
    return sect_ang
    #csvsaver.csvsaver(sect_ang,[],"SectorsSC.csv")
    #print ("Determined Sectors are:", sect_ang)
    
# def interf_sect(interf, idx_curr_MCBS, scn, np): # This function returns the overall interference matrix given a Pathloss matrix and the sectorization matrix
#     interf = np.empty((PL.shape[0],PL.shape[1])); # Initialize the interference matrix
#     PR_interf = interf; # This is a temporary matrix to hold Rx Power values due to all other APs other than AP of interest
#     #print PL[1,:]
#     #print "Next"
#     for i in range(0, PL.shape[1]):
#         PL_temp = copy.copy(PL); # This is a temporary array store
#         PL_temp[:,i] = float('nan'); # So the array now has Nan where we have our AP of interest
#         PR_interf = (10**(scn.transmit_power/10)*(10**(scn.transmit_gain_sc/10))*(10**(scn.receiver_gain/10)*10**(-3)))/(10**(PL_temp/10)); # Compute the received power on each UE-AP pair
#         interf[:,i] = np.sum(np.where(np.isnan(PR_interf), 0, PR_interf), axis=1); #Interference from other APs for a given UE-AP pair
#     return interf

# ==================
# Matrix Reorganizer
# ==================

def reorganizer(SRC_mat, IDX_mat_SC, IDX_mat_MC, num_scbs, num_mcbs, sinr_pad, np, scn):
    
    reorg_mat = np.zeros((IDX_mat_SC.shape[0], num_scbs+num_mcbs)); # The total matrix to hold the SINR values
    
    # ===================================
    # First we reorganize the Small Cells

    for i in range(0,IDX_mat_SC.shape[0]):
        for j in range(0, IDX_mat_SC.shape[1]):
            if IDX_mat_SC[i,j] != 'None':
                reorg_mat[i,int(IDX_mat_SC[i,j])] = SRC_mat[i,j]; # Reorganizing the Small cells

    # ==================================
    # We reorganize the Macro Cells
    #print reorg_mat 
    #print "======="
    for i in range(0,IDX_mat_MC.shape[0]):
        for j in range(0, IDX_mat_MC.shape[1]):
            reorg_mat[i, num_scbs + IDX_mat_MC[i,j]] = SRC_mat[i, num_scbs+j]; # Reorganizing the Macro Cells 

    #print reorg_mat
    reorg_mat = np.where(reorg_mat == 0, sinr_pad, reorg_mat)
    return reorg_mat