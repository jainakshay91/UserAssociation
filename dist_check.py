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
        sorted_mat = np.sort(src_mat,kind='mergesort')[:,:param_val]; # Sort the matrix first
        sorted_idx = np.argsort(src_mat,kind='mergesort')[:,:param_val]; #Indices of the sorted matrix
        return sorted_mat,sorted_idx # Returning the sorted matrix and the index of the requested elements in the original matrix
   
    # This function can be extended further for maximum or non-maximal/minimal scenarios
   
    elif srch_type == 'distance':
        #print src_mat[1,:]
        sorted_mat = np.sort(src_mat,kind='mergesort'); # Sort the SC distance matrix
        #print sorted_mat[1,:]
        sorted_idx = np.where(sorted_mat<=200,np.argsort(src_mat,kind='mergesort'),'None'); # Indices of the SCs that are within 200m and can impact the UE through interference
        #print sorted_idx[1]
        return np.where(sorted_mat>200,0,sorted_mat),sorted_idx # Return Sorted Matrix and the indices 
    
# ==============================
# Interference Matrix Calculator
# ==============================

def interf(PL, scn, np): # This function returns the overall interference matrix given a Pathloss matrix
    interf = np.empty((PL.shape[0],PL.shape[1])); # Initialize the interference matrix
    PR_interf = interf; # This is a temporary matrix to hold Rx Power values due to all other APs other than AP of interest
    #print PL[1,:]
    #print "Next"
    for i in range(0, PL.shape[1]):
        PL_temp = copy.copy(PL); # This is a temporary array store
        PL_temp[:,i] = float('nan'); # So the array now has Nan where we have our AP of interest
        PR_interf = (10**(scn.transmit_power/10)*(10**(scn.transmit_gain_sc/10))*(10**(scn.receiver_gain/10)*10**(-3)))/(10**(PL_temp/10)); # Compute the received power on each UE-AP pair
        interf[:,i] = np.sum(np.where(np.isnan(PR_interf), 0, PR_interf), axis=1); #Interference from other APs for a given UE-AP pair
    return interf

    

