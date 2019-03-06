#!/usr/bin/env	python

# ==================================================================================================================================
# This is a function file enables the program to check if the ISD rule, according to the considered specifications, is satisfied
# This function file also consists of a grid creator function to place the Macro BS
# This function file also consists of other miscellaneous functions
# ==================================================================================================================================

# ======================
# Import Necessary Files
# ======================

from bp_assister import bp_assist

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

def breakpt_dist (bs_ht, usr_ht, fc, dist, flag_sc, np): # Generates the breakpoint distance parameter
    # We first initiate some local parameters to calculate the breakpoint distance
    c = 3e8; # This is the speed of light 

    # Calculating the effective environmental height
    if flag_sc: # This is for small cells
        eff_ht = 1; # Effective environment height 
    else: # This is when we have an urban macro
        # We have to calculate the probability for the effective environmental height
        bp = bp_assist(); # breakpoint distance
        prob_1m = 1/(1+bp.bp_assister(dist,usr_ht)); # Probability of effective environmental height being 1m
        if prob_1m > 0.5:
            eff_ht = 1; # Effective environment height
        else: 
            eff_ht = 12 + ((usr_ht-1.5)-12)*np.random_integers(np.floor((usr_ht-13.5)/3.)-1)/(np.floor((usr_ht-13.5)/3.));
    
    # Final Breakpoint distance calculation
    bs_eff = bs_ht - eff_ht; 
    usr_eff = usr_ht - eff_ht; 
    bp_dist = 4*bs_eff*usr_eff*fc/c; # Breakpoint dist 
    return bp_dist

# ===============================
# SINR Calculator per Application
# ===============================

#def sinr_gen (mc_locs, sc_locs, usr_locs_eMBB, usr_locs_URLLC, usr_locs_mMTC): # Generates the SINR per application


    
    


    

    

