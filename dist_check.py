# ==> This is a function file enables the program to check if the ISD rule, according to the considered specifications, is satisfied
# ==> This function file also consists of a grid creator function to place the Macro BS

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

def gridder(locs_interim,MCBS_intersite,np): # Create the grid through random permutation 
    #print locs_interim.shape[0]
    locs_MCBS = np.empty([locs_interim.shape[0]**2,2]); # Create the 3-D vector for holding the X-Y coordinates of the MCBS 
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
        
