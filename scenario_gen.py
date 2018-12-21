# ==> This file enables the scenario generation process for the network to be analyzed.


# Import the necessary libraries

import os.path
import dist_check


# Load/Generate the Macro Cell base station locations

def macro_cell(simulation_area,MCBS_intersite,np):
    # We distribute the Macro BSs as a grid
#    num_macro_BS = simulation_area/MCBS_intersite; # Number of BSs on the grid
    offset = MCBS_intersite/2; # Offset param
    locs_interim = np.arange(offset, np.sqrt(simulation_area).astype(int), MCBS_intersite); # Range of numbers from 0 to the end of the grid area, with intersite distance spacing 
    print locs_interim
    locs_MCBS = dist_check.gridder(locs_interim,MCBS_intersite,np); # Calling a permutation function that generates the grid
    return locs_MCBS

# Load/Generate the Small Cell base station locations

def small_cell(num, MCBS_locs, SCBS_intersite,SCBS_per_MCBS,MCBS_intersite,np):
    offset = MCBS_intersite/2; # Offset param
    while True:	
	locs_SCBS_x = np.random.uniform(MCBS_locs[num,0] - offset,MCBS_locs[num,0]+ offset,(SCBS_per_MCBS,1)); # Generating the X coordinate of the small cells for a given macro cell
	locs_SCBS_y = np.random.uniform(MCBS_locs[num,1]-offset,MCBS_locs[num,1]+offset,(SCBS_per_MCBS,1)); # Generating the Y coordinate of the small cells for a given macro cell
	locs_SCBS = np.concatenate((locs_SCBS_x, locs_SCBS_y), axis=1); 
	if dist_check.checker(locs_SCBS,SCBS_intersite,np)==1:
		break
    return locs_SCBS

# Load/Generate the User locations

def user_dump(UE_density, MCBS_intersite, simulation_area,np):
    usr_locs = np.random.uniform(0,np.sqrt(simulation_area),(int(UE_density*simulation_area/1e6),2)); # We obtain a set of UE locations 	
    return usr_locs
