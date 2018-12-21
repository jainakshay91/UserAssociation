#!/usr/bin/env python


# Import the necessary binaries

import numpy as np
import matplotlib.pyplot as plt
import scenario_gen
# from gurobipy import *
from scenario_var import *


# Main Code Begins here

# ==> Macro cell placement 

macro_cell_locations = scenario_gen.macro_cell(simulation_area, MCBS_intersite, np); # Get the macro cell locations

#print macro_cell_locations
#print "==================="

# ==> Small cell placement

SCBS_per_MCBS = np.random.randint(3,10,size=macro_cell_locations.shape[0]); # Randomly choosing number of SCBS within an MCBS domain in the range 3 to 1
#print SCBS_per_MCBS
locs_SCBS = []; # We create an empty list of numpy arrays 
for i in range(0,macro_cell_locations.shape[0]):
    small_cell_locations = scenario_gen.small_cell(i, macro_cell_locations, SCBS_intersite, SCBS_per_MCBS[i], MCBS_intersite, np); #Get the small cell locations for each macro cell domain 
    locs_SCBS.append(small_cell_locations); # Store the small cell locations in the list of numpy arrays 
    #print locs_SCBS[:,:,i]
    #print "==================="
#print locs_SCBS
# ==> Get the user locations 
usr_locations = scenario_gen.user_dump(UE_density,MCBS_intersite,simulation_area,np); 

# Plots

plt.plot(macro_cell_locations[:,0], macro_cell_locations[:,1],'rs'); # Plot the macro cells
for j in range(0,macro_cell_locations.shape[0]):
    print_element = locs_SCBS[j]; #Accessing the numpy array of SC locations corresponding to the Macro Cell    
    plt.plot(print_element[:,0], print_element[:,1], 'b*'); # Plot the small cells
plt.plot(usr_locations[:,0],usr_locations[:,1],'k+')
plt.show() # Show the small cells
