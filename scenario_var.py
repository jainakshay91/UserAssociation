# This file consists of all the scenario variables


# Simulation area and number of BSs, UEs
simulation_area = 1e6; #The area is in square meters; The shape is also a square 
MCBS_intersite = 200; #Intersite distance for Macro BS
SCBS_intersite = 20; #Intersite distance for Small cell BS
#SCBS_per_MCBS = np.random.randint(3,10,size=1); # Create a set for number of small cells per macro BS
UE_density = 1250; #Number of UEs per sq. km
#UE_density_eMBB = 24000; #Number of URLLC UEs per Macro BS
ant_height_MCBS = 25; # Height above rooftop in meters
ant_gain_MCBS = 17; # dBi gain 
max_tnsmtpow_MCBS = 49; # dBm gain per band (in 20 MHz)

   
