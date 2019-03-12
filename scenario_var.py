# ========================================
# This file defines the Scenario Variables
# ========================================

c = 3e8; # Speed of light in m/s
fc_mc = 3.5e9; # Carrier frequency of the MC
fc_sc = 27e9; # Carrier frequency of the SC
fc_bh_sc = 73e9; # Carrier frequency for the wireless BH in SC
usr_ht = 1.5; # User height
simulation_area = 1e6; #The area is in square meters; The shape is also a square 
MCBS_intersite = 200; #Intersite distance for Macro BS
SCBS_intersite = 20; #Intersite distance for Small cell BS
#SCBS_per_MCBS = np.random.randint(3,10,size=1); # Create a set for number of small cells per macro BS
UE_density_eMBB = 10; #Number of eMBB devices per TRP
UE_density_URLLC = (4000/1e6); #Number of URLLC devices per sq. m
UE_density_mMTC = 24000; #Number of mMTC devices per Macro 
#UE_density_eMBB = 24000; #Number of URLLC UEs per Macro BS
ant_height_MCBS = 25; # Height above rooftop in meters
ant_gain_MCBS = 17; # dBi gain 
max_tnsmtpow_MCBS = 49; # dBm gain per band (in 20 MHz)
transmit_gain_sc = 30; # This value is in dBi
receiver_gain = 15; # This value is in dBi
transmit_power = 30; # This value is in dBm
other_losses = 20; # This is in dB (due to cable losses, penetration, reflection, etc. )
sc_bw = 1e9; # 1 GHz bandwidth for the small cells
mc_bw = 20*1e6; # 20 MHz bandwidth for the LTE macro cells
N = -174; # This is the noise spectral density in dbm/Hz
min_num_hops = 1; # If a local breakout exists
max_num_hops = 4; # Maximum hops to the IMS core
wl_bh_bp = 0.25*MCBS_intersite; # This is the distance beyond which a wired backhaul should be used (Can be changed later to the specifications)   
