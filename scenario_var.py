# ========================================
# This file defines the Scenario Variables
# ========================================

class scenario_var:
	
	c = 3e8; # Speed of light in m/s
	fc_mc = 3.55e9; # Carrier frequency of the MC (Qualcomm white paper)
	fc_sc = 27e9; # Carrier frequency of the SC
	fc_bh_sc = 73e9; # Carrier frequency for the wireless BH in SC
	usr_ht = 1.5; # User height
	bs_ht_sc = 10; # Small cell height
	bs_ht_mc = 25; # Macro cell height
	simulation_area = 0.36*1e6; #The area is in square meters; The shape is also a square 
	MCBS_intersite = 200; #Intersite distance for Macro BS
	SCBS_intersite = 20; #Intersite distance for Small cell BS
	#SCBS_per_MCBS = np.random.randint(3,10,size=1); # Create a set for number of small cells per macro BS
	#UE_density_eMBB = 10; #Number of eMBB devices per TRP (5GPPP white paper)
	UE_density_URLLC = (4000/1e6); #Number of URLLC devices per sq. m (5GPPP white paper)
	UE_density_mMTC = 24000; #Number of mMTC devices per Macro (5GPPP white paper)
	#UE_density_eMBB = 24000; #Number of URLLC UEs per Macro BS
	ant_gain_MCBS = 17; # dBi gain [5G PPP doc]
	max_tnsmtpow_MCBS = 49; # dBm gain per band (in 20 MHz) [5G PPP doc]
	transmit_gain_sc = 30; # This value is in dBi [5G PPP doc]
	receiver_gain = 14; # This value is in dBi for UE-SC network [KTH Paper]
	rx_mc_gain = 0; # This value is in dB for UE-MC network [5G PPP doc] 
	transmit_power = 23; # This value is in dBm [5G PPP doc]
	#mMTC_tnsmt_power = 20; # This value is in dBm 
	#other_losses = 20; # This is in dB (due to cable losses, penetration, reflection, etc. )
	sc_bw = 1e9; # 1 GHz bandwidth for the small cells
	mc_bw = 20*1e6; # 20 MHz bandwidth for the UE on LTE macro cells
	N = -174; # This is the noise spectral density in dbm/Hz
	min_num_hops = 1; # If a local breakout exists
	max_num_hops = 4; # Maximum hops to the IMS core
	wl_bh_bp = 0.25*MCBS_intersite; # This is the distance beyond which a wired backhaul should be used (Can be changed later to the specifications)   
	num_appl_types = 3; # We current have a broad category of 3 application types
	max_num_appl_UE = 3; # Maximum number of applications on any given UE
	num_users_min = 150; # Minimum number of users in the scenario
	num_users_max = 300; # Maximum number of users in the scenario
	user_steps_siml = 25; # For the simulation we increase users in steps of 50
	eMBB_minrate = 1e8; # 100 mbps minimum required data rate for most eMBB applications
	fib_BH_capacity = 1e9; # 1Gbps of fibre backhaul capacity (Find a reference for this)
	fib_BH_MC_capacity = 10e9; # 10Gbps of fiber backhaul for MCs
	wl_link_delay = 1*1e-3; # 1 ms link delay for the wireless link [Mona Jaber Paper] 
	wrd_link_delay = 1*1e-3; # 1-7 ms link delay for the wired link [Mona Jaber Paper]
	eMBB_latency_req = 3*1e-3; # 3 ms link latency requirement for the eMBB applications
	avg_fib_BH_capacity = 1.379*1e9; # This is the average backhaul usage for SC
	avg_fib_BH_MC_capacity = 9.325*1e9; # This is the average backhaul usage for MC 
	perct_incr = [0.1, 0.2, 0.5, 1.0]; # This is the various percentages of increment of BH 
	MCMC_iter = 100; # Number of Monte Carlo Iterations
	num_Subcarriers_MCBS = 1200; # LTE number of subcarriers
	num_Subcarriers_SCBS = 3300; # 5G NR number of subcarriers 
	usr_scbw = 2*1e8; # 100 MHz bandwidth per user 
	mMTC_bw = 180*1e3; # Device Bandwidth (Guard Band Operations considered)
	mMTC_maxrate = 1e6; # Device data rate
	eNB_bw = 80*1e6; # Bandwidth for the Macro Cells (Qualcomm Media Release)
	mMTC_maxrate = [1e3, 1e4]; # Device data rate
	BW_SC = [50*1e6, 100*1e6, 200*1e6]; # Small cell bandwidth values (All values are in MHz)
	BW_MC = [1.5*1e6, 3*1e6, 5*1e6, 10*1e6, 20*1e6]; # Macro cell bandwidth values (All values are in MHz)
	beam_hpbw_rx = 45 # Assuming a HPBW of 45 degrees at the receiver (UE) [KTH Paper]
	#beam_hpbw_tx = 30 # Assuming a HPBW of 30 degrees at the transmitter (Small Cells)
 	
