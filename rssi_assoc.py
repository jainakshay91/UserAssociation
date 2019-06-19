#!/usr/bin/env python

# ==============================
# Import the Necessary Libraries
# ==============================

import collections

# =============================
# Baseline Association Function
# =============================

def baseline_assoc(SNR_eMBB, SNR_mMTC, sinr_eMBB, sinr_mMTC, BHCAP_SC, BHCAP_MC, num_SCBS, num_MCBS, np, scn):


	# ==================================
	# Compute the Highest Received Power

	#print RX_eMBB.shape
	#print sinr_eMBB.shape
	SNR_eMBB = np.where(np.isnan(SNR_eMBB) == True, -300 ,SNR_eMBB); # Taking care of Nan values

	SNR_max_eMBB = np.flip(np.sort(SNR_eMBB, axis = 1),axis=1); # Sorted Matrix for eMBB 
	idx_max_eMBB = np.flip(np.argsort(SNR_eMBB, axis = 1),axis=1); # Maximum received power for each eMBB application
	
	#idx_max_mMTC = np.argsort(RX_mMTC, axis = 1); # Maximum received power for each mMTC application 
	#rx_max_mMTC = np.sort(RX_mMTC, axis = 1); # Sorted Matrix for mMTC
	#sinr_max_eMBB = np.empty((sinr_eMBB.shape[0],1))
	#for i in range(0,sinr_eMBB.shape[0]):
		#if sinr_eMBB[i,idx_max_eMBB[i,0]] == pad_value:
	#	sinr_max_eMBB[i] = sinr_eMBB[i,idx_max_eMBB[i,0]]; # Capture the SINR from the BS which is the closest for the eMBB
	
	sinr_max_eMBB = np.empty((sinr_eMBB.shape[0],sinr_eMBB.shape[1]))
	for i in range(sinr_max_eMBB.shape[0]):
		for j in range(sinr_max_eMBB.shape[1]):
			sinr_max_eMBB[i,j] = sinr_eMBB[i,idx_max_eMBB[i,j]]; # Developing the SINR matrix

	# ===> Establish the Resource Pool Variables

	num_BS = num_MCBS + num_SCBS; # We calculate the total number of Base stations

	access_bw = np.empty((num_BS,1)); # Initializing the Access BW resource
	bcap_sc = np.ones((num_SCBS,1))*BHCAP_SC; # Small Cell Backhaul Capacity variable
	bcap_mc = np.ones((num_MCBS,1))*BHCAP_MC; # Macro Cell Backhaul Capacity variable

	access_bw[0:num_SCBS,0] = scn.sc_bw; # Small Cell Bandwidth
	access_bw[num_SCBS:num_BS,0] = scn.eNB_bw; # Macro Cell Bandwidth


	#print "Received Power ==>"
	#print RX_eMBB
	#print "Max Received Power ====>"
	#print rx_max_eMBB[:,0]
	#print "SINR ====>"
	#print sinr_max_eMBB[:,0]
	#for j in range(0,sinr_mMTC.shape[0]):
	#	sinr_max_mMTC[i] = sinr_mMTC[i,idx_max_mMTC[i,0]]; # Capture the SINR from the BS which is the closest for the mMTC
	
	counter_eMBB = collections.Counter(idx_max_eMBB[:,0]); # Counts the frequency of occurence of each BS as being the strongest BS for eMBB type
	#counter_mMTC = collections.Counter(idx_max_mMTC[:,0]); # Counts the frequency of occurence of each BS as being the strongest BS for mMTC type

	#Data_Rate_eMBB_scbw = scn.usr_scbw*np.log2(1+10**(SNR_max_eMBB[:,0]/10)); # Data rate at each eMBB application when attaching to the AP with best RSSI
	#Data_Rate_eMBB_fscbw = scn.sc_bw*np.log2(1+10**(SNR_max_eMBB[:,0]/10)); # Data rate with 1GHz BW for each eMBB application
	#Data_Rate_mMTC = scn.mMTC_bw*np.log2(1+10**(rx_max_eMBB[:,0]/10)); # mMTC Data rate with standard data rate 

	#print Data_Rate_eMBB_scbw

	Data_rate_sinr_eMBB_scbw= np.empty((sinr_max_eMBB.shape[0], sinr_max_eMBB.shape[1])); # Initialize the Data Rate matrix

	for i in range(sinr_max_eMBB.shape[0]):
		for j in range(sinr_max_eMBB.shape[1]):
			if j <= num_SCBS:
				Data_rate_sinr_eMBB_scbw[i,j] = scn.usr_scbw*np.log2(1+10**(sinr_max_eMBB[i,j]/10)); # SINR based Data Rate
			else:
				Data_rate_sinr_eMBB_scbw[i,j] = scn.mc_bw*np.log2(1+10**(sinr_max_eMBB[i,j]/10)); # SINR based Data Rate

	#Data_rate_sinr_eMBB_fscbw = scn.sc_bw*np.log2(1+10**(sinr_max_eMBB/10)); # SINR based Data Rate for full bandwidth
	#Data_rate_sinr_mMTC = scn.mMTC_bw*np.log2(1+10**(sinr_max_mMTC/10)); # SINR based Data Rate for mMTC

	#print Data_rate_sinr_eMBB_scbw
	
	# ================================
	# Baseline Association Methodology

	Tot_Datarate = 0; # Total Data Rate 
	Accepted_Users = 0; # Number of Accepted Users

	for i in range(sinr_eMBB.shape[0]):
		for j in range(sinr_eMBB.shape[1]):
			if Data_rate_sinr_eMBB_scbw[i,j] >= scn.eMBB_minrate:
				if idx_max_eMBB[i,j] <= num_SCBS: # If its a Small Cell
					if access_bw[idx_max_eMBB[i,j],0] >= scn.usr_scbw and bcap_sc[idx_max_eMBB[i,j],0] >= scn.eMBB_minrate:
						Tot_Datarate = Tot_Datarate + Data_rate_sinr_eMBB_scbw[i,j]; # Update the Total network throughput
						access_bw[idx_max_eMBB[i,j],0] = access_bw[idx_max_eMBB[i,j],0] - scn.usr_scbw; # Update the available access bw 
						bcap_sc[idx_max_eMBB[i,j],0] = bcap_sc[idx_max_eMBB[i,j],0] - Data_rate_sinr_eMBB_scbw[i,j]; # Update the available bhaul capacity
						Accepted_Users = Accepted_Users + 1; # Increment the number of users that have been accepted into the system
				  	else:
				  		pass
				elif idx_max_eMBB[i,j] > num_SCBS: # If its a macro cell
					if access_bw[idx_max_eMBB[i,j],0] >= scn.mc_bw and bcap_mc[idx_max_eMBB[i,j] - num_SCBS,0] >= scn.eMBB_minrate:
						Tot_Datarate = Tot_Datarate + Data_rate_sinr_eMBB_scbw[i,j]; # Update the Total network throughput
						access_bw[idx_max_eMBB[i,j],0] = access_bw[idx_max_eMBB[i,j],0] - scn.mc_bw; # Update the Available access bw
						bcap_mc[idx_max_eMBB[i,j] - num_SCBS,0] = bcap_mc[idx_max_eMBB[i,j] - num_SCBS,0] - Data_rate_sinr_eMBB_scbw[i,j];  
						Accepted_Users = Accepted_Users + 1; # Increment the number of users that have been accepted into the system 
				break
			else:
				continue

	#  access_bw >= scn.usr_scbw and ((bcap_sc >= scn.eMBB_minrate and idx_max_eMBB[i,j]<=num_SCBS) or (bcap_mc >= scn.eMBB_minrate and idx_max_eMBB[i,j]>num_SCBS)):
				
	print "Generated the Baseline Data"

	#return Data_Rate_eMBB_scbw, Data_Rate_eMBB_fscbw, Data_Rate_mMTC, Data_rate_sinr_eMBB_scbw, Data_rate_sinr_eMBB_fscbw, Data_rate_sinr_mMTC
	return Tot_Datarate, Accepted_Users


