#!/usr/bin/env python

# ==============================
# Import the Necessary Libraries
# ==============================

import collections

# =============================
# Baseline Association Function
# =============================

def baseline_assoc(SNR_eMBB, SNR_mMTC, sinr_eMBB, sinr_mMTC, np, scn):

	# ==================================
	# Compute the Highest Received Power

	#print RX_eMBB.shape
	#print sinr_eMBB.shape
	SNR_eMBB = np.where(np.isnan(SNR_eMBB) == True, -300 ,SNR_eMBB); # Taking care of Nan values

	SNR_max_eMBB = np.flip(np.sort(SNR_eMBB, axis = 1),axis=1); # Sorted Matrix for eMBB 
	idx_max_eMBB = np.flip(np.argsort(SNR_eMBB, axis = 1),axis=1); # Maximum received power for each eMBB application
	#idx_max_mMTC = np.argsort(RX_mMTC, axis = 1); # Maximum received power for each mMTC application 
	#rx_max_mMTC = np.sort(RX_mMTC, axis = 1); # Sorted Matrix for mMTC
	sinr_max_eMBB = np.empty((sinr_eMBB.shape[0],1))
	for i in range(0,sinr_eMBB.shape[0]):
		#if sinr_eMBB[i,idx_max_eMBB[i,0]] == pad_value:
		sinr_max_eMBB[i] = sinr_eMBB[i,idx_max_eMBB[i,0]]; # Capture the SINR from the BS which is the closest for the eMBB
		

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

	Data_Rate_eMBB_scbw = scn.usr_scbw*np.log2(1+10**(SNR_max_eMBB[:,0]/10)); # Data rate at each eMBB application when attaching to the AP with best RSSI
	Data_Rate_eMBB_fscbw = scn.sc_bw*np.log2(1+10**(SNR_max_eMBB[:,0]/10)); # Data rate with 1GHz BW for each eMBB application
	#Data_Rate_mMTC = scn.mMTC_bw*np.log2(1+10**(rx_max_eMBB[:,0]/10)); # mMTC Data rate with standard data rate 

	print Data_Rate_eMBB_scbw

	Data_rate_sinr_eMBB_scbw = scn.usr_scbw*np.log2(1+10**(sinr_max_eMBB/10)); # SINR based Data Rate
	Data_rate_sinr_eMBB_fscbw = scn.sc_bw*np.log2(1+10**(sinr_max_eMBB/10)); # SINR based Data Rate for full bandwidth
	#Data_rate_sinr_mMTC = scn.mMTC_bw*np.log2(1+10**(sinr_max_mMTC/10)); # SINR based Data Rate for mMTC

	print Data_rate_sinr_eMBB_scbw
	
	print "Generated the Baseline Data"

	#return Data_Rate_eMBB_scbw, Data_Rate_eMBB_fscbw, Data_Rate_mMTC, Data_rate_sinr_eMBB_scbw, Data_rate_sinr_eMBB_fscbw, Data_rate_sinr_mMTC
	return Data_Rate_eMBB_scbw, Data_Rate_eMBB_fscbw, Data_rate_sinr_eMBB_scbw, Data_rate_sinr_eMBB_fscbw



