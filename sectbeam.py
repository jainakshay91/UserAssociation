#!/usr/bin/env python

import csvsaver
# This file contains the functions where the beamforming and Macro Cell sectorization has been performed for 
# interference calculation purposes

# ====================================================
# MCBS Sectorization and Interference Compute Function
# ====================================================

def MCBS_sectorizer(np,scn,num_mcbs,mcbs_locs,usr_locs):

	#===> Compute the angular displacement for determining the influence of an AP
	theta_diff = np.zeros((usr_locs.shape[0],1)) # This matrix holds the angular displacement of each user with a given AP
	sector_mat = np.zeros((usr_locs.shape[0], num_mcbs), dtype=int) # This matrix holds the sector number to which a given user belongs to for a given MC AP
	theta_mat =  np.zeros((usr_locs.shape[0], num_mcbs))
	for i in range(num_mcbs):
		x_diff = usr_locs[:,0] - mcbs_locs[i,0] # Compute the Perpendicular
		y_diff = usr_locs[:,1] - mcbs_locs[i,1] # Compute the base
		theta_diff[:,0] = np.degrees(np.arctan2(y_diff,x_diff)) # Computes the vector of angles each UE makes with a MCBS
		theta_mat[:,i] = theta_diff[:,0]
		# sector_mat[np.nonzero(theta_diff<=60 or theta_diff>300),i] = 1 # Sector 1 for MCBS i is between +60 and -60
		# sector_mat[np.nonzero(theta_diff>60 or theta_diff<=180),i] = 2 # Sector 1 for MCBS i is between +60 and 180
		# sector_mat[np.nonzero(theta_diff>180 or theta_diff<=300),i] = 3 # Sector 1 for MCBS i is between +180 and +300
		sector_mat[np.nonzero(np.all(np.hstack(((theta_diff<=60)*1,(theta_diff>-60)*1)), axis = 1)),i] = 1 # Sector 1 for MCBS i is between +60 and -60
		sector_mat[np.nonzero(np.all(np.hstack(((theta_diff>60)*1,(theta_diff<=180)*1)), axis = 1)),i] = 2 # Sector 2 for MCBS i is between +60 and 180
		sector_mat[np.nonzero(np.all(np.hstack(((theta_diff>-180)*1,(theta_diff<=-60)*1)), axis = 1)),i] = 3 # Sector 3 for MCBS i is between +180 and 300
	csvsaver.csvsaver(sector_mat,[], 'sector_mat.csv')
	csvsaver.csvsaver(usr_locs,[],'USERlocs.csv')
	csvsaver.csvsaver(mcbs_locs, [], 'MCBSlocs.csv')
	csvsaver.csvsaver(theta_mat, [], 'Theta.csv')
	return sector_mat
	