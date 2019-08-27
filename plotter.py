# ============================== #
# Plotting and Display Functions #
# ============================== #

# This file includes functions for plotting. It reduces the clutter in the main function file. 

# =============================
# Import the necessary binaries
# =============================

import matplotlib.pyplot as plt
import seaborn as sns
import os
from mpl_toolkits.mplot3d import Axes3D


# =======
# Plotter
# =======

def plotter(typ_plt, x_val, y_val, tick_space_x, tick_space_y, rtn_flag_x, rtn_angle_x, rtn_flag_y, rtn_angle_y, grid_flag, grid_type, grid_ax_type, title_flag, title_name, np):
	if typ_plt == 'dashline':
		plt.plot(x_val, y_val, 'r--');
		if rtn_flag_x:
			plt.xticks(np.arange(min(x_val),max(x_val),tick_space_x),rotation=rtn_angle_x);
		else:
			plt.xticks(np.arange(min(x_val),max(x_val),tick_space_x));
		if rtn_flag_y:
			plt.yticks(np.arange(min(y_val),max(y_val),tick_space_y),rotation=rtn_angle_y);
		else:
			plt.yticks(np.arange(min(y_val),max(y_val),tick_space_y));
		if grid_flag:
			plt.grid(which=grid_type,axis=grid_ax_type);
		if title_flag:
			plt.title(title_name);

	if typ_plt == 'heatmap':
		ax = sns.heatmap(x_val); 
		ax.set_title("SINR heatmap")
		ax.set_xlabel("Small Cell Access Points in consideration")
		ax.set_ylabel("Number of eMBB users")
		
		
	plt.show() 

# ====================================
# Plotting Function for Optimized data
# ====================================

def optimizer_plotter(data_plt):
	#sns.set_style("whitegrid", {'axes.grid' : False})
	f = plt.figure(figsize=(12,10))
	ax = sns.heatmap(data_plt); 
	#x = Axes3D(f)
	#ax.scatter(data_plt.shape[0], data_plt.shape[1], data_plt)
	#g = plt.figure(2)
	#ax1 = sns.heatmap(data_plt[:,:,1]);

	#h = plt.figure(3)
	#ax2 = sns.heatmap(data_plt[:,:,2]);   
	ax.set_title("User Data Rate Heatmap")
	ax.set_xlabel("Access Points")
	ax.set_ylabel("eMBB Users")
	#ax.set_zlabel("Data Rate")
	plt.savefig(os.getcwd()+"/CircularDeploy_MCSC.png")
	plt.show()	



#plt.plot(usr_lcs[0], usr_lcs[1],'k+');
#plt.plot(macro_cell_locations[:,0], macro_cell_locations[:,1],'rs'); # Plot the macro cells
#for j in range(0,macro_cell_locations.shape[0]):
#   print_element = locs_SCBS[j]; #Accessing the numpy array of SC locations corresponding to the Macro Cell    
#   plt.plot(print_element[:,0], print_element[:,1], 'b*'); # Plot the small cells
# plt.plot(usr_loc_eMBB[:,0],usr_loc_eMBB[:,1],'k+')
# plt.plot(usr_loc_URLLC[:,0],usr_loc_URLLC[:,1],'cs')
# #plt.plot(usr_loc_mMTC[:,0],usr_loc_mMTC[:,1],'go')

