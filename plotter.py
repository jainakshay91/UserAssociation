# ============================== #
# Plotting and Display Functions #
# ============================== #

# This file includes functions for plotting. It reduces the clutter in the main function file. 

# =============================
# Import the necessary binaries
# =============================

import matplotlib.pyplot as plt
import scipy
from scipy.stats import norm
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

# =====================================
# Geographical Heatmap creator Function
# =====================================

def hmap_creator(usr_lcs, mc_lcs, sc_lcs, rate_mat, np, scn):
	f,ax = plt.subplots()
	ax = plt.gca()
	
	s1, = ax.plot(usr_lcs[:,0], usr_lcs[:,1], "r+") # Plot the User locations
	s2, = ax.plot(mc_lcs[:,0],mc_lcs[:,1],"b*") # Plot the macro cell locations
	s3, = ax.plot(sc_lcs[:,0],sc_lcs[:,1],"g*") # Plot the small cell locations
	
	# Create the color range
	range_rate = np.arange(np.amin(rate_mat),np.amax(rate_mat),(np.amax(rate_mat)-np.amin(rate_mat))/7) # These are rate bands for the circular colors
	color_range = ["ko","go","bo","ro","mo","co","yo"] # These are the color codes
	circle_size = np.arange(6,13,1) # Circle size range
	# color_sel = [] # Empty list to hold the color code and circle sizes
	# # # Establish rate based circles on the plot 
	for i in range(usr_lcs.shape[0]):
		if rate_mat[i] >= range_rate[0] and rate_mat[i] < range_rate[1]:
			#ax.plot(usr_lcs[i,0],usr_lcs[i,1], color_range[0], markersize=circle_size[0], fillstyle='none')
			s4, = ax.plot(usr_lcs[i,0],usr_lcs[i,1], color_range[0], markersize=circle_size[0])
						
		elif rate_mat[i] >= range_rate[1] and rate_mat[i] < range_rate[2]:
			#ax.plot(usr_lcs[i,0],usr_lcs[i,1], color_range[1], markersize=circle_size[1], fillstyle='none')
			s5, = ax.plot(usr_lcs[i,0],usr_lcs[i,1], color_range[1], markersize=circle_size[1])
			
		elif rate_mat[i] >= range_rate[2] and rate_mat[i] < range_rate[3]:
			#ax.plot(usr_lcs[i,0],usr_lcs[i,1], color_range[2], markersize=circle_size[2], fillstyle='none')
			s6, = ax.plot(usr_lcs[i,0],usr_lcs[i,1], color_range[2], markersize=circle_size[2])
			
		elif rate_mat[i] >= range_rate[3] and rate_mat[i] < range_rate[4]:
			#ax.plot(usr_lcs[i,0],usr_lcs[i,1], color_range[3], markersize=circle_size[3], fillstyle='none')
			s7, = ax.plot(usr_lcs[i,0],usr_lcs[i,1], color_range[3], markersize=circle_size[3])
			
		elif rate_mat[i] >= range_rate[4] and rate_mat[i] < range_rate[5]:
			#ax.plot(usr_lcs[i,0],usr_lcs[i,1], color_range[4], markersize=circle_size[4], fillstyle='none')
			s8, = ax.plot(usr_lcs[i,0],usr_lcs[i,1], color_range[4], markersize=circle_size[4])
			
		elif rate_mat[i] >= range_rate[5] and rate_mat[i] < range_rate[6]:
			#ax.plot(usr_lcs[i,0],usr_lcs[i,1], color_range[5], markersize=circle_size[5], fillstyle='none')
			s9, = ax.plot(usr_lcs[i,0],usr_lcs[i,1], color_range[5], markersize=circle_size[5])
			
		else:
			#ax.plot(usr_lcs[i,0],usr_lcs[i,1], color_range[6], markersize=circle_size[6], fillstyle='none')
			s10, = ax.plot(usr_lcs[i,0],usr_lcs[i,1], color_range[6], markersize=circle_size[6])

	legend_cols = [s1, s2, s3, s4, s5, s6, s7, s8, s9, s10]
	plt.legend(legend_cols,["Users", "Macro Cells", "Small Cells", str(format(range_rate[0],'0.6e'))+'--'+str(format(range_rate[1],'0.6e')),str(format(range_rate[1],'.6e'))+'--'+str(format(range_rate[2],'.6e')),str(format(range_rate[2],'0.6e'))+'--'+str(format(range_rate[3],'0.6e')),str(format(range_rate[3],'0.6e'))+'--'+str(format(range_rate[4],'0.6e')),str(format(range_rate[4],'0.6e'))+'--'+str(format(range_rate[5],'0.6e')),str(format(range_rate[5],'0.6e'))+'--'+str(format(range_rate[6],'0.6e'))],loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol = 5)	
	plt.title("Heatmap of Individual User Data Rates (DC+BHCAP+LAT Scenario)")
	plt.show()

# =============================
# Histogram and Scatter Plotter
# =============================

def hist_plotter(rate_matrix_DC, rate_matrix_SA, rate_matrix_DC_BHCAP, rate_matrix_SA_BHCAP, rate_matrix_SA_LAT, rate_matrix_SA_MRT, rate_matrix_DC_MRT, rate_matrix_DC_LAT, rate_matrix_SA_MRT_LAT, rate_matrix_DC_MRT_LAT, rate_matrix_SA_BHCAP_LAT, rate_matrix_DC_BHCAP_LAT, np,scn):
	y1,binEdges1 = np.histogram(rate_matrix_DC,bins=200)
	y2,binEdges2 = np.histogram(rate_matrix_SA,bins=200)
	y3,binEdges3 = np.histogram(rate_matrix_DC_BHCAP,bins=200)
	y4,binEdges4 = np.histogram(rate_matrix_SA_BHCAP,bins=200)
	y5,binEdges5 = np.histogram(rate_matrix_DC_MRT,bins=200)
	y6,binEdges6 = np.histogram(rate_matrix_DC_LAT,bins=200)
	y7,binEdges7 = np.histogram(rate_matrix_DC_BHCAP_LAT,bins=200)
	y8,binEdges8 = np.histogram(rate_matrix_DC_MRT_LAT,bins=200)
	y9,binEdges9 = np.histogram(rate_matrix_SA_MRT,bins=200)
	y10,binEdges10 = np.histogram(rate_matrix_SA_LAT,bins=200)
	y11,binEdges11 = np.histogram(rate_matrix_SA_BHCAP_LAT,bins=200)
	y12,binEdges12 = np.histogram(rate_matrix_SA_MRT_LAT,bins=200)
	
	bincenters1 = 0.5*(binEdges1[1:]+binEdges1[:-1])
	bincenters2 = 0.5*(binEdges2[1:]+binEdges2[:-1])
	bincenters3 = 0.5*(binEdges3[1:]+binEdges3[:-1])
	bincenters4 = 0.5*(binEdges4[1:]+binEdges4[:-1])
	bincenters5 = 0.5*(binEdges5[1:]+binEdges5[:-1])
	bincenters6 = 0.5*(binEdges6[1:]+binEdges6[:-1])
	bincenters7 = 0.5*(binEdges7[1:]+binEdges7[:-1])
	bincenters8 = 0.5*(binEdges8[1:]+binEdges8[:-1])
	bincenters9 = 0.5*(binEdges9[1:]+binEdges9[:-1])
	bincenters10 = 0.5*(binEdges10[1:]+binEdges10[:-1])
	bincenters11 = 0.5*(binEdges11[1:]+binEdges11[:-1])
	bincenters12 = 0.5*(binEdges12[1:]+binEdges12[:-1])
	

	plt.plot(bincenters1,y1,'r-o',fillstyle='none')
	#plt.plot(bincenters2,y2,'b-o',fillstyle='none')
	#plt.plot(bincenters3,y3,'b-o',fillstyle='none')
	plt.plot(bincenters5,y5,'b-*',fillstyle='none')
	plt.plot(bincenters1[y1.tolist().index(np.amax(y1))],np.amax(y1),'ko')
	plt.plot(np.ones((np.arange(0,np.amax(y1)).shape[0]+1,1))*bincenters1[y1.tolist().index(np.amax(y1))],np.arange(0,np.amax(y1)+1),'k--')
	#plt.plot(bincenters2[y2.tolist().index(np.amax(y2))],np.amax(y2),'go')
	#plt.plot(np.ones((np.arange(0,np.amax(y2)).shape[0]+1,1))*bincenters2[y2.tolist().index(np.amax(y2))],np.arange(0,np.amax(y2)+1),'g--')
	#plt.plot(bincenters3[y3.tolist().index(np.amax(y3))],np.amax(y3),'go')
	#plt.plot(np.ones((np.arange(0,np.amax(y3)).shape[0],1))*bincenters3[y3.tolist().index(np.amax(y3))],np.arange(0,np.amax(y3)),'g--')
	#plt.plot(bincenters7[y7.tolist().index(np.amax(y7))],np.amax(y7),'go')
	#plt.plot(np.ones((np.arange(0,np.amax(y7)).shape[0],1))*bincenters7[y7.tolist().index(np.amax(y7))],np.arange(0,np.amax(y7)),'g--')
	plt.plot(bincenters5[y5.tolist().index(np.amax(y5))],np.amax(y5),'go')
	plt.plot(np.ones((np.arange(0,np.amax(y5)).shape[0],1))*bincenters5[y5.tolist().index(np.amax(y5))],np.arange(0,np.amax(y5)),'g--')
		

	#plt.plot(bincenters3,y3,'k--')
	#lt.plot(bincenters4,y4,'g-o')
	#plt.plot(bincenters5,y5,'m--')
	#plt.plot(bincenters6,y6,'c--')
	#plt.plot(bincenters7,y7,'r-*')
	#plt.plot(bincenters8,y8,'b:')
	#plt.plot(bincenters9,y9,'k-*')
	#plt.plot(bincenters10,y10,'g:')
	#plt.plot(bincenters11,y11,'m-*')
	#plt.plot(bincenters12,y12,'c:')
	
	plt.legend(["Dual Connectivity","Dual Connectivity + Minimum Rate"])
	plt.xlabel('Throughput(bps)')
	plt.ylabel('Number of Users')
	plt.title('Histogram of User Distribution')
	plt.grid()
	plt.show()


def scatter_plotter(rate_matrix_DC,rate_matrix_SA,np,scn):
	plt.scatter(np.arange(1,len(rate_matrix_DC) +1),rate_matrix_DC,alpha=0.5)
	plt.plot(np.arange(1,len(rate_matrix_DC) +1),np.ones((np.arange(1,len(rate_matrix_DC) +1).shape[0],1))*1e8, 'r--')
	plt.xlabel('Users')
	plt.ylabel('Throughput (in bps)')
	plt.title("Data Rate Scatter Plot of Users (DC Scenario)")
	plt.legend(["100 Mbps"])
	plt.show()

# ========================================
# Accepted and Rejected User Visualization
# ========================================

def accepted_user_plotter(accepted_usr_list_baseline, accepted_usr_list_SA, accepted_usr_list_DC, accepted_usr_list_DC_MRT,accepted_usr_list_DC_BHCAP,accepted_usr_list_DC_LAT,accepted_usr_list_DC_BHCAP_LAT,accepted_usr_list_SA_MRT,accepted_usr_list_SA_LAT,accepted_usr_list_SA_BHCAP,accepted_usr_list_SA_BHCAP_LAT,accepted_usr_list_SA_MRT_LAT,accepted_usr_list_DC_MRT_LAT,np,scn):
	actual_user_list = [500,600,700,800,900,1000]
	labels = ['500','600','700','800','900','1000']
	baseline_ar = -1*accepted_usr_list_baseline + actual_user_list;
	SA_ar = -1*accepted_usr_list_SA + actual_user_list
	DC_ar = -1*accepted_usr_list_DC + actual_user_list
	DC_MRT_ar = -1*accepted_usr_list_DC_MRT + actual_user_list
	DC_BHCAP_ar = -1*accepted_usr_list_DC_BHCAP + actual_user_list
	DC_LAT_ar = -1*accepted_usr_list_DC_LAT + actual_user_list
	SA_MRT_ar = -1*accepted_usr_list_SA_MRT + actual_user_list
	SA_LAT_ar = -1*accepted_usr_list_SA_LAT + actual_user_list
	SA_BHCAP_ar = -1*accepted_usr_list_SA_BHCAP + actual_user_list
	SA_BHCAP_LAT_ar = -1*accepted_usr_list_SA_BHCAP_LAT + actual_user_list
	SA_MRT_LAT_ar = -1*accepted_usr_list_SA_MRT_LAT + actual_user_list
	DC_MRT_LAT_ar = -1*accepted_usr_list_DC_MRT_LAT + actual_user_list
	
	x = np.arange(len(labels)) # The label locations
	width = 0.15 # Width of the bars
	f,ax = plt.subplots()
	r1 = ax.bar(x - width/2, baseline_ar, width, label='Baseline')
	r2 = ax.bar(x - 5*width/12, SA_ar, width, label='SA')
	r3 = ax.bar(x - width/3, DC_ar, width, label='DC')
	r4 = ax.bar(x - width/4, DC_MRT_ar, width, label='DC+MRT')
	r5 = ax.bar(x - width/6, DC_BHCAP_ar, width, label='DC+BHCAP')
	r6 = ax.bar(x - width/12, DC_LAT_ar, width, label='DC+LAT')
	r7 = ax.bar(x + width/12, SA_MRT_ar, width, label='SA+MRT')
	r8 = ax.bar(x + width/6, SA_LAT_ar, width, label='SA+LAT')
	r9 = ax.bar(x + width/4, SA_BHCAP_ar, width, label='SA+BHCAP')
	r1 = ax.bar(x + width/3, SA_BHCAP_LAT_ar, width, label='SA+BHCAP+LAT')
	r1 = ax.bar(x + 5*width/12, SA_MRT_LAT_ar, width, label='SA+MRT+LAT')
	r1 = ax.bar(x + width/2, DC_MRT_LAT_ar, width, label='DC+MRT+LAT')

	ax.set_ylabel('Number of Unaccepted users')
	ax.set_title("User Acceptance")
	ax.set_xticks(x)
	ax.set_xticklabels(labels)
	ax.legend()
	f.tight_layout()
	plt.show()
#plt.plot(usr_lcs[0], usr_lcs[1],'k+');
#plt.plot(macro_cell_locations[:,0], macro_cell_locations[:,1],'rs'); # Plot the macro cells
#for j in range(0,macro_cell_locations.shape[0]):
#   print_element = locs_SCBS[j]; #Accessing the numpy array of SC locations corresponding to the Macro Cell    
#   plt.plot(print_element[:,0], print_element[:,1], 'b*'); # Plot the small cells
# plt.plot(usr_loc_eMBB[:,0],usr_loc_eMBB[:,1],'k+')
# plt.plot(usr_loc_URLLC[:,0],usr_loc_URLLC[:,1],'cs')
# #plt.plot(usr_loc_mMTC[:,0],usr_loc_mMTC[:,1],'go')

	#dist_names = ['rayleigh', 'rice','norm','expon']
	# val_dist =[]
	# # ===> Perform the KS Test for distribution fit

	# dist_results = []
	# for i in range(len(dist_names)):
	# 	dist = getattr(scipy.stats, dist_names[i])
	# 	param = dist.fit(y)
	# 	D,p = scipy.stats.kstest(y,dist_names[i],args=param)
	# 	dist_results.append((dist_names[i],p))

	# dist_selected, p = (max(dist_results,key=lambda item:item[1]))
	# dist_fin = getattr(scipy.stats, dist_selected)
	# val_dist = dist_fin.rvs(*(dist_fin.fit(y))[:-2], loc = param[-2], scale=param[-1], size=len(y))
	# plt.hist(val_dist, alpha=0.5)
	# plt.hist(y, alpha=0.5)
	# plt.legend(loc='upper right')