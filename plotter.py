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
from matplotlib import cm
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import matplotlib.ticker as ticker

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

def hmap_creator(usr_lcs, mc_lcs, sc_lcs, rate_mat, connect_info_mat, np, scn):
	f,ax = plt.subplots()
	ax = plt.gca()
	#print usr_lcs.shape[0]
	#print rate_mat 

	s1, = ax.plot(usr_lcs[:,0], usr_lcs[:,1], "r*", markersize=12) # Plot the User locations
	s2, = ax.plot(mc_lcs[:,0],mc_lcs[:,1],"k^", markersize=12) # Plot the macro cell locations
	s3, = ax.plot(sc_lcs[:,0],sc_lcs[:,1],"g^", markersize=8) # Plot the small cell locations
	
	for i in range(connect_info_mat.shape[0]):
		for j in range(connect_info_mat.shape[1]):
			if connect_info_mat[i,j] == 1 and j<sc_lcs.shape[0]:
				ax.plot([usr_lcs[i,0],sc_lcs[j,0]],[usr_lcs[i,1],sc_lcs[j,1]],'c-')
			elif connect_info_mat[i,j] == 1 and j>=sc_lcs.shape[0]:
				ax.plot([usr_lcs[i,0],mc_lcs[j-sc_lcs.shape[0],0]],[usr_lcs[i,1],mc_lcs[j-sc_lcs.shape[0],1]],'m-')
	
	#Create the color range
	# range_rate = np.arange(np.amin(rate_mat),np.amax(rate_mat),(np.amax(rate_mat)-np.amin(rate_mat))/7) # These are rate bands for the circular colors
	# color_range = ['#ffa07a','m','b','r','#daa520','#b22222','#8b0000'] # These are the color codes
	# circle_size = np.arange(6,13,1) # Circle size range
	# # color_sel = [] # Empty list to hold the color code and circle sizes
	# # # # Establish rate based circles on the plot 
	# for i in range(usr_lcs.shape[0]):
	# 	if rate_mat[i] >= range_rate[0] and rate_mat[i] < range_rate[1]:
	# 		#ax.plot(usr_lcs[i,0],usr_lcs[i,1], color_range[0], markersize=circle_size[0], fillstyle='none')
	# 		s4, = ax.plot(usr_lcs[i,0],usr_lcs[i,1], marker = 'o', markeredgecolor = color_range[0], markerfacecolor = color_range[0], markersize=circle_size[0])
						
	# 	elif rate_mat[i] >= range_rate[1] and rate_mat[i] < range_rate[2]:
	# 		#ax.plot(usr_lcs[i,0],usr_lcs[i,1], color_range[1], markersize=circle_size[1], fillstyle='none')
	# 		s5, = ax.plot(usr_lcs[i,0],usr_lcs[i,1], marker = 'o', markeredgecolor = color_range[1], markerfacecolor = color_range[1], markersize=circle_size[1])
			
	# 	elif rate_mat[i] >= range_rate[2] and rate_mat[i] < range_rate[3]:
	# 		#ax.plot(usr_lcs[i,0],usr_lcs[i,1], color_range[2], markersize=circle_size[2], fillstyle='none')
	# 		s6, = ax.plot(usr_lcs[i,0],usr_lcs[i,1], marker = 'o', markeredgecolor = color_range[2], markerfacecolor = color_range[2], markersize=circle_size[2])
			
	# 	elif rate_mat[i] >= range_rate[3] and rate_mat[i] < range_rate[4]:
	# 		#ax.plot(usr_lcs[i,0],usr_lcs[i,1], color_range[3], markersize=circle_size[3], fillstyle='none')
	# 		s7, = ax.plot(usr_lcs[i,0],usr_lcs[i,1], marker = 'o',markeredgecolor = color_range[3], markerfacecolor = color_range[3], markersize=circle_size[3])
			
	# 	elif rate_mat[i] >= range_rate[4] and rate_mat[i] < range_rate[5]:
	# 		#ax.plot(usr_lcs[i,0],usr_lcs[i,1], color_range[4], markersize=circle_size[4], fillstyle='none')
	# 		s8, = ax.plot(usr_lcs[i,0],usr_lcs[i,1], marker = 'o', markeredgecolor = color_range[4], markerfacecolor = color_range[4], markersize=circle_size[4])
			
	# 	elif rate_mat[i] >= range_rate[5] and rate_mat[i] < range_rate[6]:
	# 		#ax.plot(usr_lcs[i,0],usr_lcs[i,1], color_range[5], markersize=circle_size[5], fillstyle='none')
	# 		s9, = ax.plot(usr_lcs[i,0],usr_lcs[i,1], marker = 'o', markeredgecolor = color_range[5], markerfacecolor = color_range[5], markersize=circle_size[5])
			
	# 	else:
	# 		#ax.plot(usr_lcs[i,0],usr_lcs[i,1], color_range[6], markersize=circle_size[6], fillstyle='none')
	# 		s10, = ax.plot(usr_lcs[i,0],usr_lcs[i,1], marker = 'o', markeredgecolor = color_range[6], markerfacecolor = color_range[6], markersize=circle_size[6])

	# #legend_cols = [ s2, s3, s4, s5, s6, s7, s8, s9, s10]
	# legend_cols = [ s2, s3, s4, s6, s7, s8, s10]
	# plt.legend(legend_cols,["Macro Cells", "Small Cells", str(format(range_rate[0],'0.6e'))+'--'+str(format(range_rate[1],'0.6e')),str(format(range_rate[1],'.6e'))+'--'+str(format(range_rate[2],'.6e')),str(format(range_rate[2],'0.6e'))+'--'+str(format(range_rate[3],'0.6e')),str(format(range_rate[3],'0.6e'))+'--'+str(format(range_rate[4],'0.6e')),str(format(range_rate[4],'0.6e'))+'--'+str(format(range_rate[5],'0.6e')),str(format(range_rate[5],'0.6e'))+'--'+str(format(range_rate[6],'0.6e'))],loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol = 4)	
	plt.title("Heatmap of Individual User Data Rates (DC+MRT Scenario)")
	plt.show()

# =============================
# Histogram and Scatter Plotter
# =============================

def hist_plotter(rate_matrix_DC, rate_matrix_SA, rate_matrix_DC_BHCAP, rate_matrix_SA_BHCAP, rate_matrix_SA_LAT, rate_matrix_SA_MRT, rate_matrix_DC_MRT, rate_matrix_DC_LAT, rate_matrix_SA_MRT_LAT, rate_matrix_DC_MRT_LAT, rate_matrix_SA_BHCAP_LAT, rate_matrix_DC_BHCAP_LAT, np,scn):
	#rc_mat_DC = np.array(rate_matrix_DC).reshape(())


	# y1,binEdges1 = np.histogram(rate_matrix_DC,bins=200)
	# y2,binEdges2 = np.histogram(rate_matrix_SA,bins=200)
	# y3,binEdges3 = np.histogram(rate_matrix_DC_BHCAP,bins=200)
	# y4,binEdges4 = np.histogram(rate_matrix_SA_BHCAP,bins=200)
	# y5,binEdges5 = np.histogram(rate_matrix_DC_MRT,bins=200)
	# y6,binEdges6 = np.histogram(rate_matrix_DC_LAT,bins=200)
	# y7,binEdges7 = np.histogram(rate_matrix_DC_BHCAP_LAT,bins=200)
	# y8,binEdges8 = np.histogram(rate_matrix_DC_MRT_LAT,bins=200)
	# y9,binEdges9 = np.histogram(rate_matrix_SA_MRT,bins=200)
	# y10,binEdges10 = np.histogram(rate_matrix_SA_LAT,bins=200)
	# y11,binEdges11 = np.histogram(rate_matrix_SA_BHCAP_LAT,bins=200)
	# y12,binEdges12 = np.histogram(rate_matrix_SA_MRT_LAT,bins=200)
	
	# bincenters1 = 0.5*(binEdges1[1:]+binEdges1[:-1])
	# bincenters2 = 0.5*(binEdges2[1:]+binEdges2[:-1])
	# bincenters3 = 0.5*(binEdges3[1:]+binEdges3[:-1])
	# bincenters4 = 0.5*(binEdges4[1:]+binEdges4[:-1])
	# bincenters5 = 0.5*(binEdges5[1:]+binEdges5[:-1])
	# bincenters6 = 0.5*(binEdges6[1:]+binEdges6[:-1])
	# bincenters7 = 0.5*(binEdges7[1:]+binEdges7[:-1])
	# bincenters8 = 0.5*(binEdges8[1:]+binEdges8[:-1])
	# bincenters9 = 0.5*(binEdges9[1:]+binEdges9[:-1])
	# bincenters10 = 0.5*(binEdges10[1:]+binEdges10[:-1])
	# bincenters11 = 0.5*(binEdges11[1:]+binEdges11[:-1])
	# bincenters12 = 0.5*(binEdges12[1:]+binEdges12[:-1])
	
	# #print np.sum(y5)

	# plt.plot(bincenters1[np.where(y1!=0)],y1[np.where(y1!=0)],'rx',fillstyle='none', markersize=8)
	# plt.plot(bincenters2[np.where(y2!=0)],y2[np.where(y2!=0)],'b^',fillstyle='none', markersize=8)
	# #plt.plot(bincenters3,y3,'b-o',fillstyle='none')
	# plt.plot(bincenters5[np.where(y5!=0)],y5[np.where(y5!=0)],'ko',fillstyle='none', markersize=8)

	n_bins = 300; # Number of bins for the histogram
	fig, ax = plt.subplots()
	#ax.set_title("User Distribution CDF")
	#print len(rate_matrix_DC)
	#print time_DC[:,5]
	n1, bins1, patches1 = ax.hist(rate_matrix_DC, n_bins,density=True, histtype='step',
                           cumulative=True, label='DC')
	n2, bins2, patches2 = ax.hist(rate_matrix_SA, n_bins,density=True, histtype='step',
                          cumulative=True, label='SA')
	n3, bins3, patches3 = ax.hist(rate_matrix_DC_MRT, n_bins,density=True, histtype='step',
                          cumulative=True, label='DC+MRT')

	#ax.set_xlabel('Throughput(bps)')
	
	#plt.plot(bincenters1[y1.tolist().index(np.amax(y1))],np.amax(y1),'rs')
	#plt.plot(np.ones((np.arange(0,np.amax(y1)).shape[0]+1,1))*bincenters1[y1.tolist().index(np.amax(y1))],np.arange(0,np.amax(y1)+1),'k--')
	#plt.plot(bincenters2[y2.tolist().index(np.amax(y2))],np.amax(y2),'bs')
	#plt.plot(np.ones((np.arange(0,np.amax(y2)).shape[0]+1,1))*bincenters2[y2.tolist().index(np.amax(y2))],np.arange(0,np.amax(y2)+1),'g--')
	#plt.plot(bincenters3[y3.tolist().index(np.amax(y3))],np.amax(y3),'go')
	#plt.plot(np.ones((np.arange(0,np.amax(y3)).shape[0],1))*bincenters3[y3.tolist().index(np.amax(y3))],np.arange(0,np.amax(y3)),'g--')
	#plt.plot(bincenters7[y7.tolist().index(np.amax(y7))],np.amax(y7),'go')
	#plt.plot(np.ones((np.arange(0,np.amax(y7)).shape[0],1))*bincenters7[y7.tolist().index(np.amax(y7))],np.arange(0,np.amax(y7)),'g--')
	#plt.plot(bincenters5[y5.tolist().index(np.amax(y5))],np.amax(y5),'ks')
	#plt.plot(np.ones((np.arange(0,np.amax(y5)).shape[0],1))*bincenters5[y5.tolist().index(np.amax(y5))],np.arange(0,np.amax(y5)),'g--')
		

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
	
	# plt.legend(["DC","SA","DC+MRT"])
	# plt.xlabel('Throughput(bps)')
	# plt.ylabel('Number of Users')
	# plt.title('User Distribution')
	#plt.grid()
	plt.close()

	f, ax1 = plt.subplots(figsize=(8,4))
	ax1.plot(bins1[:-1]/1e9,n1,'r-', label="DC")
	ax1.plot(bins2[:-1]/1e9,n2,'b-', label="SA")
	ax1.plot(bins3[:-1]/1e9,n3,'k-', label="DC+MRT")
	ax1.plot([0.1]*len(np.arange(0,1.1,0.1)), np.arange(0,1.1,0.1), 'g--', label="Minimum Rate = 100 Mbps")
	ax1.plot(bins3[0]/1e9, n3[0], 'ko', markersize=12, label="Minimum Rate with DC+MRT: "+str(format(bins3[0]/1e6,'0.2f'))+" Mbps")
	ax1.set_title("User Distribution CDF", fontsize = 14)
	ax1.set_xlabel('Throughput(Gbps)', fontsize = 12)
	ax1.set_ylim(0,1,0.1)
	ax1.set_xlim(min(min(bins1[:-1]/1e9), min(bins2[:-1]/1e9), min(bins3[:-1]/1e9)),max(max(bins1[:-1]/1e9), max(bins2[:-1]/1e9), max(bins3[:-1]/1e9)) )
	ax1.legend(prop={'size': 12})
	ax1.yaxis.set_ticks_position('none')
	ax1.xaxis.set_ticks_position('none')
	axins = zoomed_inset_axes(ax1, 15.5, loc=10, bbox_to_anchor=(1101.,405.))
	axins.plot(bins1[:-1]/1e9,n1,'r-')
	axins.plot(bins2[:-1]/1e9,n2,'b-')
	axins.plot(bins3[:-1]/1e9,n3,'k-')
	axins.plot([0.1]*len(np.arange(0,1.1,0.1)), np.arange(0,1.1,0.1), 'g--')
	axins.plot(bins3[0]/1e9, n3[0], 'ko', markersize=8)
	axins.set_ylim(0.01,0.03)
	axins.set_xlim(0.06,0.12)
	axins.yaxis.set_ticks_position('none')
	axins.xaxis.set_ticks_position('none')
	axins.yaxis.set_ticks(np.arange(0.01,0.035,0.01))
	axins.xaxis.set_ticks(np.arange(0.06,0.13,0.02))
	axins.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.2f'))
	#plt.yticks(visible = False)
	#plt.xticks(visible = False)
	mark_inset(ax1, axins, loc1=2, loc2=4, fc="none", ec="0.5")
	#ax1.grid()
	
	plt.show()

def scatter_plotter(rate_matrix_DC,rate_matrix_DC_MRT,np,scn):
	
	f,ax = plt.subplots(2)
	f.suptitle('Data Rate Scatter Plot of Users -- DC Scenario (top); DC+MRT Scenario (bottom)')
	ax[0].scatter(np.arange(1,len(rate_matrix_DC) +1),rate_matrix_DC,alpha=0.5)
	ax[0].plot(np.arange(1,len(rate_matrix_DC) +1),np.ones((np.arange(1,len(rate_matrix_DC) +1).shape[0],1))*1e8, 'r--')
	#ax[0].xlabel('Users')
	#ax[0].ylabel('Throughput (in bps)')
	#ax[0].title('a')
	ax[0].legend(["100 Mbps"])
	ax[1].scatter(np.arange(1,len(rate_matrix_DC_MRT) +1),rate_matrix_DC_MRT,alpha=0.5)
	ax[1].plot(np.arange(1,len(rate_matrix_DC_MRT) +1),np.ones((np.arange(1,len(rate_matrix_DC_MRT) +1).shape[0],1))*1e8, 'r--')
	#ax[1].xlabel('Users')
	#ax[1].ylabel('Throughput (in bps)')
	#ax[1].title('b')
	ax[1].legend(["100 Mbps"])
	f.text(0.5, 0.04, 'Users', ha='center')
	f.text(0.04, 0.5, 'Throughput (in bps)', va='center', rotation='vertical')

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

# ============================================
# BH Utilization and Latency Provision Plotter
# ============================================

def bhutil_latprov_plotter(bhutil_val_DC, bhutil_val_DC_BHCAP, bhutil_val_DC_BHCAP_LAT, avail_bh, latprov_DC, latprov_DC_LAT, latprov_DC_MRT_LAT, latprov_DC_BHCAP_LAT, np, scn):

	# ====> BH Utilization Plots

	bhvbh = [item for sublist in bhutil_val_DC_BHCAP for item in sublist]
	bhvbh_DC = [item for sublist in bhutil_val_DC for item in sublist]
	avbh = [item for sublist in avail_bh for item in sublist]
	#print avbh
	print np.amax(np.array(avbh))
	tot_avail_bh = avbh + [scn.fib_BH_MC_capacity]*(len(bhvbh)-len(avail_bh))
	#print tot_avail_bh
	#f,axs = plt.subplots(2)
	#f.suptitle('Backhaul Resource Utilization -- Constrained (top) and Unconstrained (bottom) BH')
	# axs[0].bar(np.arange(len(bhvbh)), [x1 - x2 for (x1, x2) in zip(bhvbh, tot_avail_bh)])
	# #ax.bar(np.arange(len(bhvbh)), bhvbh)
	# #ax.plot(np.arange(len(bhvbh)), tot_avail_bh, 'r--')
	# axs[0].set_ylim(-1*(max(tot_avail_bh)+1e9), 1e9)
	# axs[0].grid()
	# #axs[0].set_title('Backhaul Resource Utilization -- Constrained BH')
	# #axs[0].set_xlabel('(a)')
	# #axs[0].set_ylabel('Demand to Available BW Difference (bps)')

	x = np.arange(len(bhvbh))
	width = 0.35 # Width of the bars


	fig, ax = plt.subplots()

	l1 = [x1 - x2 for (x1, x2) in zip(bhvbh, tot_avail_bh)]
	l2 = [x1 - x2 for (x1, x2) in zip(bhvbh_DC, tot_avail_bh)]
	ax.bar(x - width/2, [a2/1e9 for a2 in l1] , width, label='Constrained Backhaul')
	ax.bar(x + width/2, [a1/1e9 for a1 in l2], width, label='Unconstrained Backhaul')

		#ax.set_xticklabels(labels, rotation = 90)
	
	ax.plot(np.arange(len(bhvbh)), [-1*np.amax(np.array(avbh))/1e9]*len(bhvbh),'b-', label='Maximum Available SC capacity')
	ax.plot(np.arange(len(bhvbh)), [-1*scn.fib_BH_MC_capacity/1e9]*len(bhvbh), 'k-', label='Maximum Available MC capacity')
	
	handles,labels = ax.get_legend_handles_labels()

	handles = [handles[2], handles[3], handles[0], handles[1]]
	labels = [labels[2], labels[3], labels[0], labels[1]]

	ax.grid()
	ax.set_ylabel('Demand to Available BW Difference (Gbps)')
	ax.set_xlabel('Access Points')
	ax.set_title('Backhaul Resource Utilization')
	ax.set_xticks(x)
	plt.xticks(rotation=90)

	#ax.set_xticklabels(ax.get_xticklabels(), rotation=90)

	ax.legend(handles, labels, loc="best")
	for i in range(len(bhvbh)-len(avail_bh)):
		ax.get_xticklabels()[len(bhvbh)-1 -i].set_color("red") 

	# axs[1].bar(np.arange(len(bhvbh_DC)), [x1 - x2 for (x1, x2) in zip(bhvbh_DC, tot_avail_bh)])
	# #ax.bar(np.arange(len(bhvbh)), bhvbh)
	# #ax.plot(np.arange(len(bhvbh)), tot_avail_bh, 'r--')
	# axs[1].set_ylim(-1*(max(tot_avail_bh)+1e9), max(tot_avail_bh)+1e9)
	# axs[1].grid()
	# #axs[1].set_title('Backhaul Resource Utilization -- Unconstrained BH')
	# #axs[1].set_xlabel('(b)')
	# #axs[1].set_ylabel('Demand to Available BW Difference (bps)')


	# f.text(0.5, 0.04, 'Access Points', ha='center')
	#f.text(0.04, 0.5, 'Demand to Available BW Difference (bps)', va='center', rotation='vertical')
	fig.tight_layout()
	plt.show()
	plt.close("all")
	# ====> Latency Plot
	#print latprov_DC_LAT
	lprov_DC_LAT = np.empty((latprov_DC_LAT.shape[0],2))
	for i in range(latprov_DC_LAT.shape[0]):
		temp = latprov_DC_LAT[i,np.nonzero(latprov_DC_LAT[i,:])]
		if temp.shape[1] == 2:
			lprov_DC_LAT[i,:] = temp
		else:
			lprov_DC_LAT[i,0] = temp
			lprov_DC_LAT[i,1] = temp
	#print lprov_DC_LAT[:,0].shape
	plt.scatter(np.arange(1,lprov_DC_LAT.shape[0]+1), lprov_DC_LAT[:,0], c = 'b', marker = 'o',  alpha = 0.5)
	plt.scatter(np.arange(1,lprov_DC_LAT.shape[0]+1), lprov_DC_LAT[:,1], c = 'b', marker = 'o',  alpha = 0.5)
	plt.plot(np.arange(latprov_DC_LAT.shape[0]), [scn.eMBB_latency_req]*latprov_DC_LAT.shape[0],'r-')
	plt.xlabel('Users')
	plt.ylabel('Latency (in seconds)')
	plt.title("Latency Scatter Plot of Users (DC LAT Scenario)")
	plt.legend(["3 ms"])
	plt.show()
def infeasible_iter_counter(iters_infeas, iters_infeas_DC, iters_infeas_DC_MRT, iters_infeas_SA_MRT_LAT, iters_infeas_SA_BHCAP_MRT, iters_infeas_DC_BHCAP_MRT, iters_infeas_DC_BHCAP_MRT_LAT, iters_infeas_SA_MRT , iters_timeout, iters_timeout_DC, iters_timeout_DC_MRT, iters_timeout_SA_MRT_LAT, iters_timeout_SA_BHCAP_MRT, iters_timeout_DC_BHCAP_MRT, iters_timeout_DC_BHCAP_MRT_LAT, iters_timeout_SA_MRT , iters_infeas_SA_MRT_BHCAP_LAT, iters_timeout_SA_MRT_BHCAP_LAT, iters_infeas_DC_MRT_LAT,iters_timeout_DC_MRT_LAT, iters_infeas_SA_BHCAP, iters_timeout_SA_BHCAP, iters_infeas_SA_LAT, iters_timeout_SA_LAT,
		iters_infeas_SA_BHCAP_LAT, iters_timeout_SA_BHCAP_LAT, iters_infeas_DC_BHCAP, iters_timeout_DC_BHCAP, iters_infeas_DC_LAT, iters_timeout_DC_LAT,
		iters_infeas_DC_BHCAP_LAT, iters_timeout_DC_BHCAP_LAT, np,scn):

	labels = ['SA', 'DC', 'SA + CB', 'DC + CB', 'SA + CPL', 'DC + CPL', 'SA + CPL + CB', 'DC + CPL + CB', 'SA + MRT', 'DC + MRT', 'SA + MRT + CPL', 'DC + MRT + CPL', 'SA + CB + MRT', 'DC + CB + MRT', 'SA + CB + MRT + CPL', 'DC + CB + MRT + CPL']

	x = np.arange(len(labels))
	width = 0.25 # Width of the bars

	fig, ax = plt.subplots()

	rects1 = ax.bar(x - width/2, [iters_infeas, iters_infeas_DC, iters_infeas_SA_BHCAP, iters_infeas_DC_BHCAP, iters_infeas_SA_LAT, iters_infeas_DC_LAT, iters_infeas_SA_BHCAP_LAT, iters_infeas_DC_BHCAP_LAT, iters_infeas_SA_MRT, iters_infeas_DC_MRT, iters_infeas_SA_MRT_LAT, iters_infeas_DC_MRT_LAT, iters_infeas_SA_BHCAP_MRT, iters_infeas_DC_BHCAP_MRT, iters_infeas_SA_MRT_BHCAP_LAT, iters_infeas_DC_BHCAP_MRT_LAT], width, label='Infeasible Iterations')
	rects2 = ax.bar(x + width/2, [iters_timeout, iters_timeout_DC, iters_timeout_SA_BHCAP, iters_timeout_DC_BHCAP, iters_timeout_SA_LAT, iters_timeout_DC_LAT, iters_timeout_SA_BHCAP_LAT, iters_timeout_DC_BHCAP_LAT, iters_timeout_SA_MRT, iters_timeout_DC_MRT, iters_timeout_SA_MRT_LAT, iters_timeout_DC_MRT_LAT, iters_timeout_SA_BHCAP_MRT, iters_timeout_DC_BHCAP_MRT, iters_timeout_SA_MRT_BHCAP_LAT, iters_timeout_DC_BHCAP_MRT_LAT], width, label='Timed out iterations')

	ax.set_ylabel('Number of Iterations')
	ax.set_title('Infeasible and Timed out Iterations')
	ax.set_xticks(x)
	ax.set_xticklabels(labels, rotation = 90)
	ax.legend()

	def autolabel(rects):
		"""Attach a text label above each bar in *rects*, displaying its height."""
		for rect in rects:
			height = rect.get_height()
			ax.annotate('{}'.format(height),
				xy=(rect.get_x() + rect.get_width() / 2, height),
				xytext=(0, 2),  # 3 points vertical offset
				textcoords="offset points",
				ha='center', va='bottom')


	autolabel(rects1)
	autolabel(rects2)

	fig.tight_layout()
	plt.show()

# ===========
# CDF Builder
# ===========


def timecdf(time_DC, time_DC_MRT , time_SA_MRT , time_DC_MRT_BHCAP , time_DC_MRT_BHCAP_LAT , time_DC_MRT_LAT , time_SA_MRT_BHCAP , time_SA_MRT_BHCAP_LAT , time_SA_MRT_LAT ,time_SA, np, scn ):
	#plt.close('all')
	#print time_SA_MRT[:,5]
	n_bins = 300; # Number of bins for the histogram
	#fig, axs = plt.subplots()
	fig, axs = plt.subplots()
	#fig.suptitle("CDF for Optimizer Processing Times at Maximum User Density")
	#print time_DC[:,5]
	n1, bins1, patches1 = axs.hist(time_DC[:,4], n_bins,density=True, histtype='step',
                           cumulative=True, label='Empirical', color='r')
	seq = np.array([0])
	seq_bins1 = np.array([bins1[0]])
	n1 = np.concatenate((seq,n1), axis =0)
	bins1 = np.concatenate((seq_bins1,bins1), axis =0)

	n5, bins5, patches5 = axs.hist(time_SA[:,4], n_bins,density=True, histtype='step',
                           cumulative=True, label='Empirical', color='c')
	
	seq_bins5 = np.array([bins5[0]])
	n5 = np.concatenate((seq,n5), axis =0)
	bins5 = np.concatenate((seq_bins5,bins5), axis =0)
	#axs[0,0].legend()
	#axs[0,0].set_xlim(min(min(bins1), min(bins5)), max(max(bins1), max(bins5)))
	#axs[0,0].grid()
	#ax1.plot(len(bins1),np.ones((1,len(bins1))),'g-')


	n2, bins2, patches2 = axs.hist(time_DC_MRT_BHCAP[:,4], n_bins,density=True, histtype='step',
                           cumulative=True, label='Empirical', color='k')
	
	seq_bins2 = np.array([bins2[0]])
	n2 = np.concatenate((seq,n2), axis =0)
	bins2 = np.concatenate((seq_bins2,bins2), axis =0)

	#ax2.plot(len(bins2),np.ones((1,len(bins2))),'g-')	
	#len_plt = max(len(bins1), len(bins2))
	#print n1
	#axs[1,0].set_title("DC+MRT+BHCAP")
	#axs[1,0].set_xlim(min(bins2), max(bins2))
	#axs[1,0].grid()
	
	n3, bins3, patches3 = axs.hist(time_DC_MRT[:,4], n_bins,density=True, histtype='step',
                           cumulative=True, label='Empirical', color='b')
	
	seq_bins3 = np.array([bins3[0]])
	n3 = np.concatenate((seq,n3), axis =0)
	bins3 = np.concatenate((seq_bins3,bins3), axis =0)
	#xs[0,1].set_title("DC+MRT")
	#axs[0,1].set_xlim(min(bins3), max(bins3))
	#axs[0,1].grid()

	n4, bins4, patches4 = axs.hist(time_DC_MRT_BHCAP_LAT[:,4], n_bins,density=True, histtype='step',
                           cumulative=True, label='Empirical', color='g')
	
	seq_bins4 = np.array([bins4[0]])
	n4 = np.concatenate((seq,n4), axis =0)
	bins4 = np.concatenate((seq_bins4,bins4), axis =0)

	n6, bins6, patches6 = axs.hist(time_DC_MRT_LAT[:,4], n_bins,density=True, histtype='step',
                           cumulative=True, label='Empirical', color='g')
	
	seq_bins6 = np.array([bins6[0]])
	n6 = np.concatenate((seq,n6), axis =0)
	bins6 = np.concatenate((seq_bins6,bins6), axis =0)

	n7, bins7, patches7 = axs.hist(time_SA_MRT[:,4], n_bins,density=True, histtype='step',
                           cumulative=True, label='Empirical', color='g')
	
	seq_bins7 = np.array([bins7[0]])
	n7 = np.concatenate((seq,n7), axis =0)
	bins7 = np.concatenate((seq_bins7,bins7), axis =0)
	#axs[1,1].set_title("DC+MRT+BHCAP+LAT")
	#axs[1,1].set_xlim(min(bins4), max(bins4))
	#axs[1,1].grid()

	plt.close()
	f, ax1 = plt.subplots(2,1)
	ax1[0].plot(bins1[:-1],n1,'r-', label="DC")
	ax1[0].plot(bins5[:-1],n5,'c-', label="SA")
	ax1[1].plot(bins7[:-1],n7,'c-', label="SA+MRT")
	ax1[1].plot(bins2[:-1],n2,'b-', label="DC+MRT+BHCAP")
	ax1[1].plot(bins3[:-1],n3,'k-', label="DC+MRT")
	ax1[1].plot(bins6[:-1],n6,'m-', label="DC+MRT+LAT")
	ax1[1].plot(bins4[bins4<=600],n4[bins4[:-1]<=600],'g-', label="DC+MRT+BHCAP+LAT")
	#labels_y_1 = [str(min(min(bins1), min(bins5))),'0.2','0.4','0.6','0.8','1.0']
	# labels = [item.get_text() for item in ax1[0].get_yticklabels()]
	# labels = [str(format(min(min(n1),min(n5)),'0.3e')),'0.2','0.4','0.6','0.8','1.0']
	# #print labels
	# ax1[0].set_yticklabels(labels)
	#print n1
	#print bins1
	# extraticks = [min(min(n1), min(n5))]
	# ax1[0].set_yticks(list(ax1[0].get_yticks()) + extraticks)
	# #[0].yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.3f'))
	ax1[0].set_xlim(min(min(bins1), min(bins5))-0.1, max(max(bins1), max(bins5))+0.1)
	ax1[0].set_ylim(0,1)
	ax1[0].grid(alpha = 0.2, linestyle = '--')
	ax1[0].set_xlabel("Processing time (in seconds)")

	# extraticks_y = [min(min(n2),min(n3),min(n4))]
	#extraticks_x = [min(min(bins2),min(bins3),min(bins4),min(bins6))-3]
	# ax1[1].set_yticks(list(ax1[1].get_yticks()) + extraticks_y)
	#ax1[1].set_xticks(list(ax1[1].get_xticks()) + extraticks_x)
	ax1[1].set_xlim(0, 600)
	ax1[1].set_ylim(0,1)
	ax1[1].grid(alpha = 0.2, linestyle = '--')
	ax1[1].set_xlabel("Processing time (in seconds)")
	#ax1[1,].set_xlim(min(bins3), max(bins3))
	#ax1[1,1].set_xlim(min(bins4), max(bins4))
	#ax1[0].xaxis.set_ticks(np.arange(min(min(bins1), min(bins5))-0.1,max(max(bins1), max(bins5))+0.1,0.5))
	#ax1[0].yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.2f'))
	ax1[0].xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.2f'))
	#ax1[0].xaxis.set_ticks(np.arange(min(min(bins2),min(bins3),min(bins4))-3,600,100))

	# #ax1.plot([0.1]*len(np.arange(0,1.1,0.1)), np.arange(0,1.1,0.1), 'g--', label="Minimum Rate = 100 Mbps")
	# #ax1.plot(bins3[0]/1e9, n3[0], 'ko', markersize=12, label="Minimum Rate with DC+MRT: "+str(format(bins3[0]/1e6,'0.2f'))+" Mbps")
	f.suptitle("CDF for Optimizer Processing Times at Maximum User Density", fontsize = 14)
	# ax1.set_xlabel('Processing Time (in seconds)', fontsize = 12)
	# ax1.set_ylim(0,1,0.1)
	# ax1.set_xlim(min(min(bins1[:-1]), min(bins2[:-1]), min(bins3[:-1]), min(bins4[:-1])),max(max(bins1[:-1]), max(bins2[:-1]), max(bins3[:-1]), max(bins4[:-1]) ) )
	ax1[0].legend(loc=4, prop={'size': 12})
	#ax1[0,1].legend(prop={'size': 12})
	ax1[1].legend(loc=1, prop={'size': 12})
	#ax1[1,1].legend(prop={'size': 12})
	
	# ax1.yaxis.set_ticks_position('none')
	# ax1.xaxis.set_ticks_position('none')
	# axins = zoomed_inset_axes(ax1, 0.5, loc=10, bbox_to_anchor=(1201.,405.))
	# axins.plot(bins1[:-1],n1,'r-')
	# #axins.plot(bins2[:-1],n2,'b-')
	# #axins.plot(bins3[:-1],n3,'k-')
	# #axins.plot(bins4[:-1],n4,'g-')
	# #axins.plot([0.1]*len(np.arange(0,1.1,0.1)), np.arange(0,1.1,0.1), 'g--')
	# #axins.plot(bins3[0]/1e9, n3[0], 'ko', markersize=8)
	# axins.set_ylim(0,1)
	# axins.set_xlim(0,5.5)
	# axins.yaxis.set_ticks_position('none')
	# axins.xaxis.set_ticks_position('none')
	# #axins.yaxis.set_ticks(np.arange(0,0.03,0.01))
	# #axins.xaxis.set_ticks(np.arange(0.09,0.12,0.01))
	# #axins.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.2f'))
	# #plt.yticks(visible = False)
	# #plt.xticks(visible = False)
	# mark_inset(ax1, axins, loc1=2, loc2=4, fc="none", ec="0.5")
	# #ax1.grid()
	
	#bincenters1 = 0.5*(bins1[1:]+bins1[:-1])
	#bincenters2 = 0.5*(bins2[1:]+bins2[:-1])

	#plt.plot(bincenters1,n1,'r-', bincenters2, n2, 'k-', fillstyle='none')

	plt.show()
# 
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