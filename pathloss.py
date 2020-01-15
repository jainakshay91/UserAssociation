#!/usr/bin/env

# ====> This file consists of the pathloss generator function for all the considered scenarios

#from dsc import breakpt_dist
#from dsc import los_prob_var_gen
import los_probability

# ============================================
# CI Model Pathloss 
# ============================================

def pathloss_CI(scn, dist, np, d3d, dsc, sc_flag):
   
    # =====================================================
    # We implement the NYU CI model for full distance range
    
    los_flag = 0 # Line of Sight and Non Line of Sight flag 
    if sc_flag == 0 or sc_flag == 1:
        FSPL = 20*np.log10((4*np.pi*np.where(sc_flag,scn.fc_sc,scn.fc_mc))/scn.c); # Calculate the Free Space Path loss

        # =====================================================================
        # We consider a LOS scenario if the los probability is greater than 50%
        
        #print ("The Threshold for SC LOS-NLOS is:", scn.tau_sc[tau_flag])
        #print ("The Threshold for MC LOS-NLOS is:", scn.tau_mc[tau_flag])

        if (np.random.rand(1) <= los_probability.los_prob(np,dist,sc_flag)  and sc_flag == 1) or (np.random.rand(1) <= los_probability.los_prob(np,dist,sc_flag) and sc_flag == 0):
            los_flag = 1 # Return the los flag for the plots later
            #print ("LOS with flag:",sc_flag)
            n = np.where(sc_flag, 2.1, 2.0); # For LOS scenarios UMa has PLE = 2.0 and UMi at 28 and 73GHz has PLE = 2.1
            SF_dev = np.where(sc_flag, 4.4, 2.4); # For LOS Scenarios UMa has a SF dev of 2.4 and UMi at 28 and 73 GHz has SF dev of 4.4 
            shadowing = np.random.normal(0,SF_dev);
            if dist < 1:
                return (FSPL + shadowing), los_flag  # Below 1m it is PL is just FSPL in CI model
            else: 
                PL_CI = FSPL + 10*n*np.log10(d3d) + shadowing; # CI model Pathloss 
            return PL_CI, los_flag

        else:
            los_flag = 0 # Return the los flag for the plots later (NLOS)
            #print ("NLOS with flag:",sc_flag)
            n = np.where(sc_flag, 3.2, 2.9); # For NLOS scenarios UMa has PLE = 2.0 and UMi at 28 and 73GHz has PLE = 2.1
            SF_dev = np.where(sc_flag, 8.0, 5.7); # For NLOS Scenarios UMa has a SF dev of 2.4 and UMi at 28 and 73 GHz has SF dev of 4.4 
            shadowing = np.random.normal(0,SF_dev);
            if dist < 1:
                return (FSPL + shadowing), los_flag  # Below 1m it is PL is just FSPL in CI model
            else: 
                PL_CI = FSPL + 10*n*np.log10(d3d) + shadowing; # CI model Pathloss 
            return PL_CI, los_flag
            
    elif sc_flag == 2:
        
        #if los_probability.los_prob_sc(np,dist) >= 0.5: # Small Cells will always be in LOS for a MC
        los_flag = 1
        FSPL = 20*np.log10((4*np.pi*scn.fc_bh_sc)/scn.c); # Calculate the Free Space Path loss for Backhaul
        n = 2.0; # We consider BH to be in LOS scenario with a pathloss exponent of 2.1
        SF_dev = 4.2; # Standard deviation for Shadow Fading
        shadowing = np.random.normal(0, SF_dev);
        if dist < 1:
            return (FSPL+shadowing), los_flag; # Unusual between SC and MC but can happen
        else: 
            PL_SC_MC_CI = FSPL + 10*n*np.log10(d3d) + shadowing; # CI model Pathloss between SC and MC
        return PL_SC_MC_CI, los_flag
        # else:
        #     los_flag = 1
        #     FSPL = 20*np.log10((4*np.pi*scn.fc_bh_sc)/scn.c); # Calculate the Free Space Path loss for Backhaul
        #     n = 3.5; # We consider BH to be in LOS scenario with a pathloss exponent of 2.1
        #     SF_dev = 7.9; # Standard deviation for Shadow Fading
        #     shadowing = np.random.normal(0, SF_dev);
        #     if dist < 1:
        #         return (FSPL+shadowing); # Unusual between SC and MC but can happen
        #     else: 
        #         PL_SC_MC_CI = FSPL + 10*n*np.log10(d3d) + shadowing; # CI model Pathloss between SC and MC
        #     return PL_SC_MC_CI, los_flag




# ============================================
# 3GPP Small Cell Pathloss 
# ============================================

# def pathloss_SC_3GPP(scn, dist, np, d3d, dsc):
   
#     # =====================================================================
#     # We consider a LOS scenario if the los probability is greater than 50%
    
#     pathloss_sc = 0; # Declaring the pathloss variable
#     #bs_ht = 10; # Small cell base station height is 10m
#     if los_probability.los_prob_sc(np,dist) >= 0.5:
#         bp_dst = dsc.breakpt_dist(scn, dist, 1, np); # Breakpoint distance 
#         if dist>=10 and dist<bp_dst:
#             pathloss_sc = 32.4 + 21*np.log10(d3d)+20*np.log10(scn.fc_sc); 
#         elif dist >= bp_dst and dist <= 5000:
#             pathloss_sc = 32.4 + 40*np.log10(d3d) + 20*np.log10(scn.fc_sc) -9.5*np.log10((bp_dst)**2 + (scn.bs_ht_sc - scn.usr_ht)**2)
#         shadowing = np.random.normal(0,4); # We introduce shadowing
#         return (pathloss_sc+shadowing) # We return the total large scale fading

#     # ===================================================
#     # We now consider the N-LOS scenario
    
#     else:
#         bp_dst = dsc.breakpt_dist(scn, dist, 1, np); # Breakpoint distance 
#         if dist>=10 and dist<=5000:
#             if dist>=10 and dist<bp_dst:
#                 los_sc = 32.4 + 21*np.log10(d3d)+20*np.log10(scn.fc_sc); 
#             elif dist >= bp_dst and dist <= 5000:
#                 los_sc = 32.4 + 40*np.log10(d3d) + 20*np.log10(scn.fc_sc) -9.5*np.log10((bp_dst)**2 + (scn.bs_ht_sc - scn.usr_ht)**2);
#             nlos_sc = 35.3*np.log10(d3d)+22.4+21.3*np.log10(scn.fc_sc)-0.3*(scn.usr_ht-1.5);
#             pathloss_sc = np.maximum(los_sc,nlos_sc)      
#             shadowing = np.random.normal(0,7.82); #Shadowing in NLOS scenarios
#     return (pathloss_sc+shadowing)


# =====================================================
# 3GPP Macro Cell Pathloss
# =====================================================


# def pathloss_MC_3GPP(scn, dist, np, d3d, dsc):

#     # ==================================================
#     # We consider a LOS scenario if the los probability is greater than 50%
    
#     #bs_ht = 25; # Macro cell base station height is 25m
#     if los_probability.los_prob_mc(np,dist,dsc.los_prob_var_gen(scn.usr_ht)) >= 0.5:
#         bp_dst = dsc.breakpt_dist(scn, dist, 0, np); # Breakpoint distance 
#         if dist>=10 and dist<bp_dst:
#             pathloss_mc = 32.4 + 20*np.log10(d3d)+20*np.log10(scn.fc); 
#         elif dist >= bp_dst and dist <= 5000:
#             pathloss_mc = 32.4 + 40*np.log10(d3d) + 20*np.log10(scn.fc) -10*np.log10((bp_dst)**2 + (scn.bs_ht - scn.usr_ht)**2)
#         shadowing = np.random.normal(0,4); # We introduce shadowing
#         return (pathoss_mc+shadowing) # We return the total large scale fading

#     # ===================================================
#     # We now consider the N-LOS scenario
    
#     else:
#         bp_dst = dsc.breakpt_dist(scn, dist, 0, np); # Breakpoint distance 
#         if dist>=10 and dist<=5000:
#             if dist>=10 and dist<bp_dst:
#                 los_mc = 32.4 + 20*np.log10(d3d)+20*np.log10(scn.fc); 
#             elif dist >= bp_dst and dist <= 5000:
#                 los_mc = 32.4 + 40*np.log10(d3d) + 20*np.log10(scn.fc) -10*np.log10((bp_dst)**2 + (scn.bs_ht - scn.usr_ht)**2);
#             nlos_mc = 39.08*np.log10(d3d)+13.54+20*np.log10(scn.fc)-0.6*(scn.usr_ht-1.5);
#             pathloss_mc = np.maximum(los_sc,nlos_sc)      
#             shadowing = np.random.normal(0,7.8); #Shadowing in NLOS scenarios
#     return (pathloss_mc+shadowing)
        
