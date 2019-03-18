#!/usr/bin/env

# ====> This file consists of the pathloss generator function for all the considered scenarios

#from dsc import breakpt_dist
#from dsc import los_prob_var_gen
import los_probability

# ============================================
# Small Cell Pathloss 
# ============================================

def pathloss_SC(scn, dist, np, d3d, dsc):
   
    # ====================================================
    # We consider a LOS scenario if the los probability is greater than 50%
    
    pathloss_sc = 0; # Declaring the pathloss variable
    #bs_ht = 10; # Small cell base station height is 10m
    if los_probability.los_prob_sc(np,dist) >= 0.5:
        bp_dst = dsc.breakpt_dist(scn, dist, 1, np); # Breakpoint distance 
        if dist>=10 and dist<bp_dst:
            pathloss_sc = 32.4 + 21*np.log10(d3d)+20*np.log10(scn.fc_sc); 
        elif dist >= bp_dst and dist <= 5000:
            pathloss_sc = 32.4 + 40*np.log10(d3d) + 20*np.log10(scn.fc_sc) -9.5*np.log10((bp_dst)**2 + (scn.bs_ht_sc - scn.usr_ht)**2)
        shadowing = np.random.normal(0,4); # We introduce shadowing
        return (pathloss_sc+shadowing) # We return the total large scale fading

    # ===================================================
    # We now consider the N-LOS scenario
    
    else:
        bp_dst = dsc.breakpt_dist(scn, dist, 1, np); # Breakpoint distance 
        if dist>=10 and dist<=5000:
            if dist>=10 and dist<bp_dst:
                los_sc = 32.4 + 21*np.log10(d3d)+20*np.log10(scn.fc_sc); 
            elif dist >= bp_dst and dist <= 5000:
                los_sc = 32.4 + 40*np.log10(d3d) + 20*np.log10(scn.fc_sc) -9.5*np.log10((bp_dst)**2 + (scn.bs_ht_sc - scn.usr_ht)**2);
            nlos_sc = 35.3*np.log10(d3d)+22.4+21.3*np.log10(scn.fc_sc)-0.3*(scn.usr_ht-1.5);
            pathloss_sc = np.maximum(los_sc,nlos_sc)      
            shadowing = np.random.normal(0,7.82); #Shadowing in NLOS scenarios
    return (pathloss_sc+shadowing)


# =====================================================
# Macro Cell Pathloss
# =====================================================


def pathloss_MC(scn, dist, np, d3d, dsc):

    # ==================================================
    # We consider a LOS scenario if the los probability is greater than 50%
    
    #bs_ht = 25; # Macro cell base station height is 25m
    if los_probability.los_prob_mc(np,dist,dsc.los_prob_var_gen(scn.usr_ht)) >= 0.5:
        bp_dst = dsc.breakpt_dist(scn, dist, 0, np); # Breakpoint distance 
        if dist>=10 and dist<bp_dst:
            pathloss_mc = 32.4 + 20*np.log10(d3d)+20*np.log10(scn.fc); 
        elif dist >= bp_dst and dist <= 5000:
            pathloss_mc = 32.4 + 40*np.log10(d3d) + 20*np.log10(scn.fc) -10*np.log10((bp_dst)**2 + (scn.bs_ht - scn.usr_ht)**2)
        shadowing = np.random.normal(0,4); # We introduce shadowing
        return (pathoss_mc+shadowing) # We return the total large scale fading

    # ===================================================
    # We now consider the N-LOS scenario
    
    else:
        bp_dst = dsc.breakpt_dist(scn, dist, 0, np); # Breakpoint distance 
        if dist>=10 and dist<=5000:
            if dist>=10 and dist<bp_dst:
                los_mc = 32.4 + 20*np.log10(d3d)+20*np.log10(scn.fc); 
            elif dist >= bp_dst and dist <= 5000:
                los_mc = 32.4 + 40*np.log10(d3d) + 20*np.log10(scn.fc) -10*np.log10((bp_dst)**2 + (scn.bs_ht - scn.usr_ht)**2);
            nlos_mc = 39.08*np.log10(d3d)+13.54+20*np.log10(scn.fc)-0.6*(scn.usr_ht-1.5);
            pathloss_mc = np.maximum(los_sc,nlos_sc)      
            shadowing = np.random.normal(0,7.8); #Shadowing in NLOS scenarios
    return (pathloss_mc+shadowing)
        
