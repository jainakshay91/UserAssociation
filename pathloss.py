#!/usr/bin/env

# ====> This file consists of the pathloss generator function for all the considered scenarios

from dist_check import breakpt_dist
from dist_check import los_prob_var_gen
from los_probability import *

# ============================================
# Small Cell Pathloss 
# ============================================

def pathloss_SC(usr_ht, fc, dist, np, d3d):
   
    # ====================================================
    # We consider a LOS scenario if the los probability is greater than 50%
    
    bs_ht = 10; # Small cell base station height is 10m
    if los_probability.los_prob_sc(np,dist) >= 0.5:
        bp_dst = dist_check.breakpt_dist(bs_ht, usr_ht, fc, dist, 1, np); # Breakpoint distance 
        if dist>=10 and dist<bp_dst:
            pathloss_sc = 32.4 + 21*np.log10(d3d)+20*np.log10(fc); 
        else if dist >= bp_dst and dist <= 5000:
            pathloss_sc = 32.4 + 40*np.log10(d3d) + 20*np.log10(fc) -9.5*np.log10((bp_dst)**2 + (bs_ht - usr_ht)**2)
        shadowing = np.random.normal(0,4); # We introduce shadowing
        return (pathoss_sc+shadowing) # We return the total large scale fading

    # ===================================================
    # We now consider the N-LOS scenario
    
    else:
        bp_dst = dist_check.breakpt_dist(bs_ht, usr_ht, fc, dist, 1, np); # Breakpoint distance 
        if dist>=10 and dist<=5000:
            if dist>=10 and dist<bp_dst:
                los_sc = 32.4 + 21*np.log10(d3d)+20*np.log10(fc); 
            else if dist >= bp_dst and dist <= 5000:
                los_sc = 32.4 + 40*np.log10(d3d) + 20*np.log10(fc) -9.5*np.log10((bp_dst)**2 + (bs_ht - usr_ht)**2);
            nlos_sc = 35.3*np.log10(d3d)+22.4+21.3*np.log10(fc)-0.3*(usr_ht-1.5);
            pathloss_sc = np.maximum(los_sc,nlos_sc)      
            shadowing = np.random.normal(0,7.82); #Shadowing in NLOS scenarios
    return (pathloss_sc+shadowing)


# =====================================================
# Macro Cell Pathloss
# =====================================================


def pathloss_MC(usr_ht, np, dist, d3d, fc):

    # ==================================================
    # We consider a LOS scenario if the los probability is greater than 50%
    
    bs_ht = 25; # Small cell base station height is 10m
    if los_probability.los_prob_mc(np,dist,dist_check.los_prob_var_gen(usr_ht)) >= 0.5:
        bp_dst = dist_check.breakpt_dist(bs_ht, usr_ht, fc, dist, 1, np); # Breakpoint distance 
        if dist>=10 and dist<bp_dst:
            pathloss_mc = 32.4 + 20*np.log10(d3d)+20*np.log10(fc); 
        else if dist >= bp_dst and dist <= 5000:
            pathloss_mc = 32.4 + 40*np.log10(d3d) + 20*np.log10(fc) -10*np.log10((bp_dst)**2 + (bs_ht - usr_ht)**2)
        shadowing = np.random.normal(0,4); # We introduce shadowing
        return (pathoss_mc+shadowing) # We return the total large scale fading

    # ===================================================
    # We now consider the N-LOS scenario
    
    else:
        bp_dst = dist_check.breakpt_dist(bs_ht, usr_ht, fc, dist, 1, np); # Breakpoint distance 
        if dist>=10 and dist<=5000:
            if dist>=10 and dist<bp_dst:
                los_mc = 32.4 + 20*np.log10(d3d)+20*np.log10(fc); 
            else if dist >= bp_dst and dist <= 5000:
                los_mc = 32.4 + 40*np.log10(d3d) + 20*np.log10(fc) -10*np.log10((bp_dst)**2 + (bs_ht - usr_ht)**2);
            nlos_mc = 39.08*np.log10(d3d)+13.54+20*np.log10(fc)-0.6*(usr_ht-1.5);
            pathloss_mc = np.maximum(los_sc,nlos_sc)      
            shadowing = np.random.normal(0,7.8); #Shadowing in NLOS scenarios
    return (pathloss_mc+shadowing)
        
