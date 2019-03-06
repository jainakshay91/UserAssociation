#!/usr/bin/env

# ====================================================================================================
# This file provides class with the necessary sub functions for the calculation of breakpoint distance
# ====================================================================================================

class bp_assist:
    
    def bp_assister(self, dist, usr_ht):
        if usr_ht < 13: 
            C = 0;
        elif usr_ht >= 13 and usr_ht <= 23: 
            C = ((usr_ht-13)/10)**1.5*self.g(dist)
        return C

    def g(self, dist):
        if dist <18:
            G = 0; 
        else: 
            G = ((5/4)*(dist/100)**3)*np.exp(-1*(dist/150));
        return G

