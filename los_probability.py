#!/usr/bin/env

# =========================
# LOS Probability Generator
# =========================

# Los Probability for UMi Street Canyon (Small Cell)

def los_prob_sc (np,dist):
    if dist<=18:
        los_prob = 1; # The UE is in a LOS zone surely given that the distance is less than 18m
    else:
        los_prob = (18/dist)*(np.exp(-1*(dist/36)))*(1-18/dist); # Probability of a UE being in LOS zone given that the distance is beyond 18m
    return los_prob


# Los Probability for the UMa (Macro Cell in Urban Environments)

def los_prob_mc(np,dist,ht):
    if dist<=18:
        los_prob = 1; # The UE is in LOS always given that the distance is less than 18m
    else: 
        los_prob = (18/dist + np.exp(-1*dist/63)*(1-18/dist))*(1 + los_prob_var_gen(ht)*(5/4)*((dist/100)**3)*np.exp(-1*dist/150)); # Probability of a UE being in LOS given that the distance between UE and MC is more than 18m
    return los_prob



