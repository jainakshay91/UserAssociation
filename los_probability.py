# =========================
# LOS Probability Generator
# =========================

import dist_check as dsc
from scenario_var import scenario_var 

# ==================================================
# Los Probability for UMi Street Canyon (Small Cell)


scn = scenario_var();
def los_prob(np, dist,sc_flag):
    if sc_flag == 1: # Small Cell
        los_prob = 0; # Probability holder variable
        if dist<=18:
            los_prob = 1; # The UE is in a LOS zone surely given that the distance is less than 18m
            #print ("SC within dist:", los_prob)
            return los_prob # Return the los probability
        else:
            for i in range(0,1000):
                intr_prob = (18/dist)*(np.exp(-1*(dist/36)))*(1-18/dist); # Probability of a UE being in LOS zone given that the distance is beyond 18m
                los_prob = los_prob + intr_prob; # We sum and average to obtain the average probability value   
            #print ("Small Cell LOS Probability", los_prob/i)
            return los_prob/i # We return the expected value of LOS Probability

    elif sc_flag == 0: # Macro Cell
        los_prob = 0; # Los probability holder variable
        C = dsc.los_prob_var_gen(scn.usr_ht); # Compute the Necessary Variables
        if dist<=18:
            los_prob = 1; # The UE is in LOS always given that the distance is less than 18m
            #print ("MC within dist:", los_prob)
            return los_prob
        else: 
            for i in range(0,1000):
                intr_prob = (18/dist + np.exp(-1*dist/63)*(1-18/dist))*(1 + C*(5/4)*((dist/100)**3)*np.exp(-1*dist/150)); # Probability of a UE being in LOS given that the distance between UE and MC is more than 18m
                los_prob = los_prob + intr_prob
            #print ("Macro Cell LOS Probability",los_prob/i )
            return los_prob/i # Return the average which will be close to the expected value         



# def los_prob_sc (np,dist):
#     # We take the expected value, i.e. a value over a 1000 iterations
#     los_prob = 0; # Probability holder variable
#     if dist<=18:
#         los_prob = 1; # The UE is in a LOS zone surely given that the distance is less than 18m
#         print ("SC within dist:", los_prob)
#         return los_prob # Return the los probability
#     else:
#         for i in range(0,1000):
#             intr_prob = (18/dist)*(np.exp(-1*(dist/36)))*(1-18/dist); # Probability of a UE being in LOS zone given that the distance is beyond 18m
#             los_prob = los_prob + intr_prob; # We sum and average to obtain the average probability value   
#         print ("Small Cell LOS Probability", los_prob/i)
#         return los_prob/i # We return the expected value of LOS Probability


# # ==============================================================
# # Los Probability for the UMa (Macro Cell in Urban Environments)

# def los_prob_mc(np,dist,C):
#     # We take the expected value, i.e. an average over 1000 iterations
#     los_prob = 0; # Los probability holder variable
#     if dist<=18:
#         los_prob = 1; # The UE is in LOS always given that the distance is less than 18m
#         #print ("MC within dist:", los_prob)
#         return los_prob
#     else: 
#         for i in range(0,1000):
#             intr_prob = (18/dist + np.exp(-1*dist/63)*(1-18/dist))*(1 + C*(5/4)*((dist/100)**3)*np.exp(-1*dist/150)); # Probability of a UE being in LOS given that the distance between UE and MC is more than 18m
#             los_prob = los_prob + intr_prob
#         #print ("Macro Cell LOS Probability",los_prob/i )
#         return los_prob/i # Return the average which will be close to the expected value 



