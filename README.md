# UserAssociation
This repository consists of codes for the Policy based User Association method that is being developed. Currently the code for mMTC and eMBB is available for use. You can clone the mMTCOptimizer Branch. 

The main tasks done by this codebase are explained in the article: "User Association and Resource Allocation in 5G (AURA-5G): A Joint Optimization Framework", Elesvier Computer Networks. 

Specifically, the program generates the scenario and simulates the wireless evironment based on the parameters specified in the article mentioned above. The data is then fed to the Gurobi optimizer which generates feasible solutions. This metadata is then fed into the visualization toolbox developed, which assists in performing the necessary data analysis. 

The program can be run from CLI, simply by: 

python MCMC.py \\
python plotgen.py

Note: Be careful about the directories for storing the metadata. Also there is a Telegram bot that can be created which can receive the status messages from the code. To do this, you need to go to Telegram, create a bot using botfather, and then use the token and insert it into the Token area that is left blank in the code. 

For further queries, feel free to write me at: akshay.jain@upc.edu

Collaborators: Dr. Elena Lopez-Aguilera, Dr. Ilker Demirkol \\
Work Funded by: EU H2020 MSCA-ITN Project Grant # 675806. \\
Special Thanks: Dr. H Birkan Yilmaz
