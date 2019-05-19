#!/usr/bin/env python 

# =============================
# Import the necessary binaries
# =============================

import subprocess 
import time
import os, sys
import json, requests
from multiprocessing import Pool
import numpy as np
import signal 


# =====================================
# Check Presence of Storage Directories
# =====================================

def path_checker():
	flag = -1; # Initialize the flag variable 
	path = os.getcwd() + '/Data'; # This is the path we have to check for
	subpath = os.getcwd() + '/Data/Temp'; # This is the subdirectory to store data  
	if os.path.isdir(path):
		if os.path.isdir(subpath):
			flag = 0; # We do not need to generate the scenario data again
		else:
			flag = 1; # Generate the Data if the Subpath does not exist
	else:
		flag = 1; # Generate the Data if the Path does not exist 
	return flag 

# ==========================
# Parallel Process Function
# ==========================

def parallel_executor(iter_num):
	subprocess.call(['python',os.path.join(os.getcwd(),"main.py"), '-iter', str(iter_num)])

# ==================================
# Create a Telegram Bot Communicator
# ==================================

TOKEN = "849800908:AAHHCf4rI24sAcNv-QwKiSqX1wYSxfHLDdA"
URL = "https://api.telegram.org/bot{}/".format(TOKEN)

def get_url(url):
	response = requests.get(url)
	content = response.content.decode("utf8")
	return content

def get_json_from_url(url):
	content = get_url(url)
	js = json.loads(content)
	return js

def get_updates():
	url = URL + "getUpdates"
	js = get_json_from_url(url)
	return js

def last_chat_id(updates):
	num_updates = len(updates["result"])
	last_update = num_updates - 1
	chat_id = updates["result"][last_update]["message"]["chat"]["id"]
	return chat_id

def send_message(text, chat_id):
	url = URL + "sendMessage?text={}&chat_id={}".format(text,chat_id)
	get_url(url)



# ======================
# Monte Carlo Simulation
# ======================


sys.path.append(os.getcwd()); # Add current working directory to python path
os.chdir(os.getcwd()); # Change to the current working directory
chat_frequency = 10; # Select the divider so as to obtain timely update messages
num_processors = int(int(subprocess.check_output(['nproc']))/2); # Number of Processors to be utilized 
MCMC_iter = 16; # Number of Monte Carlo Iterations


# =============
# Main Function 

if __name__ == '__main__':

	dat_gen_flag = path_checker(); # Get the Data generation flag value

	if dat_gen_flag == 1:
		file_indexer = 0; # For File Indexing
		for i in range(0, MCMC_iter/num_processors):
			print "Entering Round " + str(i) + " of Processing"
			print "------------------------------"
			print ""
			idx_range = np.arange(file_indexer, file_indexer + num_processors); # Data file Index numbers
			pool = Pool(processes = num_processors); # Creates a pool of 10 parallel processes to be done
			pool.map(parallel_executor,idx_range.tolist()); # Maps the function to parallel processes
			file_indexer = file_indexer + num_processors; # Increase the Iterator number
			print file_indexer
		pool.close()
		pool.join()
		
	print "Entering the Optimizer"
	for i in range(MCMC_iter):
		try:
			#subprocess.check_call(['python',os.path.join(os.getcwd(),"main.py")]); # Open Main File for Generating the scenario
			subprocess.check_call(['python',os.path.join(os.getcwd(),"optimizer_func.py"),'-iter', str(i) ,'-minRate', '0','-dual', '0','-bhaul', '0','-latency', '1'])
			chat = last_chat_id(get_updates()) # Get the Bot Chat ID
			if i%chat_frequency == 0:
				try:
					message = "Execution of Iteration " + str(i) + " Completed"
					send_message(message,chat) # Send the Message 
				except(RuntimeError, TypeError, NameError, IndexError):
					pass
		except:
			message = "Programme has encountered an error"
			send_message(message, chat) # Send the message if an error has been encountered in the code
			message = "Ending the Processing for Debugging"
			send_message(message, chat) # Send the End process message
			sys.exit('Error Encountered')