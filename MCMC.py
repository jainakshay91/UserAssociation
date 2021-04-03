#!/usr/bin/env python 

# =============================
# Import the necessary binaries
# =============================

import subprocess 
import time
import os, sys
import json, requests
from multiprocessing import Pool, Process
import numpy as np
import signal 
from scenario_var import scenario_var 
import logging as lg

# =====================================
# Check Presence of Storage Directories
# =====================================

def path_checker():
	#print "here"
	#print os.getcwd()
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
	#print flag
	return flag 

# ==================================
# Create a Telegram Bot Communicator
# ==================================

TOKEN = ""
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

# ==========================
# Parallel Process Function
# ==========================

def parallel_executor(iter_num):
	print ("Iteration number:", iter_num)
	subprocess.call(['python',os.path.join(os.getcwd(),"main.py"), '-iter', str(iter_num), '-interf', str(0)])

def Single_assoc(MCMC_iter, chat_frequency):
	for i in range(MCMC_iter):
		chat = last_chat_id(get_updates()) # Get the Bot Chat ID
		try:
			print MCMC_iter
			print chat_frequency
			#subprocess.check_call(['python',os.path.join(os.getcwd(),"main.py")]); # Open Main File for Generating the scenario
			subprocess.call(['python',os.path.join(os.getcwd(),"optimizer_func.py"),'-iter', str(i) ,'-minRate', '0','-dual', '0','-bhaul', '0','-latency', '0', '-mipGP', '0'])
			#chat = last_chat_id(get_updates()) # Get the Bot Chat ID
			if i%chat_frequency == 0:
				try:
					message = "Execution of Iteration " + str(i) + " Completed for SA"
					send_message(message,chat) # Send the Message 
				except(RuntimeError, TypeError, NameError, IndexError):
					pass
		except:
			message = "Programme has encountered an error"
			send_message(message, chat) # Send the message if an error has been encountered in the code
			message = "Ending the Processing for Debugging"
			send_message(message, chat) # Send the End process message
			sys.exit('Error Encountered')

def Dual_assoc(MCMC_iter, chat_frequency):
	for i in range(MCMC_iter):
		chat = last_chat_id(get_updates()) # Get the Bot Chat ID
		try:
			#subprocess.check_call(['python',os.path.join(os.getcwd(),"main.py")]); # Open Main File for Generating the scenario
			subprocess.call(['python',os.path.join(os.getcwd(),"optimizer_func.py"),'-iter', str(i) ,'-minRate', '0','-dual', '1','-bhaul', '0','-latency', '0', '-mipGP', '0'])
			#chat = last_chat_id(get_updates()) # Get the Bot Chat ID
			if i%chat_frequency == 0:
				try:
					message = "Execution of Iteration " + str(i) + " Completed for DA"
					send_message(message,chat) # Send the Message 
				except(RuntimeError, TypeError, NameError, IndexError):
					pass
		except:
			message = "Programme has encountered an error"
			send_message(message, chat) # Send the message if an error has been encountered in the code
			message = "Ending the Processing for Debugging"
			send_message(message, chat) # Send the End process message
			sys.exit('Error Encountered')

def DA_MRT(MCMC_iter, chat_frequency):
	for i in range(MCMC_iter):
		chat = last_chat_id(get_updates()) # Get the Bot Chat ID
		try:
			#subprocess.check_call(['python',os.path.join(os.getcwd(),"main.py")]); # Open Main File for Generating the scenario
			subprocess.call(['python',os.path.join(os.getcwd(),"optimizer_func.py"),'-iter', str(i) ,'-minRate', '1','-dual', '1','-bhaul', '0','-latency', '0', '-mipGP', '1'])
			#chat = last_chat_id(get_updates()) # Get the Bot Chat ID
			if i%chat_frequency == 0:
				try:
					message = "Execution of Iteration " + str(i) + " Completed for DA + MRT"
					send_message(message,chat) # Send the Message 
				except(RuntimeError, TypeError, NameError, IndexError):
					pass
		except:
			message = "Programme has encountered an error"
			send_message(message, chat) # Send the message if an error has been encountered in the code
			message = "Ending the Processing for Debugging"
			send_message(message, chat) # Send the End process message
			sys.exit('Error Encountered')

def DA_BHCAP(MCMC_iter, chat_frequency):
	for i in range(MCMC_iter):
		chat = last_chat_id(get_updates()) # Get the Bot Chat ID
		try:
			#subprocess.check_call(['python',os.path.join(os.getcwd(),"main.py")]); # Open Main File for Generating the scenario
			subprocess.call(['python',os.path.join(os.getcwd(),"optimizer_func.py"),'-iter', str(i) ,'-minRate', '0','-dual', '1','-bhaul', '1','-latency', '0', '-mipGP', '1'])
			#chat = last_chat_id(get_updates()) # Get the Bot Chat ID
			if i%chat_frequency == 0:
				try:
					message = "Execution of Iteration " + str(i) + " Completed for DA + BHCAP"
					send_message(message,chat) # Send the Message 
				except(RuntimeError, TypeError, NameError, IndexError):
					pass
		except:
			message = "Programme has encountered an error"
			send_message(message, chat) # Send the message if an error has been encountered in the code
			message = "Ending the Processing for Debugging"
			send_message(message, chat) # Send the End process message
			sys.exit('Error Encountered')

def DA_BHCAP_LAT(MCMC_iter, chat_frequency):
	for i in range(MCMC_iter):
		chat = last_chat_id(get_updates()) # Get the Bot Chat ID
		try:
			#subprocess.check_call(['python',os.path.join(os.getcwd(),"main.py")]); # Open Main File for Generating the scenario
			subprocess.call(['python',os.path.join(os.getcwd(),"optimizer_func.py"),'-iter', str(i) ,'-minRate', '0','-dual', '1','-bhaul', '1','-latency', '1', '-mipGP', '1'])
			#chat = last_chat_id(get_updates()) # Get the Bot Chat ID
			if i%chat_frequency == 0:
				try:
					message = "Execution of Iteration " + str(i) + " Completed for DA + BHCAP + LAT"
					send_message(message,chat) # Send the Message 
				except(RuntimeError, TypeError, NameError, IndexError):
					pass
		except:
			message = "Programme has encountered an error"
			send_message(message, chat) # Send the message if an error has been encountered in the code
			message = "Ending the Processing for Debugging"
			send_message(message, chat) # Send the End process message
			sys.exit('Error Encountered')

def DA_LAT(MCMC_iter, chat_frequency):
	for i in range(MCMC_iter):
		chat = last_chat_id(get_updates()) # Get the Bot Chat ID
		try:
			#subprocess.check_call(['python',os.path.join(os.getcwd(),"main.py")]); # Open Main File for Generating the scenario
			subprocess.call(['python',os.path.join(os.getcwd(),"optimizer_func.py"),'-iter', str(i) ,'-minRate', '0','-dual', '1','-bhaul', '0','-latency', '1', '-mipGP', '0'])
			#chat = last_chat_id(get_updates()) # Get the Bot Chat ID
			if i%chat_frequency == 0:
				try:
					message = "Execution of Iteration " + str(i) + " Completed for DA + LAT"
					send_message(message,chat) # Send the Message 
				except(RuntimeError, TypeError, NameError, IndexError):
					pass
		except:
			message = "Programme has encountered an error"
			send_message(message, chat) # Send the message if an error has been encountered in the code
			message = "Ending the Processing for Debugging"
			send_message(message, chat) # Send the End process message
			sys.exit('Error Encountered')	

def DA_MRT_LAT(MCMC_iter, chat_frequency):
	for i in range(MCMC_iter):
		chat = last_chat_id(get_updates()) # Get the Bot Chat ID
		try:
			#subprocess.check_call(['python',os.path.join(os.getcwd(),"main.py")]); # Open Main File for Generating the scenario
			subprocess.call(['python',os.path.join(os.getcwd(),"optimizer_func.py"),'-iter', str(i) ,'-minRate', '1','-dual', '1','-bhaul', '0','-latency', '1', '-mipGP', '0'])
			#chat = last_chat_id(get_updates()) # Get the Bot Chat ID
			if i%chat_frequency == 0:
				try:
					message = "Execution of Iteration " + str(i) + " Completed for DA + MRT + LAT"
					send_message(message,chat) # Send the Message 
				except(RuntimeError, TypeError, NameError, IndexError):
					pass
		except:
			message = "Programme has encountered an error"
			send_message(message, chat) # Send the message if an error has been encountered in the code
			message = "Ending the Processing for Debugging"
			send_message(message, chat) # Send the End process message
			sys.exit('Error Encountered')	

def SA_MRT(MCMC_iter, chat_frequency):
	for i in range(MCMC_iter):
		chat = last_chat_id(get_updates()) # Get the Bot Chat ID
		try:
			#subprocess.check_call(['python',os.path.join(os.getcwd(),"main.py")]); # Open Main File for Generating the scenario
			subprocess.call(['python',os.path.join(os.getcwd(),"optimizer_func.py"),'-iter', str(i) ,'-minRate', '1','-dual', '0','-bhaul', '0','-latency', '0', '-mipGP', '0'])
			#chat = last_chat_id(get_updates()) # Get the Bot Chat ID
			if i%chat_frequency == 0:
				try:
					message = "Execution of Iteration " + str(i) + " Completed for SA + MRT"
					send_message(message,chat) # Send the Message 
				except(RuntimeError, TypeError, NameError, IndexError):
					pass
		except:
			message = "Programme has encountered an error"
			send_message(message, chat) # Send the message if an error has been encountered in the code
			message = "Ending the Processing for Debugging"
			send_message(message, chat) # Send the End process message
			sys.exit('Error Encountered')	

def SA_BHCAP(MCMC_iter, chat_frequency):
	for i in range(MCMC_iter):
		chat = last_chat_id(get_updates()) # Get the Bot Chat ID
		try:
			#subprocess.check_call(['python',os.path.join(os.getcwd(),"main.py")]); # Open Main File for Generating the scenario
			subprocess.call(['python',os.path.join(os.getcwd(),"optimizer_func.py"),'-iter', str(i) ,'-minRate', '0','-dual', '0','-bhaul', '1','-latency', '0', '-mipGP', '1'])
			#chat = last_chat_id(get_updates()) # Get the Bot Chat ID
			if i%chat_frequency == 0:
				try:
					message = "Execution of Iteration " + str(i) + " Completed for SA + BHCAP"
					send_message(message,chat) # Send the Message 
				except(RuntimeError, TypeError, NameError, IndexError):
					pass
		except:
			message = "Programme has encountered an error"
			send_message(message, chat) # Send the message if an error has been encountered in the code
			message = "Ending the Processing for Debugging"
			send_message(message, chat) # Send the End process message
			sys.exit('Error Encountered')	

def SA_BHCAP_LAT(MCMC_iter, chat_frequency):
	for i in range(MCMC_iter):
		chat = last_chat_id(get_updates()) # Get the Bot Chat ID
		try:
			#subprocess.check_call(['python',os.path.join(os.getcwd(),"main.py")]); # Open Main File for Generating the scenario
			subprocess.call(['python',os.path.join(os.getcwd(),"optimizer_func.py"),'-iter', str(i) ,'-minRate', '0','-dual', '0','-bhaul', '1','-latency', '1', '-mipGP', '1'])
			#chat = last_chat_id(get_updates()) # Get the Bot Chat ID
			if i%chat_frequency == 0:
				try:
					message = "Execution of Iteration " + str(i) + " Completed for SA + BHCAP + LAT"
					send_message(message,chat) # Send the Message 
				except(RuntimeError, TypeError, NameError, IndexError):
					pass
		except:
			message = "Programme has encountered an error"
			send_message(message, chat) # Send the message if an error has been encountered in the code
			message = "Ending the Processing for Debugging"
			send_message(message, chat) # Send the End process message
			sys.exit('Error Encountered')	

def SA_LAT(MCMC_iter, chat_frequency):
	for i in range(MCMC_iter):
		chat = last_chat_id(get_updates()) # Get the Bot Chat ID
		try:
			#subprocess.check_call(['python',os.path.join(os.getcwd(),"main.py")]); # Open Main File for Generating the scenario
			subprocess.call(['python',os.path.join(os.getcwd(),"optimizer_func.py"),'-iter', str(i) ,'-minRate', '0','-dual', '0','-bhaul', '0','-latency', '1', '-mipGP', '0'])
			#chat = last_chat_id(get_updates()) # Get the Bot Chat ID
			if i%chat_frequency == 0:
				try:
					message = "Execution of Iteration " + str(i) + " Completed for SA + LAT"
					send_message(message,chat) # Send the Message 
				except(RuntimeError, TypeError, NameError, IndexError):
					pass
		except:
			message = "Programme has encountered an error"
			send_message(message, chat) # Send the message if an error has been encountered in the code
			message = "Ending the Processing for Debugging"
			send_message(message, chat) # Send the End process message
			sys.exit('Error Encountered')	


def SA_MRT_LAT(MCMC_iter, chat_frequency):
	for i in range(MCMC_iter):
		chat = last_chat_id(get_updates()) # Get the Bot Chat ID
		try:
			#subprocess.check_call(['python',os.path.join(os.getcwd(),"main.py")]); # Open Main File for Generating the scenario
			subprocess.call(['python',os.path.join(os.getcwd(),"optimizer_func.py"),'-iter', str(i) ,'-minRate', '1','-dual', '0','-bhaul', '0','-latency', '1', '-mipGP', '0'])
			#chat = last_chat_id(get_updates()) # Get the Bot Chat ID
			if i%chat_frequency == 0:
				try:
					message = "Execution of Iteration " + str(i) + " Completed for SA + MRT + LAT"
					send_message(message,chat) # Send the Message 
				except(RuntimeError, TypeError, NameError, IndexError):
					pass
		except:
			message = "Programme has encountered an error"
			send_message(message, chat) # Send the message if an error has been encountered in the code
			message = "Ending the Processing for Debugging"
			send_message(message, chat) # Send the End process message
			sys.exit('Error Encountered')	

def DA_MRT_BHCAP(MCMC_iter, chat_frequency):
	for i in range(MCMC_iter):
		chat = last_chat_id(get_updates()) # Get the Bot Chat ID
		try:
			#subprocess.check_call(['python',os.path.join(os.getcwd(),"main.py")]); # Open Main File for Generating the scenario
			subprocess.call(['python',os.path.join(os.getcwd(),"optimizer_func.py"),'-iter', str(i) ,'-minRate', '1','-dual', '1','-bhaul', '1','-latency', '0', '-mipGP', '1'])
			#chat = last_chat_id(get_updates()) # Get the Bot Chat ID
			if i%chat_frequency == 0:
				try:
					message = "Execution of Iteration " + str(i) + " Completed for DA + MRT + BHCAP"
					send_message(message,chat) # Send the Message 
				except(RuntimeError, TypeError, NameError, IndexError):
					pass
		except:
			message = "Programme has encountered an error"
			send_message(message, chat) # Send the message if an error has been encountered in the code
			message = "Ending the Processing for Debugging"
			send_message(message, chat) # Send the End process message
			sys.exit('Error Encountered')	

def DA_MRT_BHCAP_LAT(MCMC_iter, chat_frequency):
	for i in range(MCMC_iter):
		chat = last_chat_id(get_updates()) # Get the Bot Chat ID
		try:
			#subprocess.check_call(['python',os.path.join(os.getcwd(),"main.py")]); # Open Main File for Generating the scenario
			subprocess.call(['python',os.path.join(os.getcwd(),"optimizer_func.py"),'-iter', str(i) ,'-minRate', '1','-dual', '1','-bhaul', '1','-latency', '1', '-mipGP', '1'])
			#chat = last_chat_id(get_updates()) # Get the Bot Chat ID
			if i%chat_frequency == 0:
				try:
					message = "Execution of Iteration " + str(i) + " Completed for DA + MRT + BHCAP + LAT"
					send_message(message,chat) # Send the Message 
				except(RuntimeError, TypeError, NameError, IndexError):
					pass
		except:
			message = "Programme has encountered an error"
			send_message(message, chat) # Send the message if an error has been encountered in the code
			message = "Ending the Processing for Debugging"
			send_message(message, chat) # Send the End process message
			sys.exit('Error Encountered')	


def SA_MRT_BHCAP(MCMC_iter, chat_frequency):
	for i in range(MCMC_iter):
		chat = last_chat_id(get_updates()) # Get the Bot Chat ID
		try:
			#subprocess.check_call(['python',os.path.join(os.getcwd(),"main.py")]); # Open Main File for Generating the scenario
			subprocess.call(['python',os.path.join(os.getcwd(),"optimizer_func.py"),'-iter', str(i) ,'-minRate', '1','-dual', '0','-bhaul', '1','-latency', '0', '-mipGP', '0'])
			#chat = last_chat_id(get_updates()) # Get the Bot Chat ID
			if i%chat_frequency == 0:
				try:
					message = "Execution of Iteration " + str(i) + " Completed for SA + MRT + BHCAP"
					send_message(message,chat) # Send the Message 
				except(RuntimeError, TypeError, NameError, IndexError):
					pass
		except:
			message = "Programme has encountered an error"
			send_message(message, chat) # Send the message if an error has been encountered in the code
			message = "Ending the Processing for Debugging"
			send_message(message, chat) # Send the End process message
			sys.exit('Error Encountered')	

def SA_MRT_BHCAP_LAT(MCMC_iter, chat_frequency):
	for i in range(MCMC_iter):
		chat = last_chat_id(get_updates()) # Get the Bot Chat ID
		try:
			#subprocess.check_call(['python',os.path.join(os.getcwd(),"main.py")]); # Open Main File for Generating the scenario
			subprocess.call(['python',os.path.join(os.getcwd(),"optimizer_func.py"),'-iter', str(i) ,'-minRate', '1','-dual', '0','-bhaul', '1','-latency', '1', '-mipGP', '0'])
			#chat = last_chat_id(get_updates()) # Get the Bot Chat ID
			if i%chat_frequency == 0:
				try:
					message = "Execution of Iteration " + str(i) + " Completed for SA + MRT + BHCAP + LAT"
					send_message(message,chat) # Send the Message 
				except(RuntimeError, TypeError, NameError, IndexError):
					pass
		except:
			message = "Programme has encountered an error"
			send_message(message, chat) # Send the message if an error has been encountered in the code
			message = "Ending the Processing for Debugging"
			send_message(message, chat) # Send the End process message
			sys.exit('Error Encountered')	

# ======================
# Monte Carlo Simulation
# ======================


sys.path.append(os.getcwd()); # Add current working directory to python path
os.chdir(os.getcwd()); # Change to the current working directory
chat_frequency = 10; # Select the divider so as to obtain timely update messages
#num_processors = int(int(subprocess.check_output(['nproc']))/2)*2; # Number of Processors to be utilized 
num_processors = 2
scn = scenario_var();
MCMC_iter = scn.MCMC_iter; # Number of Monte Carlo Iterations


# =============
# Main Function 

if __name__ == '__main__':

	dat_gen_flag = path_checker(); # Get the Data generation flag value

	if dat_gen_flag == 1:
		#print "In the Generator"
		file_indexer = 0; # For File Indexing
		pool = Pool(processes = num_processors); # Creates a pool of 10 parallel processes to be done
		for i in range(0, MCMC_iter/num_processors):
			print "Entering Round " + str(i) + " of Processing"
			print "------------------------------"
			print ""
			idx_range = np.arange(file_indexer, file_indexer + num_processors); # Data file Index numbers
			pool.map(parallel_executor,idx_range.tolist()); # Maps the function to parallel processes. 
			file_indexer = file_indexer + num_processors; # Increase the Iterator number
			print file_indexer
		pool.close()
		pool.join()
		
	print "Entering the Optimizer"
	
	# =====================================================
	# Multiple Processes for Parallel Scenario Optimization

	p1 = Process(target = Single_assoc, args = (MCMC_iter,chat_frequency))
	p2 = Process(target = Dual_assoc, args = (MCMC_iter, chat_frequency))
	p3 = Process(target = DA_MRT, args = (MCMC_iter, chat_frequency))
	p4 = Process(target = DA_BHCAP, args = (MCMC_iter, chat_frequency))
	p5 = Process(target = DA_BHCAP_LAT, args = (MCMC_iter, chat_frequency))
	p6 = Process(target = DA_LAT, args = (MCMC_iter, chat_frequency))
	p7 = Process(target = SA_MRT, args = (MCMC_iter, chat_frequency))
	p8 = Process(target = SA_LAT, args = (MCMC_iter, chat_frequency))
	p9 = Process(target = SA_BHCAP_LAT, args = (MCMC_iter, chat_frequency))
	p10 = Process(target = SA_BHCAP, args = (MCMC_iter, chat_frequency))
	p11 = Process(target = DA_MRT_LAT, args = (MCMC_iter, chat_frequency))
	p12 = Process(target = SA_MRT_LAT, args = (MCMC_iter, chat_frequency))
	p13 = Process(target = DA_MRT_BHCAP, args = (MCMC_iter, chat_frequency))
	p14 = Process(target = DA_MRT_BHCAP_LAT, args = (MCMC_iter, chat_frequency))
	p15 = Process(target = SA_MRT_BHCAP, args = (MCMC_iter, chat_frequency))
	p16 = Process(target = SA_MRT_BHCAP_LAT, args = (MCMC_iter, chat_frequency))


	p1.start()
	p2.start()
	p3.start()
	p4.start()
	p5.start()
	p6.start()
	p7.start()
	p8.start()
	p9.start()
	p10.start()
	p11.start()
	p12.start()
	p13.start()
	p14.start()
	p15.start()
	p16.start()

	p1.join()
	p2.join()
	p3.join()
	p4.join()
	p5.join()
	p6.join()
	p7.join()
	p8.join()
	p9.join()
	p10.join()
	p11.join()
	p12.join()
	p13.join()
	p14.join()
	p15.join()
	p16.join()
	
	

	#for i in range(MCMC_iter):
	#	try:
	#		#subprocess.check_call(['python',os.path.join(os.getcwd(),"main.py")]); # Open Main File for Generating the scenario
	#		subprocess.check_call(['python',os.path.join(os.getcwd(),"optimizer_func.py"),'-iter', str(i) ,'-minRate', '0','-dual', '0','-bhaul', '0','-latency', '1'])
	#		chat = last_chat_id(get_updates()) # Get the Bot Chat ID
	#		if i%chat_frequency == 0:
	#			try:
	#				message = "Execution of Iteration " + str(i) + " Completed"
	#				send_message(message,chat) # Send the Message 
	#			except(RuntimeError, TypeError, NameError, IndexError):
	#				pass
	#	except:
	#		message = "Programme has encountered an error"
	#		send_message(message, chat) # Send the message if an error has been encountered in the code
	#		message = "Ending the Processing for Debugging"
	#		send_message(message, chat) # Send the End process message
	#		sys.exit('Error Encountered')
