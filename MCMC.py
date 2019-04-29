#!/usr/bin/env python 

# =============================
# Import the necessary binaries
# =============================

import numpy as np
import subprocess 
import time
import os, sys
import json, requests


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

for i in range(2):
	#subprocess.check_call(['python',os.path.join(os.getcwd(),"main.py")]); # Open Main File for Generating the scenario
	subprocess.check_call(['python',os.path.join(os.getcwd(),"optimizer_func.py"),'-iter', str(i) ,'-minRate', '1','-dual', '1','-bhaul', '0','-latency', '1'])
	chat = last_chat_id(get_updates()) # Get the Bot Chat ID
	message = "Execution of Iteration " + str(i) + " Completed"
	send_message(message,chat) # Send the Message 