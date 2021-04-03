#!/usr/bin/env python

# This script allows the user to utilize the functions stated here in and save any intended Numpy matrix data as a CSV file 

# ==================
# Import the Modules
# ==================

import csv
import pandas as pd
import numpy as np


def csvsaver(data, col_names, filname):
	dframe = {} # Empty dictionary for the pandas Dataframe
	if len(col_names)!=0:
		for i in range(len(col_names)):
			dframe[col_names[i]] = data[:,i]

		df = pd.DataFrame(dframe)
		df.to_csv(filname, index=False)

	else:
		for i in range(data.shape[1]):
			dframe[str(i)] = data[:,i]
		df = pd.DataFrame(dframe)
		df.to_csv(filname, index=False)