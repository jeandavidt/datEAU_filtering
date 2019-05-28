################ UNIVARIATE METHOD FOR DATA ###############################
# This file is a port-over from Janelcy Alferes' Matlab scripts

# This script contains the different steps to follow to apply the univariate 
# method for on-line data treatment.  It's a type script with the different 
#step for an example of parameter X of a Sensor. 

################ Importing the required libraries #########################
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import register_matplotlib_converters

import FaultDetection
import ModelCalibration
import OutlierDetection
import outlierdetection_Online_EWMA
import PlottingTools
import Smoother
import TreatedData
from DataCoherence import Data_Coherence
from DefaultSettings import DefaultParam

register_matplotlib_converters()



##########################################################################

########################  TIME SERIES : RAW DATA  #########################

##########################################################################
#The Times dictionnary keeps track of the time as the method unfolds to see
#which parts of the script take the longest to execute
Times={}
Times['ini'] = time.time()

#Import the raw data

path = '../sample_data/influentdata.csv'
raw_data =pd.read_csv(path, sep=';')
raw_data.datetime = pd.to_datetime(raw_data.datetime)
raw_data.set_index('datetime', inplace=True, drop=True)
Times['import_done'] = time.time()

#Add new data
#path1 = 'G:\Documents\......\New_data.csv'
#other_data = pd.read_csv(path1,sep=';')
#other_data.datetime = pd.to_datetime(other_data.datetime)
#other_data.set_index('datetime', inplace=True, drop=True)
#data = pd.concat([raw_data,other_data],axis=1) 
 

resamp_data= raw_data.asfreq('2 min')
data = resamp_data.fillna(method='ffill')
Times['resample_done'] = time.time()


 
parameters_list = []
for column in data.columns:
    if (('Unit' not in column) & ('equipment' not in column)):
        parameters_list.append(column)
print('Parameters are {}'.format(parameters_list))

#Plot raw data 
title = 'Pre-processed Data'
PlottingTools.plotRaw_D(data, parameters_list,title)
Times['plot raw'] = time.time()
# -------------------------------------------------------------------------
# ----------------------------------X--------------------------------------
# -------------------------------------------------------------------------

#####################Generate default parameters###########################
# Selection of the period of the data series to be treated

channel = 'COD' # Variable to be filtered

T0 = data.first_valid_index()
TF = data.last_valid_index()

##########################################################################

##########################  DATA FILTERING     ###########################

########################## OUTLIERS DETECTION  ###########################

# Load default parameters

paramX = DefaultParam()

# Set parameters: Example
paramX['nb_reject']= 100  

################ Select a subset of the data for calibration ###############

# The susbset should be as large as possible to better represent the system
# and sensor behavior

Tini = '15 January 2018'
Tfin = '15 February 2018'

CalibX = data.loc[Tini:Tfin,:].copy()
#Plot calibration data 
title = 'Calibration subset'
PlottingTools.plotRaw_D(CalibX, [channel],title)
Times['parameters set'] = time.time()
#################Test the dataset for missing values, NaN, etc.############
flag = Data_Coherence(data, paramX)
print(flag)
answer = None
while answer not in ("y", "n"):
    answer = input("Continue?")
    if answer == "y":
         pass
    elif answer == "n":
         exit()
    else:
    	print("Please enter y or n.")
Times['Data coherence checked'] = time.time()
############################## Outlier detection ##########################

paramX['OutlierDetectionMethod'] = "Online_EWMA"

data, paramX = OutlierDetection.outlier_detection(data, CalibX, channel, paramX)
Times['outlier detection done'] = time.time()
# Plot the outliers detected
PlottingTools.Plot_Outliers(data, channel)
Times['Outliers plotted'] = time.time()

###########################################################################

###########################  DATA SMOOTHER   ##############################

###########################################################################

#####################Generate default parameters###########################

# Set parameters
paramX['h_smoother']    = 10

# Data filtation ==> kernel_smoother fucntion.

data = Smoother.kernel_smoother(data, channel, paramX)
Times['data smoothed'] = time.time()
# Plot filtered data
PlottingTools.Plot_Filtered(data, channel)
plt.show()
fault_detect_time = time.time()
Times['smoothed data plotted'] = time.time()

##########################################################################

##############################FAULT DETECTION#############################

##########################################################################
#data = pickle.load(open('smooth.p','rb'))
#paramX = pickle.load(open('parameters.p','rb'))

#Definition range (min and max)for Q_range: 
paramX['range_min'] = 50     #minimum real expected value of the variable
paramX['range_max'] = 550     #maximum real expected value of the variable

#Definition limit of scores: 
paramX['corr_min']= -5  
paramX['corr_max']= 5

paramX['slope_min']= -1   # maximum expected slope based on a good data series
paramX['slope_max'] = 1   # minimum expected slope based on good data series

paramX['std_min'] = -0.1    # Maximum variation between accepted data and smoothed data
paramX['std_max'] = 0.1    # Minimum variation between accepted data and smoothed data

#Calcul Q_corr, Q_std, Q_slope, Q_range: 
data = FaultDetection.D_score(data, paramX, channel)
Times['Faults detected'] = time.time()

# Plot scores
PlottingTools.Plot_DScore(data, channel, paramX)
Times['Detected faults plottted'] = time.time()
##########################################################################

##############################  TREATED DATA   ###########################

#To allow to determinate the treated data and deleted data:
Final_data = TreatedData.TreatedD(data, paramX,channel)
Times['Final data generated'] = time.time()
#plot the raw data and treated data: 
PlottingTools.plotTreatedD(Final_data, channel)
plt.show()
Times['Final data plotted'] = time.time()

Timedf = pd.DataFrame(data={'event':list(Times.keys()),'time':list(Times.values())})

# Percentage of outliers and deleted data
Intervariable = TreatedData.InterpCalculator(Final_data, channel) 


# save ('Sensor.mat')# Save the whole data 