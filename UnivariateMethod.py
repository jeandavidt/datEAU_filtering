################ UNIVARIATE METHOD FOR DATA ###############################
# This file is a port-over from Janelcy Alferes' Matlab scripts

# This script contains the different steps to follow to apply the univariate 
# method for on-line data treatment.  It's a type script with the different 
#step for an example of parameter X of a Sensor. 

################ Importing the required libraries #########################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

from OutlierDetection import Outlier_Detection, Outlier_Detection_online
import outlierdetection_Online_EWMA
from PlottingTools import plotRaw_D
from DefaultSettings import DefaultParam
from DataCoherence import Data_Coherence

# -------------------------------------------------------------------------
# -------------------------------Sensor------------------------------------
# -------------------------------------------------------------------------

#Gerenate the functions
#addpath ('DataFiltrationFramework/')
#SetFiltersPaths


##
##########################################################################

########################  TIME SERIES : RAWDATA  #########################

##########################################################################

#Import the raw data
path = '../sample_data/influentdata.csv'
#SENSOR = DataImport (path,'datEAUbaseCSVtoMAT','SENSOR.mat')
raw_data =pd.read_csv(path, sep=';')
raw_data.datetime = pd.to_datetime(raw_data.datetime)
raw_data.set_index('datetime', inplace=True, drop=True)
#Add new data
#path1 = 'G:\Documents\......\New_data.csv'
#Sen = DataImport (path1,'datEAUbaseCSVtoMAT','Sen.mat')
#SENSOR = Concatenate (SENSOR, Sen) 
#save ('SENSOR.mat')# Save the data. 
 
parameters_list = []
for column in raw_data.columns:
    if (('Unit' not in column) & ('equipment' not in column)):
        parameters_list.append(column)
print('Parameters are {}'.format(parameters_list))

#Plot raw data 
title = 'Raw Data'
plotRaw_D(raw_data, parameters_list,title)

# -------------------------------------------------------------------------
# ----------------------------------X--------------------------------------
# -------------------------------------------------------------------------

#####################Generate default parameters###########################
# Selection of the period of the data series to be treated

channel = 'COD' # Variable to be filtered

T0 = raw_data.first_valid_index()
TF = raw_data.last_valid_index()

##########################################################################

##########################  DATA FILTERING     ###########################

########################## OUTLIERS DETECTION  ###########################

# Load default parameters

paramX = DefaultParam('Online_EWMA')

# Set parameters: Example
paramX['nb_reject']= 100  

################ Select a subset of the data for calibration ###############

# The susbset should be as large as possible to better represent the system
# and sensor behavior

Tini = '15 January 2018'
Tfin = '15 February 2018'

CalibX = raw_data.loc[Tini:Tfin, [channel]].copy()
#Plot calibration data 
title = 'Calibration subset'
plotRaw_D(CalibX, [channel],title)

#################Test the dataset for missing values, NaN, etc.############
flag = Data_Coherence(CalibX, paramX)
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

############################## Outlier detection ##########################

Sensor, paramX = Outlier_Detection(Sensor, CalibX, channel, paramX)

# Plot the outliers detected
'''Plot_Outliers(Sensor, channel)

###########################################################################

###########################  DATA SMOOTHER   ##############################

###########################################################################

#####################Generate default parameters###########################

# Set parameters
paramX.h_smoother    = 30

# Data filtation ==> kernel_smoother fucntion.

[Sensor(channel).Smoothed_AD, err]=kernel_smoother(Sensor, channel, paramX)

# Plot filtered data
plotFiltered_D(Sensor, channel)

##
##########################################################################

##############################FAULT DETECTION#############################

##########################################################################

#Definition range (min and max)for Q_range: 
paramX.range_min = 2     #minimum real expected value of the variable
paramX.range_max = 4     #maximum real expected value of the variable

#Calcul Q_corr, Q_std, Q_slope, Q_range: 
Sensor(channel).Score = D_score(Sensor, paramX, channel)


#Definition limit of scores: 
paramX.corr_min= -16  
paramX.corr_max= -5

paramX.slope_min= -0.005   # maximum expected slope based on a good data series
paramX.slope_max= 0.0058   # minimum expected slope based on good data series

paramX.std_min = -0.1    # Maximum variation between accepted data and smoothed data
paramX.std_max = 0.1    # Minimum variation between accepted data and smoothed data

# Plot scores
plotD_score(Sensor, paramX, channel)

##########################################################################

##############################  TREATED DATA   ###########################

#To allow to determinate the treated data and deleted data:
Sensor(channel).Final_D = TreatedD(Sensor, paramX,channel)

#plot the raw data and treated data: 
plotTreatedD( Sensor, channel)

# Percentage of outliers and deleted data
[Sensor(channel).Intervariable] = Interpcalculator (Sensor, channel) 

#Save the param in the struct:
[Sensor(channel).param] = paramX

# save ('Sensor.mat')# Save the whole data 

#Allow to clear the different created variable in the workspace.
clear flag calibX posSensorX T Tini paramX err i
 

'''