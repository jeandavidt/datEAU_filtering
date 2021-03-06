# ############### UNIVARIATE METHOD FOR DATA ###############################
# This file is a port-over from Janelcy Alferes' Matlab scripts

# This script contains the different steps to follow to apply the univariate
# method for on-line data treatment.  It's a type script with the different
# step for an example of parameter X of a Sensor.

# ############### Importing the required libraries #########################
import time

import matplotlib.pyplot as plt

import pandas as pd
from pandas.plotting import register_matplotlib_converters

import Sensors
import FaultDetection
import ModelCalibration
import OutlierDetection
import outlierdetection_Online_EWMA
import PlottingTools
import Smoother
import TreatedData
import DataCoherence
from DefaultSettings import DefaultParam

register_matplotlib_converters()

# #########################################################################
# #######################  TIME SERIES : RAW DATA  #########################

##########################################################################
# The Times dictionnary keeps track of the time as the method unfolds to see
# which parts of the script take the longest to execute
Times = {}
Times['ini'] = time.time()

# Import the raw data

path = '../sample_data/influent3.csv'
raw_data = pd.read_csv(path, sep=';')
raw_data.datetime = pd.to_datetime(raw_data.datetime)
raw_data.set_index('datetime', inplace=True, drop=True)

Times['import_done'] = time.time()

# Add new data

# other_data = pd.read_csv(path1,sep=';')
# other_data.datetime = pd.to_datetime(other_data.datetime)
# other_data.set_index('datetime', inplace=True, drop=True)
# data = pd.concat([raw_data,other_data],axis=1)

# resamp_data= raw_data.asfreq('2 min')
# data = resamp_data.fillna(method='ffill')
# Times['resample_done'] = time.time()
data = raw_data['15-09-2017 00:00:00':'15-04-2018 00:00:00']
sensors = Sensors.parse_dataframe(data)

# Plot raw data
title = 'Pre-processed Data'
PlottingTools.plotRaw_D_mpl(data)
Times['plot raw'] = time.time()
# -------------------------------------------------------------------------
# ----------------------------------X--------------------------------------
# -------------------------------------------------------------------------
# ####################Generate default parameters##########################
# Selection of the period of the data series to be treated
sensor = sensors[0]
channel = sensor.channels['COD']  # Variable to be filtered

T0 = channel.start
TF = channel.end

# #########################################################################
# #########################  DATA FILTERING     ###########################
# ######################### OUTLIERS DETECTION  ###########################
# Load default parameters
# channel.params = DefaultParam()
# Set parameters: Example
channel.params['outlier_detection']['nb_reject'] = 100
# ############### Select a subset of the data for calibration ###############
# The subset should be as large as possible to better represent the system
# and sensor behavior

Tini = '15 January 2018'
Tfin = '15 February 2018'

channel.calib = {'start': Tini, 'end': Tfin}
start = channel.calib['start']
end = channel.calib['end']
# Plot calibration data
title = 'Calibration subset'

calib_data = channel.raw_data[start:end]

PlottingTools.plotUnivar_mpl(channel)
Times['parameters set'] = time.time()
# ################Test the dataset for missing values, NaN, etc.############

flag = DataCoherence.data_coherence(channel)

answer = None
# while answer not in ("y", "n"):
#     answer = input("Continue?")
#     if answer == "y":
#         pass
#     elif answer == "n":
#         exit()
#     else:
#         print("Please enter y or n.")
# Times['Data coherence checked'] = time.time()
# channel = DataCoherence.resample(channel, '2 min')
# flag = DataCoherence.data_coherence(channel)

# channel = DataCoherence.sort_dat(channel)
# flag = DataCoherence.data_coherence(channel)

# channel = DataCoherence.fillna(channel)
# flag = DataCoherence.data_coherence(channel)

# ############################ Outlier detection ##########################
filtration_method = "Online_EWMA"
channel.params['outlier_detection']['method'] = filtration_method

channel = OutlierDetection.outlier_detection(channel)

# Times['outlier detection done'] = time.time()
# Plot the outliers detected

# PlottingTools.plotOutliers_mpl(channel)
# plt.show()

# PlottingTools.Plot_Outliers(channel, filtration_method)(data, channel)
# Times['Outliers plotted'] = time.time()

###########################################################################
# #########################  DATA SMOOTHER   ##############################

###########################################################################
# ###################Generate default parameters###########################

# Set parameters
channel.params['data_smoother']['h_smoother'] = 10

# Data filtration ==> kernel_smoother function.

# data = Smoother.kernel_smoother(data, channel, channel.params)
channel = Smoother.kernel_smoother(channel)
# Times['data smoothed'] = time.time()

# Plot filtered data
# with open('script.json', 'w') as outfile:
#    json.dump(channel, outfile, indent=4, cls=Sensors.CustomEncoder)

PlottingTools.plotOutliers_mpl(channel)
# plt.show()
fault_detect_time = time.time()
# Times['smoothed data plotted'] = time.time()
# with open('script.json') as json_file:
#    channel = json.load(json_file, object_hook=Sensors.decode_object)

##########################################################################

# ############################FAULT DETECTION#############################

##########################################################################


# Definition range (min and max)for Q_range:
# minimum real expected value of the variable
channel.params['fault_detection_uni']['range_min'] = 50

# maximum real expected value of the variable
channel.params['fault_detection_uni']['range_max'] = 550

# Definition limit of scores:
channel.params['fault_detection_uni']['corr_min'] = -50
channel.params['fault_detection_uni']['corr_max'] = 50

# minimum expected slope based on good data series
channel.params['fault_detection_uni']['slope_min'] = -10

# maximum expected slope based on a good data series
channel.params['fault_detection_uni']['slope_max'] = 10

# Minimum variation between accepted data and smoothed data
channel.params['fault_detection_uni']['std_min'] = -10

# Maximum variation between accepted data and smoothed data
channel.params['fault_detection_uni']['std_max'] = 10

# Calcul Q_corr, Q_std, Q_slope, Q_range:
channel = FaultDetection.D_score(channel)
# Times['Faults detected'] = time.time()

# Plot scores
# PlottingTools.plotDScore_mpl(channel)
# plt.show()
# Times['Detected faults plotted'] = time.time()
##########################################################################

# ############################  TREATED DATA   ###########################

# To see the treated data and the deleted data:
channel = TreatedData.TreatedD(channel)
# Times['Final data generated'] = time.time()
# plot the raw data and treated data:
PlottingTools.plotTreatedD_mpl(channel)
plt.show()
# Times['Final data plotted'] = time.time()

# Timedf = pd.DataFrame(data={'event': list(Times.keys()), 'time': list(Times.values())})

# Percentage of outliers and deleted data
filtration_method = channel.info['current_filtration_method']
print(channel.info['filtration_results'][filtration_method])


# save ('Sensor.mat')# Save the whole data'''
