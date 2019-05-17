def DefaultParam(method='Online_EWMA'):

    # This function generates a default param structure holding all required
    # parameters for the filter It is possible to modify any parameter here,
    # but we strongly suggest to keep this script untouched and to update
    # manually the parameters in your own Main script For example, the two
    # following lines of code initialize all parameters and update only the
    # parameter NB_REJECT:
    #   >> param = DefaultParam('Online_EWMA')
    #   >> param['nb_reject'] = 80

    # One outlier detection method is currently available in Python: 
    #   - Online_EWMA (default), based on Janelcy's work
    #   - EWMA (matlab version only so far): a simple exponentially weighted moving       average without recovery algorithm
    #   - NeuralNetwork (matlab version only so far): An experimental approach
    #     through neural networks to detect unpredictable datapoints
    OutlierDetectionMethod = method
    param = {}
    

    # At the moment, defaults parameters are provided for Online_EWMA only
    if OutlierDetectionMethod is 'Online_EWMA':
        param['OutlierDetectionMethod'] = 'Online_EWMA'
    
        # Multiplicative factor that drives calculation of the prediction interval
        # At each time step, the confidence interval is computed as
        #    Forecast +/- [nb_s * 125 * Mean_Absolute_Deviation]
        # In other words, a large value of nb_s (ie nb_s = 10) accepts most datapoints and rejects
        # only the most obvious outliers while a small value of nb_s (ie nb_s = 1) is much more
        # restrictive
        param['nb_s'] = 3   

        # Number of consecutive rejected data needed to reinitialization the 
        # outlier detection method  If nb_reject data are reject this is called an 
        # out of control
        param['nb_reject'] = 100  

        # Number of data before the last rejected data (the last of nb_reject data) 
        # where the outlier detection method is reinitialization for a forward 
        # application 
        param['nb_backward']   = 15

        # Mean absolute deviation used to start and reinitialization the outlier 
        # detection method
        param['MAD_ini']       = 10

        # Minimum mean absolute deviation to be used If the computed MAD falls
        # below, it is replaced by min_MAD A specific example of this
        # occurence is if a constant value appears in the time series Under
        # this circumstance and without a min_MAD value, the MAD will fall to
        # zero and all datapoint different from the constant will be flag as
        # outlier until the next restart
        # The default value of 0 means that min_MAD will be initialized in the
        # ModelCalibration function
        param['min_MAD']       = 0

        # The smoother parameter defines how much datapoints are used to smooth a
        # specific value Datapoints between [i-h_smoother : i+h_smoother] are used
        # in the weighting formula If h_smoother == 0, an automatic calibration of
        # the parameter is attempted (not tested by CG yet)
        param['h_smoother']    = 5

        # Show some statistics about the filtering process
        param['ShowStats'] = True

        # Displays some warning and error messages when TRUE 
        param['Verbose'] = True

        # Permitted variation of the timestep to be considered constant is the
        # variation between a timestep and the median timestep is smaller than the
        # parameter, the timestep is considered constant Otherwise, it is
        # considered variable and must be used with caution: the filter currently
        # assumes a constant timestep
        param['DT_RelRol'] = 0.01

        # Set to TRUE if the filtering must be restarted from scratch If set to
        # FALSE, a sequential filtering is performed and new filtered data is
        # either apped to existing one or replaces it
        param['restart'] = True

        # If a serie of data is refiltered, the exponential moving average filter
        # must be applied to a number of datapoints in the so-called warmup period
        # The period of the filter is defined by N in the equation: 
        #           ALPHA = 1/(1+N)
        # In theory, 86# of the warmup is done after N datapoints are filtered To
        # get closer to 100#, the parameter N_Reset allows to use more than one
        # period, thus more datapoints based on the calibrated parameter ALPHA
        # No value larger than 4 or 5 should be used, since no improvement can be
        # observed 
        param['N_Reset'] = 2
        
    elif OutlierDetectionMethod is 'NeuralNetwork':
        raise Exception('NeuralNetwork not implemented yet.')
        #param = SetparamNN
        #param['lambda']  = 0.2
        #param['sigma']   = 3
        #param['OutlierDetectionMethod'] = 'NeuralNetwork' 

    elif OutlierDetectionMethod is 'EWMA':
        raise Exception('EWMA not implemented yet.')
        #param['lambda'] = 0.2
        #param['sigma'] = 3
        #param['OutlierDetectionMethod'] = 'EWMA' 

    else:
        raise Exception('Unknown outlier detection method: {}'.format(OutlierDetectionMethod))


    #Parameters for the fault detection method: 

    #Definition of window for the run_test test of the fault detection:
    param['moving_window'] = 1000

    #Definition reading interval: 
    param['reading_interval'] = 5 # Value chose by RP Can be changed 


    #This parameter allows to 
    param['affmobilerange'] = None

    #Definition of window for the mobilerange test of the fault detection
    param['affmobilewindow'] =None

    #This parameter allows to select the difference of y This one is
    #differente about each sensor 
    param['affdy'] = None


    #Definition of Range (Max and Min): These one will be decided by the
    #operator
    param['range_min'] = None
    param['range_max'] = None

    #Limit for the whole data feature calculation For now, every limit (Min and Max) are equal to None because it's the operator 
    #who will decide of these limits 

    param['corr_min'] = None  
    param['corr_max'] = None

    param['slope_min']= None  
    param['slope_max']= None   

    param['std_min'] = None 
    param['std_max'] = None 

    param['range_min']= None 
    param['range_max'] = None

    return param

