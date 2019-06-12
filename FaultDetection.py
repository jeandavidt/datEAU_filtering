def D_score(channel):
    import pandas as pd
    import numpy as np
    # This function allows to calculate the data feature. Inside this function,
    # two functions are used: Quality_data and single_sample_runs_test.

    # The INPUT:
    # AcceptedData: This variable are the accepted data provinding of the
    # outliers detection
    # Smoothed_ACCECPTEDDATA: This variable are the smoothed
    # data provinding of the kernel smoother process. It's a pre-treatment.
    # param: This variable corresponds chosen parameters by the operator.
    # Then definition of this one are in the function DefaultParam.
    # calib: This variable corresponds data interval chosen for the calibration
    # of model in the outliers detection.

    # The OUTPUT:
    # Q_corr: check the indepence of the residuals on different intervals.
    # Evaluates whether residuals are randomly distributed.
    # The calcul is carried out by the function single_sample_runs_test.
    # Q_slope: Slopes calculated between two smoothed data.
    # Gives information about the dynamics of the data and helps detection of too sudden.
    # Q_std: Standard devaition of residuals (Difference bewteen accepted data
    # and smoothed data). Estimation of variance of the data, large standard
    # deviation of residuals can be a sign of faulty data.
    # Q_range: Investigates if the data lies inside the expected range.
    # Q_mobilerange: Investigates if the data lies inside the expected range
    # with a mobile window.

    # Initialization of INUPUTS:
    filtration_method = channel.info['current_filtration_method']
    df = channel.filtered[filtration_method]
    AD = np.array(df['Accepted']).flatten()
    Smoothed_AD = np.array(df['Smoothed_AD']).flatten()
    params = channel.params

    nb_data = len(AD)  # number of data used to calculate
    # the statistics (Runs test and standard deviation).
    # These statistics are calculated with the two others functions.

    # Calcul Q_Corr: Single sample runs test:
    # Q_corr = single_sample_runs_test(AD,Smoothed_AD,nb_data,param)
    Q_corr = single_sample_runs_test(AD, Smoothed_AD, nb_data, params)
    # Check the indepence of the residuals

    # Others data features: the slope, the residual standard
    # deviation and the locally realistic range:

    [Q_slope, Q_std, Q_range] = Quality_D(AD, Smoothed_AD, nb_data, params)

    ##
    # Generation of outputs

    df['Q_corr'] = Q_corr
    df['Q_slope'] = Q_slope
    df['Q_std'] = Q_std
    df['Q_range'] = Q_range
    channel.filtered[filtration_method] = df
    return channel

def single_sample_runs_test(AcceptedData, SMOOTHED_ACCEPTEDDATA, nb_data, params):
    import numpy as np
    # This functions check the indepence of the residuals on different
    # intervals in observing the residual sign.

    # The INPUT of this function are:
    # AcceptedData: This variable are the accepted data provinding of the
    # outliers detection
    # Smoothed_ACCECPTEDDATA: This variable are the smoothed data
    # provinding of the kernel smoother process. It's a pre-treatment.
    # nb_data: This varaible represents the data number of the sample.

    # h_smoother:The variable is a chosen paramater in the function
    # DefaultParam.
    # #The OUPUT of this function are:
    # Q_corr: check the indepence of the residuals on different intervals.
    # Evaluates whether residuals are randomly distributed.
    # The calculation is carried out by the function single_sample_runs_test.

    h_smoother = params['data_smoother']['h_smoother']
    moving_window = params['fault_detection_uni']['moving_window']
    # Definition of the work window

    # Number of intervals in the data serie: The function floor carries an
    # around
    # nb_interval=floor(nb_data/moving_window)

    # Creation of the matrices:
    t = np.zeros((len(AcceptedData),))  # Sign of the residuals
    run = np.zeros((len(AcceptedData),))  # Sign changes
    # run_test=zeros(nb_interval,1) #Number of run in the interval
    Q_corr = np.full((len(AcceptedData),), np.nan)  # Run test values

    for i in range(h_smoother, len(AcceptedData) - h_smoother):
        # i=h_smoother+1:length(AcceptedData)-h_smoother #start at h+1

        t[i] = np.sign(AcceptedData[i] - SMOOTHED_ACCEPTEDDATA[i])
        # Determination of the residual sign between the smoothed data and the AccepetedData
        run[i] = abs(t[i] - t[i - 1]) / 2
        # Determination of where there is a sign change in the data serie

    # Run_test is (R - (N/2)) / (SQRT (N/2))
    # R: Number of sign changes
    # N: Number data point in this window

    res = AcceptedData - SMOOTHED_ACCEPTEDDATA
    Sign_change = res[0:-1] * res[1:]

    for i in range(moving_window - 1, nb_data - 1):
        # i=moving_window:nb_data-1
        y = Sign_change[i - moving_window + 1:i]
        r = np.sum(np.sign(y[np.all([y < 0], axis=0)]))
        Q_corr[i] = (abs(r) - (moving_window / 2)) / np.sqrt(moving_window / 2)
    return Q_corr

def Quality_D(AcceptedData, SMOOTHED_ACCEPTEDDATA, nb_data, params):
    import numpy as np

    # This function calculates the slope, the residuals standard deviation and
    # check for the realistic range of the data
    # The INPUT of this function are:
    # SMOOTHED_ACCEPTEDDATA:This variable are the smoothed data
    # provinding of the kernel smoother process. It's a pre-treatment.
    # transAcceptedData/transSMOOTHED_ACCEPTEDDATA:Logarithmic transformation
    # to stabilize the residual standard deviation, if it is necessary
    # nb_data: This varaible represents the data number of the sample.

    # h_smoother:The variable is a chosen paramater in the function DefaultParam.
    # reading_interval:The reading_interval is used to calculate the slope
    # Vmax: minimum real expected value of the variable
    # Vmin: maximum real expected value of the variable
    # The OUTPUT of this function are:
    # Q_slope: Slopes calculated between two smoothed data.
    # Gives information about the dynamics of the data and helps detection of too sudden.

    # Q _std: Standard devaition of residuals (Difference bewteen accepted data
    # and smoothed data). Estimation of variance of the data, large standard
    # deviation of residuals can be a sign of faulty data.

    # Q_range: Investigates if the data lies inside the expected range.
    # Value of the score: 1=accepted data, 0= rejected data

    # #Number of intervals in the data serie. The function floor allow to
    # calculate the round.
    # nb_interval=floor(length(SMOOTHED_ACCEPTEDDATA)/nb_data)

    # Moving window:
    moving_window = params['fault_detection_uni']['moving_window']

    # Creation of the matrixes
    Q_slope = np.zeros((len(SMOOTHED_ACCEPTEDDATA),))
    # S lopes calculated between two smoothed data
    Q_std = np.zeros((len(SMOOTHED_ACCEPTEDDATA),))  # Residuals standard deviation
    Q_range = np.ones((len(SMOOTHED_ACCEPTEDDATA),))  # Check of the locally realistic range

    # Calculation of the slope: "Q_slope"

    for i in range(moving_window - 1, nb_data - 1):  # i=moving_window:nb_data-1
        Q_slope[i] = (
            (SMOOTHED_ACCEPTEDDATA[i] - SMOOTHED_ACCEPTEDDATA[i - 1]) / (
                params['fault_detection_uni']['reading_interval']))

    # Logarithmic transformation to stabilize the residual standard deviation, if it is necessary

    transAcceptedData = np.log(AcceptedData)
    transSMOOTHED_ACCEPTEDDATA = np.log(SMOOTHED_ACCEPTEDDATA)

    # Calculation of the residual: "Q_range"

    for i in range(moving_window - 1, nb_data - 1):  # i=moving_window:nb_data-1
        if (SMOOTHED_ACCEPTEDDATA[i] > params['fault_detection_uni']['range_max']) or \
                (SMOOTHED_ACCEPTEDDATA[i] < params['fault_detection_uni']['range_min']):
            # Check of the locally realistic range. If Q_range=0, the data is outside the locally realistic range
            Q_range[i] = 0

    # Calculation of the residual standard deviation: "Q_std"
    # difference between the accepted data and the smoothed data
    # Evaluate if residuals are randomly distributed

    for i in range(moving_window - 1, nb_data - 1):  # i=moving_window:nb_data-1
        Q_std[i] = transAcceptedData[i] - transSMOOTHED_ACCEPTEDDATA[i]

    return [Q_slope, Q_std, Q_range]
