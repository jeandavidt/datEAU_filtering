def outlier_detection(channel):
    import pandas as pd
    # This function serves as a relay for all Outlier Detection methods
    # implemented. It only transfer the inputs to the chosen function and
    # return the outputs provided.

    # The whole of INPUT and OUTPUT are explained in the three functions :
    # OutlierDetection_EWMA, Outlier_Detection_NeuralNetworks, Outlier_Detection_online
    print('Hey! Fitting has started')
    channel.filtered = {}
    params = channel.params
    calibperiod = channel.calib
    most_recent = channel.info['most_recent_series']
    if most_recent == 'raw':
        data = channel.raw_data
    else:
        data = pd.DataFrame(channel.processed_data[most_recent])

    if 'method' not in params['outlier_detection']:
        raise Exception('Outlier detection: No method selected (params["outlier_detection"]["method"] does not exist')

    if params['outlier_detection']['method'] == 'EWMA':
        raise Exception('Not implemented yet')
        # data, params = OutlierDetection_EWMA(data, calibperiod, channel, params)
    elif params['outlier_detection']['method'] == 'NeuralNetwork':
        raise Exception('Not implemented yet')
        # data, params = Outlier_Detection_NeuralNetworks(data, calibperiod, channel, params)
    elif params['outlier_detection']['method'] == 'Online_EWMA':

        channel.filtered['Online_EWMA'], channel.params = Outlier_Detection_online(data, calibperiod, params)
    else:
        raise Exception('Outlier detection: unknown method')

    return channel

def Outlier_Detection_online(data, calibperiod, params):
    from ModelCalibration import ModelCalib, lambda_determination, objFun_alpha_z, objFun_alpha_MAD, objFun_alpha_MADini
    from outlierdetection_Online_EWMA import Outlier_Detection_Online_EWMA, calc_z, calc_forecast
    import pandas as pd
    # DATA : Original data to filter.
    #           Column 1 = date of the observation in Matlab format.
    #           Column 2 = raw observation.
    # PARAM : Structure of paramseters. Should be initialized by the function
    #         DefaultParam.m and the calibration from the function ModelCalib.m
    #         should be done to ensure that all paramseters are properly
    #         initialized.
    # Calibperiod: The times series selected in your general script for the
    # calibration model
    # channel: It's the number of paramseter which you want to treat in your
    # structure.

    # OUTPUT:
    # The Output are explained in each function below:

    # ###############Automatic calibration of some paramseters.#################
    try:
        start = pd.to_datetime(calibperiod['start'])
    except Exception:
        raise Exception('Did not find the start of calib period in channel')
    try:
        end = pd.to_datetime(calibperiod['end'])
    except Exception:
        raise Exception('Did not find the end of calib period in channel')
    calibdata = data[start:end]
    params = ModelCalib(calibdata, params)

    # #######################Find the outliers#################################
    newdata = Outlier_Detection_Online_EWMA(data, params)
    
    return newdata, params
