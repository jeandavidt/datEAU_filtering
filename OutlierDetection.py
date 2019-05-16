def OutlierDetection(Data, calibperiod, Channel, param):
    # This function serves as a relay for all Outlier Detection methods
    # implemented. It only transfer the inputs to the chosen function and
    # return the outputs provided.

    #The whole of INPUT and OUTPUT are explained in the three functions :
    #OutlierDetection_EWMA, Outlier_Detection_NeuralNetworks, Outlier_Detection_online

    if 'OutlierDetectionMethod' in param:
        raise Exception('Outlier detection: No method selected (param.OutlierDetectionMethod does not exist')
    
    if param['OutlierDetectionMethod'] is 'EWMA':
        raise Exception('Not implemented yet')
        #Data, param = OutlierDetection_EWMA(Data, calibperiod, Channel, param)
    elif param['OutlierDetectionMethod'] is 'NeuralNetwork':
        raise Exception('Not implemented yet')
        #Data, param = Outlier_Detection_NeuralNetworks(Data, calibperiod, Channel, param)
    elif param['OutlierDetectionMethod'] is'Online_EWMA':
        Data, param = Outlier_Detection_online(Data, calibperiod, Channel, param)
    else:
        raise Exception('Outlier detection: unknown method')
    return Data, param

def Outlier_Detection_online(Data, calibperiod, Channel, param):
    import ModelCalibration
    # DATA : Original data to filter. 
    #           Column 1 = date of the observation in Matlab format.
    #           Column 2 = raw observation.
    # PARAM : Structure of parameters. Should be initialized by the function
    #         DefaultParam.m and the calibration from the function ModelCalib.m
    #         should be done to ensure that all parameters are properly
    #         initialized.
    #Calibperiod: The times series selected in your general script for the
    #calibration model
    #Channel: It's the number of parameter which you want to treat in your
    #structure. 

    # OUTPUT:
    #The Output are explained in each function below: 

    ################Automatic calibration of some parameters.#################
    param = ModelCalib(calibperiod,param)


    ########################Find the outliers################################## 
    
    Data = outlierdetection_Online_EWMA(Data, param )

    return Data, param 


