def calc_z(datapoint, alpha, z_prev):
    z=[0,0,0] 
    z[0] = (alpha * datapoint) + ((1 - alpha) * z_prev[0])
    z[1] = alpha * z[0] + (1 - alpha) * z_prev[1]
    z[2] = alpha * z[1] + (1 - alpha) * z_prev[2]
    return z

def calc_forecast(alpha,z):
    a = 3 * z[0] - 3 * z[1] + z[2]
    b = (alpha / (2 * (1 - alpha)**2)) * ((6 - 5*alpha) * z[0] - 2 * (5- 4*alpha) * z[1] + (4- 3*alpha) * z[2])
    c = (alpha/(1-alpha))**2 * (z[0]-2*z[1]+z[2])
    return a + b + 0.5 * c

def Outlier_Detection_Online_EWMA(newData, param):
    import numpy as np
    import pandas as pd

    #warning 'off'
    # ## CG Modification: Definition of the Outlier vector has changed!!! An
    #                     outlier flag is 1 and an accepted value flag is 0.
    #                     This is consistent with all other outlier detection
    #                     methods implemented.
    # INPUT:
    # DATA : Original data to filter. 
    #           Column 1 = date of the observation in Matlab format.
    #           Column 2 = raw observation.
    # PARAM : Structure of parameters. Should be initialized by the function
    #         DefaultParam.m and the calibration from the function ModelCalib.m
    #         should be done to ensure that all parameters are properly
    #         initialized.
    # OUTPUT:
    # ACCEPTED_DATA : Data without outliers, but unfiltered.
    # SEC_RESULTS : A structure containing secondary results. Specifically:
    #   Orig :      Original dataset, for reference
    #   forecast_outlier : Forecast of the data based on the outlier filter.
    #   UpperLimit_outlier : limit above which an observation becomes an
    #                        outlier.
    #   LowerLimit_outlier : Limit under which an observation becomes an
    #                        outlier.
    #   outlier     : Detected outliers. outlier(i) = 1 for detected outlier.
    #                                    outlier(i) = 0 for accepted data.
    #   out_of_control_outlier : Data was in or out of control. 
    #                               = 1 for "Out of control"
    #                               = 0 for "In control"
    #   reini_outlier : Indicates points of reinitialization of the outlier
    #                   filter when out of control.


    #A third-order smoothing model is used to predict the forecast value at time
    #t+1. 

    #A first-order smoothing model is used to predict the forecast mean absolute deviation at
    #time t+1. 

    name = newData.columns[0]
    n_dat = len(newData)
    RawData = np.array(newData).flatten()
    #i_0 = 0
    #i_F = n-1

    #DATA : Time serie with the time and its corresponding value
    #ALPHA_Z and ALPHA_MAD : Smoothing parameters
    K = param['nb_s'] # /**/!!!!! : Number of standard deviation unit used for the calculation of the prediction interval

    MAD_ini = param['MAD_ini'] # Initial mean absolute deviation used to start or reinitiate the outlier detection method
    nb_reject = param['nb_reject'] # Number of consecutive rejected data needed to reinitiate
    #the outlier detection method.  When nb_reject data are rejected, this is called an out-of-control.
    nb_backward = param['nb_backward'] # Number of data before the last rejected data(the last of nb_reject data) where the outlier detection method 
    #is reinitialization for a forward application.
    alpha_z    = param['lambda_z']
    alpha_MAD  = param['lambda_MAD']
    min_MAD    = param['min_MAD']

    # Creation of the matrices

    # Accepted data and the forecast values which replaced the outliers. The
    # vector is initialized to NaN to prevent zeros from appearing at the
    # beginning and the  of the vector.
    AcceptedData = np.full((n_dat,),np.nan)

    # smoothed values
    z          = np.zeros([3,1])  
    z_previous = np.zeros([3,1])

    # mean absolute deviations
    MAD = np.full((n_dat,),np.nan)

    # forecast error standard deviation values
    s = np.full((n_dat,),np.nan)

    # OUTLIER : quality indicator => Data accepted = 0, Data rejected = 1
    outlier = np.zeros((n_dat,)) 

    # Limits of the prediction interval
    LowerLimit = np.full((n_dat,),np.nan)
    UpperLimit = np.full((n_dat,),np.nan)

    # Matrices used to recover data lost after a backward application of the 
    # outlier detection method. Used in the Backward_Method.m script. 
    ##ok<*NASGU> 
    back_AcceptedData = np.full((n_dat,),np.nan) # Accepted data and the forecast values which replaced the outliers
    back_LowerLimit   = np.full((n_dat,),np.nan) # Lower limit values of the prediction interval
    back_UpperLimit   = np.full((n_dat,),np.nan) # Upper limit values of the prediction interval
    back_outlier      = np.full((n_dat,),0.) # Quality indicator => Data rejected=0, Data accepted=1
    forecast     = np.full((n_dat,),np.nan) # Forecast values
    out_of_control    = np.full((n_dat,),0.) # If out_of_control(i) = 1, there is an out of control situation.
    reini             = np.full((n_dat,),1.)# It indicates where the reinitialisation occurs

    #smoothed value
    z_previous[0] = RawData[0]
    z_previous[1] = RawData[0]
    z_previous[2] = RawData[0]

    z = calc_z(RawData[1], alpha_z, z_previous)

    #forecast value
    forecast = calc_forecast(alpha_z,z)

    MAD[2] = MAD_ini     # Initial MAD value
    s[2] = 1.25 * MAD[2] # Initial forecast error standard deviation value
    LowerLimit[2] = forecast - K * s[2] # Lower limit value
    UpperLimit[2] = forecast + K * s[2] # Upper limit value

    # Application of the outlier detection method
    for i in range(2,n_dat-3): # Without reinitialization
        if (RawData[i] < UpperLimit[i]) and  (RawData[i] > LowerLimit[i]):
            # The data is inside the prediction interval
            AcceptedData[i]=RawData[i] # Accepted data

            outlier[i] = 0 # Quality indicator- Accepted=1

            # Calculation of the smoothed value
            z_previous = z
                        
            z = calc_z(AcceptedData[i], alpha_z, z_previous)

            # Calculation of the MAD
            MAD[i+1] = abs(alpha_MAD * (AcceptedData[i] - forecast)) + (1 - alpha_MAD) * MAD[i] 
            MAD[i+1] = max([MAD[i+1],min_MAD])

            # Calculation of the forecast error standard deviation
            s[i+1] = 1.25 * MAD[i+1]  

            # Calculation of the model parameter value
            forecast = calc_forecast(alpha_z,z)

            # Calculation of the prediction interval for the next data
            LowerLimit[i+1] = forecast - K * s[i+1]
            UpperLimit[i+1] = forecast + K * s[i+1]

        else:
            # The data is out of the prediction interval = Outlier

            # Outlier are replaced by the forecast value
            AcceptedData[i] = forecast 
            
            outlier[i] = 1

            # Last MAD value kept
            MAD[i+1] = MAD[i]
            MAD[i+1] = max([MAD[i+1],min_MAD])

            # Last forecast error standard deviation value
            s[i+1] = s[i]

            # Last value kept for the next limit
            LowerLimit[i+1] = forecast - K * s[i+1]
            UpperLimit[i+1] = forecast + K * s[i+1]
        


    # Detection of outliers with reinitialization if an "out-of-control"
    # situation is encountered.
    for i in range(nb_reject, n_dat - 3):
        if (RawData[i] < UpperLimit[i]) and (RawData[i] > LowerLimit[i]): 
            # The data is inside the prediction interval
            
            AcceptedData[i] = RawData[i] # Accepted data

            outlier[i] = 0 # Quality indicator: Accepted = 1

            # Calculation of the smoothed value
            z_previous = z
            z = calc_z(AcceptedData[i], alpha_z, z_previous)

            # Calculation of the mean absolute deviation
            MAD[i+1] = abs(alpha_MAD * (AcceptedData[i] - forecast)) + (1 - alpha_MAD) * MAD[i]
            MAD[i+1] = max([MAD[i+1],min_MAD])

            # Calculation of the forecast error standard deviation value 
            s[i+1] = 1.25*MAD[i+1]

            # Calculation of the forecast value
            forecast = calc_forecast(alpha_z,z)

            # Calculation of the prediction interval for the next data
            LowerLimit[i+1] = forecast - K * s[i+1]
            UpperLimit[i+1] = forecast + K * s[i+1]
        else:
            # The data is out of the prediction interval = Outlier

            AcceptedData[i] = forecast # Outlier are replaced by the forecast value
            # outlier(i) = 0 # Quality indicator- outlier=0
            outlier[i] = 1 # Quality indicator- outlier=0

            MAD[i+1] = MAD[i] # Last MAD kept
            MAD[i+1] = max([MAD[i+1],min_MAD])

            s[i+1] = s[i] # Last forecast error standard deviation value kept

            # Last value kept for the next limit
            LowerLimit[i+1] = forecast - K * s[i+1]
            UpperLimit[i+1] = forecast + K * s[i+1]
        

        # Reinitialization in case of an out of control, which caused by nb_reject data detected as outlier

        # The outlier detection method is reinitiated nb_reject data later and 
        # is applied backward to recovered the lost data. After that,
        # it is applied forward again nb_backward data before the last data 
        # rejected(the last of nb_reject data).

        # Backward application of the outlier detection method 
        if out_of_control[i-nb_reject:i].sum() == 0:
            # if    outlier((i-nb_reject):i) == 0
            if outlier[i-nb_reject:i].sum() == nb_reject:
                ####### BACKWARD METHOD ########
                # If nb_reject data are detected as outlier, there is 
                # a reinitialization of the outlier detection method

                # In_control = 0, out_of_control = 1. It marks where the out of 
                # control starts and where the reinitialization is done
                out_of_control[i-nb_reject:i] = 1
                reini[i] = 1

                # Backward reinitialization
                # smoothed value
                z_previous = [RawData[i+3],RawData[i+3],RawData[i+3]]

                z = calc_z(RawData[i+2], alpha_z, z_previous)
                 
                z_previous = z
                z = calc_z(RawData[i+1], alpha_z, z_previous)

                forecast = calc_forecast(alpha_z, z)
                
                # Initial MAD value
                MAD[i] = MAD_ini
                # Initial forecast error standard deviation value
                s[i]=1.25*MAD[i]

                back_LowerLimit[i] = forecast - K * s[i]   # Lower limit value
                back_UpperLimit[i] = forecast + K * s[i]   # Upper limit value

                # Backward application of the outlier detection method
                for q in range(nb_reject):
                    f=(i-q)+1 #Used for backward

                    if (RawData[f] > back_UpperLimit[f]) or (RawData[f] < back_LowerLimit[f]): 
                        # The data is out of the prediction interval and marked as "Outlier"

                        back_AcceptedData[f] = forecast # Outlier is replaced by the forecast value
                        # back_outlier(f) = 0    # Quality indicator - outlier=0
                        back_outlier[f] = 1    # Quality indicator - outlier=0

                        MAD[f-1] = MAD[f]  # Last MAD kept
                        MAD[f-1] = max([MAD[f-1],min_MAD])

                        s[f-1] = s[f]      # Forecast error standard deviation value kept

                        # Last value kept for the next limit
                        back_LowerLimit[f-1] = forecast - K * s[f-1]
                        back_UpperLimit[f-1] = forecast + K * s[f-1]
                    else:
                        #If the data is inside the prediction interval
                        back_AcceptedData[f] = RawData[f]    # Accepted data
                        # back_outlier[f] = 1    # Quality indicator -Accepted=1
                        back_outlier[f] = 0    # Quality indicator -Accepted=1

                        # Calculation of the smoothed value
                        z_previous = z
                        z = calc_z(RawData[f], alpha_z, z_previous)

                        # Calculation of the MAD
                        MAD[f-1] = abs(alpha_MAD * (back_AcceptedData[f] - forecast)) +(1 - alpha_MAD) * MAD[f]    
                        MAD[f-1] = max([MAD[f-1],min_MAD])

                        # Calculation of the  forecast error standard deviation value 
                        s[f-1] = 1.25 * MAD[f-1]   

                        # Calculation of the forecast value
                        
                        forecast = calc_forecast(alpha_z, z)

                        # Calculation of the prediction interval for the next data
                        back_LowerLimit[f-1] = forecast - K * s[f-1]
                        back_UpperLimit[f-1] = forecast + K * s[f-1]
                    


                # Forward application of the outlier detection method
                # The forward reinitialization is done nb_backward data before the last 
                # rejected data which has caused an out of control situation

                # Forward reinitialization
                # Smoothed value
                prev_raw = RawData[i-nb_backward-4]
                z_previous = [prev_raw, prev_raw, prev_raw]

                prev_raw2 = RawData[i-nb_backward-3]
                z = calc_z(prev_raw2, alpha_z, z_previous)

                prev_raw3 = RawData[i-nb_backward-2]
                z_previous = z     
                z = calc_z(prev_raw3, alpha_z, z_previous)
           
                prev_raw4 = RawData[i-nb_backward-1]
                z_previous = z
                z = calc_z(prev_raw4, alpha_z, z_previous)
           
                # Model parameter value
                forecast = calc_forecast(alpha_z, z)

                MAD[i-nb_backward] = MAD_ini # Initial MAD value
                s[i-nb_backward] = 1.25 * MAD[i-nb_backward]   # Initial forecast error standard deviation value

                LowerLimit[i-nb_backward] = forecast - K * s[i-nb_backward] # Lower limit value
                UpperLimit[i-nb_backward] = forecast + K * s[i-nb_backward] # Upper limit value   

                # Forward application of the outlier detection method
                for k in range(i-nb_backward-1,i):
                    if (RawData[k] < UpperLimit[k]) or (RawData[k] > LowerLimit[k]): 
                        # The data is out of the prediction interval = Outlier 

                        AcceptedData[k] = forecast # Outlier is replaced by the forecast
                        # outlier(k) = 0     # Quality indicator- outlier=0
                        outlier[k] = 1     # Quality indicator- outlier=0

                        MAD[k+1] = MAD[k]  # Last MAD kept
                        MAD[k+1] = max([MAD[k+1],min_MAD])
                        
                        s[k+1] = s[k]      # Last forecast error standard deviation value kept

                        # Last value kept for the next limit
                        LowerLimit[k+1] = forecast - K * s[k+1]
                        UpperLimit[k+1] = forecast + K * s[k+1]

                    else:
                        #The data is inside the prediction interval
                        AcceptedData[k] = RawData[k]   # Accepted data

                        # outlier(k) = 1 # Quality indicator -Accepted=1
                        outlier[k] = 0 # Quality indicator -Accepted=1

                        # Calculation of the smoothed value
                        z_previous = z
                        z = calc_z(AcceptedData[k], alpha_z, z_previous)

                        MAD[k+1] = abs(alpha_MAD * (AcceptedData[k] - forecast)) + (1 - alpha_MAD) * MAD[k] # Calculation of the MAD
                        MAD[k+1] = max([MAD[k+1],min_MAD])

                        s[k+1] = 1.25 * MAD[k+1]   # Calculation of the forecast error standard deviation value 

                        # Calculation of the model parameter value
                        
                        forecast = calc_forecast(alpha_z, z)

                        # Calculation of the prediction interval for the next data
                        LowerLimit[k+1] = forecast - K * s[k+1]
                        UpperLimit[k+1] = forecast + K * s[k+1]
                    


                #After the backward application of the outlier detection method, only the first half of the data between where the forward and
                #backward applications are reinitialize is kept .  After the forward application of the outlier detection method, only the second half of the data between where the forward and
                #backward applications are reinitialize is kept. In this way, the prediction interval is adapted to the data serie before the application pf the outlier detection method. 
                strt=i-nb_reject+1
                mid = int(i-np.floor(nb_backward/2))
                AcceptedData[strt:mid] = back_AcceptedData[strt:mid]
                UpperLimit[strt:mid]   = back_UpperLimit[strt:mid]
                LowerLimit[strt:mid]   = back_LowerLimit[strt:mid]
                outlier[strt:mid]      = back_outlier[strt:mid]
            
        


    # Filter last



    # Record the position of the next data to filter at next iteration.

    #index0 = len(RawData) - 3

    # Generation of outputs
    Sec_data = {
        name+'_forecast':forecast,
        name+'_UpperLimit_outlier':UpperLimit,
        name+'_LowerLimit_outlier':LowerLimit,
        name+'_outlier': [x==1 for x in outlier],
        name+'_out_of_control_outlier': out_of_control,
        name+'_reini_outlier':reini
    }
    Sec_Results = pd.DataFrame(index = newData.index, data =Sec_data)

    newData[name+'_Accepted'] = AcceptedData
    newData = pd.concat([newData, Sec_Results],axis=1)

    return newData

