def ModelCalib(data, param):
    # New optimization procedure implemented in alpha_z_determination which
    # does both lambda_z and lambda_MAD together

    # This function takes a data time serie and auto-calibrate the outlier
    # filter. The dataset is assumed to be of good quality (no fault sensor and
    # limited amount of outliers) and without significant gaps (missing time
    # steps or NaN). 
    # This smoother is highly inspired from the textbook Introduction to
    # Statistical Quality Control, 6th edition by Douglas Montgomery, section
    # 10.4.2
    # DATA :        The input data matrice of format Nx2. First column is a
    #               date vector compliant with Matlab internal representation
    #               of date (use of DATENUM returns the proper format). The
    #               second column is the signal to be treated.
    # PARAM :       A structure containing all parameters (general and
    #               advanced) to be set by the user. See "DefaultParam.m"
    param['lambda_z'], param['lambda_MAD'], param['MAD_ini'], min_MAD = lambda_determination(data, param)     
    #Automatic estimation: 
    # # if param.h_smoother == 0
    # #     param.h_smoother = h_kernel_determination(AcceptedData) 
    # #
    if param['min_MAD'] == 0:
        param['min_MAD'] = min_MAD

    return param

def lambda_determination(data, param):
    from scipy.optimize import minimize 
    import numpy as np
    # To do : add comments on the optimisation procedure! 

    #This function calculates the right value of lamda_z in the model used to
    #calculate the forecast value of the variable

    #The RMSE is calculated on the forecast and the measured value to choose the
    #optmimum value of the alpha_z 

    #Raw data selection
    db = np.array(data).flatten()

    # The fminsearch provides an unconstrained optimization of lambda_z which
    # is faster and more precise than the previous systematic method. However,
    # in some particular cases, it might be necessary to return to the original
    # method (i.e. optimal lambda_z not between 0 and 1).

    alpha_0 = 1

    result = minimize(objFun_alpha_z, alpha_0, args=(db), bounds=(0.01,1))
    log_lambda_z = result.x[0]
    # put error handling code here in case the optimization fails
    lambda_z   = np.exp(-(log_lambda_z**2))


    result = minimize(objFun_alpha_MAD, alpha_0, args=(lambda_z, db))
    log_lambda_MAD=result.x[0]
    # put error handling code here in case the optimization fails
    lambda_MAD = np.exp(-(log_lambda_MAD**2))

    MAD_ini = param['MAD_ini']
    result = minimize(objFun_alpha_MADini, MAD_ini,args=(lambda_MAD,lambda_z,db,'err'))
    MAD_ini = result.x

    min_MAD = objFun_alpha_MADini(MAD_ini,lambda_MAD,lambda_z,db,'min_MAD')


    # MAD_ini = exp(log_lambda_MAD(2))

    # If the fminsearch optimization failed, the following procedure ensures
    # that lambda_z and lambda_MAD are properly computed. The following
    # procedure is slower on the average case.
    '''if (FLAG ~= 1)
        #Creation of the matrixes
        z=zeros(length(db),3,100)
        a=zeros(len(db))
        b=zeros(len(db))
        c=zeros(len(db))
        forecast=zeros(len(db))
        square_err=zeros(length(db),100) 
        RMSE=zeros(100,1)
        RMSE_z=zeros(100,1)

        alpha_z_vect = 0.01:0.01:1
        for n=1:100
            #The RMSE is calculated for alpha_z values from 0.01 to 1 with an interval of 0.01
            alpha_z = alpha_z_vect(n) 

            #Model initialisation
            #Smoothed value
            z(1,1)=db(1) 
            z(1,2)=db(1)
            z(1,3)=db(1)

            z(2,1)=alpha_z*db(2)+(1-alpha_z)*z(1,1)
            z(2,2)=alpha_z*z(2,1)+(1-alpha_z)*z(1,2)
            z(2,3)=alpha_z*z(2,2)+(1-alpha_z)*z(1,3)

            #Model parameter value
            a(2)=3*z(2,1)-3*z(2,2)+z(2,3)
            b(2)=(alpha_z/(2*(1-alpha_z)^2))*((6-5*alpha_z)*z(2,1)-2*(5-4*alpha_z)*z(2,2)+(4-3*alpha_z)*z(2,3))
            c(2)=(alpha_z/(1-alpha_z))^2*(z(2,1)-2*z(2,2)+z(2,3))

            #Forecast value
            forecast(3)=(a(2)+b(2)*1+(1/2)*c(2)*1^2)

            # Data on which the filter is applied
            i = 3:length(db)

            #Calculation of the smoothed value
            z(i,1,n)=alpha_z*db(i)+(1-alpha_z)*z(i-1,1,n) 
            z(i,2,n)=alpha_z*z(i,1,n)+(1-alpha_z)*z(i-1,2,n)
            z(i,3,n)=alpha_z*z(i,2,n)+(1-alpha_z)*z(i-1,3,n)

            #Calculation of the model parameter value
            a(i)=3*z(i,1,n)-3*z(i,2,n)+z(i,3,n)
            b(i)=(alpha_z/(2*(1-alpha_z)^2))*((6-5*alpha_z)*z(i,1,n)-2*(5-4*alpha_z)*z(i,2,n)+(4-3*alpha_z)*z(i,3,n))
            c(i)=(alpha_z/(1-alpha_z))^2*(z(i,1,n)-2*z(i,2,n)+z(i,3,n))

            #Calculation of the forecast value
            forecast(i+1,n)=(a(i)+b(i)*1+(1/2)*c(i)*1^2)

            #square error between the measured  and the forecast values
            square_err(i,n)=(db(i)-forecast(i,n)).^2

            #Calculation of the RMSE
            RMSE_z(n)=sum(square_err(:,n))/length(db)
        
        
        [~,b]= min(RMSE_z)
        lambda_z = alpha_z_vect(b)
        lambda_MAD = alpha_MAD_determination(data,lambda_z)'''

    return lambda_z, lambda_MAD, MAD_ini, min_MAD


def objFun_alpha_z(log_alpha_z,db):
    import numpy as np
    alpha_z   = np.exp(-(log_alpha_z**2))

    z           = np.zeros(len(db),3)  # Smoothed values
    a           = np.zeros(len(db),)      # Model parameter values 
    b           = np.zeros(len(db),)      # Model parameter values
    c           = np.zeros(len(db),)      # Model parameter values 
    forecast    = np.zeros(len(db),)      # Forecast values

    #Model initialisation
    z[1,0] = db[0]
    z[1,1] = db[0]
    z[1,2] = db[0]

    for i in range(1,len(db)):
        z[i,0] = alpha_z * db[i]  + (1-alpha_z) * z[i-1,0]
        z[i,1] = alpha_z * z[i,0] + (1-alpha_z) * z[i-1,1]
        z[i,2] = alpha_z * z[i,1] + (1-alpha_z) * z[i-1,2]

    a[1:] = 3 * z[1:,0] - 3 * z[1:,1] + z[1:,2]
    b[1:] = (alpha_z / (2 * (1 - alpha_z)^2)) * ((6 - 5 * alpha_z) * z[1:,0] - 2 * (5 - 4 * alpha_z) * z[1:,1] + (4 - 3 * alpha_z) * z[1:,2])
    c[1:] = (alpha_z / (1 - alpha_z))**2 * (z[1:,0] - 2 * z[1:,1] + z[1:,2])

    forecast[2:] = a[1:-1] + b[1:-1] + 0.5 * c[1:-1]

    # square error between the measured  and the forecast values                         
    square_err = (db - forecast)**2

    RMSE_z = np.sqrt((square_err/len(db)).sum())    # Calculation of the RMSE
    return RMSE_z

def objFun_alpha_MAD(log_alpha_MAD, alpha_z, db):
    import numpy as np
    alpha_MAD = np.exp(-(log_alpha_MAD**2))

    z            = np.zeros(len(db),3)  # Smoothed values
    a            = np.zeros(len(db))      # Model parameter values 
    b            = np.zeros(len(db))      # Model parameter values
    c            = np.zeros(len(db))      # Model parameter values 
    forecast     = np.zeros(len(db))      # Forecast values
    MAD          = np.zeros(len(db))
    forecast_MAD = np.zeros(len(db))


    #Model initialisation
    z[0,0] = db[0]
    z[0,1] = db[0]
    z[0,2] = db[0]

    z[1:,0] = alpha_z * db[1:]  + (1-alpha_z) * z[:-1,0]
    z[1:,1] = alpha_z * z[1:,0] + (1-alpha_z) * z[:-1,1]
    z[1:,2] = alpha_z * z[1:,1] + (1-alpha_z) * z[:-1,2]

    for i in range(1,len(db)-1):
        z[i,0] = alpha_z * db[i]  + (1-alpha_z) * z[i-1,0]
        z[i,1] = alpha_z * z[i,0] + (1-alpha_z) * z[i-1,1]
        z[i,2] = alpha_z * z[i,1] + (1-alpha_z) * z[i-1,2]    

    a[1:] = 3 * z[1:,0] - 3 * z[1:,1] + z[1:,2]
    b[1:] = (alpha_z / (2 * (1 - alpha_z)**2)) * ((6 - 5 * alpha_z) * z[1:,0] - 2 * (5 - 4 * alpha_z) * z[1:,1] + (4 - 3 * alpha_z) * z[1:,2])
    c[1:] = (alpha_z / (1 - alpha_z))**2 * (z[1:,0] - 2 * z[1:,1] + z[1:,2])

    forecast[2:] = a[1:-1] + b[1:-1] + 0.5 * c[1:-1]

    # square error between the measured  and the forecast values    
    err = db - forecast
    square_err = err**2

    #Calculation of the forecast MAD
    for i in range(2,len(db)):

        MAD[i]=abs(alpha_MAD*err[i])+(1-alpha_MAD)*MAD[i-1] #Calculation of the MAD
        forecast_MAD[i+1]=MAD[i]#Calculation of the forecast MAD
        square_err[i]=(forecast_MAD[i]-abs(err[i]))**2#Calculation of the square error between the forecast MAD and the absolute deviation between the forecast value and the measured value

    RMSE_mad = (square_err/len(db)).sum() #Calculation of the RMSE

        
    RMSE = np.sqrt(RMSE_mad)
    return RMSE

def objFun_alpha_MADini(MADini, alpha_MAD,alpha_z, db, obj):
    import numpy as np
    z            = np.zeros(len(db),3)  # Smoothed values
    a            = np.zeros(len(db))      # Model parameter values 
    b            = np.zeros(len(db))      # Model parameter values
    c            = np.zeros(len(db))      # Model parameter values 
    forecast     = np.zeros(len(db))      # Forecast values
    MAD          = np.zeros(len(db))

    MAD[0] = MADini
    MAD[1] = MADini

    #Model initialisation
    z[0,0] = db[0]
    z[0,1] = db[0]
    z[0,2] = db[0]

    z[1:,0] = alpha_z * db[1:]  + (1-alpha_z) * z[:-1,0]
    z[1:,1] = alpha_z * z[1:,0] + (1-alpha_z) * z[:-1,1]
    z[1:,2] = alpha_z * z[1:,1] + (1-alpha_z) * z[:-1,2]

    for i in range(1,len(db)-1):
        z[i,0] = alpha_z * db[i]  + (1-alpha_z) * z[i-1,0]
        z[i,1] = alpha_z * z[i,0] + (1-alpha_z) * z[i-1,1]
        z[i,2] = alpha_z * z[i,1] + (1-alpha_z) * z[i-1,2]
        
    a[1:] = 3 * z[1:,0] - 3 * z[1:,1] + z[1:,2]
    b[1:] = (alpha_z / (2 * (1 - alpha_z)**2)) * ((6 - 5 * alpha_z) * z[1:,0] - 2 * (5 - 4 * alpha_z) * z[1:,1] + (4 - 3 * alpha_z) * z[1:,2])
    c[1:] = (alpha_z / (1 - alpha_z))**2 * (z[1:0] - 2 * z[1:,1] + z[1:,2])

    forecast[2:] = a[1:-1] + b[1:-1] + 0.5 * c[1:-1]

    # error between the measured  and the forecast values    
    err = db - forecast

    # Calculation of the forecast MAD
    for i in range(2,len(db)):
        MAD[i]=abs(alpha_MAD*err[i])+(1-alpha_MAD)*MAD[i-1] #Calculation of the MAD
    
    err = abs(MADini - np.mean(MAD[2:]))
    min_MAD = np.median(MAD)
    if obj =='err':
        return err
    elif obj == 'min_MAD':
        return min_MAD
    else:
        raise Exception('Must be either "err" or "min_MAD')