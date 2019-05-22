def TreatedD(df, param, name ):
    import numpy as np
    import pandas as pd
    #SMOOTHED_ACCEPTEDDATA
    #This function allows to validate or not the smoother data. For that, we
    #compare the data feature calculation (Q_corr, Q_std, Q_slope, Q_Range and
    #Q_mobilerange). 
    #How accepted a data: 
    #0: deleted data 1: treated data. 

    #INPUT: 
    #nb_data: data number in the serie
    #Data: raw data
    #SMOOTHED_ACCEPTEDDATA: data smoothed for a parameter
    #Q_corr: a calculated score
    #Q_std: a calcultated score
    #Q_range: a calculated score
    #Q_slope: a calculated score
    #Q_mobilerange: a calculated score
    #param: parameters which find in the function "default_param"

    #OUPUT: 
    #a structure with the treated and deleted data. Two matrix: 
    #Treateddata: treated data
    #Deleteddata: deleted data 


    #Initialisation of Inputs: Here, it takes back the different scores and the
    #smoothed data. 
    Raw = np.array(df[name]).flatten()
    outlier = np.array(df[name+"_outlier"]).flatten()
    Smoothed_AD = np.array(df[name+"_Smoothed_AD"]).flatten()
    Q_corr = np.array(df[name+"_Qcorr"]).flatten()
    Q_slope = np.array(df[name+"_Qslope"]).flatten()
    Q_std = np.array(df[name+"_Qstd"]).flatten()
    Q_range = np.array(df[name+"_Qrange"]).flatten()

    # Test correlation limits:
    Val_corr_min = Q_corr > param['corr_min']
    Val_corr_max = Q_corr < param['corr_max']
    Val_slope_min = Q_slope > param['slope_min']
    Val_slope_max = Q_slope < param['slope_max']
    Val_std_min = Q_std > param['std_min']
    Val_std_max = Q_std < param['std_max']


    # Aggregate values that comply both with minimum correlation and maximum correlation.
    idx_corr = Val_corr_min * Val_corr_max
    idx_slope = Val_slope_min * Val_slope_max
    idx_std = Val_std_min * Val_std_max
    idx_range = Q_range

    idx_tot = idx_corr * idx_slope * idx_std * idx_range


    #Generation of outputs: 
    Treateddata =  Smoothed_AD * idx_tot.astype('int')
    Deleteddata = Smoothed_AD * ~idx_tot.astype('int')
    
    for i in range(len(Smoothed_AD)):#i=1:length (Smoothed_AD):
        
        if idx_tot[i] == 0:
            Treateddata[i] = np.nan
        
        
        elif idx_tot[i] == 1:
            Deleteddata[i] = np.nan
        
    
    
    #Generalization of OUPUTS:
    Final_D = pd.DataFrame(data={
        name+"_raw":Raw,
        name+'_outlier': outlier,
        name+'_Treated':Treateddata, 
        name+'_Deleted':Deleteddata
        })
  
    return Final_D

def InterpCalculator(final_df, name):
    import numpy as np
    import pandas as pd
    #This function allows to calculate different variable to interprate the
    #data filtration. 

    #Input: 
    # Data: Raw data 
    # Outlier: the outlier obtained
    # DataValidated: data 

    #Output:
    # PercenOutlier: Outlier percentage
    #LosingData: number of data lose. 

    #Initialization of INPUTS:
    DATA = np.array(final_df[name+'_raw']).flatten()
    Outlier = np.array(final_df[name+'_outlier']).flatten()
    Deleteddata = np.array(final_df[name+'_Deleted']).flatten()

    Datatot = len(DATA)

    #Percentage Outliers:
    Out = Outlier[Outlier==1]
    Outtot = len(Out)
    PercenOutlier = ( Outtot/ Datatot) * 100

    #Percentage data losing
    Deleted = len(Deleteddata[Deleteddata>0])

    PerceletedData = (100-((Datatot - Deleted)/ Datatot) * 100)

    #Generalization Outputs: 
    Intervariable={}
    Intervariable[name+'_PercenOutlier'] = PercenOutlier
    Intervariable[name+'_PercenLosingData'] = PerceletedData

    


    return Intervariable