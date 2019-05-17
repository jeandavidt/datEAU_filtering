def Data_Coherence(Data, param):
    import warnings
    # This function allows to detect most common data problems. It does not
    # alter the database, but returns meaningful errors or warning codes.
    # DATA : The data to filter is a two-columns matrix. The first column
    #        contains the date in the internal Matlab format. The second
    #        contains the observation to filter.
    # PARAM: The structure containing the parameters of the filter. Some
    #        parameters allow to classify problematic situations as warning or
    #        errors.
    # FLAG : The flag returns a code that deps on the DATA. If no error or
    #        warning is encontered, 0 is returned. A warning has a value > 0
    #        and an error has a value < 0. Multiple warnings/errors are
    #        returned in a vector.
    #   0: No error was encontered.
    #   1: NaN were detected in the dates or in the observations. A NAN
    #      prevents the outlier filter to work properly, but the filter can
    #      recover once more than NB_REJECT real values are present. The
    #      weigthed average filter do not filter NAN values (they are rejected
    #      from calculation).
    #   2: A variable time step was detected. The filter is not designed
    #      for varying time step. 
    #   3: A large gap in data was detected. The width of the gap is compared
    #      to the parameter nb_reject. The user should fill the gap with NAN
    #      values. These values will not be filtered, but will prevent
    #      unpredictable or undesirable effect from the filter.

    #   4: Negative time step was detected. T_(i+1) < T_(i). This could cause
    #      improper behavior of the filters. A common fix is to sort data with
    #      respect to time and to keep unique value for each time step.

    # Additional tests on raw data can be made here. absolute FLAG code should be
    # sequential, so errors can be converted into warning once a solution is
    # proposed.

    import pandas as pd
    import numpy as np

    nb_reject = param['nb_reject']

    definition_0 = 'No error was encontered'
    definition_1 = 'NaN were detected in the dates or in the observations. A NAN prevents the outlier filter to work properly, but the filter can recover once more than NB_REJECT real values are present. The weigthed average filter do not filter NAN values (they are rejected from calculation).'
    definition_2 = 'A variable time step was detected. The filter is not designed for varying time step. '
    definition_3 ='A large gap in data was detected. The width of the gap is compared to the parameter nb_reject. The user should fill the gap with NAN values. These values will not be filtered, but will prevent unpredictable or undesirable effect from the filter.'
    definition_4 ='Negative time step was detected. T_(i+1) < T_(i). This could cause improper behavior of the filters. A common fix is to sort data with respect to time and to keep unique value for each time step.'

    flag = {}

    # Check for NaN
    n_nulls = Data.isnull().sum()
    if n_nulls.any():
        if param['Verbose']:
            warnings.warn('DataCoherence warning: NaN values are present in the dataset')
        
        flag[1] = definition_1
    
    # Check for variable time step
    Time = pd.Series(Data.index.astype('int64')/10**9)
    dT = Time.diff().dropna()
    
    maxDT = dT.max()
    minDT = dT.min()
    medDT = np.median(dT)
    if ((maxDT > (1 + param['DT_RelRol']) * medDT) or (minDT < (1 - param['DT_RelRol']) * medDT)):
        # Check if the largest variation in the timestep is too important
        if maxDT > nb_reject * minDT:
            if param['Verbose']:
                warnings.warn('DataCoherence warning: Large gap is present in the dataset')
            
            flag[2] = definition_2
        
        if param['Verbose']:
            warnings.warn('DataCoherence warning: the timestep is not constant')
        
        flag[3] = definition_3

    if minDT < 0:
        if param['Verbose']:
            warnings.warn('DataCoherence warning: Negative time step found.')
            
        flag[4] = definition_4
    
    if not flag:
        flag[0] = definition_0

    return flag 