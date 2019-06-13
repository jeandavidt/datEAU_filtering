def TreatedD(channel):
    # def TreatedD(df, param, name):
    import numpy as np
    import pandas as pd
    # SMOOTHED_ACCEPTEDDATA
    # This function allows to validate or not the smoother data. For that, we
    # compare the data feature calculation (Q_corr, Q_std, Q_slope, Q_Range and
    # Q_mobilerange).
    # How accepted a data:
    # 0: deleted data 1: treated data.

    # INPUT:
    # nb_data: data number in the serie
    # Data: raw data
    # SMOOTHED_ACCEPTEDDATA: data smoothed for a parameter
    # Q_corr: a calculated score
    # Q_std: a calcultated score
    # Q_range: a calculated score
    # Q_slope: a calculated score
    # Q_mobilerange: a calculated score
    # param: parameters which find in the function "default_param"

    # OUPUT:
    # a structure with the treated and deleted data. Two matrix:
    # Treateddata: treated data
    # Deleteddata: deleted data

    # Initialisation of Inputs: Here, it takes back the different scores and the
    # smoothed data.
    filtration_method = channel.info['current_filtration_method']
    df = channel.filtered[filtration_method]
    params = channel.params

    raw = np.array(channel.raw_data['raw']).flatten()
    raw_out = channel.raw_data.join(df['outlier'], how='left').dropna()
    outlier = np.array(raw_out['raw'].loc[raw_out['outlier']]).flatten()
    Smoothed_AD = np.array(df["Smoothed_AD"]).flatten()
    Q_corr = np.array(df["Q_corr"]).flatten()
    Q_slope = np.array(df["Q_slope"]).flatten()
    Q_std = np.array(df["Q_std"]).flatten()
    Q_range = np.array(df["Q_range"]).flatten()

    # Test correlation limits:
    with np.errstate(invalid='ignore'):
        Val_corr_min = Q_corr > params['fault_detection_uni']['corr_min']
        Val_corr_max = Q_corr < params['fault_detection_uni']['corr_max']
        Val_slope_min = Q_slope > params['fault_detection_uni']['slope_min']
        Val_slope_max = Q_slope < params['fault_detection_uni']['slope_max']
        Val_std_min = Q_std > params['fault_detection_uni']['std_min']
        Val_std_max = Q_std < params['fault_detection_uni']['std_max']

    # Aggregate values that comply both with minimum correlation and maximum correlation.
    idx_corr = Val_corr_min * Val_corr_max
    idx_slope = Val_slope_min * Val_slope_max
    idx_std = Val_std_min * Val_std_max
    idx_range = Q_range

    idx_tot = idx_corr * idx_slope * idx_std * idx_range

    # Generation of outputs:
    Treateddata = Smoothed_AD * idx_tot.astype('int')
    Deleteddata = Smoothed_AD * ~idx_tot.astype('int') * -1

    for i in range(len(Smoothed_AD)):  # i=1:length (Smoothed_AD):
        if idx_tot[i] == 0:
            Treateddata[i] = np.nan
        elif idx_tot[i] == 1:
            Deleteddata[i] = np.nan

    # Generalization of OUPUTS:
    df['treated'] = Treateddata
    df['deleted'] = Deleteddata

    channel.info['filtration_results'][filtration_method] = {}
    stats = channel.info['filtration_results'][filtration_method]
    stats['percent_outlier'], stats['percent_loss'] = InterpCalculator(raw, outlier, Deleteddata)

    channel.info['filtration_results'][filtration_method] = stats
    channel.filtered[filtration_method] = df
    return channel


def InterpCalculator(raw, outlier, deleted):
    import numpy as np
    import pandas as pd
    # This function allows to calculate different variable to interprate the
    # data filtration.

    # Input:
    # Data: Raw data
    # Outlier: the outlier obtained
    # DataValidated: data

    # Output:
    # PercenOutlier: Outlier percentage
    # LosingData: number of data lose.

    # Initialization of INPUTS:
    DATA = np.array(raw.flatten())
    Outlier = np.array(outlier).flatten()
    Deleteddata = np.array(deleted).flatten()

    Datatot = len(DATA)

    # Percentage Outliers:
    Out = Outlier[Outlier == 1]
    Outtot = len(Out)
    PercenOutlier = (Outtot / Datatot) * 100

    # Percentage data losing
    mask = ~np.isnan(Deleteddata)
    Deleted = len(Deleteddata[mask > 0])

    PerceletedData = (100 - ((Datatot - Deleted) / Datatot) * 100)

    # Generalization Outputs:
    return PercenOutlier, PerceletedData
