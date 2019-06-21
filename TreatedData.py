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
    dat = channel.filtered[filtration_method].copy(deep=True)
    params = channel.params

    corr_min = params['fault_detection_uni']['corr_min']
    corr_max = params['fault_detection_uni']['corr_max']
    slope_min = params['fault_detection_uni']['slope_min']
    slope_max = params['fault_detection_uni']['slope_max']
    std_min = params['fault_detection_uni']['std_min']
    std_max = params['fault_detection_uni']['std_max']
    range_min = params['fault_detection_uni']['range_min']
    range_max = params['fault_detection_uni']['range_max']

    if 'raw' not in dat.columns:
        dat = dat.join(channel.raw_data['raw'], how='left')

    dat['val_corr'] = dat['Q_corr'].between(corr_min, corr_max)
    dat['val_slope'] = dat['Q_slope'].between(slope_min, slope_max)
    dat['val_std'] = dat['Q_std'].between(std_min, std_max)
    dat['val_range'] = dat['Q_slope'].between(slope_min, slope_max)
    dat['val_range'] = dat['Smoothed_AD'].between(range_min, range_max)
    dat['treated'] = dat.val_corr & dat.val_slope & dat.val_std & dat.val_range
    dat['deleted'] = ~ dat.treated
    dat['treated'] = (dat['treated'] * dat['Smoothed_AD']).replace(0, np.nan)
    dat['deleted'] = (dat['deleted'] * dat['Smoothed_AD']).replace(0, np.nan)

    n_del = (dat['deleted'] > 0).sum()
    n_dat = len(dat)
    n_outlier = (dat['outlier'] > 0).sum()
    dat.drop(['val_corr', 'val_slope', 'val_std', 'val_range', 'raw'], axis=1)

    channel.filtered[filtration_method] = dat
    channel.info['filtration_results'][filtration_method] = {}
    channel.info['filtration_results'][filtration_method]['percent_outlier'] = n_outlier / n_dat * 100
    channel.info['filtration_results'][filtration_method]['percent_loss'] = (n_del) / n_dat * 100
    channel.info['send_to_multivar'] = 'treated'
    return channel
