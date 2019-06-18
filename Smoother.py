def kernel_smoother(channel):
    # A kernel smoother with a bandwith H is used to smooth the data serie. The
    # filter does not consider the time step between observations. If the
    # timestep is not constant or close to constant, unrealistic results may
    # arise.
    # Some cross-validation procedure of the kernel was available in previous
    # version, but was removed due to lack of comment on the procedure.
    #
    # ACCEPTEDDATA  : A vector containing data filtered for outliers.
    # H             : Parameter to compute the bandwidth of the filter. Large
    #                 value indicate that the filter weights the smoothed data
    #                 over a large quantity of observations while small value
    #                 smooths much less.
    # SMOOTHED_ACCEPTEDDATA : The final smooth data without outlier.
    # ERR           : Difference between the smoothed data and the accepted
    #                 data.
    import pandas as pd
    import numpy as np

    # Initilization of INPUT:

    param = channel.params
    h = param['data_smoother']['h_smoother']
    method = channel.info['current_filtration_method']
    df = channel.filtered[method].copy(deep=True)
    if 'Accepted' not in df.columns:
        raise Exception('Outlier detection has not been done yet.')
    AD = np.array(df['Accepted']).flatten()
    n_dat = len(AD)
    # Smoothed values
    SmoothedAD = np.full((n_dat,), np.nan)
    for col in ['Q_range', 'Q_slope', 'Q_std', 'Q_range', 'treated', 'deleted']:
        if col in df.columns:
            df = df.drop(col, axis=1)
            print('cols dropped by smoother')
    idx = np.linspace(-h, h, 2 * h + 1)
    idx = [int(i) for i in idx]

    # Denominator weight vector of all filtered values
    W_DENUM = [1 / np.sqrt((2 * np.pi)) * np.exp(-np.square(-idx[i] / h) / 2) for i in idx]
    sum_denum = np.sum(W_DENUM)
    # Calculation of the smoothed data
    for x in range(h, len(AD) - h):
        # Weighting the observations (numerator of the filtered value)
        W_NUM = [AD[int(i + x)] * W_DENUM[i] for i in idx]

        # The NaN values must be removed from the filter. REAL_IDX is the index
        # of the real values to weight.
        # x = x[~n.isnan(x)]
        real_idx = []
        for i in idx:
            if not np.isnan(W_NUM[i]):
                real_idx.append(i)
        for i in idx:
            if i not in real_idx:
                W_NUM.pop(i)
        sum_num = np.sum(W_NUM)

        # Calculation of the smoothed value
        SmoothedAD[x] = sum_num / sum_denum

    # Calculation of the residuals between the smoothed data and the accepted data
    err = SmoothedAD - AD

    df['Smoothed_AD'] = SmoothedAD
    df['err'] = err
    channel.filtered[method] = df
    return channel
