def data_coherence(channel):
    import warnings
    # This function allows to detect most common data problems. It does not
    # alter the database, but returns meaningful errors or warning codes.
    # DATA : The data to filter is a two-columns matrix. The first column
    #        contains the date in the internal Matlab format. The second
    #        contains the observation to filter.
    # PARAM: The structure containing the parameters of the filter. Some
    #        parameters allow to classify problematic situations as warning or
    #        errors.
    # FLAG : The flag returns a code that depends on the DATA. If no error or
    #        warning is encountered, 0 is returned. A warning has a value > 0
    #        and an error has a value < 0. Multiple warnings/errors are
    #        returned in a vector.
    #   0: No error was encountered.
    #   1: NaN were detected in the dates or in the observations. A NAN
    #      prevents the outlier filter to work properly, but the filter can
    #      recover once more than NB_REJECT real values are present. The
    #      weighted average filter do not filter NAN values (they are rejected
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

    series = channel.info['last-processed']
    if series == 'raw':
        data = channel.raw_data
    else:
        data = pd.DataFrame(channel.processed_data[series])

    nb_reject = channel.params['outlier_detection']['nb_reject']
    param = channel.params

    definition_0 = 'No error was encountered'
    definition_1 = 'NaN were detected in the dates or in the observations.'
    # A NAN prevents the outlier filter to work properly, but the filter
    # can recover once more than NB_REJECT real values are present.
    # The weighted average filter do not filter NAN values (they are rejected from calculation).

    definition_2 = 'A variable time step was detected.'
    # The filter is not designed for varying time step. '

    definition_3 = 'A large gap in data was detected. The width of the gap is compared to the parameter nb_reject.'
    # The user should fill the gap with NAN values. These values will not be
    # filtered, but will prevent unpredictable or undesirable effect from the filter.'

    definition_4 = 'Negative time step was detected. T_(i+1) < T_(i).'
    # This could cause improper behavior of the filters. A common fix
    # is to sort data with respect to time and to keep unique value for each time step.'

    flag = {}

    # Check for NaN
    n_nulls = data.isnull().sum()
    if n_nulls.any():
        if param['general']['Verbose']:
            warnings.warn('DataCoherence warning: NaN values are present in the dataset')
        flag[1] = definition_1

    # Check for variable time step
    Time = pd.Series(data.index.astype('int64') / 10 ** 9)
    dT = Time.diff().dropna()

    maxDT = dT.max()
    minDT = dT.min()
    medDT = np.median(dT)
    if (maxDT > (1 + param['data_coherence']['DT_RelRol']) * medDT) or \
            (minDT < (1 - param['data_coherence']['DT_RelRol']) * medDT):
        # Check if the largest variation in the timestep is too important
        if maxDT > nb_reject * minDT:
            if param['general']['Verbose']:
                warnings.warn('DataCoherence warning: Large gap is present in the dataset')
            flag[2] = definition_2

        if param['general']['Verbose']:
            warnings.warn('DataCoherence warning: the timestep is not constant')
        flag[3] = definition_3

    if minDT < 0:
        if param['general']['Verbose']:
            warnings.warn('DataCoherence warning: Negative time step found.')
        flag[4] = definition_4

    if not flag:
        if param['general']['Verbose']:
            warnings.warn('DataCoherence warning: No error was encountered.')
        flag[0] = definition_0

    return flag


def sort_dat(channel):
    import pandas as pd
    series = channel.info['last-processed']
    if series == 'raw':
        data = channel.raw_data
    else:
        data = pd.DataFrame(channel.processed_data[series])
    grouped = data.groupby(level=0)
    sorted_data = grouped.last()
    sorted_data.columns = ['sorted']
    if series == 'raw':
        channel.processed_data = sorted_data
    else:
        channel.processed_data['sorted'] = sorted_data

    channel.info['last-processed'] = 'sorted'
    channel.info['send_to_multivar'] = 'sorted'
    return channel


def resample(channel, timestep):
    import pandas as pd
    series = channel.info['last-processed']
    if series == 'raw':
        data = channel.raw_data
    else:
        data = pd.DataFrame(channel.processed_data[series])
    data = data[~data.index.duplicated()]
    resampled = data.asfreq(timestep)  # write seconds as 'x s', minutes as 'x min'
    resampled.columns = ['resampled']

    channel.processed_data = resampled
    channel.info['last-processed'] = 'resampled'
    channel.info['send_to_multivar'] = 'resampled'
    return channel


def fillna(channel):
    import pandas as pd
    series = channel.info['last-processed']
    if series == 'raw':
        data = channel.raw_data
    else:
        data = pd.DataFrame(channel.processed_data[series])
    filled = data.fillna(method='ffill')
    filled.columns = ['filled']
    if series == 'raw':
        channel.processed_data = filled
    else:
        channel.processed_data['filled'] = filled
    channel.info['last-processed'] = 'filled'
    channel.info['send_to_multivar'] = 'filled'
    return channel
