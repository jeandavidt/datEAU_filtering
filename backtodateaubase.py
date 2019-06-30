import pandas as pd
import numpy as np

path = '../sample_data/influent3short.csv'
raw_data = pd.read_csv(path, sep=';')
raw_data.datetime = pd.to_datetime(raw_data.datetime)


def pypull_to_dateaubase(df):
    n_cols = len(df.columns)
    date_and_time = []
    sampling = []
    pars = []
    values = []
    units = []
    for column in df.columns:
        if 'datetime' in column:
            pass
        else:
            project, location, equipment, parameter, unit = column.split('-')
            for row in range(len(df[column])):
                date_and_time.append(df.iat[row, df.columns.get_loc('datetime')])
                sampling.append(location)
                pars.append('{} from {}'.format(parameter, equipment))
                values.append(df.iat[row, df.columns.get_loc(column)])
                units.append(unit)

    newdf = pd.DataFrame(
        data={
            'Date and Time': date_and_time,
            'Sampling point': sampling,
            'Parameter / from': pars,
            'Value': values,
            'Unit': units,
        },
    )

'''params_list = ['CODf','COD','NH4_N','K']
sampling_point = 'Primary settling tank effluent'''
def stackparams(df_input, params_list, sampling_point):
    df_list = []
    for param in params_list:
        df_list.append(pypull_to_dateaubase(df_input, param, sampling_point))
    df = pd.concat(df_list, ignore_index=True)
    return df

'''test2 = stackparams(raw_data,params_list,sampling_point)
print(len(test2))
test2.set_index('Date and Time', drop=True, inplace=True)
test2.to_csv('unfiltered_data.csv',sep=';')'''
