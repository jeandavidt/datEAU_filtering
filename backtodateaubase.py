import pandas as pd
import numpy as np

path = '../sample_data/influentdata.csv'
raw_data = pd.read_csv(path,sep=';')
raw_data.datetime = pd.to_datetime(raw_data.datetime)


def backtodateaubase(input_df, param,sampling_point):
    df = input_df.copy(deep=True)
    df = df[['datetime',param,param+' Unit', param+' equipment']]
    df['Sampling Point'] = sampling_point
    df[param+' equipment'] = param +' from '+ df[param+' equipment']
    df.columns = ['Date and Time', 'Value', 'Unit', 'Parameter / from','Sampling point']
    df=df[['Date and Time','Sampling point','Parameter / from','Value','Unit']]
    return df


'''params_list = ['CODf','COD','NH4_N','K']
sampling_point = 'Primary settling tank effluent''''
def stackparams(df_input, params_list, sampling_point):
    df_list = []
    for param in params_list:
        df_list.append(backtodateaubase(df_input,param,sampling_point))
    df = pd.concat(df_list, ignore_index=True)
    return df

'''test2 = stackparams(raw_data,params_list,sampling_point)
print(len(test2))
test2.set_index('Date and Time', drop=True, inplace=True)
test2.to_csv('unfiltered_data.csv',sep=';')'''