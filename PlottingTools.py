def plotRaw_D(df):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from pandas.plotting import register_matplotlib_converters

    register_matplotlib_converters()

    _, ax = plt.subplots(figsize=(12,8))
   
    sensors=[]
    units = []
    
    for column in df.columns:
        sensors.append(column.split('-')[-2])
        units.append(column.split('-')[-1])
        ax.plot(df[column],alpha=0.8)
    sensors=[sensor.replace('_','-')for sensor in sensors]
    plt.legend([sensors[i]+' ('+units[i]+')' for i in range(len(sensors))])
    plt.xticks(rotation=45)
    plt.ylim(bottom=0)
    plt.title("Raw data")
    plt.show(block=False)
   
def plotlyRaw_D(df):
    import plotly
    import plotly.graph_objs as go
    import pandas as pd
    import numpy as np
     
    traces=[]
    for column in df.columns:
        equipment = column.split('-')[-3]
        parameter=column.split('-')[-2]
        unit = column.split('-')[-1]
        
        trace = go.Scattergl(
            x=df.index,
            y=df[column],
            name=" ".join([equipment, parameter, unit]),
            mode='lines+markers',
            marker=dict(
                opacity=0
            ))
        traces.append(trace)
    layout=go.Layout(dict(
        title='Raw data',
        yaxis=dict(title='Value'),
        xaxis=dict(title='Date and Time')
        )
    )
    figure=go.Figure(data=traces,layout=layout)
    return figure

def plotlyUnivar(channel):
    import plotly
    import plotly.graph_objs as go
    import pandas as pd
    import numpy as np
     
    traces=[]
    raw = channel.raw_data
    
    trace = go.Scattergl(
            x=raw.index,
            y=raw['raw'],
            name='Raw data',
            mode='lines+markers',
            marker=dict(
                opacity=0
            ))
    traces.append(trace)

    if channel.processed_data is not None:
        for col in channel.processed_data.columns:
            if col in ['filled', 'sorted','resampled']:
                trace = go.Scattergl(
                    x=channel.processed_data.index,
                    y=channel.processed_data[col],
                    name=col,
                    mode='lines+markers',
                    marker=dict(
                        opacity=0
                    ),
                )
                traces.append(trace)

    if channel.calib is not None:
        calib_start = pd.to_datetime(channel.calib['start'])
        calib_end = pd.to_datetime(channel.calib['end'])
        most_recent = channel.info['most_recent_series']
        if most_recent =='raw':
            df = channel.raw_data
        else:
            df = channel.processed_data
        trace = go.Scattergl(
                x=df[calib_start:calib_end].index,
                y=df[calib_start:calib_end][most_recent],
                name='Calibration series',
                mode='lines+markers',
                marker=dict(
                    opacity=0
                ),
            )
        traces.append(trace)

    layout=go.Layout(dict(
        title='Data preparation',
        yaxis=dict(title='Value'),
        xaxis=dict(title='Date and Time')
        )  
    )

    figure=go.Figure(data=traces,layout=layout)
    return figure

def Plot_Outliers(df,var_name):

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from pandas.plotting import register_matplotlib_converters

    register_matplotlib_converters()

    _, ax = plt.subplots(figsize=(12,8))

    AD = df[var_name+'_Accepted']
    outlier = df.loc[df[var_name+'_outlier'],[var_name]]
    ub = df[var_name+'_UpperLimit_outlier']
    lb = df[var_name+'_LowerLimit_outlier']
    raw=df[var_name]
    
    ax.plot(outlier,'rx')
    ax.plot(lb,'b')
    ax.plot(ub,'g')
    
    ax.plot(AD,'purple')
    ax.plot(raw,'k')
    
    plt.xlabel('Time')
    plt.xticks(rotation=45)
    plt.ylabel(var_name)  
    plt.xticks(rotation=45)
    plt.legend(['Outliers','LowerLimit', 'UpperLimit','Accepted Data','Raw'])
    plt.show(block=False)

def Plot_Filtered(df, var_name):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from pandas.plotting import register_matplotlib_converters
    register_matplotlib_converters()

    _, ax = plt.subplots(figsize=(12,8))

    ax.plot(df.loc[df[var_name+'_outlier'],[var_name]], 'xk', markersize=8)
    ax.plot(df[var_name+'_Accepted'], 'ok', markersize=4)
    ax.plot(df[var_name+'_UpperLimit_outlier'], 'None',c='red', linestyle='-', linewidth=1)
    ax.plot(df[var_name+'_LowerLimit_outlier'], 'None',c='blue', linestyle='-', linewidth=1)
    ax.plot(df[var_name+'_Smoothed_AD'], 'None',c='green', linestyle='-', linewidth=1)
    ax.plot(df[var_name+'_smoothed_Pandas'], 'None',c='purple', linestyle='-', linewidth=1)
    ax.set_facecolor('white')
    plt.xlabel('Time')
    plt.xticks(rotation=45)
    plt.ylabel(var_name)  
    plt.legend(['Outliers','Accepted Data', 'Upper Limit', 'Lower Limit','Filtered','Filtered_pandas'])

    plt.show(block = False)

def Plot_DScore(df, name, param):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from pandas.plotting import register_matplotlib_converters

    register_matplotlib_converters()

    _, axes = plt.subplots(figsize=(12,8),nrows=4, ncols=1)
    #This function allows to create several plots with the data feature. 
    #For each data features, you need to change the limits. The whole are
    #defined in the DefaultParam function. 

    #GRAPHICS

    axes_list = axes.flatten()
    ax0 = axes_list[0]
    
    ax0.plot(df[name+'_Qcorr'],linewidth=2)
    ax0.set(ylabel='Runs test value')
    ax0.plot([df.first_valid_index(),df.last_valid_index()],[param['corr_max'], param['corr_max']],c='r', linewidth=1)
    ax0.plot([df.first_valid_index(),df.last_valid_index()],[param['corr_min'], param['corr_min']],c='r', linewidth=1)
    ax0.set_xticks([])

    ax1 = axes_list[1]
    ax1.plot(df[name+'_Qslope'],linewidth=2)

    # ylabel(sprintf('#s','Slope (', units, ')'),'fontsize',10)
    ax1.set(ylabel='Slope [mg/L*s]')
    ax1.plot([df.first_valid_index(),df.last_valid_index()],[param['slope_max'], param['slope_max']],c='r', linewidth=1)
    ax1.plot([df.first_valid_index(),df.last_valid_index()],[param['slope_min'], param['slope_min']],c='r', linewidth=1)
    ax1.set_xticks([])

    ax2 = axes_list[2]
    ax2.plot(df[name+'_Qstd'],linewidth=2)
    
    ax2.set(ylabel='Std ln[mg/L]')
    ax2.plot([df.first_valid_index(),df.last_valid_index()],[param['std_max'], param['std_max']],c='r', linewidth=1)
    ax2.plot([df.first_valid_index(),df.last_valid_index()],[param['std_min'], param['std_min']],c='r', linewidth=1)
    ax2.set_xticks([])

    ax3 = axes_list[3]
    ax3.plot(df[name+'_Smoothed_AD'],linewidth=2)
    ax3.set(ylabel='Range [mg/L]')
    ax3.plot([df.first_valid_index(),df.last_valid_index()],[param['range_max'], param['range_max']],c='r', linewidth=1)
    ax3.plot([df.first_valid_index(),df.last_valid_index()],[param['range_min'], param['range_min']],c='r', linewidth=1)
    ax3.set_xticks([])

    plt.show(block=False)

def plotTreatedD(df, name):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from pandas.plotting import register_matplotlib_converters
    register_matplotlib_converters()

    _, ax = plt.subplots(figsize=(12,8))
    #This function allows to plot the DataValidated with also the raw data


    ax.plot(df[name+'_raw'],'k')
    ax.plot(df[name+"_Treated"], '-g', markersize=6, markerfacecolor='r')
    # plot(Time, DeletedD, 'Or', 'markersize',6, 'markerfacecolor', 'r')
    ax.set(xlabel='Temps')
    plt.xticks(rotation=45)
    ax.set(ylabel=name)
    plt.legend(['Raw Data','Treated Data'])
    plt.show(block=False)

