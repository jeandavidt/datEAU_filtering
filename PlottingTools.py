def plotRaw_D_mpl(df):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from pandas.plotting import register_matplotlib_converters

    register_matplotlib_converters()

    _, ax = plt.subplots(figsize=(12,8))
   
    names=[]
    for column in df.columns:
        equipment = column.split('-')[-3]
        parameter=column.split('-')[-2]
        unit = column.split('-')[-1]
        name=" ".join([equipment, parameter, unit])
        names.append(name)
        ax.plot(df[column])
    
    plt.legend(names)
    plt.xticks(rotation=45)
    plt.ylim(bottom=0)
    plt.title("Raw data")
    plt.show(block=False)
   
def plotRaw_D_plotly(df):
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

def plotUnivar_mpl(channel):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from pandas.plotting import register_matplotlib_converters

    register_matplotlib_converters()

    _, ax = plt.subplots(figsize=(12,8))
     
    raw = channel.raw_data
    
    plt.plot(raw)

    if channel.processed_data is not None:
        for col in channel.processed_data.columns:
            if col in ['filled', 'sorted','resampled']:
                plt.plot(channel.processed_data[col])
                

    if channel.calib is not None:
        calib_start = pd.to_datetime(channel.calib['start'])
        calib_end = pd.to_datetime(channel.calib['end'])
        most_recent = channel.info['most_recent_series']
        if most_recent =='raw':
            df = channel.raw_data
        else:
            df = channel.processed_data
        plt.plot(df[most_recent][calib_start:calib_end])

    plt.title('Data preparation')
    plt.ylabel=('Value'),
    plt.xlabel('Date and Time')

    plt.show(block=False)

def plotUnivar_plotly(channel):
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

def plotOutliers_mpl(channel):

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from pandas.plotting import register_matplotlib_converters

    register_matplotlib_converters()

    _, ax = plt.subplots(figsize=(12,8))
    filtration_method = channel.info['current_filtration_method']
    df = channel.filtered[filtration_method]
    raw = channel.raw_data
    AD = df['Accepted']
    raw_out =raw.join(df['outlier'],how='left').dropna()
    outlier = raw_out['raw'].loc[raw_out['outlier']]
    
    ub = df['UpperLimit_outlier']
    lb = df['LowerLimit_outlier']

    legend_list=[]
    if 'Smoothed_AD' in df.columns:
        smooth =df['Smoothed_AD']
        ax.plot(smooth,'g')
        legend_list.append('Smooth')
    
    ax.plot(outlier,'rx')
    legend_list.append('Outliers')
    ax.plot(lb,'b')
    legend_list.append('Lower bound')
    ax.plot(ub,'r')
    legend_list.append('Upper bound')

    
    ax.plot(AD,'orange')
    legend_list.append('Accepted')
    ax.plot(channel.raw_data.index, channel.raw_data['raw'],'grey','o')
    legend_list.append('Raw')
    
    plt.xlabel('Time')
    #plt.xticks(rotation=45)
    #plt.ylabel(channel.parameter)  
   

    plt.legend(legend_list)
    plt.show(block=False)

def plotOutliers_plotly(channel):
    import pandas as pd
    import numpy as np
    import plotly
    import plotly.graph_objs as go

    filtration_method = channel.info['current_filtration_method']
    df = channel.filtered[filtration_method]
    raw = channel.raw_data['raw']
    AD = df['Accepted']
    outlier = raw.loc[df['outlier']]
    ub = df['UpperLimit_outlier']
    lb = df['LowerLimit_outlier']

    to_plot = {'Raw': raw, 'Upper Bound': ub,'Lower Bound': lb,'Accepted':AD,'Outliers':outlier,}
    if 'Smoothed_AD' in df.columns:
        to_plot['Smooth']=df['Smoothed_AD']
    
    traces=[]
    #REFERENCE : plotly.colors.DEFAULT_PLOTLY_COLORS
    '''[
        'rgb(31, 119, 180)',    #blue
        'rgb(255, 127, 14)',    #orange
        'rgb(44, 160, 44)' ,    #green
        'rgb(214, 39, 40)',     #red
        'rgb(148, 103, 189)',   #purple
        'rgb(140, 86, 75)',     #taupe
        'rgb(227, 119, 194)',   #pink
        'rgb(127, 127, 127)',   #middle grey
        'rgb(188, 189, 34)',    #greenish yellow
        'rgb(23, 190, 207)',    #azure
    ]'''
    for name, series in to_plot.items():
        
        trace = go.Scattergl(
                x=series.index,
                y=series,
                name=name)
        if name =='Accepted':
            trace['mode']='lines+markers'
            trace['marker']=dict(opacity=0,color='rgb(255, 127, 14)')#orange
        elif name == 'Upper Bound':
            trace['mode']='lines+markers'
            trace['marker']=dict(opacity=0,color='rgb(214, 39, 40)') #red
        elif name == 'Lower Bound':
            trace['mode']='lines+markers'
            trace['marker']=dict(opacity=0,color='rgb(31, 119, 180)')#blue
        elif name == 'Smooth':
            trace['mode']='lines+markers'
            trace['marker']=dict(opacity=0,color='rgb(44, 160, 44)')#green
        elif name == 'Raw':
            trace['mode'] = 'markers'
            trace['marker']=dict(opacity=0.8, color='rgb(127, 127, 127)',size=2)
        elif name == 'Outliers':
            trace['mode'] = 'markers'
            trace['marker']=dict(opacity=1, color='black',size=8,symbol='x')
        traces.append(trace)
    
    layout=go.Layout(dict(
        title='Outlier Detection',
        yaxis=dict(title='Value'),
        xaxis=dict(title='Date and Time')
        )  
    )
    figure=go.Figure(data=traces,layout=layout)
    return figure

def plotDScore_mpl(channel):
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
    
    filtration_method = channel.info['current_filtration_method']
    df = channel.filtered[filtration_method]
    
    params = channel.params

    corr_max = params['fault_detection_uni']['corr_max']
    corr_min = params['fault_detection_uni']['corr_min']

    slope_max = params['fault_detection_uni']['slope_max']
    slope_min = params['fault_detection_uni']['slope_min']

    std_max=params['fault_detection_uni']['std_max']
    std_min=params['fault_detection_uni']['std_min']

    range_max = params['fault_detection_uni']['range_max']
    range_min = params['fault_detection_uni']['range_max']

    ax0 = axes_list[0]
    ax0.plot(df['Qcorr'],linewidth=2)
    ax0.set(ylabel='Runs test value')

    ax0.plot([df.first_valid_index(), df.last_valid_index()],[corr_max, corr_max],c='r', linewidth=1)
    ax0.plot([df.first_valid_index(), df.last_valid_index()],[corr_min, corr_min],c='r', linewidth=1)
    ax0.set_xticks([])

    ax1 = axes_list[1]
    ax1.plot(df['Qslope'],linewidth=2)

    # ylabel(sprintf('#s','Slope (', units, ')'),'fontsize',10)
    ax1.set(ylabel='Slope [mg/L*s]')
    ax1.plot([df.first_valid_index(), df.last_valid_index()],[slope_max, slope_max],c='r', linewidth=1)
    ax1.plot([df.first_valid_index(), df.last_valid_index()],[slope_min, slope_min],c='r', linewidth=1)
    ax1.set_xticks([])

    ax2 = axes_list[2]
    ax2.plot(df['Qstd'],linewidth=2)
    
    ax2.set(ylabel='Std ln[mg/L]')
    ax2.plot([df.first_valid_index(), df.last_valid_index()],[std_max, std_max],c='r', linewidth=1)
    ax2.plot([df.first_valid_index(), df.last_valid_index()],[std_min, std_min],c='r', linewidth=1)
    ax2.set_xticks([])

    ax3 = axes_list[3]
    ax3.plot(df['Smoothed_AD'],linewidth=2)
    ax3.set(ylabel='Range [mg/L]')
    ax3.plot([df.first_valid_index(), df.last_valid_index()],[range_max, range_max],c='r', linewidth=1)
    ax3.plot([df.first_valid_index(), df.last_valid_index()],[range_min, range_min],c='r', linewidth=1)
    ax3.set_xticks([])

    plt.show(block=False)

def plotDScore_plotly(channel):
    import pandas as pd
    import numpy as np
    import plotly
    import plotly.graph_objs as go
    from plotly import tools

    #This function allows to create several plots with the data feature. 
    #For each data features, you need to change the limits. The whole are
    #defined in the DefaultParam function. 
    filtration_method = channel.info['current_filtration_method']
    df = channel.filtered[filtration_method]
    
    params = channel.params

    corr_max = params['fault_detection_uni']['corr_max']
    corr_min = params['fault_detection_uni']['corr_min']

    slope_max = params['fault_detection_uni']['slope_max']
    slope_min = params['fault_detection_uni']['slope_min']

    std_max=params['fault_detection_uni']['std_max']
    std_min=params['fault_detection_uni']['std_min']

    range_max = params['fault_detection_uni']['range_max']
    range_min = params['fault_detection_uni']['range_max']

    #DEFINING VARIABLES

    trace1a = go.Scattergl(
        x=[df.first_valid_index(), df.last_valid_index()],
        y=[corr_max, corr_max],
        xaxis='x1',
        yaxis='y1',
        mode='lines',
        line=dict(
            color=('rgb(205, 12, 24)'),
            width=1,
        )
    )

    trace1b = go.Scattergl(
        x=[df.first_valid_index(), df.last_valid_index()],
        y=[corr_min, corr_min],
        xaxis='x1',
        yaxis='y1',
        mode='lines',
        line=dict(
            color=('rgb(205, 12, 24)'),
            width=1,
        )
    )

    trace1c = go.Scattergl(
        x=df.index,
        y=df['Q_corr'],
        xaxis='x1',
        yaxis='y1',
        mode='lines',
        line=dict(
            color=('rgb(22, 96, 167)'),
            width=2,
        )
    )

    trace2a = go.Scattergl(
        x=[df.first_valid_index(), df.last_valid_index()],
        y=[slope_max, slope_max],
        xaxis='x1',
        yaxis='y2',
        mode='lines',
        line=dict(
            color=('rgb(205, 12, 24)'),
            width=1,
        )
    )

    trace2b = go.Scattergl(
        x=[df.first_valid_index(), df.last_valid_index()],
        y=[slope_min, slope_min],
        xaxis='x1',
        yaxis='y2',
        mode='lines',
        line=dict(
            color=('rgb(205, 12, 24)'),
            width=1,
        )
    )

    trace2c = go.Scattergl(
        x=df.index,
        y=df['Qslope'],
        xaxis='x1',
        yaxis='y2',
        mode='lines',
        line=dict(
            color=('rgb(22, 96, 167)'),
            width=2,
        )
    )
    # ax1.set(ylabel='Slope [mg/L*s]')
    
    trace3a = go.Scattergl(
        x=[df.first_valid_index(), df.last_valid_index()],
        y=[std_max, std_max],
        xaxis='x1',
        yaxis='y3',
        mode='lines',
        line=dict(
            color=('rgb(205, 12, 24)'),
            width=1,
        )
    )

    trace3b = go.Scattergl(
        x=[df.first_valid_index(), df.last_valid_index()],
        y=[std_min, std_min],
        xaxis='x1',
        yaxis='y3',
        mode='lines',
        line=dict(
            color=('rgb(205, 12, 24)'),
            width=1,
        )
    )

    trace3c = go.Scattergl(
        x=df.index,
        y=df['Qstd'],
        xaxis='x1',
        yaxis='y3',
        mode='lines',
        line=dict(
            color=('rgb(22, 96, 167)'),
            width=2,
        )
    )
    # ax2.set(ylabel='Std ln[mg/L]')
    
    trace4a = go.Scattergl(
        x=[df.first_valid_index(), df.last_valid_index()],
        y=[range_max, range_max],
        xaxis='x1',
        yaxis='y4',
        mode='lines',
        line=dict(
            color=('rgb(205, 12, 24)'),
            width=1,
        )
    )

    trace4b = go.Scattergl(
        x=[df.first_valid_index(), df.last_valid_index()],
        y=[range_min, range_min],
        xaxis='x1',
        yaxis='y4',
        mode='lines',
        line=dict(
            color=('rgb(205, 12, 24)'),
            width=1,
        )
    )

    trace4c = go.Scattergl(
        x=df.index,
        y=df['Qstd'],

        mode='lines',
        line=dict(
            color=('rgb(22, 96, 167)'),
            width=2,
        )
    )
    # ax3.set(ylabel='Range [mg/L]')
    
    fig = tools.make_subplots(
        rows=4,
        cols=1, 
        specs=[[{}], [{}], [{}]],
        shared_xaxes=True, 
        shared_yaxes=False,
        vertical_spacing=0.1)    
    fig.append_trace(trace1a,1,1)
    fig.append_trace(trace1b,1,1)
    fig.append_trace(trace1c,1,1)
        
    fig.append_trace(trace2a,2,1)
    fig.append_trace(trace2b,2,1)
    fig.append_trace(trace2c,2,1)

    fig.append_trace(trace3a,3,1)
    fig.append_trace(trace3b,3,1)
    fig.append_trace(trace3c,3,1)
    
    fig.append_trace(trace4a,4,1)
    fig.append_trace(trace4b,4,1)
    fig.append_trace(trace4c,4,1)

    return fig

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

