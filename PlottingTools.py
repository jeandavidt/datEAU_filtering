def plotRaw_D_mpl(df):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from pandas.plotting import register_matplotlib_converters

    register_matplotlib_converters()

    _, ax = plt.subplots(figsize=(12, 8))

    names = []
    for column in df.columns:
        equipment = column.split('-')[-3]
        parameter = column.split('-')[-2]
        unit = column.split('-')[-1]
        name = " ".join([equipment, parameter, unit])
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
    from itertools import cycle

    traces = []
    axes = []
    colors = []
    color_palette = [
        'rgb(31, 119, 180)',    # blue
        'rgb(255, 127, 14)',    # orange
        'rgb(44, 160, 44)',    # green
        'rgb(214, 39, 40)',     # red
        'rgb(148, 103, 189)',   # purple
        'rgb(140, 86, 75)',     # taupe
        'rgb(227, 119, 194)',   # pink
        'rgb(127, 127, 127)',   # middle grey
        'rgb(188, 189, 34)',    # greenish yellow
        'rgb(23, 190, 207)',    # azure
    ]
    color_cycle = cycle(color_palette)
    dash_styles = ['solid', 'dash', 'dot', 'dash', 'longdash', 'dashdot', 'longdashdot']
    dash_cycle = cycle(dash_styles)

    for column in df.columns:
        project, location, equipment, parameter, unit = column.split('-')
        name = '{} ({})'.format(parameter, unit)
        if name not in axes:
            axes.append(name)
            del dash_cycle
            dash_cycle = cycle(dash_styles)
        else:
            pass
        trace = go.Scattergl(
            x=df.index,
            y=df[column],
            yaxis='y{}'.format(axes.index(name) + 1),
            name='{}-{} ({})'.format(equipment, parameter, unit),
            mode='lines+markers',
            line=dict(
                dash=next(dash_cycle)
            ),
            marker=dict(
                opacity=0
            ),
        )
        traces.append(trace)

    n_axes = len(axes)
    layout_axes = []
    ax_pos = 0
    for i in range(n_axes):
        color = next(color_cycle)
        '''anchor='free',
        overlaying='y',
        side='left',
        position=0.15'''
        ax_pos = i * 0.075
        layout_axes.append({
            'yaxis{}'.format(i + 1): dict(
                title=axes[i],
                titlefont=dict(
                    color=color
                ),
                tickfont=dict(
                    color=color
                ),
                anchor='free',
                side='left',
                position=ax_pos
            )
        })
    layout = {
        'title': 'Raw uploaded data',
        'xaxis': {
            'domain': [0.075 * (n_axes - 1), 1],
            'title': 'Date and time'
        }
    }
    for ax in layout_axes:
        layout = {**layout, **ax}

    figure = go.Figure(data=traces, layout=layout)
    return figure


def plotUnivar_mpl(channel):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from pandas.plotting import register_matplotlib_converters

    register_matplotlib_converters()

    _, ax = plt.subplots(figsize=(12, 8))

    raw = channel.raw_data

    plt.plot(raw)

    if channel.processed_data is not None:
        for col in channel.processed_data.columns:
            if col in ['filled', 'sorted', 'resampled']:
                plt.plot(channel.processed_data[col])

    if channel.calib is not None:
        calib_start = pd.to_datetime(channel.calib['start'])
        calib_end = pd.to_datetime(channel.calib['end'])
        most_recent = channel.info['last-processed']
        if most_recent == 'raw':
            df = channel.raw_data
        else:
            df = channel.processed_data
        plt.plot(df[most_recent][calib_start:calib_end])

    plt.title('Data preparation')
    plt.ylabel('Value'),
    plt.xlabel('Date and Time')
    plt.show(block=False)

def plotUnivar_plotly(channel):
    import plotly
    import plotly.graph_objs as go
    import pandas as pd
    import numpy as np

    traces = []
    raw = channel.raw_data
    trace = go.Scattergl(
        x=raw.index,
        y=raw['raw'],
        name='Raw data',
        mode='lines+markers',
        marker=dict(
            opacity=0
        )
    )
    traces.append(trace)

    if channel.processed_data is not None:
        for col in channel.processed_data.columns:
            if col in ['filled', 'sorted', 'resampled']:
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
        most_recent = channel.info['last-processed']
        if most_recent == 'raw':
            df = channel.raw_data
        elif channel.processed_data is not None:
            df = channel.processed_data
        else:
            df = channel.raw_data
            most_recent = 'raw'
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

    layout = go.Layout(
        dict(
            title='Data preparation',
            yaxis=dict(title='Value'),
            xaxis=dict(title='Date and Time')
        )
    )
    figure = go.Figure(data=traces, layout=layout)
    return figure

def plotOutliers_mpl(channel):
    import datetime
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    from pandas.plotting import register_matplotlib_converters

    register_matplotlib_converters()

    _, ax = plt.subplots(figsize=(12, 8))
    filtration_method = channel.info['current_filtration_method']
    df = channel.filtered[filtration_method].copy(deep=True)
    df.index.name = 'index'
    df.reset_index(inplace=True, drop=False)
    df.index.name = 'index'
    df['t'] = df['index'].apply(lambda x: pd.Timestamp(str(x)))
    df.set_index(df['t'], drop=True)

    raw = channel.raw_data.copy(deep=True)
    raw.index.name = 'index'
    raw.reset_index(inplace=True, drop=False)
    raw.index.name = 'index'
    raw['t'] = raw['index'].apply(lambda x: pd.Timestamp(str(x)))
    raw.set_index(raw['t'], drop=True)
    raw_out = raw.join(df['outlier'], how='left').dropna()

    AD = df['Accepted']
    outlier = raw_out['raw'].loc[raw_out['outlier']]

    ub = df['UpperLimit_outlier']
    lb = df['LowerLimit_outlier']

    legend_list = []
    ax.plot(raw.index, raw['raw'], 'grey', 'o')
    legend_list.append('Raw')
    ax.plot(outlier, 'rx')
    legend_list.append('Outliers')
    ax.plot(df.index, lb, 'b')
    legend_list.append('Lower bound')
    ax.plot(df.index, ub, 'r')
    legend_list.append('Upper bound')
    ax.plot(df.index, AD, 'orange')
    legend_list.append('Accepted')
    if 'Smoothed_AD' in df.columns:
        smooth = df['Smoothed_AD']
        ax.plot(df.index, smooth, 'g')
        legend_list.append('Smooth')

    plt.xlabel('Time')
    plt.xticks(rotation=45)
    plt.ylabel(channel.parameter)

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
    raw_out = channel.raw_data.join(df['outlier'], how='left').dropna()
    outlier = raw_out['raw'].loc[raw_out['outlier']]

    ub = df['UpperLimit_outlier']
    lb = df['LowerLimit_outlier']

    to_plot = {
        'Raw': raw,
        'Upper Bound': ub,
        'Lower Bound': lb,
        'Accepted': AD,
        'Outliers': outlier
    }
    if 'Smoothed_AD' in df.columns:
        to_plot['Smooth'] = df['Smoothed_AD']

    traces = []
    # REFERENCE : plotly.colors.DEFAULT_PLOTLY_COLORS
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
            name=name
        )
        if name == 'Accepted':
            trace['mode'] = 'lines+markers'
            trace['marker'] = dict(opacity=0, color='rgb(255, 127, 14)')  # orange
        elif name == 'Upper Bound':
            trace['mode'] = 'lines+markers'
            trace['marker'] = dict(opacity=0, color='rgb(214, 39, 40)')  # red
        elif name == 'Lower Bound':
            trace['mode'] = 'lines+markers'
            trace['marker'] = dict(opacity=0, color='rgb(31, 119, 180)')  # blue
        elif name == 'Smooth':
            trace['mode'] = 'lines+markers'
            trace['marker'] = dict(opacity=0, color='rgb(44, 160, 44)')  # green
        elif name == 'Raw':
            trace['mode'] = 'markers'
            trace['marker'] = dict(opacity=0.8, color='rgb(127, 127, 127)', size=2)  # grey
        elif name == 'Outliers':
            trace['mode'] = 'markers'
            trace['marker'] = dict(opacity=1, color='black', size=8, symbol='x')
        traces.append(trace)

    layout = go.Layout(
        dict(
            title='Outlier Detection',
            yaxis=dict(title='Value'),
            xaxis=dict(title='Date and Time')
        )
    )
    figure = go.Figure(data=traces, layout=layout)
    return figure

def plotDScore_mpl(channel):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from pandas.plotting import register_matplotlib_converters

    register_matplotlib_converters()

    _, axes = plt.subplots(figsize=(12, 8), nrows=4, ncols=1)
    # This function allows to create several plots with the data feature.
    # For each data features, you need to change the limits. The whole are
    # defined in the DefaultParam function.
    # GRAPHICS
    axes_list = axes.flatten()

    filtration_method = channel.info['current_filtration_method']
    df = channel.filtered[filtration_method].copy(deep=True)

    df.reset_index(inplace=True)
    df['t'] = df['index'].apply(lambda x: pd.Timestamp(str(x)))
    df.set_index(df['t'], drop=True)

    params = channel.params

    corr_max = params['fault_detection_uni']['corr_max']
    corr_min = params['fault_detection_uni']['corr_min']

    slope_max = params['fault_detection_uni']['slope_max']
    slope_min = params['fault_detection_uni']['slope_min']

    std_max = params['fault_detection_uni']['std_max']
    std_min = params['fault_detection_uni']['std_min']

    range_max = params['fault_detection_uni']['range_max']
    range_min = params['fault_detection_uni']['range_max']

    ax0 = axes_list[0]
    ax0.plot(df['Q_corr'], linewidth=2)
    ax0.set(ylabel='Runs test value')

    ax0.plot(
        [df.first_valid_index(), df.last_valid_index()],
        [corr_max, corr_max],
        c='r',
        linewidth=1)
    ax0.plot(
        [df.first_valid_index(), df.last_valid_index()],
        [corr_min, corr_min],
        c='r',
        linewidth=1)
    ax0.set_xticks([])

    ax1 = axes_list[1]
    ax1.plot(df['Q_slope'], linewidth=2)

    ax1.set(ylabel='Slope [mg/L*s]')
    ax1.plot(
        [df.first_valid_index(), df.last_valid_index()],
        [slope_max, slope_max],
        c='r',
        linewidth=1)
    ax1.plot(
        [df.first_valid_index(), df.last_valid_index()],
        [slope_min, slope_min],
        c='r',
        linewidth=1)
    ax1.set_xticks([])

    ax2 = axes_list[2]
    ax2.plot(df['Q_std'], linewidth=2)
    ax2.set(ylabel='Std ln[mg/L]')
    ax2.plot(
        [df.first_valid_index(), df.last_valid_index()],
        [std_max, std_max],
        c='r',
        linewidth=1)
    ax2.plot(
        [df.first_valid_index(), df.last_valid_index()],
        [std_min, std_min],
        c='r',
        linewidth=1)
    ax2.set_xticks([])

    ax3 = axes_list[3]
    ax3.plot(df['Smoothed_AD'], linewidth=2)
    ax3.set(ylabel='Range [mg/L]')
    ax3.plot(
        [df.first_valid_index(), df.last_valid_index()],
        [range_max, range_max],
        c='r',
        linewidth=1)
    ax3.plot(
        [df.first_valid_index(), df.last_valid_index()],
        [range_min, range_min],
        c='r',
        linewidth=1)
    ax3.set_xticks([])

    plt.show(block=False)

def plotTreatedD_plotly(channel):
    import pandas as pd
    import numpy as np
    import plotly
    import plotly.graph_objs as go

    filtration_method = channel.info['current_filtration_method']
    df = channel.filtered[filtration_method]
    raw = channel.raw_data['raw']
    treated = df['treated']
    deleted = df['deleted']
    to_plot = {
        'Raw': raw,
        'Treated': treated,
        'Deleted': deleted,
    }
    traces = []
    for name, series in to_plot.items():
        trace = go.Scattergl(
            x=series.index,
            y=series,
            name=name
        )
        if name == 'Treated':
            trace['mode'] = 'lines+markers'
            trace['marker'] = dict(opacity=0, color='rgb(44, 160, 44)')  # green
        elif name == 'Deleted':
            trace['mode'] = 'lines+markers'
            trace['marker'] = dict(opacity=0, color='rgb(214, 39, 40)')  # red
        elif name == 'Raw':
            trace['mode'] = 'markers'
            trace['marker'] = dict(opacity=0.8, color='rgb(127, 127, 127)', size=5)  # grey
        traces.append(trace)

    layout = go.Layout(
        dict(
            title='Treated univariate data',
            yaxis=dict(title='Value'),
            xaxis=dict(title='Date and Time')
        )
    )
    figure = go.Figure(data=traces, layout=layout)
    return figure


def plotTreatedD_mpl(channel):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from pandas.plotting import register_matplotlib_converters
    register_matplotlib_converters()

    filtration_method = channel.info['current_filtration_method']

    raw = channel.raw_data['raw'].copy(deep=True)
    df = channel.filtered[filtration_method][['treated', 'deleted']].copy(deep=True)
    df.index.name = 'index'
    df.reset_index(inplace=True, drop=False)
    df.index.name = 'index'
    df['t'] = df['index'].apply(lambda x: pd.Timestamp(str(x)))
    df.set_index(df['t'], drop=True)

    raw = channel.raw_data.copy(deep=True)
    raw.index.name = 'index'
    raw.reset_index(inplace=True, drop=False)
    raw.index.name = 'index'
    raw['t'] = raw['index'].apply(lambda x: pd.Timestamp(str(x)))
    raw.set_index(raw['t'], drop=True)

    treated = df['treated']
    deleted = df['deleted']
    _, ax = plt.subplots(figsize=(12, 8))
    # This function allows to plot the DataValidated with also the raw data
    ax.plot(raw.index, raw['raw'], 'k')
    ax.plot(treated, '-g', markersize=6, markerfacecolor='g')
    ax.plot(deleted, 'r', markersize=6, markerfacecolor='r')
    ax.set(xlabel='Time')
    plt.xticks(rotation=45)
    ax.set(ylabel=channel.parameter)
    plt.legend(['Raw', 'Treated', 'Deleted'])
    plt.show(block=False)


def plotD_plotly(params, testID, start=None, end=None, channel=None):
    import pandas as pd
    import numpy as np
    import plotly
    import plotly.graph_objs as go
    from plotly import tools

    # This function allows to create several plots with the data feature.
    # For each data features, you need to change the limits. The whole are
    # defined in the DefaultParam function.
    if channel:
        filtration_method = channel.info['current_filtration_method']
        df = channel.filtered[filtration_method]
        start = df.first_valid_index()
        end = df.last_valid_index()

    min_val = params[0]
    max_val = params[1]
    titles = {
        'Q_corr': 'Residual correlation',
        'Q_slope': 'Slope',
        'Q_std': 'Standard deviation',
        'Q_range': 'Data range'
    }
    # if testID == 'Q_std':
    #     min_val = 10 ** min_val
    #     max_val = 10 ** max_val

    # DEFINING VARIABLES
    traces = []
    trace1a = go.Scattergl(
        x=[start, end],
        y=[min_val, min_val],
        xaxis='x1',
        yaxis='y1',
        name='Min. threshold',
        mode='lines',
        line=dict(
            color=('rgb(205, 12, 24)'),
            width=1,
        )
    )
    traces.append(trace1a)
    trace1b = go.Scattergl(
        x=[start, end],
        y=[max_val, max_val],
        xaxis='x1',
        yaxis='y1',
        mode='lines',
        name='Max. threshold',
        line=dict(
            color=('rgb(205, 12, 24)'),
            width=1,
        )
    )
    traces.append(trace1b)
    if channel:
        if testID in df.columns:
            if testID == 'Q_range':
                name = 'Smoothed_AD'
            else:
                name = testID
            trace1c = go.Scattergl(
                x=df.index,
                y=df[name],
                xaxis='x1',
                yaxis='y1',
                name=name,
                mode='lines',
                line=dict(
                    color=('rgb(22, 96, 167)'),
                    width=2,
                )
            )
            traces.append(trace1c)

    layout = go.Layout(
        title=titles[testID],
        autosize=True,
        # width=800,
        height=250,
        margin=go.layout.Margin(
            l=50,
            r=50,
            b=50,
            t=50,
            pad=4
        ),
        xaxis=dict(
            zeroline=False,
            showline=False,
            ticks='',
            showticklabels=False
        ),
    )

    figure = go.Figure(layout=layout, data=traces)
    # if testID == 'Q_std':
    #     figure.update(dict(layout=dict(yaxis=dict(type='log'))))
    return figure


def show_pca_mpl(df, limits, svd, model):
    import pandas as pd
    import numpy as np
    from matplotlib import cm
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 10), nrows=1, ncols=1)
    plt.rc('lines', linewidth=1)
    color = iter(cm.plasma(np.linspace(0, 1, 4)))
    start = model['start_cal']
    end = model['end_cal']
    ax.plot(df['t_1'], df['t_2'], 'o', markersize=0.5, c=next(color))
    ax.plot(
        df['t_1'].loc[start:end],
        df['t_2'].loc[start:end],
        'o', markersize=0.5, c=next(color)
    )
    ax.set(
        ylabel='PC 2',
        xlabel='PC 1',
        title='Principal components of calibration and complete data sets'
    )
    # ### drawing the ellipse

    ellipse_a = np.sqrt(limits['T2']) * model['t_hat_stdev'][0]
    ellipse_b = np.sqrt(limits['T2']) * model['t_hat_stdev'][1]
    t = np.linspace(0, 2 * np.pi, 100)

    ax.plot(ellipse_a * np.cos(t), ellipse_b * np.sin(t), c=next(color))
    ax.grid(which='major', axis='both')
    ax.legend(['complete', 'calibration', 'limit {}'.format(limits['alpha'])])
    plt.gca().set_aspect('equal')
    plt.show()

def ini_multivar_plotly(df, start=None, end=None):
    import pandas as pd
    import plotly
    import plotly.graph_objs as go
    from itertools import cycle

    traces = []
    axes = []
    colors = []
    color_palette = [
        'rgb(31, 119, 180)',    # blue
        'rgb(255, 127, 14)',    # orange
        'rgb(44, 160, 44)',    # green
        'rgb(214, 39, 40)',     # red
        'rgb(148, 103, 189)',   # purple
        'rgb(140, 86, 75)',     # taupe
        'rgb(227, 119, 194)',   # pink
        'rgb(127, 127, 127)',   # middle grey
        'rgb(188, 189, 34)',    # greenish yellow
        'rgb(23, 190, 207)',    # azure
    ]
    color_cycle = cycle(color_palette)
    dash_styles = ['solid', 'dash', 'dot', 'dash', 'longdash', 'dashdot', 'longdashdot']
    dash_cycle = cycle(dash_styles)
    for column in df.columns:
        series, project, location, equipment, parameter, unit = column.split('-')
        name = '{} ({})'.format(parameter, unit)
        if name not in axes:
            axes.append(name)
            del dash_cycle
            dash_cycle = cycle(dash_styles)
        else:
            pass
        trace = go.Scattergl(
            x=df[column].index,
            y=df[column],
            yaxis='y{}'.format(axes.index(name) + 1),
            name='{}: {}-{} ({})'.format(series, equipment, parameter, unit),
            mode='lines+markers',
            line=dict(
                dash=next(dash_cycle)
            ),
            marker=dict(
                opacity=0
            )
        )
        traces.append(trace)
    n_axes = len(axes)
    layout_axes = []
    ax_pos = 0
    for i in range(n_axes):
        color = next(color_cycle)
        '''anchor='free',
        overlaying='y',
        side='left',
        position=0.15'''
        ax_pos = i * 0.10
        layout_axes.append({
            'yaxis{}'.format(i + 1): dict(
                title=axes[i],
                titlefont=dict(
                    color=color
                ),
                tickfont=dict(
                    color=color
                ),
                anchor='free',
                side='left',
                position=ax_pos
            )
        })
    layout = {
        'title': 'Multivariate data preparation',
        'xaxis': {
            'domain': [0.10 * (n_axes - 1), 1],
            'title': 'Date and time'
        }
    }
    for ax in layout_axes:
        layout = {**layout, **ax}
    if start is not None and end is not None:
        calib_shape = {
            'type': 'rect',
            'xref': 'x',
            'yref': 'paper',
            'x0': start,
            'x1': end,
            'y0': 0,
            'y1': 1,
            'line': {
                'color': 'rgba(214, 39, 40, 1)',
                'width': 1
            },
            'fillcolor': 'rgba(214, 39, 40, 0.3)'
        }
        layout['shapes'] = [calib_shape]
    figure = go.Figure(layout=layout, data=traces)
    return figure
