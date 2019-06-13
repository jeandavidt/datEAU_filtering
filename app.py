import base64
import io
import json
import time

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import flask
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate


import DataCoherence
import DefaultSettings
import FaultDetection
import OutlierDetection
import PlottingTools
import Sensors
import Smoother

# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__)
app.config['suppress_callback_exceptions'] = True


########################################################################
# Table Building helper function #
########################################################################


def build_param_table(_id):
    table = dash_table.DataTable(
        id=_id,
        columns=[
            {'name': 'Parameter', 'id': 'Parameter'},
            {'name': 'Value', 'id': 'Value'},
            {'name': '', 'id': 'blank'}
        ],
        n_fixed_rows=1,
        style_table={
            'maxHeight': '300',
            'overflowY': 'scroll'
        },
        style_cell_conditional=[
            {'if': {'column_id': 'Parameter'},
                'minWidth': '50%', 'maxWidth': '50%', 'textAlign': 'left'},
            {'if': {'column_id': 'Value'},
                'minWidth': '50%', 'maxWidth': '50%', 'textAlign': 'left'},
        ],
        style_header={
            'backgroundColor': 'white',
            'fontWeight': 'bold'
        },
        editable=True,
        style_as_list_view=True,
    )
    return table


def small_button(_id, label):
    button = html.Button(
        id=_id,
        children=[label],
        style={
            'height': '24px',
            'padding': '0 10px',
            'font-size': '9px',
            'font-weight': '500',
            'line-height': '24px',
        })
    return button


def small_graph(_id, title):
    graph = dcc.Graph(
        id=_id,
        figure={
            'layout': go.Layout(
                title=title,
                autosize=False,
                width=800,
                height=250,
                margin=go.layout.Margin(
                    l=50,
                    r=50,
                    b=30,
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
        },
    )
    return graph


########################################################################
# APP LAYOUT #
########################################################################


app.layout = html.Div([
    dcc.Store(id='session-store'),
    dcc.Store(id='sensors-store'),
    dcc.Store(id='modif-store'),
    dcc.Store(id='coherence-store'),
    dcc.Store(id='new-params-store'),
    dcc.Store(id='multivariate-store'),
    html.H1(dcc.Markdown('dat*EAU* filtration'), id='header'),
    dcc.Tabs([
        dcc.Tab(label='Data Import', value='import', children=[
            html.Div([
                html.H3('Import data'),
                html.Div([
                    dcc.Upload(
                        id='upload-data',
                        children=html.Button(
                            'Upload Raw Data',
                            id='upload-button',
                            className='button-primary'
                        )
                    ),
                    html.Button(
                        id='select-all-series-import',
                        children=['Select all series']
                    ),
                ], style={'width': '40%', 'columnCount': 2}),
                html.H6(id='import-message'),
                html.Div(id='upload-graph-location'),
                html.Div('Use the Box Select tool to choose a range of dates to analyze.'),
                html.Br(),
                dcc.DatePickerRange(id='import-dates'),
                html.Br(),
                html.Br(),
                html.Div(id='upload-dropdown'),
                html.Div(id="button-location")
            ], id='tab1_content')
        ]),
        dcc.Tab(label='Univariate filter', value='univar', children=[
            html.Div([
                html.H3('Univariate fault detection'),
                'Pick a series to anlayze.',
                html.Br(),
                # dcc.Upload(
                #    id='upload-sensor-data',
                #    children=html.Button(
                #        'Upload Processed Data',
                #        id='upload-sensor-button',
                #    )
                # ),
                html.Br(),
                dcc.Dropdown(
                    id='select-series',
                    multi=False,
                    placeholder='Pick a series to analyze here.',
                    options=[]),
                html.Br(),
                'Pick an outlier detection method.',
                html.Br(),
                dcc.Dropdown(
                    id='select-method',
                    multi=False,
                    placeholder='Pick a method to detect outliers.',
                    options=[{'label': 'Online EWMA', 'value': 'Online_EWMA'}],
                    value='Online_EWMA'),
                html.Br()]),
            html.Div(
                id='uni-up',
                children=[
                    html.Div(
                        id='uni-up-left',
                        children=[
                            html.Button(id='check-coherence', children='Check data coherence'),
                            html.Div(
                                id='faults',
                                children=[
                                    html.Table([
                                        html.Tr([
                                            html.Th('Status'),
                                            html.Th('Message')
                                        ]),
                                        html.Tr([
                                            html.Td(id='err0-status'),
                                            html.Td(id='err0-msg', children=[])
                                        ]),
                                        html.Tr([
                                            html.Td(id='err1-status'),
                                            html.Td(id='err1-msg', children=[])
                                        ]),
                                        html.Tr([
                                            html.Td(id='err2-status'),
                                            html.Td(id='err2-msg', children=[])
                                        ], style={'line-height': '24px'}),
                                        html.Tr([
                                            html.Td(id='err3-status'),
                                            html.Td(id='err3-msg', children=[])
                                        ]),
                                        html.Tr([
                                            html.Td(id='err4-status'),
                                            html.Td(id='err4-msg', children=[])
                                        ])
                                    ], style={
                                        'font-size': '9px',
                                        'font-weight': '200',
                                        'line-height': '12px'
                                    }),
                                    html.Button(id='sort-button', children=['Sort indices']),
                                    html.Br(),
                                    html.Br(),
                                    html.Button(id='resample-button', children=['Resample']),
                                    dcc.Input(
                                        id='sample-freq',
                                        placeholder='frequency (min)',
                                        type='number',
                                        value=2,
                                        min=0.01,
                                        step=1
                                    ),
                                    'min',
                                    html.Br(),
                                    html.Br(),
                                    html.Button(id='fillna-button', children=['Fill blank rows']),
                                    html.Br(),
                                    html.Br(),
                                    html.Button(className='button-alert', id='reset-button', children=['Reset to raw']),
                                ],
                            ),
                        ], style={'width': '20%', 'display': 'inline-block', 'float': 'left'}),
                    html.Div(
                        id='uni-up-center',
                        children=[
                            dcc.Graph(id='initial_uni_graph'),
                            html.P([
                                'Use the Box Select tool to',
                                ' choose a range of data with which to',
                                ' fit the outlier detection model.'
                            ]),
                            html.Br(),
                            dcc.DatePickerRange(id='fit-range'),
                            html.Button(
                                className='button-primary',
                                id='fit-button',
                                children='Fit outlier filter'),
                        ], style={'width': '55%', 'display': 'inline-block'}
                    ),
                    html.Div(
                        id='uni-up-right',
                        children=[
                            html.H6('Channel Parameters'),
                            dcc.Tabs(
                                parent_className='custom-tabs',
                                className='custom-tabs-container',
                                children=[
                                    dcc.Tab(
                                        className='custom-tab',
                                        selected_className='custom-tab--selected',
                                        id='outlier-param-tab',
                                        label='Outliers',
                                        children=[
                                            build_param_table('outlier-param-table')
                                        ]
                                    ),
                                    dcc.Tab(
                                        className='custom-tab',
                                        selected_className='custom-tab--selected',
                                        id='data_smoother-param-tab',
                                        label='Smoother',
                                        children=[
                                            build_param_table('data_smoother-param-table')
                                        ]
                                    ),
                                    dcc.Tab(
                                        className='custom-tab',
                                        selected_className='custom-tab--selected',
                                        id='fault_detection_uni-param-tab',
                                        label='Faults',
                                        children=[
                                            build_param_table('fault_detection_uni-param-table')
                                        ]
                                    )
                                ]
                            ),
                            html.Br(),
                            html.Button(id='save-params-button', children=['Save Parameters']),
                        ], style={'width': '25%', 'display': 'inline-block', 'float': 'right'}
                    ),
                ]
            ),
            html.Br(),
            html.Div(
                id='uni-down',
                children=[
                    dcc.Tabs([
                        dcc.Tab(
                            id='uniOutlier-graph-tab',
                            label='Outliers',
                            children=[
                                html.Div(
                                    id='uni-outlier-left',
                                    children=[
                                        html.Br(),
                                        html.Br(),
                                        html.Button(
                                            className='button-primary',
                                            id='smooth-button',
                                            children='Smoothen data'),
                                    ], style={'width': '20%', 'display': 'inline-block', 'float': 'left'}
                                ),
                                html.Div(
                                    id='uni-outlier-center',
                                    children=[
                                        dcc.Graph(id='uni-outlier-graph')
                                    ], style={'width': '80%', 'display': 'inline-block'}
                                )
                            ]),
                        dcc.Tab(
                            id='uniFaults-graph-tab',
                            label='Faults',
                            children=[
                                html.Div(
                                    id='uni-faults-left',
                                    children=[
                                        html.P('Correlation'),
                                        dcc.RangeSlider(
                                            id='corr-slide',
                                            updatemode='mouseup',
                                            allowCross=False,
                                            min=-25,
                                            max=25,
                                            value=[-5, 5]
                                        ),
                                        html.P(id='corr-vals'),
                                        html.Br(),
                                        html.P('Slope Limits'),
                                        dcc.RangeSlider(
                                            id='slope-slide',
                                            updatemode='mouseup',
                                            allowCross=False,
                                            min=-2,
                                            max=2,
                                            step=0.01,
                                            value=[-1, 1]
                                        ),
                                        html.P(id='slope-vals'),
                                        html.Br(),
                                        html.P('Standard devation limits'),
                                        dcc.RangeSlider(
                                            id='std-slide',
                                            updatemode='mouseup',
                                            allowCross=False,
                                            min=-5,
                                            max=2,
                                            step=0.01,
                                            value=[-2, 1]
                                        ),
                                        html.P(id='std-vals'),
                                        html.Br(),
                                        html.P('Range limits'),
                                        dcc.RangeSlider(
                                            id='range-slide',
                                            updatemode='mouseup',
                                            allowCross=False,
                                            min=0,
                                            max=2000,
                                            value=[10, 300]
                                        ),
                                        html.P(id='range-vals'),
                                        html.Br(),
                                        html.Br(),
                                        html.Button(id='detect_faults-uni', children='Detect Faults'),
                                        html.Br(),
                                        html.Br(),
                                        html.Button(id='treated-uni', children='Get treated data'),
                                        html.Br(),
                                        html.Br(),
                                    ], style={'width': '20%', 'display': 'inline-block', 'float': 'left'}
                                ),
                                html.Div(
                                    id='uni-faults-field',
                                    children=[
                                        small_graph('uni-corr-graph', 'Correlation test'),
                                        small_graph('uni-slope-graph', 'Slope test'),
                                        small_graph('uni-std-graph', 'Standard deviation test'),
                                        small_graph('uni-range-graph', 'Range test'),
                                    ], style={'textAlign': 'center', 'width': '80%', 'display': 'inline-block'}
                                )
                            ],
                        )
                    ]),
                ]
            ),
            html.Hr(),
            html.Button(id='send-to-multivar', children=['Send to multivariate']),
        ]),
        dcc.Tab(label='Multivariate filter', value='multivar', children=[
            html.Br(),
            dcc.Dropdown(
                id='multivar-select-dropdown',
                multi=True,
                placeholder='Pick a series to analyze here.',
                options=[]
            ),
            html.Div(id='multivar-top-left', children=[
                dcc.Graph(id='multivar-select-graph')
            ], style={'width': '70%', 'display': 'inline-block', 'float': 'left'}),
            html.Div(id='multivar-top-right', children=[
                html.Br(),
                html.H6('Parameters'),
                'Min explained variance',
                dcc.Input(
                    id='multivar-min-exp-var',
                    type='number',
                    min=0,
                    max=1,
                    step=0.01,
                    value=0.95),
                html.Br(),
                'Alpha',
                dcc.Input(
                    id='multivar-alpha',
                    type='number',
                    min=0,
                    max=1,
                    step=0.01,
                    value=0.95),
                html.Br(),
                html.Hr(),
                html.H6('Calibration period'),
                dcc.DatePickerRange(id='multivar-select-range'),
                html.Br(),
                html.Br(),
                html.Button(
                    id='multivar-calibrate-button',
                    children='Calibrate model'),
            ], style={'width': '30%', 'display': 'inline-block', 'float': 'right'}),
            html.Br(),
            html.Div(id='multivariate-bottom-left', children=[
                dcc.Graph(id='multivariate-pca-graph')
            ], style={'width': '50%', 'display': 'inline-block', 'float': 'left'}),
            html.Div(id='multivariate-bottom-right', children=[
                dcc.Graph(id='multivariate-faults-graph')
            ], style={'width': '50%', 'display': 'inline-block', 'float': 'right'}),
            html.Button(id='multi-save-accepted-button', children='Save accepted to CSV')
        ])
    ], id="tabs", value='import'),
    html.Div(id='output-data-upload', style={'display': 'none'}),
    html.Div(id='tabs-content'),
    html.Hr(),
    # dcc.Link(
    #    'Save sensor data for later analysis.',
    #    id='save-sensors-link',
    # ),
], id='applayout')


########################################################################
# HELPER FUNCTIONS #
########################################################################


def transform_value(value):
    return 10 ** value

def parse_contents(contents, filename):
    _, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')), sep=';')
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))

        df.datetime = pd.to_datetime(df.datetime)
        df.set_index('datetime', inplace=True, drop=True)
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])
    return df.to_json(date_format='iso', orient='split')


def get_channel(data, channel_props):
    sensors = json.loads(data, object_hook=Sensors.decode_object)
    for sensor in sensors:
        project, location, equipment, parameter, _ = channel_props.split('-')
        if (
            sensor.project == project and sensor.location == location and sensor.equipment == equipment
        ):
            return sensor.channels[parameter]


def get_sensors_and_channel(serialized_data, channel_props):
    sensors = json.loads(serialized_data, object_hook=Sensors.decode_object)
    for sensor in sensors:
        project, location, equipment, _, _ = channel_props.split('-')
        if (
            sensor.project == project and sensor.location == location and sensor.equipment == equipment
        ):
            sensor_index = sensors.index(sensor)
            channel = get_channel(serialized_data, channel_props)
    return sensors, sensor_index, channel


def is_int(s):
    try:
        float(s)
        if '.' not in s:
            return True
        else:
            return False
    except ValueError:
        return False


def is_float(s):
    try:
        float(s)
        if '.' in s:
            return True
        else:
            return False
    except ValueError:
        return False


def parse_parameter(parameter):
    par = str(parameter)
    if 'True' in par or 'true' in par:
        return True
    elif 'False' in par or 'false' in par:
        return False
    elif is_int(par):
        return int(par)
    elif is_float(par):
        return float(par)
    else:
        return par


########################################################################
# IMPORT TAB #
########################################################################


@app.callback(
    Output('output-data-upload', 'children'),
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename')]
)
def update_output(contents, filename):
    if contents is not None:
        data = parse_contents(contents, filename)
        return data


@app.callback(
    Output('upload-graph-location', 'children'),
    [Input('output-data-upload', 'children')],
    [State('upload-button', 'n_clicks')])
def update_upload_fig(data, n_clicks):
    if not n_clicks:
        return dcc.Graph(id='upload-graph')
    else:
        df = pd.read_json(data, orient='split')
        figure = PlottingTools.plotRaw_D_plotly(df)
        figure.update(dict(layout=dict(clickmode='event+select')))

        return dcc.Graph(id='upload-graph', figure=figure),


@app.callback(
    [Output('import-dates', 'start_date'),
        Output('import-dates', 'end_date')],
    [Input('upload-graph', 'selectedData')])
def add_interval(selection):
    if 'range' not in selection:
        raise PreventUpdate
    else:
        start = selection['range']['x'][0]
        end = selection['range']['x'][1]
        return start, end


@app.callback(
    [Output('upload-dropdown', 'children')],
    [Input('output-data-upload', 'children'),
        Input('select-all-series-import', 'n_clicks')],
    [State('upload-button', 'n_clicks')])
def show_series_list(data, all_inputs, n_clicks):
    if n_clicks == 0:
        raise PreventUpdate
    if not data:
        raise PreventUpdate
    else:
        ctx = dash.callback_context
        trigger = ctx.triggered[0]['prop_id'].split('.')[0]

        df = pd.read_json(data, orient='split')
        columns = df.columns
        labels = [column.split('-')[-2] + ' ' + column.split('-')[-1] for column in columns]
        options = [{'label': labels[i], 'value':columns[i]} for i in range(len(columns))]
        if trigger == 'select-all-series-import':
            select = [i['value'] for i in options]
        else:
            select = None
        return [
            html.Div(
                id='test',
                children=[
                    dcc.Dropdown(
                        id='series-selection',
                        multi=True,
                        placeholder='Select series to analyze here.',
                        options=options,
                        value=select
                    ),
                    html.Br()
                ]
            )
        ]


@app.callback(
    [Output('button-location', 'children')],
    [Input('series-selection', 'value')])
def check_if_ready_to_save(series):
    if (series is not None):
        return[
            html.Button(
                id='save-button',
                children='Save data for analysis',
                className='button-primary'
            )
        ]
    else:
        return [
            html.Div('You must select at least one time series to continue')
        ]


@app.callback(
    Output('session-store', 'data'),
    [Input('save-button', 'n_clicks')],
    [State('output-data-upload', 'children'),
        State('series-selection', 'value'),
        State('import-dates', 'start_date'),
        State('import-dates', 'end_date')]
)
def store_raw(click, data, series, start, end):
    if not click:
        raise PreventUpdate
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    df = pd.read_json(data, orient='split')
    df.index = pd.to_datetime(df.index)
    if not start:
        filtered = df
    filtered = df.loc[start:end, series]
    to_save = filtered.to_json(date_format='iso', orient='split')
    return to_save


########################################################################
# UNIVARIATE TAB #
########################################################################


@app.callback(
    Output('select-series', 'options'),
    [Input('sensors-store', 'data')])
def show_univar_list(data):
    if not data:
        raise PreventUpdate
    else:
        sensors = json.loads(data, object_hook=Sensors.decode_object)
        labels = []
        columns = []
        for sensor in sensors:
            project = sensor.project
            location = sensor.location
            print(sensor.channels)
            for channel in sensor.channels.values():
                equipment = channel.equipment
                parameter = channel.parameter
                unit = channel.unit

                labels.append('{} ({})'.format(parameter, unit))
                columns.append('-'.join([project, location, equipment, parameter, unit]))
        options = [{'label': labels[i], 'value':columns[i]} for i in range(len(columns))]

        return options


@app.callback(
    Output('sensors-store', 'data'),
    [Input('session-store', 'data'),
        # Input('upload-sensor-data', 'contents'),
        Input('modif-store', 'data')])
# [State('upload-sensor-data', 'filename')])
def create_sensors(original_data, modif_data):  # sensor_upload, sensor_filename
    ctx = dash.callback_context
    trigger = ctx.triggered[0]['prop_id'].split('.')[0]
    if trigger == 'upload-sensor-data':
        '''_, content_string = sensor_upload.split(',')
        decoded = base64.b64decode(content_string)
        try:
            if 'json' in sensor_filename:
                # Assume that the user uploaded a JSON file
                sensors = json.load(io.BytesIO(decoded), object_hook=Sensors.decode_object)
                serialized = json.dumps(sensors, indent=4, cls=Sensors.CustomEncoder)
                return serialized
        except Exception as e:
            print(e)
            return html.Div([
                'There was an error processing this file.'
            ])'''
    else:
        print('else is triggered too')
        if not original_data:
            raise PreventUpdate
        if modif_data == original_data:
            raise PreventUpdate
        if not modif_data:
            df = pd.read_json(original_data, orient='split')
            sensors = Sensors.parse_dataframe(df)
            serialized = json.dumps(sensors, indent=4, cls=Sensors.CustomEncoder)
            return serialized
        else:
            return modif_data


@app.callback(
    Output('modif-store', 'data'),
    [Input('fillna-button', 'n_clicks'),
        Input('resample-button', 'n_clicks'),
        Input('sort-button', 'n_clicks'),
        Input('fit-button', 'n_clicks'),
        Input('save-params-button', 'n_clicks'),
        Input('reset-button', 'n_clicks'),
        Input('fit-range', 'end_date'),
        Input('smooth-button', 'n_clicks'),
        Input('detect_faults-uni', 'n_clicks'),
        Input('select-method', 'value')],
    [State('sensors-store', 'data'),
        State('select-series', 'value'),
        State('sample-freq', 'value'),
        State('outlier-param-table', 'data'),
        State('data_smoother-param-table', 'data'),
        State('fault_detection_uni-param-table', 'data'),
        State('fit-range', 'start_date'),
        State('corr-slide', 'value'),
        State('slope-slide', 'value'),
        State('std-slide', 'value'),
        State('range-slide', 'value'), ])
def modify_sensors(
    # Inputs vars
    fillna, resamp, srt, fit, param_button, reset, calib_end, smooth, faults, filtration_method,
    # State vars
        sensor_data, channel_info, frequency, par_outlier, par_smooth,
        par_f_uni, calib_start, corr, slope, std, _range):
    ctx = dash.callback_context
    if not sensor_data:
        raise PreventUpdate
    else:
        sensors, sensor_index, channel = get_sensors_and_channel(sensor_data, channel_info)
        trigger = ctx.triggered[0]['prop_id'].split('.')[0]

        if trigger == 'fillna-button':
            channel = DataCoherence.fillna(channel)

        elif trigger == 'resample-button':
            if frequency is None:
                raise PreventUpdate
            freq = str(frequency) + ' min'
            channel = DataCoherence.resample(channel, freq)

        elif trigger == 'sort-button':
            channel = DataCoherence.sort_dat(channel)

        elif trigger == 'fit-button':
            print('Outlier detection triggered!')
            a = time.time()
            channel = OutlierDetection.outlier_detection(channel)
            print('outlier detection has finished')
            delta = time.time() - a
            print('Outlier detection took ' + str(delta) + 's')

        elif trigger == 'save-params-button':
            new_params = {
                'outlier_detection': {},
                'data_smoother': {},
                'data_coherence': {},
                'fault_detection_uni': {},
            }
            outlier_method = channel.info['current_filtration_method']
            default_params = DefaultSettings.DefaultParam(outlier_method)
            new_params['data_coherence'] = default_params['data_coherence']
            new_params['general'] = default_params['general']
            for row in par_outlier:
                valu = parse_parameter(row['Value'])
                new_params['outlier_detection'][row['Parameter']] = valu
            for row in par_smooth:
                valu = parse_parameter(row['Value'])
                new_params['data_smoother'][row['Parameter']] = valu
            for row in par_f_uni:
                valu = parse_parameter(row['Value'])
                new_params['fault_detection_uni'][row['Parameter']] = valu
            channel.params = new_params

        elif trigger == 'reset-button':
            channel.info = {'most_recent_series': 'raw'}
            channel.processed_data = None
            channel.filtered = None
            channel.params = DefaultSettings.DefaultParam
        elif trigger == 'smooth-button':
            channel = Smoother.kernel_smoother(channel)

        elif trigger == 'select-method':
            channel.info['current_filtration_method'] = filtration_method

        elif trigger == 'detect_faults-uni':
            print('correct trigger')
            channel.params['fault_detection_uni']['corr_min'] = corr[0]
            channel.params['fault_detection_uni']['corr_max'] = corr[1]
            channel.params['fault_detection_uni']['slope_min'] = slope[0]
            channel.params['fault_detection_uni']['slope_max'] = slope[1]
            channel.params['fault_detection_uni']['std_min'] = std[0]
            channel.params['fault_detection_uni']['std_max'] = std[1]
            channel.params['fault_detection_uni']['range_min'] = _range[0]
            channel.params['fault_detection_uni']['range_max'] = _range[1]
            channel = FaultDetection.D_score(channel)
            print('Fault detection finished')

        elif calib_start and calib_end:
            if channel.calib is not None:
                channel_start = channel.calib['start']
                channel_end = channel.calib['end']
                if channel_start == calib_start and channel_end == calib_end:
                    if (trigger == 'fit-button' or trigger == 'smooth-button' or trigger == 'detect_faults-uni'):
                        pass
                    else:
                        print('update prevented')
                        raise PreventUpdate
                else:
                    channel.calib['start'] = calib_start
                    channel.calib['end'] = calib_end
            else:
                channel.calib = {'start': calib_start, 'end': calib_end}

        sensor = sensors[sensor_index]
        sensor.channels[channel.parameter] = channel
        print('Sensor object modified')
        return json.dumps(sensors, indent=4, cls=Sensors.CustomEncoder)


@app.callback(
    Output('initial_uni_graph', 'figure'),
    [Input('select-series', 'value'),
        Input('sensors-store', 'data')]
)
def update_top_univariate_figure(value, data):
    if not value:
        raise PreventUpdate
    else:
        channel = get_channel(data, value)
        figure = PlottingTools.plotUnivar_plotly(channel)
        figure.update(dict(layout=dict(clickmode='event+select')))
        return figure


###########################################################################
# DATA COHERENCE ERROR MESSAGES ###########################################


@app.callback(
    Output('coherence-store', 'data'),
    [Input('check-coherence', 'n_clicks')],
    [State('sensors-store', 'data'),
        State('select-series', 'value')]
)
def run_data_coherence_check(click, data, channel_info):
    if not click:
        raise PreventUpdate
    else:
        channel = get_channel(data, channel_info)
        channel.params['general']['Verbose'] = False
        flag = DataCoherence.data_coherence(channel)
        return json.dumps(flag)


@app.callback(
    [Output('err0-msg', 'children'),
        Output('err0-status', 'children')],
    [Input('coherence-store', 'data')]
)
def flag_0(flag):
    if not flag:
        raise PreventUpdate
    else:
        flag = json.loads(flag)
        if '0' in flag.keys():
            return[
                'Ready to move on!',
                html.Img(src=app.get_asset_url('check.png'), width='18px')
            ]
        else:
            return[
                'You should probaly work on the data some more.',
                html.Img(src=app.get_asset_url('cross.png'), width='18px')
            ]


@app.callback(
    [Output('err1-msg', 'children'),
        Output('err1-status', 'children')],
    [Input('coherence-store', 'data')]
)
def flag1(flag):
    if not flag:
        raise PreventUpdate
    else:
        flag = json.loads(flag)
        if '1' in flag.keys():
            return[
                flag['1'],
                html.Img(src=app.get_asset_url('cross.png'), width='18px')
            ]
        else:
            return[
                html.P('There is no gap in the data'),
                html.Img(src=app.get_asset_url('check.png'), width='18px')
            ]


@app.callback(
    [Output('err2-msg', 'children'),
        Output('err2-status', 'children')],
    [Input('coherence-store', 'data')]
)
def flag2(flag):
    if not flag:
        raise PreventUpdate
    else:
        flag = json.loads(flag)
        if '2' in flag.keys():
            return[
                flag['2'],
                html.Img(src=app.get_asset_url('cross.png'), width='18px')
            ]
        else:
            return[
                html.P('The time step is constant.'),
                html.Img(src=app.get_asset_url('check.png'), width='18px')
            ]


@app.callback(
    [Output('err3-msg', 'children'),
        Output('err3-status', 'children')],
    [Input('coherence-store', 'data')]
)
def flag3(flag):
    if not flag:
        raise PreventUpdate
    else:
        flag = json.loads(flag)
        if 3 in flag.keys():
            return[
                flag[3],
                html.Img(src=app.get_asset_url('cross.png'), width='18')
            ]
        else:
            return[
                html.P('There is no large gap in the data.'),
                html.Img(src=app.get_asset_url('check.png'), width='18')
            ]


@app.callback(
    [Output('err4-msg', 'children'),
        Output('err4-status', 'children')],
    [Input('coherence-store', 'data')])
def flag4(flag):
    if not flag:
        raise PreventUpdate
    else:
        flag = json.loads(flag)
        if '4' in flag.keys():
            return[
                flag['4'],
                html.Img(src=app.get_asset_url('cross.png'), width='18')
            ]
        else:
            return[
                html.P('The data is in chronological order.'),
                html.Img(src=app.get_asset_url('check.png'), width='18')
            ]


###########################################################################
# UNIVARIATE FILTER CALIBRATION ###########################################


@app.callback(
    [Output('fit-range', 'start_date'),
        Output('fit-range', 'end_date')],
    [Input('initial_uni_graph', 'selectedData')])
def add_interval_fit(selection):
    if selection is None:
        raise PreventUpdate
    else:
        try:
            start = selection['range']['x'][0]
            end = selection['range']['x'][1]
        except KeyError:
            raise PreventUpdate
        return [start, end]


###########################################################################
# POPULATE PARAMETERS TABLE ###############################################


@app.callback(
    [Output('outlier-param-table', 'data'),
        Output('data_smoother-param-table', 'data'),
        Output('fault_detection_uni-param-table', 'data')],
    [Input('select-series', 'value'),
        Input('sensors-store', 'data')])
def fill_params_table(channel_info, data):
    if channel_info is None:
        raise PreventUpdate
    else:
        channel = get_channel(data, channel_info)
        params = channel.params
        tables_data = []
        for table in ['outlier_detection', 'data_smoother', 'fault_detection_uni']:
            table_params = list(params[table].keys())
            table_params_values = list(params[table].values())
            strings = [str(x) for x in table_params_values]
            table_data = pd.DataFrame(data={'Parameter': table_params, 'Value': strings})
            table_data = table_data.to_dict('records')
            tables_data.append(table_data)
        return tables_data


###########################################################################
# PLOT FILTERED DATA ######################################################


@app.callback(
    Output('uni-outlier-graph', 'figure'),
    [Input('select-series', 'value'),
        Input('sensors-store', 'data')])
def update_second_univariate_figure(value, data):
    if not value:
        raise PreventUpdate
    else:
        channel = get_channel(data, value)
        if channel.filtered is None:
            raise PreventUpdate
        else:
            a = time.time()
            figure = PlottingTools.plotOutliers_plotly(channel)
            figure.update(
                dict(
                    layout=dict(
                        clickmode='event+select'
                    )
                )
            )
            print('Outlier figure creation took ' + str((time.time() - a)) + 's')
            return figure


###########################################################################
# PLOT FAULT DETECTION ######################################################

@app.callback(
    [Output('uni-corr-graph', 'figure'),
        Output('uni-slope-graph', 'figure'),
        Output('uni-std-graph', 'figure'),
        Output('uni-range-graph', 'figure')],
    [Input('sensors-store', 'data'),
        Input('corr-slide', 'value'),
        Input('slope-slide', 'value'),
        Input('std-slide', 'value'),
        Input('range-slide', 'value')],
    [State('select-series', 'value'),
        State('select-method', 'value')]
)
def update_faults_figures(
        data, corr, slope, std, _range, series, filtration_method):
    if not series or not filtration_method or not data or None in corr:
        print('fig prevented')
        raise PreventUpdate
    else:
        ctx = dash.callback_context
        trigger = ctx.triggered[0]['prop_id'].split('.')[0]
        channel = get_channel(data, series)
        start = channel.raw_data.first_valid_index()
        end = channel.raw_data.last_valid_index()
        if channel.filtered:
            filtered = channel.filtered[filtration_method]
            if ('Q_corr' not in filtered.columns or
                'Q_slope' not in filtered.columns or
                    'Q_std' not in filtered.columns or
                    'Q_range' not in filtered.columns):
                print('empty figs start')
                figure1 = PlottingTools.plotD_plotly(corr, 'Q_corr', start=start, end=end, channel=None)
                figure2 = PlottingTools.plotD_plotly(slope, 'Q_slope', start=start, end=end, channel=None)
                figure3 = PlottingTools.plotD_plotly(std, 'Q_std', start=start, end=end, channel=None)
                figure4 = PlottingTools.plotD_plotly(_range, 'Q_range', start=start, end=end, channel=None)
            else:
                print('full figs start')
                figure1 = PlottingTools.plotD_plotly(corr, 'Q_corr', channel=channel)
                figure2 = PlottingTools.plotD_plotly(slope, 'Q_slope', channel=channel)
                figure3 = PlottingTools.plotD_plotly(std, 'Q_std', channel=channel)
                figure4 = PlottingTools.plotD_plotly(_range, 'Q_range', channel=channel)

            return figure1, figure2, figure3, figure4
        else:
            print('faults figure prevented')
            raise PreventUpdate

# Update the sliders text
@app.callback(Output('corr-vals', 'children'),
              [Input('corr-slide', 'value')])
def display_value_corr(value):
    if len(value) < 2:
        raise PreventUpdate
    else:
        return 'Min: {:0.2f}, Max: {:0.2f}'.format(value[0], value[1])

@app.callback(Output('slope-vals', 'children'),
              [Input('slope-slide', 'value')])
def display_value_slope(value):
    if len(value) < 2:
        raise PreventUpdate
    else:
        transform = [transform_value(x) for x in value]
        return 'Min: {:0.5f}, Max: {:0.2f}'.format(transform[0], transform[1])


@app.callback(Output('std-vals', 'children'),
              [Input('std-slide', 'value')])
def display_value_std(value):
    if len(value) < 2:
        raise PreventUpdate
    else:
        return 'Min: {:0.1f}, Max: {:0.1f}'.format(value[0], value[1])


@app.callback(Output('range-vals', 'children'),
              [Input('range-slide', 'value')])
def display_value_range(value):
    if len(value) < 2:
        raise PreventUpdate
    else:
        return 'Min: {:0.0f}, Max: {:0.0f}'.format(value[0], value[1])


@app.callback(
    Output('multivariate-store', 'data'),
    [Input('send-to-multivar', 'n_clicks')]
    [State('select-series', 'value'),
        State('select-method', 'value'),
        State('sensors-store', 'data'),
        State('multivariate-store', 'data')])
def send_to_multivar(click, channel_info, method, uni_data, multi_data):
    if not click:
        raise PreventUpdate
    else:
        sensors, sensor_index, channel = get_sensors_and_channel(uni_data, channel_info)
        if channel.filtered[method] is not None:
            df = channel.filtered[method]
            if 'Treated' in df.columns and 'Deleted' in df.columns:
                project = channel.project
                location = channel.location
                equipment = channel.equipment
                parameter = channel.parameter
                unit = channel.unit
                name = '-'.join([project, location, equipment, parameter, unit])
                currentdf = channel.raw_data['raw'].join(df[['treated, deleted']], how='left')
                currentdf.columns = [name + '-raw', name + '-treated', name + '-deleted']
                if multi_data:
                    multi_df = pd.read_json(multi_data, orient='split')
                    currentdf = multi_df.join(currentdf, how='left')
                else:
                    pass
                currentdf.to_json(date_format='iso', orient='split')
                return currentdf
            else:
                raise PreventUpdate
        else:
            raise PreventUpdate


########################################################################
# MULTIVARIATE TAB #
########################################################################


###########################################################################
# SAVE DATA ###############################################################
###########################################################################


'''@app.callback(
    Output('save-sensors-link', 'href'),
    [Input('sensors-store', 'data')])
def update_link(data):
    return '/dash/urlToDownload?value={}'.format(data)

@app.server.route('/dash/urlToDownload')
def download_json():
    value = flask.request.args.get('value')
    # create a dynamic csv or file here using `StringIO`
    # (instead of writing to the file system)
    str_io = io.StringIO()
    str_io.write(str(value))
    mem = io.BytesIO()
    mem.write(str_io.getvalue().encode('utf-8'))
    mem.seek(0)
    str_io.close()
    return flask.send_file(
        mem,
        mimetype='text/json',
        attachment_filename='downloadFile.json',
        as_attachment=True)'''


if __name__ == '__main__':
    app.run_server(debug=True)
