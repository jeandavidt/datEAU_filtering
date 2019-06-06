import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_table
import plotly.graph_objs as go
import base64
import datetime
import json
import pickle
import os

import time

import io
import pandas as pd
import numpy as np
import Sensors
import PlottingTools
import DataCoherence
import OutlierDetection

#external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__)
app.config['suppress_callback_exceptions']=True

########################################################################
                            # APP LAYOUT #
########################################################################

app.layout = html.Div([
    dcc.Store(id='session-store'),
    dcc.Store(id='sensors-store'),
    dcc.Store(id='modif-store'),
    dcc.Store(id='coherence-store'),
    dcc.Store(id='new-params-store'),
    html.H1(dcc.Markdown('dat*EAU* filtration'), id='header'),
    dcc.Tabs([
        dcc.Tab(label='Data Import', value='import', children=[
            html.Div([
            html.H3('Import data'),
            dcc.Upload(
                id='upload-data',
                children=html.Button('Upload File',id='upload-button',n_clicks=0)
            ),
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
            dcc.Dropdown(
                id='select-series',
                multi=False, 
                placeholder='Pick a series to analyze here.',
                options=[]),
                html.Br(),
            html.Div(id='uni-up', children=[
                html.Div(id='uni-up-left', children=[
                    html.Button(id='check-coherence',children='Check data coherence'),
                    
                    html.Div(id='faults',children=[
                        html.Table([
                            html.Tr([
                                html.Th('Status'),
                                html.Th('Message')
                            ]),
                            html.Tr([
                                html.Td(id='err0-status'),
                                html.Td(id='err0-msg',children=[])
                            ]),
                            html.Tr([
                                html.Td(id='err1-status'),
                                html.Td(id='err1-msg',children=[])
                            ]),
                            html.Tr([
                                html.Td(id='err2-status'),
                                html.Td(id='err2-msg',children=[])
                            ]),
                            html.Tr([
                                html.Td(id='err3-status'),
                                html.Td(id='err3-msg',children=[])
                            ]),
                            html.Tr([
                                html.Td(id='err4-status'),
                                html.Td(id='err4-msg',children=[])
                            ])
                        ], style={}),
                        html.Button(id='sort-button',children=['Sort indices'],),
                        html.Br(),
                        html.Button(id='resample-button',children=['Resample'],),
                        dcc.Input(id='sample-freq',placeholder='frequency (min)',type='number', ),
                        html.Br(),
                        html.Button(id='fillna-button',children=['Fill blank rows']),
                        html.Br(),
                        html.Button(id='reset-button',children=['Reset to raw']),
                    ],),
                ], style={'width':'20%','display':'inline-block','float':'left'}),
                html.Div(id='uni-up-center',children=[
                    dcc.Graph(id='initial_uni_graph'),
                ], style={'width':'55%','display':'inline-block',}),
                html.Div(id='uni-up-right', children=[
                    'Use the Box Select tool to choose a range of data with which to fit the outlier detection model.',
                    html.Br(),
                    dcc.DatePickerSingle(id='fit-start'),
                    dcc.DatePickerSingle(id='fit-end'),
                    html.Br(),
                    html.Button(id='fit-button', children='Fit outlier filter'),
                ], style={'width':'25%','display':'inline-block','float':'right'}),
            ]),
            html.Hr(),
            html.Div(id='uni-down', children=[
                html.Br(),
                html.Div(id='uni-low-left', children=[
                    html.H6('Parameters list'),
                    dash_table.DataTable(
                        id='parameters-list-table',
                        columns = [{'name':'Parameter','id':'Parameter'},{'name':'Value','id':'Value'},{'name':'','id':'blank'}],
                        n_fixed_rows=1,
                        style_table={
                            'maxHeight': '300',
                            'overflowY': 'scroll'
                        },
                        style_cell_conditional=[
                            {'if': {'column_id': 'Parameter'},
                            'minWidth': '20%','maxWidth':'40%', 'textAlign':'left'},
                            {'if': {'column_id': 'Value'},
                            'minWidth': '10%','maxWidth':'20%', 'textAlign':'right'},
                            {'if': {'column_id': 'blank'},
                            'minWidth': '30%=','maxWidth':'50%', 'textAlign':'left'}
                        ],
                        style_header={
                            'backgroundColor': 'white',
                            'fontWeight': 'bold'
                        },
                        editable=True,
                        style_as_list_view=True,
                    ),
                    html.Br(),
                    html.Button(id='save-params-button',children=['Save Parameters']),
                ], style={'width':'20%','display':'inline-block','float':'left'}),
                html.Div(id='uni-down-center',children=[
                    dcc.Graph(id='faults-uni-graph'),
                ], style={'width':'60%','display':'inline-block',}),
                html.Div(id='uni-down-right',children=[
                    html.Button(id='detect_faults-uni',children='Detect Faults'),
                    html.Button(id='Accept-filter', children='Accept Filter results'),
                ], style={'width':'20%','display':'inline-block','float':'right'}),
            ])
        ]),
        ]),
        dcc.Tab(label='Multivariate filter', value='multivar')
        ],id="tabs", value='import'),
    html.Div(id='output-data-upload',style={'display':'none'}),
    html.Div(id='tabs-content')
    ], id='applayout')


########################################################################
                            # HELPER FUNCTIONS #
########################################################################
def parse_contents(contents, filename):
    _, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')),sep=';')
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
    return df.to_json(date_format='iso',orient='split')

def get_channel(data, channel_props):
    sensors = json.loads(data, object_hook=Sensors.decode_object)
    for sensor in sensors:
        project, location, equipment, parameter, _ = channel_props.split('-')
        if (sensor.project == project and sensor.location == location and sensor.equipment == equipment):
            return sensor.channels[parameter]

def get_sensors_and_channel(serialized_data, channel_props):
        sensors = json.loads(serialized_data, object_hook=Sensors.decode_object)
        for sensor in sensors:
            project, location, equipment, _, _ = channel_props.split('-')
            if (sensor.project == project and sensor.location == location and sensor.equipment == equipment):
                sensor_index=sensors.index(sensor)
                channel = get_channel(serialized_data, channel_props)
        return sensors, sensor_index, channel
########################################################################
                            # IMPORT TAB #
########################################################################
@app.callback(Output('output-data-upload', 'children'),
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename')])
def update_output(contents, filename):
    if contents is not None:
        data = parse_contents(contents, filename)
        return data

@app.callback(
    Output('upload-graph-location', 'children'),
    [Input('output-data-upload', 'children')],
    [State('upload-button','n_clicks')])
def update_upload_fig(data, n_clicks):
    if n_clicks == 0:
        return dcc.Graph(id='upload-graph')
    else:
        df = pd.read_json(data, orient='split')
        figure = PlottingTools.plotlyRaw_D(df)
        figure.update(dict(layout=dict(clickmode='event+select')))
        
        return dcc.Graph(id='upload-graph', figure=figure),
            

@app.callback(
    [Output('import-dates','start_date'),
    Output('import-dates','end_date')],
    [Input('upload-graph','selectedData')])
def add_interval(selection):
    start = selection['range']['x'][0]
    end = selection['range']['x'][1]
    return start, end

@app.callback(
    [Output('upload-dropdown','children')],
    [Input('output-data-upload', 'children')],
    [State('upload-button','n_clicks')])
def show_series_list(data, n_clicks):
    if n_clicks == 0:
        raise PreventUpdate 
    else:
        df = pd.read_json(data, orient='split')
        columns = df.columns
        labels = [column.split('-')[-2]+' '+ column.split('-')[-1] for column in columns]
        options =[{'label':labels[i], 'value':columns[i]} for i in range(len(columns))]
        return [html.Div(id='test',children=[dcc.Dropdown(
            id='series-selection',
            multi=True, 
            placeholder='Select series to analyze here.',
            options=options,
            ),
            html.Br()])]
@app.callback(
    [Output('button-location','children')],
    [Input('series-selection','value')],
    [State('import-dates','start_date'),
    State('import-dates','end_date')])
def check_if_ready_to_save(series,start,end):
    if (series is not None) and (start is not None) and (end is not None):
        return[html.Button(id='save-button',children='Save data for analysis')]
    else:
        return [html.Div('You must select at least one time series, a start date and an end date to continue')]

@app.callback(Output('session-store', 'data'),
              [Input('save-button', 'n_clicks')],
              [State('output-data-upload','children'),
              State('series-selection','value'),
              State('import-dates','start_date'),
              State('import-dates','end_date')])
def store_raw(click, data, series, start, end):
    if not click:
        raise PreventUpdate
    start= pd.to_datetime(start)
    end= pd.to_datetime(end)
    df = pd.read_json(data, orient='split')
    df.index = pd.to_datetime(df.index)
    filtered = df.loc[start:end, series]
    to_save = filtered.to_json(date_format='iso',orient='split')
    return to_save
########################################################################
                            # UNIVARIATE TAB #
########################################################################
@app.callback(
    Output('select-series', 'options'),
    [Input('session-store', 'data')])
def show_univar_list(data):
    if not data:
        raise PreventUpdate
    else:
        df = pd.read_json(data, orient='split')
        columns = df.columns
        labels = [column.split('-')[-2]+' '+ column.split('-')[-1] for column in columns]
        options =[{'label':labels[i], 'value':columns[i]} for i in range(len(columns))]
        return options

@app.callback(
    Output('sensors-store', 'data'),
    [Input('session-store','data'),
    Input('modif-store','data')])
def create_sensors(original_data, modif_data):
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
            
@app.callback(Output('modif-store','data'),
[Input('fillna-button','n_clicks_timestamp'),
Input('resample-button','n_clicks_timestamp'),
Input('sort-button','n_clicks_timestamp'),
Input('fit-button','n_clicks_timestamp'),
Input('save-params-button','n_clicks_timestamp'),
Input('reset-button','n_clicks_timestamp'),
Input('fit-end','date')],
[State('sensors-store','data'),
State('select-series', 'value'),
State('sample-freq','value'),
State('parameters-list-table','data'),
State('fit-start','date'),
])
def modify_sensors(fillna,resamp,srt,fit,param_button,reset,calib_end, sensor_data,channel_info,frequency,param_data, calib_start):
    triggers = [fillna, resamp, srt,fit, param_button, reset]
    if all(x is None for x in [triggers]):
        raise PreventUpdate
    if not sensor_data:
        raise PreventUpdate
    else:
        
        sensors, sensor_index, channel = get_sensors_and_channel(sensor_data, channel_info)
        triggers = np.array(triggers, dtype=np.float64)
        trigger = np.nanmax(triggers)
        
        if trigger == fillna:
            channel = DataCoherence.fillna(channel)
        elif trigger == resamp:
            if frequency is None:
                raise PreventUpdate
            freq = str(frequency)+' min'
            channel = DataCoherence.resample(channel,freq)
        elif trigger == srt:
            channel = DataCoherence.sort_dat(channel)
        elif trigger == fit:
            channel = OutlierDetection.outlier_detection(channel)
        elif trigger == param_button:
            new_params = {}
            for row in param_data:
                new_params[row['Parameter']]=row['Value']
            channel.params = new_params
        elif trigger == reset:
            channel.info={'most_recent_series':'raw'}
            channel.processed_data=None
        
        if calib_start and calib_end:
            if channel.calib is not None:
                channel_start = channel.calib['start']
                channel_end = channel.calib['end']
                if channel_start == calib_start and channel_end == calib_end:
                    raise PreventUpdate 
                else:
                     channel.calib['start']=calib_start
                     channel.calib['end']=calib_end
            else:
                channel.calib={'start':calib_start,'end':calib_end }
        
        sensor = sensors[sensor_index]
        sensor.channels[channel.parameter] = channel
        return json.dumps(sensors, indent=4, cls=Sensors.CustomEncoder)

@app.callback(
    Output('initial_uni_graph', 'figure'),
    [Input('select-series', 'value'),
    Input('sensors-store', 'data')])
def update_top_univariate_figure(value, data):
    if not value:
        raise PreventUpdate
    else:
        channel = get_channel(data, value)
        figure = PlottingTools.plotlyUnivar(channel)
        figure.update(dict(layout=dict(clickmode='event+select')))
        return figure

###########################################################################
######################## DATA COHERENCE ERROR MESSAGES ####################

@app.callback(
    Output('coherence-store','data'),
    [Input('check-coherence','n_clicks')],
    [State('sensors-store', 'data'),
    State('select-series', 'value')])
def run_data_coherence_check(click, data, channel_info):
    if not click:
        raise PreventUpdate
    else:
        channel = get_channel(data, channel_info)
        channel.params['Verbose']=False
        flag = DataCoherence.data_coherence(channel)
        return json.dumps(flag)

@app.callback(
    [Output('err0-msg','children'),
    Output('err0-status','children')],
    [Input('coherence-store','data')])
def flag_0(flag):
    if not flag:
        raise PreventUpdate
    else:
        flag=json.loads(flag)
        if '0' in flag.keys():
            return['Ready to move on!',
            html.Img(src=app.get_asset_url('check.png'),width='24')]
        else:
            return['You should probaly work on the data some more.',
            html.Img(src=app.get_asset_url('cross.png'),width='24')]
@app.callback(
    [Output('err1-msg','children'),
    Output('err1-status','children')],
    [Input('coherence-store','data')])
def flag1(flag):
    if not flag:
        raise PreventUpdate
    else:
        flag=json.loads(flag)
        if '1' in flag.keys():
            return[flag['1'],
            html.Img(src=app.get_asset_url('cross.png'),width='24')]
        else:
            return[html.P('There is no gap in the data'),
            html.Img(src=app.get_asset_url('check.png'),width='24')]
@app.callback(
    [Output('err2-msg','children'),
    Output('err2-status','children')],
    [Input('coherence-store','data')])
def flag2(flag):
    if not flag:
        raise PreventUpdate
    else:
        flag=json.loads(flag)
        if '2' in flag.keys():
            return[flag['2'],
            html.Img(src=app.get_asset_url('cross.png'),width='24')]
        else:
            return[html.P('The time step is constant.'),
            html.Img(src=app.get_asset_url('check.png'),width='24')]
@app.callback(
    [Output('err3-msg','children'),
    Output('err3-status','children')],
    [Input('coherence-store','data')])
def flag3(flag):
    if not flag:
        raise PreventUpdate
    else:
        flag=json.loads(flag)
        if 3 in flag.keys():
            return[flag[3],
            html.Img(src=app.get_asset_url('cross.png'),width='24')]
        else:
            return[html.P('There is no large gap in the data.'),
            html.Img(src=app.get_asset_url('check.png'),width='24')]
@app.callback(
    [Output('err4-msg','children'),
    Output('err4-status','children')],
    [Input('coherence-store','data')])
def flag4(flag):
    if not flag:
        raise PreventUpdate
    else:
        flag=json.loads(flag)
        if '4' in flag.keys():
            return[flag['4'],
            html.Img(src=app.get_asset_url('cross.png'),width='24')]
        else:
            return[html.P('The data is in chronological order.'),
            html.Img(src=app.get_asset_url('check.png'),width='24')]

###########################################################################
######################## UNIVARIATE FILTER CALIBRATION ####################

@app.callback(
    [Output('fit-start','date'),
    Output('fit-end','date')],
    [Input('initial_uni_graph','selectedData')])
def add_interval_fit(selection):
    if selection is None:
        raise PreventUpdate
    else:
        start = selection['range']['x'][0]
        end = selection['range']['x'][1]
        return [start, end]

###########################################################################
######################## POPULATE PARAMETERS TABLE ########################
@app.callback(
    Output('parameters-list-table','data'),
    [Input('select-series', 'value'),
    Input('sensors-store','data')])
def fill_params_table(channel_info, data):
    if channel_info is None:
        raise PreventUpdate
    else:
        channel = get_channel(data, channel_info)
        params = channel.params    
        parameters = list(params.keys())
        values = list(params.values())
        data=pd.DataFrame(data={'Parameter':parameters,'Value':values})
        data=data.to_dict('records')
        return data
###########################################################################
######################## PLOT FILTERED DATA ###############################
@app.callback(
    Output('faults-uni-graph','figure'),
    [Input('select-series', 'value'),
    Input('sensors-store', 'data')]
)
def update_second_univariate_figure(value, data):
    if not value:
        raise PreventUpdate
    else:
        channel = get_channel(data, value)
        if channel.filtered is None:
            raise PreventUpdate
        else:
            filtration_method = channel.params['OutlierDetectionMethod']
            figure = PlottingTools.Plotly_Outliers(channel, filtration_method)
            figure.update(dict(
                layout=dict(clickmode='event+select')
            ))
            return figure
if __name__ == '__main__':
    app.run_server(debug=True)
