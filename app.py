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
import pandas as pd
import time
import uuid
import io

import pandas as pd

import Sensors
import PlottingTools

#external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__)
app.config['suppress_callback_exceptions']=True

########################################################################
                            # APP LAYOUT #
########################################################################

app.layout = html.Div([
    dcc.Store(id='session-store'),
    dcc.Store(id='sensors-store'),
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
                    html.Button(id='check-integrity-button',children='Check integrity'),
                    html.Div(id='show_faults'),
                ], style={'width':'15%','display':'inline-block','float':'left'}),
                html.Div(id='uni-up-center',children=[
                    dcc.Graph(id='initial_uni_graph'),
                ], style={'width':'60%','display':'inline-block',}),
                html.Div(id='uni-up-right', children=[
                    dcc.DatePickerRange(id='calib-range'),
                    html.Button(id='calibrate-button', children='Calibrate filter'),
                ], style={'width':'25%','display':'inline-block','float':'right'}),
            ],),
            html.Hr(),
            html.Div(id='uni-down', children=[
                html.Br(),
                html.Div(id='uni-low-left', children=[
                    html.H6('Parameters list'),
                    html.Br(),
                    html.Div(id='parameters-list'),
                ], style={'width':'20%','display':'inline-block','float':'left'}),
                html.Div(id='uni-down-center',children=[
                    dcc.Graph(id='faults-uni-graph'),
                ], style={'width':'60%','display':'inline-block',}),
                html.Div(id='uni-down-right',children=[
                    html.Button(id='detect_faults-uni',children='Detect Faults'),
                    html.Button(id='Accept-filter', children='Accept Filter results'),
                ], style={'width':'20%','display':'inline-block','float':'right'}),
            ])
        ])
        ]),
        dcc.Tab(label='Multivariate filter', value='multivar')
        ],id="tabs", value='import'),
    html.Div(id='output-data-upload',style={'display':'none'}),
    html.Div(id='tabs-content')
    ], id='applayout')

########################################################################
                            # IMPORT TAB #
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
    State('import-dates','end_date')]
)
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

@app.callback([
    Output('select-series', 'options'),
    Output('sensors-store', 'data')],
    [Input('session-store', 'data')])
def parse_data_for_analysis(data):
    if not data:
        raise PreventUpdate
    else:
        df = pd.read_json(data, orient='split')
        columns = df.columns
        labels = [column.split('-')[-2]+' '+ column.split('-')[-1] for column in columns]
        options =[{'label':labels[i], 'value':columns[i]} for i in range(len(columns))]

        sensors = Sensors.parse_dataframe(df)
        
        return [sensors, options]


if __name__ == '__main__':
    app.run_server(debug=True)
