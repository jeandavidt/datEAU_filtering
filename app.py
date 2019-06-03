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

app.layout = html.Div([
    dcc.Store(id='session-store', storage_type='session'),
    html.H1(dcc.Markdown('dat*EAU* filtration'), id='header'),
    dcc.Tabs([
        dcc.Tab(label='Data Import', value='import'),
        dcc.Tab(label='Univariate filter', value='univar'),
        dcc.Tab(label='Multivariate filter', value='multivar')
        ],id="tabs", value='import'),
    html.Div(id='output-data-upload',style={'display':'none'}),
    html.Div(id='tabs-content')
    ], id='applayout')

@app.callback(
    Output('tabs-content', 'children'),
    [Input('tabs', 'value')]
    )
def render_content(tab):
    if tab == 'import':
        return html.Div([
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
    elif tab == 'univar':
        return html.Div([
            html.H3('Tab content 2')
        ])
    elif tab == 'multivar':
        return html.Div([
            html.H3('Tab content 3')
        ])

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
            return 
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
def store_raw(click, data, options, start, end):
    if not click:
        raise PreventUpdate
    start=pd.to_datetime(start)
    end = pd.to_datetime(end)
    df = pd.read_json(data, orient='split')
    df.index = pd.to_datetime(df.index)
    filtered = df.loc[start:end, options]
    return filtered.to_json(date_format='iso',orient='split')

#df.to_json(date_format='iso',orient='split')
if __name__ == '__main__':
    app.run_server(debug=True)
