import io
import os
import pandas.util.testing as tm
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import flask

import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
# max: 760000 cells
tm.N, tm.K = 100, 10
df = tm.makeDataFrame()

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    dcc.Store(id='data-store'),
    html.Div(id='placeholder'),
    dcc.Graph(id='graph'),
    html.Button(id='save-to-store', children='Import Data'),
    html.Button(id='save-graph', children=[
        html.A('Download graph', id='download-graph-link', target='blank')
    ]),
    html.Button(id='save-data', children=[
        html.A('Download data', id='download-data-link')
    ]),
])

@app.callback(
    Output('data-store', 'data'),
    [Input('save-to-store', 'n_clicks')])
def save_to_store(click):
    if not click:
        raise PreventUpdate
    else:
        return df.to_json()


@app.callback(
    Output('graph', 'figure'),
    [Input('data-store', 'data')])
def update_figure(data):
    if not data:
        raise PreventUpdate
    else:
        df = pd.read_json(data)
        traces = []
        for column in df.columns:
            traces.append(go.Scatter(
                x=df.index,
                y=df[column],
                mode='markers',
                opacity=0.7,
                marker={
                    'size': 15,
                    'line': {'width': 0.5, 'color': 'white'}
                },
                name=column
            ))

        return {
            'data': traces,
            'layout': go.Layout(
                xaxis={'title': 'X'},
                yaxis={'title': 'Y'},
            )
        }

@app.callback(
    Output('download-graph-link', 'href'),
    [Input('save-graph', 'n_clicks')],
    [State('graph', 'figure')])
def try_img(click, figure):
    if not click:
        raise PreventUpdate
    else:
        fig_name = 'figure-{}.svg'.format(click)
        relative_filename = os.path.join('figures', fig_name)
        absolute_filename = os.path.join(os.getcwd(), relative_filename)
        pio.write_image(figure, absolute_filename)
        return '/{}'.format(relative_filename)

@app.server.route('/figures/<path:path>')
def serve_static(path):
    root_dir = os.getcwd()
    return flask.send_from_directory(
        os.path.join(root_dir, 'figures'),
        path,
        attachment_filename=path,
        as_attachment=True
    )

@app.callback(
    Output('download-data-link', 'href'),
    [Input('data-store', 'data')])
def update_link(value):
    return '/dash/urlToDownload?value={}'.format(value)

@app.server.route('/dash/urlToDownload')
def download_csv():
    value = flask.request.args.get('value')
    # create a dynamic json or file here using `StringIO`
    # (instead of writing to the file system)
    str_io = io.StringIO()
    str_io.write(str(value))
    mem = io.BytesIO()
    mem.write(str_io.getvalue().encode('utf-8'))
    mem.seek(0)
    str_io.close()
    return flask.send_file(
        mem,
        mimetype='text/csv',
        attachment_filename='download.csv',
        as_attachment=True)

if __name__ == '__main__':
    app.run_server(debug=True)
