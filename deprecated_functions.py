# @app.callback(
#    Output('download-raw-link', 'href'),
#    [Input('sql-store', 'data')])
# def update_link_rawdb(data):
#    if not data:
#        raise PreventUpdate
#    else:
#        return '/dash/download-rawdb?value={}'.format(data)


# @app.server.route('/dash/download-rawdb')
# def download_csv_rawdb():
#    value = flask.request.args.get('value')
#    df = pd.read_json(value, orient='split')
#    down = df.to_csv(sep=';')
#    str_io = io.StringIO()
#    str_io.write(str(down))
#    mem = io.BytesIO()
#    mem.write(str_io.getvalue().encode('utf-8'))
#    mem.seek(0)
#    str_io.close()
#    return flask.send_file(
#        mem,
#        mimetype='text/csv',
#        attachment_filename='download_raw.csv',
#        as_attachment=True)
#