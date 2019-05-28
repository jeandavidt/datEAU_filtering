### ATTENTION: This script only works on Windows with
### a VPN connection opened to the DatEAUbase Server
import getpass
import pandas as pd
import pyodbc
import time
import numpy as np
from matplotlib import pyplot as plt

def create_connection():
    import getpass
    username=input("Enter username")
    password =getpass.getpass(prompt="Enter password")
    config = dict(server=   '10.10.10.10', # change this to your SQL Server hostname or IP address
                port=      1433,                    # change this to your SQL Server port number [1433 is the default]
                database= 'dateaubase',
                username= username,
                password= password)
    conn_str = ('SERVER={server},{port};'   +
                'DATABASE={database};'      +
                'UID={username};'           +
                'PWD={password}')
    conn = pyodbc.connect(
        r'DRIVER={ODBC Driver 13 for SQL Server};' +
        conn_str.format(**config)
        )
    cursor = conn.cursor()
    return cursor

cursor = create_connection()

def date_to_epoch(date):
    naive_datetime = pd.to_datetime(date)
    local_datetime = naive_datetime.tz_localize(tz='US/Eastern')
    return int(local_datetime.value/10**9)
def epoch_to_pandas_datetime(epoch):
    local_time = time.localtime(epoch)
    return pd.Timestamp(*local_time[:6])


def build_query(start, end, location, parameter, equipment):
    return '''SELECT dbo.value.Timestamp,
dbo.value.Value as measurement,
dbo.parameter.Parameter as par,
dbo.unit.Unit,
dbo.equipment.Equipment_identifier as equipment,
dbo.sampling_points.Sampling_location,
dbo.project.Project_name

FROM dbo.parameter
left outer join dbo.metadata on dbo.parameter.Parameter_ID = dbo.metadata.Parameter_ID 
left outer join dbo.value on dbo.value.Metadata_ID = dbo.metadata.Metadata_ID
left outer join dbo.unit on dbo.parameter.Unit_ID = dbo.unit.Unit_ID
left outer join dbo.equipment on dbo.metadata.Equipment_ID = dbo.equipment.Equipment_ID
left outer join dbo.sampling_points on dbo.metadata.Sampling_point_ID = dbo.sampling_points.Sampling_point_ID
left outer join dbo.project on dbo.metadata.Project_ID = dbo.project.Project_ID
WHERE dbo.value.Timestamp > {}
AND dbo.value.Timestamp < {}
AND dbo.sampling_points.Sampling_location = \'{}\'
AND dbo.parameter.Parameter = \'{}\'
AND dbo.equipment.Equipment_identifier = \'{}\'
order by dbo.value.Value_ID;
'''.format(start, end, location, parameter,equipment)

def clean_up_pulled_data(df,param):
    df['datetime'] = [epoch_to_pandas_datetime(x) for x in df.Timestamp]
    df.drop(['Timestamp','Project_name','par'],axis=1, inplace=True)
    df.sort_values('datetime',axis=0, inplace=True)

    if df.Unit[0] == 'kg/m3':
        df.measurement *= 1000
        df.Unit='g/m3'
    elif df.Unit[0] == 'S/m':
        df.measurement *=10**4
        df.Unit = 'uS/cm'
    elif df.Unit[0] == 'ppm':
        df.Unit = 'g/m3'
    df.rename(columns={
        'measurement':'{} Value'.format(param),
        'Unit':'{} Unit'.format(param),
        'equipment': '{} Equipment'.format(param),
        'Sampling_location': '{} Location'.format(param),
    }, inplace=True)
    df.set_index('datetime', inplace=True, drop=True)
    df = df[~df.index.duplicated(keep='first')]
    return df


Start = date_to_epoch('2017-09-01 12:00:00')
End = date_to_epoch('2018-09-01 12:00:00')
Location = 'Primary settling tank effluent'

param_list = ['COD','CODf','NH4-N','K']
equip_list = ['Spectro_010','Spectro_010','Ammo_005','Ammo_005']

extract_list={}
for i in range(len(param_list)):
    extract_list[i] = {
        'Start':Start,
        'End':End,
        'Location':Location,
        'Parameter':param_list[i],
        'Equipment':equip_list[i]
    }
for i in range(len(extract_list)):
    query =build_query(extract_list[i]['Start'],
                           extract_list[i]['End'],
                           extract_list[i]['Location'],
                           extract_list[i]['Parameter'],
                           extract_list[i]['Equipment'])
    if i==0:
        df = pd.read_sql(query,conn)
        clean_up_pulled_data(df,extract_list[i]['Parameter'])
    else:
        temp_df = pd.read_sql(query,conn)
        clean_up_pulled_data(temp_df,extract_list[i]['Parameter'])
        df = pd.concat([df,temp_df], axis=1)
        df = df[~df.index.duplicated(keep='first')]
        

def plot_pulled_data(df):
    from pandas.plotting import register_matplotlib_converters
    register_matplotlib_converters()
    sensors=[]
    plt.figure(figsize=(12,8))
    for column in df.columns:
        if "Value" in column:
            sensors.append(column.split(' ')[0])
    for sensor in sensors:
        plt.plot(df[sensor+' Value'],alpha=0.8)
    plt.legend(sensors)
    plt.xticks(rotation=45)
    
    plt.show()

plot_pulled_data(df)

name = 'influent2'
path = r"C:\Users\Jean-David Therrien\Desktop\\"
result.to_csv(path+name+'.csv',sep=';')