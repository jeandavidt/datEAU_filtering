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


def build_query(start, end, project, location, equipment, parameter):
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
AND dbo.project.Project_name = \'{}\'
order by dbo.value.Value_ID;
'''.format(start, end, location, parameter,equipment, project)

def clean_up_pulled_data(df,project, location, equipment, parameter):
    df['datetime'] = [epoch_to_pandas_datetime(x) for x in df.Timestamp]
    df.sort_values('datetime',axis=0, inplace=True)
    parameter = parameter.replace('-','_')
    Unit = df.Unit[0]
    if Unit == 'kg/m3':
        df.measurement *= 1000
        df.Unit='g/m3'
    elif Unit == 'S/m':
        df.measurement *=10**4
        df.Unit = 'uS/cm'
    elif Unit == 'ppm':
        df.Unit = 'g/m3'
    Unit = df.Unit[0]
    df.drop(['Timestamp','Project_name','par','Unit','equipment','Sampling_location'],axis=1, inplace=True)
    df.rename(columns={
        'measurement':'{}-{}-{}-{}-{}'.format(project, location, equipment, parameter, Unit),
    }, inplace=True)
    df.set_index('datetime', inplace=True, drop=True)
    df = df[~df.index.duplicated(keep='first')]
    return df


Start = date_to_epoch('2017-09-01 12:00:00')
End = date_to_epoch('2018-09-01 12:00:00')
Location = 'Primary settling tank effluent'
Project = 'pilEAUte'

param_list = ['COD','CODf','NH4-N','K']
equip_list = ['Spectro_010','Spectro_010','Ammo_005','Ammo_005']

extract_list={}
for i in range(len(param_list)):
    extract_list[i] = {
        'Start':Start,
        'End':End,
        'Project':Project,
        'Location':Location,
        'Parameter':param_list[i],
        'Equipment':equip_list[i]
    }

def extract_data(connexion, extraction_list):
    for i in range(len(extract_list)):
        query =build_query(extract_list[i]['Start'],
                               extract_list[i]['End'],
                               extract_list[i]['Project'],
                               extract_list[i]['Location'],
                               extract_list[i]['Equipment'],
                               extract_list[i]['Parameter'])
        if i==0:
            df = pd.read_sql(query,conn)
            clean_up_pulled_data(df,
                                 extract_list[i]['Project'],
                                 extract_list[i]['Location'],
                                 extract_list[i]['Equipment'],
                                 extract_list[i]['Parameter'])
        else:
            temp_df = pd.read_sql(query,conn)
            clean_up_pulled_data(temp_df,
                                 extract_list[i]['Project'],
                                 extract_list[i]['Location'],
                                 extract_list[i]['Equipment'],
                                 extract_list[i]['Parameter'])
            df = pd.concat([df,temp_df], axis=1)
            df = df[~df.index.duplicated(keep='first')]
    return df

def plot_pulled_data(df):
    from pandas.plotting import register_matplotlib_converters
    register_matplotlib_converters()
    sensors=[]
    units = []
    plt.figure(figsize=(12,8))
    for column in df.columns:
        sensors.append(column.split('-')[-2])
        units.append(column.split('-')[-1])
        plt.plot(df[column],alpha=0.8)
    sensors=[sensor.replace('_','-')for sensor in sensors]
    plt.legend([sensors[i]+' ('+units[i]+')' for i in range(len(sensors))])
    plt.xticks(rotation=45)
    
    plt.show()

plot_pulled_data(df)

name = 'influent3'
path = r"C:\Users\Jean-David Therrien\Desktop\\"
#result.to_csv(path+name+'.csv',sep=';')