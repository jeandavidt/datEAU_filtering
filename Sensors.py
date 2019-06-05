import numpy as np
import pandas as pd
import DefaultSettings
import json

class Channel:
    def __init__(self, project, location, equipment,parameter, unit, frame=None):
        
        self.project = project
        self.location = location
        self.equipment = equipment
        self.parameter = parameter
        self.unit = unit
        column_name = '-'.join([project, location, equipment,parameter, unit])
        if frame is None:
            self.raw_data = None
        elif isinstance(frame, pd.DataFrame):
            self.raw_data = pd.DataFrame(data={'raw':frame[column_name]})
            self.start=self.raw_data.first_valid_index()
            self.end=self.raw_data.last_valid_index()
            
        elif isinstance(frame, str):
            self.raw_data = pd.read_json(frame, orient='split')
            if 'raw' not in self.raw_data.columns:
                self.raw_data = pd.DataFrame(data={'raw':self.raw_data[column_name]})
            self.start=self.raw_data.first_valid_index()
            self.end=self.raw_data.last_valid_index()
            
        self.processed_data = None
        self.params = DefaultSettings.DefaultParam()
        self.info={'most_recent_series':'raw'}

class Sensor:
    def __init__(self,project, location, equipment, frame=None):
        self.project = project
        self.location=location
        self.equipment=equipment
        self.channels={}
        if frame is None:
            self.frame=None
        elif isinstance(frame, str):
            self.frame=pd.read_json(frame,orient='split')
        else:
            self.frame=frame
        
    
    def add_channel(self, project, location, equipment,parameter, unit, frame):
        self.channels[parameter] = Channel(project, location, equipment,parameter, unit, frame)
                


def parse_dataframe(df):    
    sensor_names = []

    sensors=[]
    for column in df.columns:
        project, location, equipment, parameter, unit = column.split('-')

        if equipment not in sensor_names:
            sensor_names.append(equipment)
            match = '-'.join([project, location, equipment])
            new_sensor = Sensor(project, location, equipment, df.filter(regex=match))

            for column in df.columns:
                project, location, equipment, parameter, unit = column.split('-')
                
                if ((equipment == new_sensor.equipment ) and (parameter not in new_sensor.channels.keys())):
                    new_sensor.add_channel(new_sensor.project, new_sensor.location, new_sensor.equipment, parameter, unit, df)
            
            sensors.append(new_sensor)
    return sensors



class CustomEncoder(json.JSONEncoder):
    import Sensors
    import pandas as pd
    def default(self, o):
        if (isinstance(o, Sensor) or isinstance(o, Channel)):
            return {'__{}__'.format(o.__class__.__name__): o.__dict__}
        elif isinstance(o, pd.Timestamp):
            return {'__Timestamp__': str(o)}
        else:
            return json.JSONEncoder.default(self, o)

def decode_object(o):
    import Sensors
    if '__Channel__' in o:   
        a = Sensors.Channel(
            o['__Channel__']['project'], 
            o['__Channel__']['location'], 
            o['__Channel__']['equipment'], 
            o['__Channel__']['parameter'], 
            o['__Channel__']['unit'],
            o['__Channel__']['data'])
        a.__dict__.update(o['__Channel__'])
        return a

    elif '__Sensor__' in o:
        a = Sensors.Sensor(
            o['__Sensor__']['project'],
            o['__Sensor__']['location'],
            o['__Sensor__']['equipment'],
            o['__Sensor__']['frame'])
        a.__dict__.update(o['__Sensor__'])
        return a

    elif '__Timestamp__' in o:
        return pd.to_datetime (o['__Timestamp__'])
    else:
        return o


'''
import json
import pandas as pd
import numpy as np
import Sensors

path = '../sample_data/influent3.csv'
raw_data =pd.read_csv(path, sep=';')
raw_data.datetime = pd.to_datetime(raw_data.datetime)
raw_data.set_index('datetime', inplace=True, drop=True)

sensors = Sensors.parse_dataframe(raw_data)

serialized = json.dumps(sensors, indent=4, cls=CustomEncoder)
deserialized = json.loads(serialized, object_hook=decode_object)

with open('orig.json', 'w') as f:
    json.dump(sensors, f, sort_keys=True, indent=4, cls=CustomEncoder)
with open('deserialized.json', 'w') as g:
    json.dump(deserialized, g, sort_keys=True, indent=4, cls=CustomEncoder)

print(deserialized == sensors)
with open('orig.json', 'r') as file1:
    with open('deserialized.json', 'r') as file2:
        same = set(file1).intersection(file2)

same.discard('\n')

with open('some_output_file.txt', 'w') as file_out:
    for line in same:
        file_out.write(line)'''