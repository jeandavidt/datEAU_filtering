import numpy as np
import pandas as pd
import DefaultSettings

class Channel:
    def __init__(self, project, location, equipment,parameter, unit, frame):
        
        self.project = project
        self.location = location
        self.equipment = equipment
        self.parameter = parameter
        self.unit = unit
        column = '-'.join([project, location, equipment,parameter, unit])
        self.data = pd.DataFrame(data={'raw':frame[column]})
        self.start=self.data.first_valid_index()
        self.end=self.data.last_valid_index()
        self.params = DefaultSettings.DefaultParam()

class Sensor:
    def __init__(self,project, location, equipment):
        self.project = project
        self.location=location
        self.equipment=equipment
        self.channels={}
    
    def add_channel(self, project, location, equipment,parameter, unit, frame):
        self.channels[parameter] = Channel(project, location, equipment,parameter, unit, frame)
                


def parse_dataframe(df):    
    sensor_names = []

    sensors=[]
    for column in df.columns:
        project, location, equipment, parameter, unit = column.split('-')

        if equipment not in sensor_names:
            sensor_names.append(equipment)
            new_sensor = Sensor(project, location, equipment)

            for column in df.columns:
                project, location, equipment, parameter, unit = column.split('-')
                
                if ((equipment == new_sensor.equipment ) and (parameter not in new_sensor.channels.keys())):
                    new_sensor.add_channel(new_sensor.project, new_sensor.location, new_sensor.equipment, parameter, unit, df)
            
            sensors.append(new_sensor)
    return sensors



        

