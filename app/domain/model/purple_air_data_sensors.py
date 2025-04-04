from typing import BaseModel, List, Field

class DataSensor(BaseModel):
    """
    Modelo para los datos de un sensor de PurpleAir
    El dato de pm2.5 es en un promedio de 30 minutos, los demás son instantaneos
    """
    sensor_index: int
    name: str
    latitude: float
    longitude: float
    pm1_0: float
    pm2_5: float
    pm10_0: float
    humidity: float
    temperature: float
    pressure: float
    confidence: int 

class PurpleAirDataSensors(BaseModel):
    """Modelo para los datos de los sensores de PurpleAir"""
    data: List[DataSensor]
    zone: int

