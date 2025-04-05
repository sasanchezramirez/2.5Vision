from typing import BaseModel, List, Optional

class NetworkDataSensor(BaseModel):
    """
    Modelo para los datos de un sensor de PurpleAir
    El dato de pm2.5 es en un promedio de 30 minutos, los demás son instantaneos
    """
    sensor_index: Optional[int] = None
    name: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    pm1_0: Optional[float] = None
    pm2_5: Optional[float] = None
    pm10_0: Optional[float] = None
    humidity: Optional[float] = None
    temperature: Optional[float] = None
    pressure: Optional[float] = None
    confidence: Optional[int] = None
    network: str

class DataSensor(BaseModel):
    """Modelo para los datos de los sensores de PurpleAir"""
    data: List[NetworkDataSensor]
    zone: int

