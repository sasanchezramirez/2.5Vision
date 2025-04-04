from pydantic import BaseModel, Field
from typing import Optional

class StatsDto(BaseModel):
    """DTO para las estadísticas de PM2.5"""
    pm2_5: float = Field(alias="pm2.5")
    pm2_5_10minute: float = Field(alias="pm2.5_10minute")
    pm2_5_30minute: float = Field(alias="pm2.5_30minute")
    pm2_5_60minute: float = Field(alias="pm2.5_60minute")
    pm2_5_6hour: float = Field(alias="pm2.5_6hour")
    pm2_5_24hour: float = Field(alias="pm2.5_24hour")
    pm2_5_1week: float = Field(alias="pm2.5_1week")
    time_stamp: int

class SensorDto(BaseModel):
    """DTO para los datos del sensor"""
    sensor_index: int
    name: str
    latitude: float
    longitude: float
    pm1_0: float = Field(alias="pm1.0")
    pm2_5: float = Field(alias="pm2.5")
    pm10_0: float = Field(alias="pm10.0")
    humidity: float
    temperature: float
    pressure: float
    confidence: int
    stats: StatsDto
    stats_a: StatsDto
    stats_b: StatsDto

class SensorDataResponseDto(BaseModel):
    """DTO para la respuesta completa de PurpleAir"""
    api_version: str
    time_stamp: int
    data_time_stamp: int
    sensor: SensorDto
