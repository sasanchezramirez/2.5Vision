import requests
import logging
from typing import Final
from app.domain.gateway.purpleair_gateway import PurpleAirGateway
from app.application.settings import settings
from app.infrastructure.driven_adapter.purpleair.util.sensors_by_zone import SensorsByZoneDictionary
from app.infrastructure.driven_adapter.purpleair.dto.sensor_data_response_dto import SensorDataResponseDto
from app.domain.model.data_sensors import PurpleAirDataSensors
from app.infrastructure.driven_adapter.purpleair.mapper.purpleair_mapper import PurpleAirMapper
logger: Final[logging.Logger] = logging.getLogger("PurpleAir Service")

class PurpleAirService(PurpleAirGateway):
    def __init__(self):
        self.api_key = settings.PURPLEAIR_API_KEY
        self.base_url = settings.PURPLEAIR_BASE_URL

    def get_data_by_zone(self, zone: int) -> PurpleAirDataSensors:
        sensors_by_zone = SensorsByZoneDictionary().get_sensors_by_zone(zone)
        sensors = sensors_by_zone.sensors
        data = []
        for sensor in sensors:
            url = f"{self.base_url}/sensors/sensor/{sensor}"
            response = requests.get(url, headers={"X-API-Key": self.api_key})
            sensor_data = SensorDataResponseDto(**response.json())
            data.append(PurpleAirMapper.map_sensor_data_response_dto_to_data_sensor(sensor_data))
        return PurpleAirDataSensors(data=data, zone=zone)
