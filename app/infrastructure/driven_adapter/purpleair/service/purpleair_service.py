import requests
import logging
from typing import Final
from app.domain.gateway.purpleair_gateway import PurpleAirGateway
from app.application.settings import settings
from app.infrastructure.driven_adapter.purpleair.util.sensors_by_zone import SensorsByZoneDictionary
logger: Final[logging.Logger] = logging.getLogger("PurpleAir Service")

class PurpleAirService(PurpleAirGateway):
    def __init__(self):
        self.api_key = settings.PURPLEAIR_API_KEY
        self.base_url = settings.PURPLEAIR_BASE_URL

    def get_data_by_zone(self, zone: int) -> dict:
        sensors_by_zone = SensorsByZoneDictionary().get_sensors_by_zone(zone)
        sensors = sensors_by_zone.sensors
        data = []
        for sensor in sensors:
            url = f"{self.base_url}/sensors/sensor/{sensor}"
            response = requests.get(url, headers={"X-API-Key": self.api_key})
            data.append(response.json())
            #FALTA ORGANIZAR EL RESULTADO DE MANERA CONSOLIDADA
        return data
