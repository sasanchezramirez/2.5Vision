from abc import ABC, abstractmethod
from app.domain.model.data_sensors import PurpleAirDataSensors

class PurpleAirGateway(ABC):
    @abstractmethod
    def get_data_by_zone(self, zone: int) -> PurpleAirDataSensors:
        """
        Obtiene los datos de la zona geográfica de la red de monitoreo de PurpleAir

        Args:
            zone: Zona geográfica

        Returns:
            PurpleAirDataSensors: Datos de la zona geográfica tomados de la red de monitoreo de PurpleAir
        """
        pass