from abc import ABC, abstractmethod
from app.domain.model.data_sensors import DataSensor

class PurpleAirGateway(ABC):
    @abstractmethod
    def get_data_by_zone(self, zone: int) -> DataSensor:
        """
        Obtiene los datos de la zona geográfica de la red de monitoreo de PurpleAir

        Args:
            zone: Zona geográfica

        Returns:
            DataSensor: Datos de la zona geográfica tomados de la red de monitoreo de PurpleAir
        """
        pass