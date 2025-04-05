from abc import ABC, abstractmethod
from typing import List
from app.domain.model.data_sensors import DataSensor

class SiataGateway(ABC):
    @abstractmethod
    def get_data_by_zone(self, zone: int) -> List[DataSensor]:
        """
        Obtiene los datos de la zona geográfica.

        Args:
            zone: Zona geográfica

        Returns:
            List[DataSensor]: Datos de la zona geográfica
        """
        pass