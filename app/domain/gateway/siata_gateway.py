from abc import ABC, abstractmethod
from typing import Optional

class SiataGateway(ABC):
    @abstractmethod
    def get_data_by_zone(self, zone: int) -> dict:
        """
        Obtiene los datos de la zona geográfica.

        Args:
            zone: Zona geográfica

        Returns:
            dict: Datos de la zona geográfica
        """
        pass