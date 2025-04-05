import logging
from typing import Final, List
from app.domain.gateway.siata_gateway import SiataGateway
from app.domain.model.data_sensors import DataSensor
logger: Final[logging.Logger] = logging.getLogger("Siata Service")

class SiataService(SiataGateway):
    def __init__(self):
        pass    

    def get_data_by_zone(self, zone: int) -> List[DataSensor]:
        """
        Obtiene los datos de la zona geográfica.

        Args:
            zone: Zona geográfica

        Returns:
            List[DataSensor]: Datos de la zona geográfica
        """
        logger.info(f"Obteniendo datos de SIATA para la zona {zone}")
        return []    