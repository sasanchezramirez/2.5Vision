import logging
from typing import Final

from app.domain.gateway.persistence_gateway import PersistenceGateway

logger: Final[logging.Logger] = logging.getLogger("MasterData UseCase")

class MasterdataUseCase:
    def __init__(self, persistence_gateway: PersistenceGateway):
        self.persistence_gateway = persistence_gateway

    def get_total_images_uploaded(self) -> int:
        return self.persistence_gateway.get_total_images_uploaded()
