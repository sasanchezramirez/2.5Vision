from abc import ABC, abstractmethod
from app.domain.model.data_sensors import DataSensor
from app.domain.model.pm_estimation import PMEstimation

class EstimationMLModelGateway(ABC):
    @abstractmethod
    def estimate_pm(self, pm_data: list[DataSensor], feature_vector: list[float]) -> PMEstimation:
        """
        Estima la cantidad de material particulado presente en la imagen.

        Args:
            pm_data: Lista de datos de sensores de material particulado.
            feature_vector: Vector de características.

        Returns:
            PMEstimation: Estimación de la cantidad de material particulado presente y confianza de la estimación.
        """ 
        pass
