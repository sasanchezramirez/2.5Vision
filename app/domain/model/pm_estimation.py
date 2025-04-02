from pydantic import BaseModel, Field

class PMEstimation(BaseModel):
    """
    Modelo de dominio para representar la estimación de la cantidad de material particulado presente.
    """
    pm_estimation: float = Field(
        default=0,
        description="Estimación de la cantidad de material particulado presente"
    )
    pm_estimation_confidence: float = Field(
        default=0,
        description="Confianza en la estimación de la cantidad de material particulado presente"
    )
    pm_qualitative_estimation: str = Field(
        default="",
        description="Estimación cualitativa de la cantidad de material particulado presente"
    )