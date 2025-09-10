# TBD: Métodos de Cuantificación de Incertidumbre

*Este documento es un placeholder. Aquí se explorarán y documentarán las técnicas para medir la confianza o incertidumbre de las predicciones del modelo.*

## Técnicas Consideradas

- **Redes Neuronales Bayesianas (BNN)**
  - **Pros:** Proporcionan una distribución de probabilidad completa para los pesos del modelo.
  - **Contras:** Computacionalmente costosas de entrenar.

- **Monte Carlo Dropout**
  - **Pros:** Fácil de implementar en redes neuronales existentes. Es una aproximación a las BNN.
  - **Contras:** La calidad de la estimación de incertidumbre puede variar.

- **Ensembles**
  - **Pros:** La desviación estándar de las predicciones de un ensemble de modelos puede ser usada como una medida de incertidumbre.
  - **Contras:** Requiere entrenar múltiples modelos.
