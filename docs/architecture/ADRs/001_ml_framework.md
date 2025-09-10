# ADR-001: Elección del Framework de Machine Learning

- **Fecha:** 2025-09-09
- **Estado:** Aceptado

---

## Contexto

Necesitamos seleccionar un framework de Machine Learning principal para el desarrollo, entrenamiento y validación de nuestros modelos de estimación de PM2.5. La elección del framework impactará la velocidad de investigación, la facilidad de implementación de papers científicos, la curva de aprendizaje del equipo y el ecosistema de despliegue disponible.

---

## Decisión

Elegiremos **PyTorch** como el framework principal de Machine Learning para este proyecto.

---

## Justificación

La decisión se basa en los principios de "Learning-First" y "Portfolio-Ready" de nuestro proyecto.

- **Flexibilidad para la Investigación:** PyTorch es conocido por su API imperativa y "pythonica", lo que facilita la depuración y la implementación de arquitecturas de modelos personalizadas. Muchos papers de investigación de vanguardia en Computer Vision publican su código fuente en PyTorch, lo que acelera nuestra capacidad para replicar y adaptar el estado del arte.

- **Curva de Aprendizaje:** Para desarrolladores con una sólida base en Python, la sintaxis de PyTorch es a menudo más intuitiva que la de TensorFlow, lo que se alinea con nuestro objetivo de maximizar el aprendizaje.

- **Ecosistema en Crecimiento:** Aunque históricamente TensorFlow tenía una ventaja en producción, el ecosistema de PyTorch (TorchServe, TorchScript) ha madurado significativamente y ofrece un camino claro hacia la producción.

### Alternativas Consideradas

- **Alternativa A: TensorFlow/Keras:**
  - **Pros:** Ecosistema de producción muy maduro (TFX, TF Serving), amplia comunidad, Keras ofrece una API de alto nivel muy sencilla para empezar.
  - **Contras:** La API puede ser menos flexible para la investigación. La depuración en el modo grafo declarativo puede ser más compleja.
  - **Razón para no elegirla:** Priorizamos la flexibilidad en la investigación sobre la madurez del ecosistema de producción en esta etapa inicial del proyecto.

- **Alternativa B: Scikit-learn puro:**
  - **Pros:** Extremadamente simple para modelos clásicos. Excelente para establecer un baseline rápido.
  - **Contras:** No es adecuado para modelos de Deep Learning (CNNs) que probablemente necesitaremos para el análisis de imágenes. No es un framework de "clase mundial" para Computer Vision.
  - **Razón para no elegirla:** Aunque lo usaremos para componentes específicos (ej. `StandardScaler` o modelos base en un ensamblaje), no es lo suficientemente potente como para ser el framework *principal*.

---

## Consecuencias

### Positivas

- Podremos implementar y experimentar con arquitecturas de modelos de papers recientes de manera más eficiente.
- El equipo desarrollará habilidades en un framework de Deep Learning moderno y muy demandado en la industria.
- El código del modelo será más fácil de depurar y entender.

### Negativas

- El despliegue a producción con TorchServe puede requerir un poco más de configuración manual en comparación con el ecosistema más integrado de TensorFlow Serving.
- Nos perdemos algunas de las abstracciones de alto nivel de Keras que pueden acelerar el desarrollo de modelos más simples.
