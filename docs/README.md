# Documentación del Proyecto 2.5Vision

Bienvenido a la documentación central del proyecto 2.5Vision. Este es el punto de partida para entender tanto el contexto científico como las decisiones de arquitectura que dan forma al sistema.

## 1. Fundamento Científico

La base de nuestro modelo de Machine Learning se apoya en una investigación rigurosa del dominio del problema.

Podemos dividir el alcance en tres dominios clave: 
- El *qué*: la luz interactuando con material particulado suspendido en la atmósfera.
- El *cómo*: análisis de imágenes usando visión artificial.
- El *porqué*: los sensores fotográficos como los traductores análogos-digitales perfectos para este problema.

Daremos una respuesta fundamentada a esto en los siguientes apartados:

- **[Revisión de Literatura](./scientific_background/01_literature_review.md):** Un resumen de los papers y estudios más influyentes en la estimación de PM2.5 a través de imágenes.
- **[Teoría de Ingeniería de Características](./scientific_background/02_feature_engineering_theory.md):** Análisis de las características visuales que correlacionan con la polución atmosférica.
- **[Análisis de Benchmarks](./scientific_background/03_benchmark_analysis.md):** Comparativa de los modelos y resultados del estado del arte.
- **[Métodos de Incertidumbre](./scientific_background/04_uncertainty_methods.md):** Exploración de técnicas para cuantificar la confianza de nuestras predicciones.

## 2. Arquitectura del Sistema

Las decisiones de diseño y arquitectura están documentadas para asegurar la claridad y la mantenibilidad a largo plazo.

- **[Resumen de Arquitectura](./architecture/README.md):** Una visión general de la arquitectura elegida.
- **[Diagrama del Sistema](./architecture/system_diagram.md):** Un diagrama visual de los componentes principales y sus interacciones.
- **[Registros de Decisiones de Arquitectura (ADRs)](./architecture/ADRs/):** Justificaciones detalladas de las decisiones tecnológicas y de diseño más importantes.
