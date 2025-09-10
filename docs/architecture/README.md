# Resumen de Arquitectura - 2.5Vision

Esta sección detalla la arquitectura de alto nivel del sistema 2.5Vision.

## Principios Arquitectónicos

Nuestra arquitectura se guía por los siguientes principios, definidos en la Fase 0:

- **Simplicidad y Bajo Costo:** Para la fase inicial, priorizamos una arquitectura monolítica simple desplegada en una única instancia EC2 para mantener los costos por debajo de los $10/mes.
- **Orientada al Aprendizaje:** Las decisiones tecnológicas favorecen la exposición a conceptos fundamentales de MLOps (ej. despliegue en EC2 vs. una solución totalmente gestionada) sobre la abstracción.
- **Preparada para Evolucionar:** Aunque es un monolito, el código interno sigue principios de alta cohesión y bajo acoplamiento, facilitando una futura migración a microservicios o una arquitectura serverless.

## Componentes Principales

- **Application Layer:** Una API de FastAPI que sirve como punto de entrada para las solicitudes de predicción.
- **ML Pipeline:** El núcleo lógico que incluye preprocesamiento de imágenes, extracción de características, inferencia del modelo y validación.
- **Data Layer:** Incluye la ingesta de datos de APIs externas (PurpleAir, SIATA) y el almacenamiento de imágenes y artefactos en AWS S3.
- **Infrastructure:** Aprovisionada vía Terraform, consiste en una instancia EC2 t3.micro, un bucket S3 y métricas/logs en CloudWatch.

Para un detalle visual, consulta el [Diagrama del Sistema](./system_diagram.md).

Para las justificaciones detalladas de cada decisión, consulta nuestros [Registros de Decisiones de Arquitectura (ADRs)](./ADRs/).
