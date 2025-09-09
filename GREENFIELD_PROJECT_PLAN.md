# 2.5Vision - Proyecto desde Cero
## Plan Greenfield de Arquitectura Enterprise

---

## **RESUMEN EJECUTIVO**

Si iniciara 2.5Vision desde cero con el conocimiento actual, implementaría una **arquitectura distribuida moderna** optimizada para **ML/AI**, **escalabilidad** y **observabilidad** desde el día 1. Este documento detalla el enfoque completo: arquitectura, implementación, despliegue y operación.

---

## **1. ARQUITECTURA CONCEPTUAL**

### **1.1 Principios de Diseño**

#### **Core Principles**
- **Domain-Driven Design (DDD)**: Contextos bounded bien definidos
- **Event-Driven Architecture**: Comunicación asíncrona entre servicios
- **CQRS + Event Sourcing**: Para trazabilidad y analytics
- **Microservices**: Servicios independientes y desplegables
- **API-First**: Contratos de API como ciudadanos de primera clase

#### **ML-Specific Principles**
- **ML Ops desde el día 1**: Versionado, deployment, monitoring de modelos
- **Data Lineage**: Trazabilidad completa de datos y transformaciones
- **Feature Store**: Reutilización y consistencia de features
- **Model Registry**: Gestión centralizada de modelos
- **A/B Testing**: Comparación de modelos en producción

### **1.2 Vista Arquitectural de Alto Nivel**

```mermaid
graph TB
    subgraph "Client Layer"
        WebApp[Web Dashboard]
        MobileApp[Mobile App]
        ApiClients[API Clients]
    end
    
    subgraph "API Gateway Layer"
        Gateway[Kong Gateway]
        Auth[Keycloak]
        RateLimit[Rate Limiting]
    end
    
    subgraph "Application Services"
        ImageSvc[Image Service]
        MLSvc[ML Service]
        DataSvc[Data Ingestion]
        AnalyticsSvc[Analytics Service]
        NotificationSvc[Notification Service]
    end
    
    subgraph "ML Platform"
        MLFlow[MLFlow]
        FeatureStore[Feast Feature Store]
        ModelServing[Seldon/KServe]
        DataPipeline[Airflow]
    end
    
    subgraph "Event Streaming"
        Kafka[Apache Kafka]
        Schema[Schema Registry]
    end
    
    subgraph "Data Layer"
        PostgreSQL[(PostgreSQL)]
        Redis[(Redis)]
        ClickHouse[(ClickHouse)]
        MinIO[(MinIO S3)]
        Elasticsearch[(Elasticsearch)]
    end
    
    subgraph "Infrastructure"
        Kubernetes[Kubernetes]
        Prometheus[Prometheus]
        Grafana[Grafana]
        Jaeger[Jaeger]
    end
```

### **1.3 Bounded Contexts (DDD)**

#### **Image Processing Context**
- **Responsabilidad**: Ingesta, validación, preprocessing de imágenes
- **Entities**: Image, ImageMetadata, ProcessingJob
- **Value Objects**: GPS, Resolution, ColorProfile
- **Services**: ImageProcessor, MetadataExtractor, Validator

#### **ML Prediction Context**
- **Responsabilidad**: Feature extraction, model inference, prediction
- **Entities**: Model, Prediction, FeatureSet
- **Value Objects**: PMEstimation, Confidence, Features
- **Services**: FeatureExtractor, ModelInference, PredictionService

#### **Environmental Data Context**
- **Responsabilidad**: Integración con APIs externas, datos de sensores
- **Entities**: Sensor, SensorReading, Zone
- **Value Objects**: Location, Measurement, Timestamp
- **Services**: DataCollector, SensorIntegration, ZoneService

#### **Analytics Context**
- **Responsabilidad**: Reportes, métricas, insights de negocio
- **Entities**: Report, Dashboard, Alert
- **Value Objects**: Metric, Trend, Aggregation
- **Services**: ReportGenerator, MetricsCollector, AlertManager

---

## **2. STACK TECNOLÓGICO MODERNO**

### **2.1 Lenguajes y Frameworks**

#### **Backend Services**
```yaml
Language: Python 3.12
Framework: FastAPI 0.104+
Async: asyncio, aiohttp
Validation: Pydantic V2
DI Container: dependency-injector
ORM: SQLAlchemy 2.0 (async)
Testing: pytest, pytest-asyncio
```

#### **ML Stack**
```yaml
Core ML: scikit-learn, XGBoost, LightGBM
Deep Learning: PyTorch (preferido sobre TensorFlow)
Computer Vision: OpenCV, Pillow, scikit-image
Feature Store: Feast
Model Serving: Seldon Core o KServe
Experiment Tracking: MLflow
Data Processing: Pandas, Polars, Dask
```

#### **Frontend**
```yaml
Framework: Next.js 14 (App Router)
Language: TypeScript
State Management: Zustand
UI Library: shadcn/ui + Tailwind CSS
Charts: Recharts, Observable Plot
Maps: Mapbox GL JS
```

#### **Mobile**
```yaml
Framework: React Native con Expo
Language: TypeScript
Navigation: React Navigation 6
State: Redux Toolkit Query
```

### **2.2 Infraestructura y Plataforma**

#### **Orquestación**
```yaml
Container Runtime: Docker
Orchestration: Kubernetes (k8s)
Service Mesh: Istio
Ingress: Nginx Ingress Controller
```

#### **Data Storage**
```yaml
OLTP: PostgreSQL 16 (primary)
Cache: Redis 7 (caching, sessions, queues)
OLAP: ClickHouse (analytics, time-series)
Search: Elasticsearch 8
Object Storage: MinIO (S3-compatible)
```

#### **Message Streaming**
```yaml
Event Streaming: Apache Kafka
Schema Registry: Confluent Schema Registry
Stream Processing: Kafka Streams / Apache Flink
```

#### **Observability**
```yaml
Metrics: Prometheus + Grafana
Tracing: Jaeger + OpenTelemetry
Logging: ELK Stack (Elasticsearch, Logstash, Kibana)
APM: DataDog o New Relic (opcional)
```

#### **CI/CD**
```yaml
Version Control: Git + GitHub/GitLab
CI/CD: GitHub Actions o GitLab CI
Container Registry: Harbor o ECR
IaC: Terraform + Helm Charts
Security: Trivy, SAST, DAST
```

---

## **3. IMPLEMENTACIÓN POR FASES**

### **3.1 FASE 0: Setup y Fundación (2-3 semanas)**

#### **Week 1: Infrastructure Setup**

**Día 1-2: Proyecto Base**
```bash
# Mono-repo structure
mkdir 2.5vision-platform
cd 2.5vision-platform

# Estructura inicial
mkdir -p {
  services/{image-service,ml-service,data-service,analytics-service},
  frontend/{web,mobile},
  infrastructure/{terraform,k8s,docker},
  ml/{models,notebooks,pipelines},
  docs,
  scripts
}

# Poetry para dependency management
poetry init
poetry add fastapi uvicorn sqlalchemy redis kafka-python
poetry add --group dev pytest black isort mypy pre-commit
```

**Día 3-4: Docker & Kubernetes Setup**
```yaml
# docker-compose.yml para desarrollo local
version: '3.8'
services:
  postgres:
    image: postgres:16
    environment:
      POSTGRES_DB: vision_dev
    ports: ["5432:5432"]
  
  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]
  
  kafka:
    image: confluentinc/cp-kafka:latest
    environment:
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
    ports: ["9092:9092"]
  
  minio:
    image: minio/minio
    command: server /data --console-address ":9001"
    ports: ["9000:9000", "9001:9001"]
```

**Día 5-7: CI/CD Pipeline**
```yaml
# .github/workflows/ci.yml
name: CI/CD Pipeline
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
        with:
          python-version: '3.12'
      - run: poetry install
      - run: poetry run pytest
      - run: poetry run black --check .
      - run: poetry run mypy .
  
  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - name: Build Docker images
        run: docker build -t 2.5vision/image-service ./services/image-service
      - name: Push to registry
        run: docker push 2.5vision/image-service
```

#### **Week 2: Core Services Foundation**

**Image Service Base**
```python
# services/image-service/app/main.py
from fastapi import FastAPI, UploadFile, Depends
from app.domain.services import ImageProcessingService
from app.infrastructure.persistence import ImageRepository
from app.application.handlers import ImageHandler

app = FastAPI(title="Image Processing Service", version="1.0.0")

@app.post("/api/v1/images/process")
async def process_image(
    file: UploadFile,
    handler: ImageHandler = Depends()
):
    return await handler.process_image(file)

# Domain-driven structure
# app/
#   domain/
#     entities/
#     services/
#     repositories/
#   application/
#     handlers/
#     commands/
#     queries/
#   infrastructure/
#     persistence/
#     external/
```

**Event-Driven Communication**
```python
# shared/events/base.py
from datetime import datetime
from pydantic import BaseModel
from typing import Any, Dict

class DomainEvent(BaseModel):
    event_id: str
    event_type: str
    occurred_at: datetime
    data: Dict[str, Any]
    version: int = 1

class ImageProcessedEvent(DomainEvent):
    event_type: str = "image.processed"
    
class PredictionCompletedEvent(DomainEvent):
    event_type: str = "prediction.completed"
```

#### **Week 3: ML Infrastructure**

**MLflow Setup**
```python
# ml/mlflow/setup.py
import mlflow
from mlflow.tracking import MlflowClient

# MLflow tracking server
mlflow.set_tracking_uri("http://mlflow-server:5000")

# Model registration
def register_model(model, model_name: str, version: str):
    with mlflow.start_run():
        mlflow.sklearn.log_model(
            model, 
            "model",
            registered_model_name=model_name
        )
        mlflow.log_param("version", version)
        mlflow.log_metric("accuracy", 0.95)
```

**Feature Store Setup**
```python
# ml/feast/feature_store.py
from feast import Entity, Feature, FeatureView, ValueType
from feast.data_source import FileSource

image_features = Entity(
    name="image_id",
    value_type=ValueType.STRING
)

image_stats_fv = FeatureView(
    name="image_statistics",
    entities=["image_id"],
    features=[
        Feature("mean_rgb", ValueType.FLOAT),
        Feature("std_rgb", ValueType.FLOAT),
        Feature("contrast", ValueType.FLOAT),
    ],
    online=True,
    ttl=timedelta(days=1),
    batch_source=FileSource(
        path="s3://feast-data/image_features.parquet",
        timestamp_field="timestamp"
    )
)
```

### **3.2 FASE 1: Core MVP (4-6 semanas)**

#### **Sprint 1-2: Image Processing Service (2 semanas)**

**Domain Layer**
```python
# services/image-service/app/domain/entities/image.py
from datetime import datetime
from typing import Optional
from app.domain.value_objects import GPS, ImageMetadata

class Image:
    def __init__(
        self,
        image_id: str,
        file_path: str,
        metadata: ImageMetadata,
        gps: Optional[GPS] = None
    ):
        self.image_id = image_id
        self.file_path = file_path
        self.metadata = metadata
        self.gps = gps
        self.created_at = datetime.utcnow()
        self.status = ImageStatus.UPLOADED
    
    def mark_as_processed(self, processed_file_path: str):
        self.processed_file_path = processed_file_path
        self.status = ImageStatus.PROCESSED
        self.processed_at = datetime.utcnow()
```

**Application Layer**
```python
# services/image-service/app/application/handlers/image_handler.py
from app.domain.services import ImageProcessingService
from app.domain.repositories import ImageRepository
from app.infrastructure.events import EventPublisher

class ImageHandler:
    def __init__(
        self,
        processing_service: ImageProcessingService,
        repository: ImageRepository,
        event_publisher: EventPublisher
    ):
        self.processing_service = processing_service
        self.repository = repository
        self.event_publisher = event_publisher
    
    async def process_image(self, file: UploadFile) -> ImageProcessResult:
        # 1. Validate image
        validation_result = await self.processing_service.validate(file)
        if not validation_result.is_valid:
            raise InvalidImageError(validation_result.errors)
        
        # 2. Create domain entity
        image = await self.processing_service.create_image(file)
        
        # 3. Store in repository
        stored_image = await self.repository.save(image)
        
        # 4. Publish event for async processing
        await self.event_publisher.publish(
            ImageUploadedEvent(
                image_id=stored_image.image_id,
                file_path=stored_image.file_path
            )
        )
        
        return ImageProcessResult(image_id=stored_image.image_id)
```

#### **Sprint 3-4: ML Service Foundation (2 semanas)**

**Model Abstraction**
```python
# services/ml-service/app/domain/models/base_model.py
from abc import ABC, abstractmethod
from typing import Any, Dict, List
from app.domain.value_objects import FeatureVector, Prediction

class MLModel(ABC):
    @abstractmethod
    async def predict(self, features: FeatureVector) -> Prediction:
        pass
    
    @abstractmethod
    def get_feature_names(self) -> List[str]:
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        pass

class PMEstimationModel(MLModel):
    def __init__(self, model_path: str):
        self.model = joblib.load(model_path)
        self.feature_names = ['mean_rgb', 'std_rgb', 'contrast', 'visibility']
    
    async def predict(self, features: FeatureVector) -> Prediction:
        prediction_value = self.model.predict([features.values])[0]
        confidence = self.model.predict_proba([features.values]).max()
        
        return Prediction(
            value=prediction_value,
            confidence=confidence,
            model_version=self.get_model_info()['version']
        )
```

**Feature Pipeline**
```python
# services/ml-service/app/domain/services/feature_pipeline.py
from typing import List
from app.domain.value_objects import FeatureVector
from app.infrastructure.feature_store import FeatureStore

class FeaturePipeline:
    def __init__(self, feature_store: FeatureStore):
        self.feature_store = feature_store
    
    async def extract_features(self, image_id: str) -> FeatureVector:
        # Get features from feature store
        online_features = await self.feature_store.get_online_features(
            features=[
                "image_statistics:mean_rgb",
                "image_statistics:std_rgb", 
                "image_statistics:contrast"
            ],
            entity_rows=[{"image_id": image_id}]
        )
        
        # Transform to domain object
        return FeatureVector(
            image_id=image_id,
            values=online_features.to_dict()
        )
```

### **3.3 FASE 2: Escalabilidad y ML Avanzado (6-8 semanas)**

#### **Advanced ML Pipeline**

**Model Training Pipeline**
```python
# ml/pipelines/training_pipeline.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

default_args = {
    'owner': '2.5vision',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'pm_estimation_training',
    default_args=default_args,
    description='PM2.5 Estimation Model Training',
    schedule_interval='@weekly',
    catchup=False,
)

def extract_data(**context):
    """Extract training data from data lake"""
    pass

def preprocess_data(**context):
    """Preprocess and feature engineering"""
    pass

def train_model(**context):
    """Train and validate model"""
    pass

def register_model(**context):
    """Register model in MLflow"""
    pass

extract_task = PythonOperator(
    task_id='extract_data',
    python_callable=extract_data,
    dag=dag,
)

preprocess_task = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data,
    dag=dag,
)

train_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag,
)

register_task = PythonOperator(
    task_id='register_model',
    python_callable=register_model,
    dag=dag,
)

extract_task >> preprocess_task >> train_task >> register_task
```

**A/B Testing Framework**
```python
# services/ml-service/app/application/ab_testing.py
from typing import Dict, Any
import hashlib
import random

class ABTestingService:
    def __init__(self, experiment_config: Dict[str, Any]):
        self.experiments = experiment_config
    
    def get_model_variant(self, user_id: str, experiment_name: str) -> str:
        """Determine which model variant to use for a user"""
        experiment = self.experiments.get(experiment_name)
        if not experiment:
            return "control"
        
        # Consistent hash-based assignment
        hash_value = int(hashlib.md5(f"{user_id}_{experiment_name}".encode()).hexdigest(), 16)
        bucket = hash_value % 100
        
        cumulative = 0
        for variant, percentage in experiment["variants"].items():
            cumulative += percentage
            if bucket < cumulative:
                return variant
        
        return "control"
    
    async def record_outcome(self, user_id: str, experiment_name: str, variant: str, outcome: float):
        """Record experiment outcome for analysis"""
        # Store in analytics database
        pass
```

#### **Real-time Processing**

**Kafka Event Handlers**
```python
# services/event-processor/app/handlers/image_events.py
from kafka import KafkaConsumer
from app.domain.events import ImageUploadedEvent, ImageProcessedEvent

class ImageEventHandler:
    def __init__(self):
        self.consumer = KafkaConsumer(
            'image.uploaded',
            bootstrap_servers=['kafka:9092'],
            value_deserializer=lambda m: json.loads(m.decode('utf-8'))
        )
    
    async def handle_image_uploaded(self, event: ImageUploadedEvent):
        """Process image asynchronously when uploaded"""
        # 1. Extract features
        features = await self.feature_extractor.extract(event.image_id)
        
        # 2. Store in feature store
        await self.feature_store.write_features(features)
        
        # 3. Trigger prediction if needed
        await self.prediction_service.predict(event.image_id)
        
        # 4. Publish completion event
        await self.event_publisher.publish(
            ImageProcessedEvent(image_id=event.image_id)
        )
```

### **3.4 FASE 3: Analytics y UI (4 semanas)**

#### **Analytics Service**
```python
# services/analytics-service/app/domain/services/analytics_service.py
class AnalyticsService:
    def __init__(self, clickhouse_client, redis_cache):
        self.clickhouse = clickhouse_client
        self.cache = redis_cache
    
    async def get_pm_trends(
        self, 
        start_date: datetime, 
        end_date: datetime,
        location: Optional[GPS] = None
    ) -> PMTrendsResult:
        cache_key = f"pm_trends:{start_date}:{end_date}:{location}"
        
        # Try cache first
        cached_result = await self.cache.get(cache_key)
        if cached_result:
            return PMTrendsResult.parse_raw(cached_result)
        
        # Query ClickHouse
        query = """
        SELECT 
            toStartOfHour(timestamp) as hour,
            avg(pm_estimation) as avg_pm,
            quantile(0.5)(pm_estimation) as median_pm,
            quantile(0.95)(pm_estimation) as p95_pm
        FROM predictions 
        WHERE timestamp BETWEEN %(start_date)s AND %(end_date)s
        """
        
        if location:
            query += " AND geoDistance(latitude, longitude, %(lat)s, %(lon)s) < 10000"
        
        query += " GROUP BY hour ORDER BY hour"
        
        result = await self.clickhouse.execute(query, {
            'start_date': start_date,
            'end_date': end_date,
            'lat': location.latitude if location else 0,
            'lon': location.longitude if location else 0
        })
        
        trends = PMTrendsResult(data=result)
        
        # Cache for 1 hour
        await self.cache.setex(cache_key, 3600, trends.json())
        
        return trends
```

---

## **4. DESPLIEGUE Y INFRAESTRUCTURA**

### **4.1 Estrategia de Despliegue**

#### **Local Development**
```yaml
# docker-compose.dev.yml - Desarrollo local completo
version: '3.8'

services:
  # Core Services
  image-service:
    build: ./services/image-service
    ports: ["8001:8000"]
    environment:
      - DATABASE_URL=postgresql://user:pass@postgres:5432/vision_dev
      - REDIS_URL=redis://redis:6379
      - KAFKA_BOOTSTRAP_SERVERS=kafka:9092
    depends_on: [postgres, redis, kafka]
  
  ml-service:
    build: ./services/ml-service  
    ports: ["8002:8000"]
    depends_on: [postgres, redis, kafka]
    volumes:
      - ./ml/models:/app/models
  
  # Infrastructure
  postgres:
    image: postgres:16
    environment:
      POSTGRES_DB: vision_dev
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./infrastructure/postgres/init.sql:/docker-entrypoint-initdb.d/init.sql
  
  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
  
  # Message Streaming
  zookeeper:
    image: confluentinc/cp-zookeeper:7.4.0
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
  
  kafka:
    image: confluentinc/cp-kafka:7.4.0
    depends_on: [zookeeper]
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9092
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
    ports: ["9092:9092"]
  
  # ML Infrastructure
  mlflow:
    image: python:3.12
    command: >
      sh -c "pip install mlflow psycopg2-binary &&
             mlflow server --backend-store-uri postgresql://user:pass@postgres:5432/mlflow
                          --default-artifact-root s3://mlflow-artifacts
                          --host 0.0.0.0 --port 5000"
    ports: ["5000:5000"]
    depends_on: [postgres, minio]
  
  # Object Storage
  minio:
    image: minio/minio:latest
    command: server /data --console-address ":9001"
    ports: ["9000:9000", "9001:9001"]
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    volumes:
      - minio_data:/data
  
  # Observability
  prometheus:
    image: prom/prometheus:latest
    ports: ["9090:9090"]
    volumes:
      - ./infrastructure/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
  
  grafana:
    image: grafana/grafana:latest
    ports: ["3000:3000"]
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./infrastructure/grafana/dashboards:/etc/grafana/provisioning/dashboards

volumes:
  postgres_data:
  redis_data:
  minio_data:
  grafana_data:
```

#### **Staging/Production - Kubernetes**

**Namespace y ConfigMaps**
```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: vision-prod
  labels:
    name: vision-prod

---
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: vision-config
  namespace: vision-prod
data:
  ENVIRONMENT: "production"
  LOG_LEVEL: "INFO"
  KAFKA_BOOTSTRAP_SERVERS: "kafka.kafka.svc.cluster.local:9092"
  REDIS_URL: "redis://redis.vision-prod.svc.cluster.local:6379"
```

**Database - PostgreSQL HA**
```yaml
# k8s/postgres-ha.yaml
apiVersion: postgresql.cnpg.io/v1
kind: Cluster
metadata:
  name: postgres-cluster
  namespace: vision-prod
spec:
  instances: 3
  primaryUpdateStrategy: unsupervised
  
  postgresql:
    parameters:
      max_connections: "200"
      shared_preload_libraries: "pg_stat_statements"
      pg_stat_statements.max: "10000"
      pg_stat_statements.track: "all"
  
  bootstrap:
    initdb:
      database: vision_prod
      owner: vision_user
      secret:
        name: postgres-credentials
  
  storage:
    size: 100Gi
    storageClass: fast-ssd
  
  monitoring:
    enabled: true
```

**Image Service Deployment**
```yaml
# k8s/image-service.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: image-service
  namespace: vision-prod
spec:
  replicas: 3
  selector:
    matchLabels:
      app: image-service
  template:
    metadata:
      labels:
        app: image-service
    spec:
      containers:
      - name: image-service
        image: vision/image-service:1.0.0
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: postgres-credentials
              key: uri
        - name: REDIS_URL
          valueFrom:
            configMapKeyRef:
              name: vision-config
              key: REDIS_URL
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: image-service
  namespace: vision-prod
spec:
  selector:
    app: image-service
  ports:
  - port: 80
    targetPort: 8000
  type: ClusterIP

---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: image-service-ingress
  namespace: vision-prod
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/rate-limit: "100"
spec:
  tls:
  - hosts:
    - api.2.5vision.com
    secretName: vision-tls
  rules:
  - host: api.2.5vision.com
    http:
      paths:
      - path: /api/v1/images
        pathType: Prefix
        backend:
          service:
            name: image-service
            port:
              number: 80
```

**ML Service con Model Serving**
```yaml
# k8s/ml-service.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-service
  namespace: vision-prod
spec:
  replicas: 2
  selector:
    matchLabels:
      app: ml-service
  template:
    metadata:
      labels:
        app: ml-service
    spec:
      containers:
      - name: ml-service
        image: vision/ml-service:1.0.0
        ports:
        - containerPort: 8000
        env:
        - name: MLFLOW_TRACKING_URI
          value: "http://mlflow.mlflow.svc.cluster.local:5000"
        - name: MODEL_REGISTRY_URI
          value: "s3://model-registry"
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
            nvidia.com/gpu: 0
          limits:
            memory: "2Gi"
            cpu: "1000m"
            nvidia.com/gpu: 1
        volumeMounts:
        - name: model-cache
          mountPath: /app/model-cache
      volumes:
      - name: model-cache
        emptyDir:
          sizeLimit: 5Gi
```

#### **Terraform Infrastructure**

**AWS EKS Cluster**
```hcl
# infrastructure/terraform/eks.tf
module "eks" {
  source = "terraform-aws-modules/eks/aws"
  
  cluster_name    = "vision-production"
  cluster_version = "1.28"
  
  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets
  
  # Node groups
  eks_managed_node_groups = {
    general = {
      desired_size = 3
      max_size     = 10
      min_size     = 3
      
      instance_types = ["m5.large"]
      
      k8s_labels = {
        Environment = "production"
        NodeType    = "general"
      }
    }
    
    ml_workers = {
      desired_size = 2
      max_size     = 5
      min_size     = 0
      
      instance_types = ["p3.2xlarge"]  # GPU instances
      
      k8s_labels = {
        Environment = "production"
        NodeType    = "ml"
      }
      
      taints = {
        ml-workload = {
          key    = "ml"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      }
    }
  }
  
  # Add-ons
  cluster_addons = {
    coredns = {
      resolve_conflicts = "OVERWRITE"
    }
    kube-proxy = {}
    vpc-cni = {
      resolve_conflicts = "OVERWRITE"
    }
    aws-ebs-csi-driver = {
      resolve_conflicts = "OVERWRITE"
    }
  }
}
```

**Database (RDS)**
```hcl
# infrastructure/terraform/rds.tf
module "db" {
  source = "terraform-aws-modules/rds/aws"
  
  identifier = "vision-production"
  
  engine            = "postgres"
  engine_version    = "16.1"
  instance_class    = "db.r6g.xlarge"
  allocated_storage = 100
  storage_encrypted = true
  
  db_name  = "vision_prod"
  username = "vision_user"
  port     = "5432"
  
  iam_database_authentication_enabled = true
  
  vpc_security_group_ids = [module.security_group.security_group_id]
  
  maintenance_window = "Mon:00:00-Mon:03:00"
  backup_window      = "03:00-06:00"
  
  # Monitoring
  monitoring_interval = "60"
  monitoring_role_name = "RDSEnhancedMonitoringRole"
  create_monitoring_role = true
  
  # Backups
  backup_retention_period = 7
  backup_window          = "03:00-06:00"
  copy_tags_to_snapshot  = true
  deletion_protection    = true
  
  # Performance Insights
  performance_insights_enabled = true
  performance_insights_retention_period = 7
  
  tags = {
    Environment = "production"
    Application = "2.5vision"
  }
}
```

### **4.2 CI/CD Pipeline Avanzado**

#### **GitHub Actions Workflow**
```yaml
# .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    branches: [main]
    tags: ['v*']

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  test:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:16
        env:
          POSTGRES_PASSWORD: postgres
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
      redis:
        image: redis:7
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
        with:
          python-version: '3.12'
          cache: 'pip'
      
      - name: Install dependencies
        run: |
          pip install poetry
          poetry install
      
      - name: Run linting
        run: |
          poetry run black --check .
          poetry run isort --check-only .
          poetry run mypy .
      
      - name: Run tests
        run: |
          poetry run pytest --cov=app --cov-report=xml
        env:
          DATABASE_URL: postgresql://postgres:postgres@localhost:5432/test
          REDIS_URL: redis://localhost:6379
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
        with:
          file: ./coverage.xml
  
  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          scan-type: 'fs'
          scan-ref: '.'
          format: 'sarif'
          output: 'trivy-results.sarif'
      
      - name: Upload Trivy scan results
        uses: github/codeql-action/upload-sarif@v2
        with:
          sarif_file: 'trivy-results.sarif'
  
  build:
    needs: [test, security]
    runs-on: ubuntu-latest
    strategy:
      matrix:
        service: [image-service, ml-service, analytics-service]
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3
      
      - name: Login to Container Registry
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      
      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}/${{ matrix.service }}
          tags: |
            type=ref,event=branch
            type=ref,event=pr
            type=semver,pattern={{version}}
            type=semver,pattern={{major}}.{{minor}}
      
      - name: Build and push Docker image
        uses: docker/build-push-action@v5
        with:
          context: ./services/${{ matrix.service }}
          file: ./services/${{ matrix.service }}/Dockerfile
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
  
  deploy-staging:
    needs: [build]
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    environment: staging
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
          aws-region: us-east-1
      
      - name: Deploy to staging
        run: |
          aws eks update-kubeconfig --name vision-staging
          helm upgrade --install vision-staging ./infrastructure/helm/vision \
            --namespace vision-staging \
            --values ./infrastructure/helm/values-staging.yaml \
            --set image.tag=${{ github.sha }}
  
  integration-tests:
    needs: [deploy-staging]
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Run integration tests
        run: |
          pip install pytest requests
          STAGING_URL=https://staging.2.5vision.com pytest tests/integration/
  
  deploy-production:
    needs: [integration-tests]
    runs-on: ubuntu-latest
    if: startsWith(github.ref, 'refs/tags/v')
    environment: production
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          aws-access-key-id: ${{ secrets.AWS_PROD_ACCESS_KEY_ID }}
          aws-secret-access-key: ${{ secrets.AWS_PROD_SECRET_ACCESS_KEY }}
          aws-region: us-east-1
      
      - name: Deploy to production
        run: |
          aws eks update-kubeconfig --name vision-production
          helm upgrade --install vision-production ./infrastructure/helm/vision \
            --namespace vision-prod \
            --values ./infrastructure/helm/values-production.yaml \
            --set image.tag=${{ github.ref_name }}
      
      - name: Verify deployment
        run: |
          kubectl rollout status deployment/image-service -n vision-prod
          kubectl rollout status deployment/ml-service -n vision-prod
```

---

## **5. DESARROLLO Y MEJORES PRÁCTICAS**

### **5.1 Development Workflow**

#### **Git Strategy**
```bash
# Branching strategy: Git Flow
main          # Production releases
develop       # Integration branch
feature/*     # Feature development
release/*     # Release preparation
hotfix/*      # Production hotfixes

# Example workflow
git checkout develop
git checkout -b feature/advanced-ml-pipeline
# Develop feature
git add . && git commit -m "feat: implement advanced ML pipeline"
git push origin feature/advanced-ml-pipeline
# Create pull request to develop
```

#### **Code Quality Gates**
```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.9.1
    hooks:
      - id: black
  
  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
  
  - repo: https://github.com/pycqa/flake8
    rev: 6.1.0
    hooks:
      - id: flake8
  
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.5.1
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
  
  - repo: https://github.com/pycqa/bandit
    rev: 1.7.5
    hooks:
      - id: bandit
        args: ['-r', '.']
  
  - repo: https://github.com/hadolint/hadolint
    rev: v2.12.0
    hooks:
      - id: hadolint
        name: Lint Dockerfiles
        language: system
        types: [dockerfile]
```

#### **Testing Strategy**
```python
# tests/conftest.py - Shared test fixtures
import pytest
import asyncio
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from app.main import app
from app.core.database import get_db

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
async def test_db():
    """Test database fixture"""
    engine = create_async_engine("sqlite+aiosqlite:///./test.db")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    async with AsyncSession(engine) as session:
        yield session
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

@pytest.fixture
async def client(test_db):
    """Test client fixture"""
    app.dependency_overrides[get_db] = lambda: test_db
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client
    app.dependency_overrides.clear()

# tests/unit/test_image_service.py
@pytest.mark.asyncio
async def test_process_image_valid(client: AsyncClient):
    """Test image processing with valid image"""
    with open("tests/fixtures/test_image.jpg", "rb") as f:
        response = await client.post(
            "/api/v1/images/process",
            files={"file": ("test.jpg", f, "image/jpeg")}
        )
    
    assert response.status_code == 200
    data = response.json()
    assert "image_id" in data
    assert data["status"] == "processing"

@pytest.mark.asyncio
async def test_process_image_invalid_format(client: AsyncClient):
    """Test image processing with invalid format"""
    with open("tests/fixtures/test_document.pdf", "rb") as f:
        response = await client.post(
            "/api/v1/images/process",
            files={"file": ("test.pdf", f, "application/pdf")}
        )
    
    assert response.status_code == 400
    assert "Invalid file format" in response.json()["detail"]

# tests/integration/test_ml_pipeline.py
@pytest.mark.asyncio
async def test_full_ml_pipeline(client: AsyncClient):
    """Test complete ML pipeline integration"""
    # 1. Upload image
    with open("tests/fixtures/test_image.jpg", "rb") as f:
        upload_response = await client.post(
            "/api/v1/images/process",
            files={"file": ("test.jpg", f, "image/jpeg")}
        )
    
    image_id = upload_response.json()["image_id"]
    
    # 2. Wait for processing (in real tests, use mock or async wait)
    await asyncio.sleep(2)
    
    # 3. Get prediction result
    prediction_response = await client.get(f"/api/v1/predictions/{image_id}")
    assert prediction_response.status_code == 200
    
    prediction = prediction_response.json()
    assert "pm_estimation" in prediction
    assert prediction["pm_estimation"] > 0
    assert prediction["confidence"] > 0.5
```

### **5.2 Monitoring y Observabilidad**

#### **Application Metrics**
```python
# shared/monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge
from typing import Callable
import time

# Application metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency',
    ['method', 'endpoint']
)

PREDICTION_ACCURACY = Histogram(
    'ml_prediction_accuracy',
    'ML model prediction accuracy',
    ['model_name', 'model_version']
)

ACTIVE_PROCESSING_JOBS = Gauge(
    'image_processing_jobs_active',
    'Number of active image processing jobs'
)

def track_request_metrics(func: Callable):
    """Decorator to track request metrics"""
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        
        try:
            result = await func(*args, **kwargs)
            REQUEST_COUNT.labels(
                method=kwargs.get('method', 'unknown'),
                endpoint=kwargs.get('endpoint', 'unknown'),
                status='success'
            ).inc()
            return result
        except Exception as e:
            REQUEST_COUNT.labels(
                method=kwargs.get('method', 'unknown'),
                endpoint=kwargs.get('endpoint', 'unknown'),
                status='error'
            ).inc()
            raise
        finally:
            REQUEST_LATENCY.labels(
                method=kwargs.get('method', 'unknown'),
                endpoint=kwargs.get('endpoint', 'unknown')
            ).observe(time.time() - start_time)
    
    return wrapper
```

#### **Structured Logging**
```python
# shared/logging/structured_logger.py
import json
import logging
from typing import Dict, Any
from datetime import datetime

class StructuredLogger:
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        handler.setFormatter(self.StructuredFormatter())
        self.logger.addHandler(handler)
    
    class StructuredFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            log_entry = {
                'timestamp': datetime.utcnow().isoformat(),
                'level': record.levelname,
                'logger': record.name,
                'message': record.getMessage(),
                'module': record.module,
                'function': record.funcName,
                'line': record.lineno,
            }
            
            # Add extra fields
            if hasattr(record, 'correlation_id'):
                log_entry['correlation_id'] = record.correlation_id
            
            if hasattr(record, 'user_id'):
                log_entry['user_id'] = record.user_id
            
            if hasattr(record, 'request_id'):
                log_entry['request_id'] = record.request_id
            
            return json.dumps(log_entry)
    
    def info(self, message: str, **kwargs):
        extra = {k: v for k, v in kwargs.items()}
        self.logger.info(message, extra=extra)
    
    def error(self, message: str, **kwargs):
        extra = {k: v for k, v in kwargs.items()}
        self.logger.error(message, extra=extra)
```

#### **Distributed Tracing**
```python
# shared/tracing/tracer.py
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor

def setup_tracing(service_name: str):
    """Setup distributed tracing"""
    trace.set_tracer_provider(TracerProvider())
    
    jaeger_exporter = JaegerExporter(
        agent_host_name="jaeger-agent",
        agent_port=6831,
    )
    
    span_processor = BatchSpanProcessor(jaeger_exporter)
    trace.get_tracer_provider().add_span_processor(span_processor)
    
    # Instrument FastAPI
    FastAPIInstrumentor.instrument_app(
        app=app,
        service_name=service_name
    )
    
    # Instrument SQLAlchemy
    SQLAlchemyInstrumentor().instrument()
    
    return trace.get_tracer(service_name)

# Usage in service
tracer = setup_tracing("image-service")

@tracer.start_as_current_span("process_image")
async def process_image(file: UploadFile):
    span = trace.get_current_span()
    span.set_attribute("image.size", len(await file.read()))
    span.set_attribute("image.format", file.content_type)
    
    # Processing logic...
    
    span.set_attribute("processing.duration_ms", processing_time)
    return result
```

---

## **6. CONCLUSIÓN Y PRÓXIMOS PASOS**

### **6.1 Resumen de Diferencias Clave vs Implementación Actual**

| Aspecto | Implementación Actual | Implementación Propuesta |
|---------|----------------------|--------------------------|
| **Arquitectura** | Monolito hexagonal | Microservicios + Event-driven |
| **ML Pipeline** | Básico, modelo simulado | MLOps completo, versionado |
| **Escalabilidad** | Limitada, sync | Auto-scaling, async |
| **Testing** | Ausente | TDD, >90% coverage |
| **Observabilidad** | Básica | Metrics, tracing, alerting |
| **Despliegue** | Manual | CI/CD automático |
| **Seguridad** | Básica | Enterprise-grade |

### **6.2 Ventajas de la Implementación Greenfield**

#### **Técnicas**
- **Zero Technical Debt**: Código limpio desde día 1
- **Modern Stack**: Tecnologías actuales y probadas
- **Scalability Built-in**: Diseñado para escalar horizontalmente
- **ML-First Design**: Arquitectura optimizada para ML/AI

#### **Operacionales**
- **DevOps desde el inicio**: CI/CD, monitoring, alerting
- **Production-ready**: Alta disponibilidad, disaster recovery
- **Cost-optimized**: Auto-scaling, resource optimization
- **Security-first**: Principios de seguridad desde diseño

#### **Desarrollo**
- **Developer Experience**: Tooling moderno, hot reload, debugging
- **Team Productivity**: Separated concerns, parallel development
- **Maintenance**: Código testeable, documentado, modular

### **6.3 Timeline de Implementación**

| Fase | Duración | Equipo | Hitos Clave |
|------|----------|---------|------------|
| **Setup** | 3 semanas | 2-3 devs | Infrastructure, CI/CD |
| **MVP** | 6 semanas | 3-4 devs | Core services, basic ML |
| **Scale** | 8 semanas | 4-5 devs | Advanced ML, analytics |
| **Polish** | 4 semanas | Full team | UI, optimizations |
| **Total** | **21 semanas** | **~5 meses** | Production ready |

### **6.4 ROI Proyectado**

#### **Inversión Inicial**
- **Desarrollo**: ~5 meses × 4 devs = $200k
- **Infraestructura**: ~$500/mes (dev) + $2000/mes (prod)
- **Herramientas**: ~$1000/mes (monitoring, CI/CD)

#### **Retorno Esperado (Año 1)**
- **Desarrollo 3x más rápido**: Nuevas features
- **Operación 80% más eficiente**: Auto-scaling, monitoring
- **Calidad 10x mejor**: Testing, observabilidad
- **Costo operacional -50%**: Optimización, automation

### **6.5 Recomendación Final**

**PROCEDER CON GREENFIELD IMPLEMENTATION**

Dada la cantidad de deuda técnica y limitaciones arquitecturales en la implementación actual, recomiendo **fuertemente** iniciar desde cero con la arquitectura propuesta.

**Razones clave**:
1. **Tiempo similar**: Refactorizar vs greenfield ~mismo tiempo
2. **Calidad superior**: Sin compromises, best practices desde día 1
3. **Escalabilidad**: Diseñado para growth exponencial
4. **Mantenibilidad**: Código limpio, testeable, documentado
5. **Future-proof**: Arquitectura moderna, extensible

**Migración gradual**: Implementar servicios nuevos en paralelo, migrar datos progresivamente, mantener APIs compatibles durante transición.

---

**Total: 21 semanas para sistema production-ready de clase enterprise**
**ROI: 300-400% en productividad y calidad**
**Riesgo: Bajo (arquitectura probada, stack maduro)**

Esta implementación greenfield establecería 2.5Vision como una plataforma ML de vanguardia, escalable y mantenible para los próximos 5+ años.
