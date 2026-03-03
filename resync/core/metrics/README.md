# Resync Metrics Architecture

## Overview

O Resync utiliza um **sistema de métricas interno customizado** que é **compatível com Prometheus** mas não depende da biblioteca `prometheus-client` para funcionalidade básica.

## Arquitetura

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Code                         │
│  (Services, APIs, Agents, TWS Client, RAG, etc.)            │
└───────────────────┬─────────────────────────────────────────┘
                    │
                    │ Import & Use
                    ▼
┌─────────────────────────────────────────────────────────────┐
│              Metrics Collections                            │
│  ┌──────────────────┐  ┌─────────────────────────────┐     │
│  │ runtime_metrics  │  │    business_metrics         │     │
│  │ (infrastructure) │  │    (KPIs)                   │     │
│  └──────────────────┘  └─────────────────────────────┘     │
└───────────────────┬─────────────────────────────────────────┘
                    │
                    │ Uses
                    ▼
┌─────────────────────────────────────────────────────────────┐
│              metrics_internal.py                            │
│  ┌──────────┐  ┌──────────┐  ┌───────────────┐            │
│  │ Counter  │  │  Gauge   │  │   Histogram   │            │
│  └──────────┘  └──────────┘  └───────────────┘            │
│                                                             │
│  • Thread-safe                                              │
│  • Label support                                            │
│  • Compatible API with prometheus-client                    │
└───────────────────┬─────────────────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
┌────────────────┐    ┌────────────────────────┐
│ PostgreSQL DB  │    │  Dashboard Próprio     │
│ (persistent)   │    │  (JSON/HTTP API)       │
└────────────────┘    └────────────────────────┘
                               │
                               │ Opcional
                               ▼
                      ┌────────────────────┐
                      │ Prometheus Scraper │
                      │ (via /metrics)     │
                      └────────────────────┘
```

## Componentes

### 1. **metrics_internal.py** (Core)
Sistema interno de métricas com implementações de:
- `Counter`: Contadores que apenas incrementam
- `Gauge`: Valores que podem subir/descer
- `Histogram`: Distribuições e latências

**Características**:
- ✅ Thread-safe (locks internos)
- ✅ Label support (dimensões)
- ✅ API compatível com `prometheus-client`
- ✅ Sem dependências externas

### 2. **runtime_metrics.py** (Infraestrutura)
Métricas de infraestrutura já existentes:
```python
from resync.core.metrics import runtime_metrics

runtime_metrics.api_requests_total.inc()
runtime_metrics.cache_hits.inc()
runtime_metrics.tws_requests_failed.inc()
```

### 3. **business_metrics.py** (Novo v6.3.0)
Métricas de negócio (KPIs):
```python
from resync.core.metrics import business_metrics

# TWS Job Monitoring
business_metrics.tws_job_failures_total.labels(
    workstation="PROD01",
    jobstream="DAILY",
    job_name="BACKUP",
    error_code="ABEND0013"
).inc()

# LLM Usage
business_metrics.llm_tokens_consumed.observe(
    1245,
    labels={"provider": "ollama", "model": "qwen2.5:3b", "type": "total"}
)

# Agent Execution
business_metrics.agent_executions_total.labels(
    agent_name="diagnostic",
    status="success"
).inc()
```

### 4. **lightweight_store.py** (Persistência)
Armazena métricas no PostgreSQL para análises históricas.

### 5. **prometheus_exporter.py** (Opcional)
Endpoint `/metrics` compatível com Prometheus para scraping externo.

## Dashboards

### Dashboard Próprio (Padrão)
- **Tecnologia**: FastAPI endpoints + PostgreSQL
- **Formato**: JSON via HTTP API
- **Localização**: `/api/monitoring/dashboard`
- **Vantagens**:
  - ✅ Customizável
  - ✅ Sem dependências externas
  - ✅ Integrado com autenticação

### Prometheus (Opcional)
- **Endpoint**: `/metrics` (formato Prometheus)
- **Uso**: Scraping externo (Grafana, etc.)
- **Dependência**: `prometheus-client` (opcional)

## Diferenças: Prometheus vs. Sistema Interno

| Aspecto | prometheus-client | metrics_internal.py |
|---------|-------------------|---------------------|
| **Dependências** | Biblioteca externa | Nativo (built-in) |
| **Thread-safe** | ✅ Sim | ✅ Sim |
| **Labels** | ✅ Sim | ✅ Sim |
| **Histograms** | Buckets configuráveis | Buckets automáticos |
| **Persistência** | Apenas em memória* | PostgreSQL |
| **Dashboard** | Requer Prometheus+Grafana | Dashboard nativo |
| **Exportação** | Formato Prometheus | JSON + Prometheus |

\* Exceto se usar `PROMETHEUS_MULTIPROC_DIR`

## Exemplo Completo

```python
# 1. Import
from resync.core.metrics import business_metrics
import time

# 2. Track TWS job execution
start = time.time()
try:
    # ... execute job ...
    duration = time.time() - start

    # Success
    business_metrics.tws_job_duration_seconds.observe(
        duration,
        labels={
            "workstation": "PROD01",
            "jobstream": "DAILY",
            "job_name": "BACKUP",
            "status": "success"
        }
    )

except Exception as e:
    # Failure
    business_metrics.tws_job_failures_total.labels(
        workstation="PROD01",
        jobstream="DAILY",
        job_name="BACKUP",
        error_code=str(e)[:20]
    ).inc()
```

## Adicionando Novas Métricas

### 1. Em business_metrics.py:
```python
my_new_metric = create_counter(
    "resync_my_metric_total",
    "Description of my metric",
    labels=["dimension1", "dimension2"],
)
```

### 2. Uso no código:
```python
from resync.core.metrics import business_metrics

business_metrics.my_new_metric.labels(
    dimension1="value1",
    dimension2="value2"
).inc()
```

### 3. Consulta no dashboard:
```bash
curl http://localhost:8000/api/monitoring/dashboard
```

## Migration Guide (prometheus-client → metrics_internal)

Se você tinha código usando `prometheus-client` diretamente:

### Antes:
```python
from prometheus_client import Counter, Gauge, Histogram

my_counter = Counter("my_metric", "desc", labelnames=["label1"])
my_counter.labels(label1="value").inc()
```

### Depois:
```python
from resync.core.metrics_internal import create_counter

my_counter = create_counter("my_metric", "desc", labels=["label1"])
my_counter.labels(label1="value").inc()
```

**Compatibilidade**: A API é 99% compatível! 🎉

## FAQ

### Q: Por que não usar prometheus-client diretamente?
**A**: Para reduzir dependências externas e ter controle total sobre armazenamento (PostgreSQL) e visualização (dashboard customizado).

### Q: Posso ainda exportar para Prometheus?
**A**: Sim! O endpoint `/metrics` continua funcionando se `prometheus-client` estiver instalado.

### Q: As métricas persistem após restart?
**A**: Sim! Via `lightweight_store.py` (PostgreSQL). Métricas em memória são perdidas, mas os valores agregados ficam no banco.

### Q: Qual a performance?
**A**: ~10μs por `inc()` com labels. Thread-safe via locks otimizados.

## Support

- **Documentação**: Este README
- **Código**: `resync/core/metrics/`
- **Dashboards**: `/api/monitoring/dashboard`
- **Prometheus**: `/metrics` (opcional)
