# Plano de Refatoração: Service Discovery Patterns Reutilizáveis

## Visão Geral

Este plano detalha a extração e reutilização dos padrões implementados em `service_discovery.py` para outros componentes do projeto Resync.

**Origem:** `resync/core/service_discovery.py` (51.695 bytes, v7.2-prod)

---

## 1. Padrões Identificados no service_discovery.py

### 1.1 Backend Interface Pattern (ABC)
- **Localização:** Linhas 344-356
- **Descrição:** Interface abstrata para múltiplos backends de discovery
- **Métodos abstratos:**
  - `register_service()`
  - `deregister_service()`
  - `discover_services()`
  - `watch_service()`
  - `health_check()`
  - `close()`

### 1.2 Load Balancing Strategies
- **Localização:** Linhas 144-150, 910-927
- **Estratégias disponíveis:**
  - `ROUND_ROBIN` - Rotação cíclica
  - `RANDOM` - Seleção aleatória
  - `LEAST_CONNECTIONS` - Menos conexões ativas
  - `WEIGHTED_RANDOM` - Aleatório ponderado
  - `LATENCY_BASED` - Baseado em latência

### 1.3 Circuit Breaker por Instância
- **Localização:** Linhas 320-323, 617-618, 907-908, 1024-1028
- **Características:**
  - Tracking de `_circuit_open_until` (monotonic time)
  - Threshold configurável de falhas consecutivas
  - Recovery timeout
  - Métricas por instância

### 1.4 Health Checks com Métricas
- **Localização:** Linhas 987-1050
- **Características:**
  - Falhas consecutivas
  - Response time average (média móvel exponencial)
  - Intervalo configurável
  - TTL de instância

### 1.5 Prometheus Metrics
- **Localização:** Linhas 59-127
- **Métricas de baixa cardinalidade:**
  - `_prom_registrations_total`
  - `_prom_discoveries_total`
  - `_prom_health_checks_total`
  - `_prom_lb_decisions_total`
  - `_prom_active_services` (Gauge)
  - `_prom_service_instances` (Gauge)
  - `_prom_service_healthy_instances` (Gauge)

### 1.6 OpenTelemetry Tracing
- **Localização:** Throughout
- **Spans em operações request-like:**
  - `sdm.backend.consul.register`
  - `sdm.get_service_instances`
  - `sdm.health_check`

### 1.7 HTTP Client Pooling
- **Localização:** Linhas 213-219, 371-382
- **Características:**
  - `httpx.AsyncClient` reutilizável
  - `httpx.Limits` com max_connections, keepalive
  - Timeout configurável

### 1.8 Workers Background
- **Localização:** Linhas 1063-1159
- **Workers:**
  - `_discovery_worker` - Refresh de cache
  - `_health_worker` - Health checks
  - `_backend_health_worker` - Health do backend
  - `_metrics_worker` - Logging de métricas

### 1.9 Context Manager Borrow
- **Localização:** Linhas 963-981
- **Descrição:** Incremento/decremento automático de conexões

### 1.10 Semáforos para Concurrency
- **Localização:** Linhas 621-622
- **Descrição:** `_disc_sem`, `_hc_sem` para controle de concorrência

---

## 2. Arquivos Novos a Criar

### 2.1 `resync/core/backends/__init__.py`
```
exports: BackendInterface, BackendHealth
```

### 2.2 `resync/core/backends/base.py`
```python
class BackendInterface(ABC):
    @abstractmethod
    async def register(...) -> bool: ...
    @abstractmethod
    async def deregister(...) -> bool: ...
    @abstractmethod
    async def discover(...) -> list[ServiceInstance]: ...
    @abstractmethod
    async def health_check() -> bool: ...
    @abstractmethod
    async def close() -> None: ...
```

### 2.3 `resync/core/load_balancing.py`
```python
class LoadBalancingStrategy(str, Enum):
    ROUND_ROBIN = "round_robin"
    RANDOM = "random"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_RANDOM = "weighted_random"
    LATENCY_BASED = "latency_based"

class LoadBalancer(Generic[T]):
    def select(candidates: list[T], strategy: Strategy) -> T | None: ...
    def with_connections(candidates: list[T], conn_counts: dict[str, int], strategy: Strategy) -> T | None: ...
```

### 2.4 `resync/core/factories/http_client.py`
```python
class HTTPClientFactory:
    @staticmethod
    def create_client(base_url: str, config: HTTPClientConfig) -> httpx.AsyncClient: ...
    @staticmethod
    def create_pooled_limits(config: HTTPClientConfig) -> httpx.Limits: ...
```

### 2.5 `resync/core/workers.py`
```python
class BaseWorker(ABC):
    async def start() -> None: ...
    async def stop() -> None: ...
    async def run_loop() -> None: ...

class WorkerManager:
    def register_worker(name: str, worker: BaseWorker) -> None: ...
    async def start_all() -> None: ...
    async def stop_all() -> None: ...
```

### 2.6 `resync/core/metrics_utils.py`
```python
def get_or_create_metric(factory, name: str, *args, **kwargs): ...

# Reusable metric definitions
prom_discovery_total = ...
prom_health_checks = ...
prom_lb_decisions = ...
```

---

## 3. Arquivos a Atualizar

### 3.1 `resync/core/service_discovery.py`
- **Mudanças:**
  - Importar de `backends.base`
  - Importar de `load_balancing`
  - Importar de `metrics_utils`
  - Importar de `factories.http_client`
  - Remover código duplicado

### 3.2 `resync/core/redis_init.py`
- **Benefícios:**
  - Backend interface para Redis providers
  - Health checks unificados
  - HTTP client pooling

### 3.3 `resync/core/redis_strategy.py`
- **Benefícios:**
  - Load balancing strategies
  - Circuit breaker instance-level

### 3.4 `resync/core/smart_pooling.py`
- **Benefícios:**
  - Load balancing strategies (ROUND_ROBIN, LEAST_CONNECTIONS, etc)
  - Context manager borrow pattern

### 3.5 `resync/core/resilience.py`
- **Benefícios:**
  - Circuit breaker com GC de estado
  - Métricas Prometheus padronizadas

### 3.6 `resync/core/circuit_breaker_registry.py`
- **Benefícios:**
  - Métricas padronizadas
  - OpenTelemetry tracing

---

## 4. Ordem de Implementação

### Fase 1: Foundation (Semana 1)
1. Criar `resync/core/backends/base.py`
2. Criar `resync/core/backends/__init__.py`
3. Criar `resync/core/load_balancing.py`
4. Criar `resync/core/metrics_utils.py`

### Fase 2: HTTP & Workers (Semana 2)
5. Criar `resync/core/factories/http_client.py`
6. Criar `resync/core/workers.py`

### Fase 3: Integração (Semana 3)
7. Atualizar `service_discovery.py` para usar módulos novos
8. Atualizar `redis_init.py` para usar backends pattern

### Fase 4: Expansão (Semana 4)
9. Atualizar `smart_pooling.py` com load balancing
10. Atualizar `resilience.py` com métricas

---

## 5. Critérios de Sucesso

- [ ] 80% de redução de código duplicado em service discovery
- [ ] Backends Redis/Kubernetes funcionando com interface comum
- [ ] Load balancing strategies reutilizáveis em smart_pooling
- [ ] Métricas Prometheus padronizadas
- [ ] OpenTelemetry tracing consistente

---

## 6. Riscos e Mitigações

| Risco | Mitigação |
|-------|-----------|
| Breaking changes em APIs | Manter backwards compatibility |
| Performance overhead |lazy initialization |
| Complexidade adicional | Documentação clara |

---

## 7. Dependências Externas

- `httpx` (já presente)
- `prometheus-client` (já presente)
- `opentelemetry-api` (já presente)
- `pydantic` (já presente)

---

## 8. Testes Necessários

- Testes unitários para LoadBalancer
- Testes de integração para BackendInterface
- Testes de stress para workers
- Testes de regressão para service_discovery.py

## Visão Geral

Este plano detalha a extração e reutilização dos padrões implementados em `service_discovery.py` para outros componentes do projeto Resync.

**Origem:** `resync/core/service_discovery.py` (51.695 bytes, v7.2-prod)

---

## 1. Padrões Identificados no service_discovery.py

### 1.1 Backend Interface Pattern (ABC)
- **Localização:** Linhas 344-356
- **Descrição:** Interface abstrata para múltiplos backends de discovery
- **Métodos abstratos:**
  - `register_service()`
  - `deregister_service()`
  - `discover_services()`
  - `watch_service()`
  - `health_check()`
  - `close()`

### 1.2 Load Balancing Strategies
- **Localização:** Linhas 144-150, 910-927
- **Estratégias disponíveis:**
  - `ROUND_ROBIN` - Rotação cíclica
  - `RANDOM` - Seleção aleatória
  - `LEAST_CONNECTIONS` - Menos conexões ativas
  - `WEIGHTED_RANDOM` - Aleatório ponderado
  - `LATENCY_BASED` - Baseado em latência

### 1.3 Circuit Breaker por Instância
- **Localização:** Linhas 320-323, 617-618, 907-908, 1024-1028
- **Características:**
  - Tracking de `_circuit_open_until` (monotonic time)
  - Threshold configurável de falhas consecutivas
  - Recovery timeout
  - Métricas por instância

### 1.4 Health Checks com Métricas
- **Localização:** Linhas 987-1050
- **Características:**
  - Falhas consecutivas
  - Response time average (média móvel exponencial)
  - Intervalo configurável
  - TTL de instância

### 1.5 Prometheus Metrics
- **Localização:** Linhas 59-127
- **Métricas de baixa cardinalidade:**
  - `_prom_registrations_total`
  - `_prom_discoveries_total`
  - `_prom_health_checks_total`
  - `_prom_lb_decisions_total`
  - `_prom_active_services` (Gauge)
  - `_prom_service_instances` (Gauge)
  - `_prom_service_healthy_instances` (Gauge)

### 1.6 OpenTelemetry Tracing
- **Localização:** Throughout
- **Spans em operações request-like:**
  - `sdm.backend.consul.register`
  - `sdm.get_service_instances`
  - `sdm.health_check`

### 1.7 HTTP Client Pooling
- **Localização:** Linhas 213-219, 371-382
- **Características:**
  - `httpx.AsyncClient` reutilizável
  - `httpx.Limits` com max_connections, keepalive
  - Timeout configurável

### 1.8 Workers Background
- **Localização:** Linhas 1063-1159
- **Workers:**
  - `_discovery_worker` - Refresh de cache
  - `_health_worker` - Health checks
  - `_backend_health_worker` - Health do backend
  - `_metrics_worker` - Logging de métricas

### 1.9 Context Manager Borrow
- **Localização:** Linhas 963-981
- **Descrição:** Incremento/decremento automático de conexões

### 1.10 Semáforos para Concurrency
- **Localização:** Linhas 621-622
- **Descrição:** `_disc_sem`, `_hc_sem` para controle de concorrência

---

## 2. Arquivos Novos a Criar

### 2.1 `resync/core/backends/__init__.py`
```
exports: BackendInterface, BackendHealth
```

### 2.2 `resync/core/backends/base.py`
```python
class BackendInterface(ABC):
    @abstractmethod
    async def register(...) -> bool: ...
    @abstractmethod
    async def deregister(...) -> bool: ...
    @abstractmethod
    async def discover(...) -> list[ServiceInstance]: ...
    @abstractmethod
    async def health_check() -> bool: ...
    @abstractmethod
    async def close() -> None: ...
```

### 2.3 `resync/core/load_balancing.py`
```python
class LoadBalancingStrategy(str, Enum):
    ROUND_ROBIN = "round_robin"
    RANDOM = "random"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_RANDOM = "weighted_random"
    LATENCY_BASED = "latency_based"

class LoadBalancer(Generic[T]):
    def select(candidates: list[T], strategy: Strategy) -> T | None: ...
    def with_connections(candidates: list[T], conn_counts: dict[str, int], strategy: Strategy) -> T | None: ...
```

### 2.4 `resync/core/factories/http_client.py`
```python
class HTTPClientFactory:
    @staticmethod
    def create_client(base_url: str, config: HTTPClientConfig) -> httpx.AsyncClient: ...
    @staticmethod
    def create_pooled_limits(config: HTTPClientConfig) -> httpx.Limits: ...
```

### 2.5 `resync/core/workers.py`
```python
class BaseWorker(ABC):
    async def start() -> None: ...
    async def stop() -> None: ...
    async def run_loop() -> None: ...

class WorkerManager:
    def register_worker(name: str, worker: BaseWorker) -> None: ...
    async def start_all() -> None: ...
    async def stop_all() -> None: ...
```

### 2.6 `resync/core/metrics_utils.py`
```python
def get_or_create_metric(factory, name: str, *args, **kwargs): ...

# Reusable metric definitions
prom_discovery_total = ...
prom_health_checks = ...
prom_lb_decisions = ...
```

---

## 3. Arquivos a Atualizar

### 3.1 `resync/core/service_discovery.py`
- **Mudanças:**
  - Importar de `backends.base`
  - Importar de `load_balancing`
  - Importar de `metrics_utils`
  - Importar de `factories.http_client`
  - Remover código duplicado

### 3.2 `resync/core/redis_init.py`
- **Benefícios:**
  - Backend interface para Redis providers
  - Health checks unificados
  - HTTP client pooling

### 3.3 `resync/core/redis_strategy.py`
- **Benefícios:**
  - Load balancing strategies
  - Circuit breaker instance-level

### 3.4 `resync/core/smart_pooling.py`
- **Benefícios:**
  - Load balancing strategies (ROUND_ROBIN, LEAST_CONNECTIONS, etc)
  - Context manager borrow pattern

### 3.5 `resync/core/resilience.py`
- **Benefícios:**
  - Circuit breaker com GC de estado
  - Métricas Prometheus padronizadas

### 3.6 `resync/core/circuit_breaker_registry.py`
- **Benefícios:**
  - Métricas padronizadas
  - OpenTelemetry tracing

---

## 4. Ordem de Implementação

### Fase 1: Foundation (Semana 1)
1. Criar `resync/core/backends/base.py`
2. Criar `resync/core/backends/__init__.py`
3. Criar `resync/core/load_balancing.py`
4. Criar `resync/core/metrics_utils.py`

### Fase 2: HTTP & Workers (Semana 2)
5. Criar `resync/core/factories/http_client.py`
6. Criar `resync/core/workers.py`

### Fase 3: Integração (Semana 3)
7. Atualizar `service_discovery.py` para usar módulos novos
8. Atualizar `redis_init.py` para usar backends pattern

### Fase 4: Expansão (Semana 4)
9. Atualizar `smart_pooling.py` com load balancing
10. Atualizar `resilience.py` com métricas

---

## 5. Critérios de Sucesso

- [ ] 80% de redução de código duplicado em service discovery
- [ ] Backends Redis/Kubernetes funcionando com interface comum
- [ ] Load balancing strategies reutilizáveis em smart_pooling
- [ ] Métricas Prometheus padronizadas
- [ ] OpenTelemetry tracing consistente

---

## 6. Riscos e Mitigações

| Risco | Mitigação |
|-------|-----------|
| Breaking changes em APIs | Manter backwards compatibility |
| Performance overhead |lazy initialization |
| Complexidade adicional | Documentação clara |

---

## 7. Dependências Externas

- `httpx` (já presente)
- `prometheus-client` (já presente)
- `opentelemetry-api` (já presente)
- `pydantic` (já presente)

---

## 8. Testes Necessários

- Testes unitários para LoadBalancer
- Testes de integração para BackendInterface
- Testes de stress para workers
- Testes de regressão para service_discovery.py

