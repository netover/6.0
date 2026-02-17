# Documentação de Módulos - Resync v6.2.0

Este documento detalha cada módulo do projeto Resync, sua responsabilidade e arquivos principais.

---

## Índice

1. [Arquivos Raiz](#1-arquivos-raiz)
2. [API Layer](#2-api-layer)
3. [Core Layer](#3-core-layer)
4. [Services Layer](#4-services-layer)
5. [Knowledge Layer](#5-knowledge-layer)
6. [Models](#6-models)
7. [Workflows](#7-workflows)
8. [Scripts](#8-scripts)

---

## 1. Arquivos Raiz

| Arquivo | Descrição |
|---------|-----------|
| `main.py` | Entry point da aplicação FastAPI |
| `app_factory.py` | Factory para criação da app FastAPI |
| `settings.py` | Configurações globais (Pydantic Settings) |
| `settings_types.py` | Tipos customizados para settings |
| `settings_validators.py` | Validadores para settings |
| `csp_validation.py` | Validação de Content Security Policy |

---

## 2. API Layer

### 2.1 Routes (`resync/api/routes/`)

#### Admin Routes (`routes/admin/`)
| Arquivo | Descrição |
|---------|-----------|
| `main.py` | Endpoints administrativos principais |
| `v2.py` | Endpoints admin v2 |
| `users.py` | Gerenciamento de usuários |
| `settings_manager.py` | Configurações dinâmicas |
| `backup.py` | Backup e restore |
| `prompts.py` | Gerenciamento de prompts |
| `rag_stats.py` | Estatísticas do RAG |
| `semantic_cache.py` | Cache semântico admin |
| `rag_reranker.py` | Configuração de reranking |
| `feedback_curation.py` | Curadoria de feedback |
| `connectors.py` | Gerenciamento de conectores |
| `teams.py` | Integração Teams admin |
| `teams_notifications_admin.py` | Admin de notificações |
| `teams_webhook_admin.py` | Admin de webhooks Teams |
| `tws_instances.py` | Gestão de instâncias TWS |
| `threshold_tuning.py` | Ajuste de thresholds |
| `environment.py` | Variáveis de ambiente |
| `config.py` | Configurações |

#### Agent Routes (`routes/agents/`)
| Arquivo | Descrição |
|---------|-----------|
| `agents.py` | Endpoints de execução de agentes |

#### Core Routes (`routes/core/`)
| Arquivo | Descrição |
|---------|-----------|
| `auth.py` | Autenticação e login |
| `chat.py` | Endpoint de chat |
| `health.py` | Health checks |
| `status.py` | Status do sistema |

#### Monitoring Routes (`routes/monitoring/`)
| Arquivo | Descrição |
|---------|-----------|
| `dashboard.py` | Dashboard de monitoramento |
| `metrics.py` | Métricas Internas (formato Prometheus) |
| `metrics_dashboard.py` | Dashboard de métricas |
| `observability.py` | Observabilidade |
| `ai_monitoring.py` | Monitoramento de AI |
| `admin_monitoring.py` | Monitoramento admin |

#### RAG Routes (`routes/rag/`)
| Arquivo | Descrição |
|---------|-----------|
| `query.py` | Consulta RAG |
| `upload.py` | Upload de documentos |

#### Knowledge Routes (`routes/knowledge/`)
| Arquivo | Descrição |
|---------|-----------|
| `ingest_api.py` | API de ingestão |

#### Enterprise Routes (`routes/enterprise/`)
| Arquivo | Descrição |
|---------|-----------|
| `enterprise.py` | Endpoints enterprise |
| `gateway.py` | API Gateway |

#### Demais Routes
| Arquivo | Descrição |
|---------|-----------|
| `endpoints.py` | Endpoints gerais |
| `cache.py` | Cache API |
| `audit.py` | Endpoints de audit |
| `performance.py` | Performance |
| `a2a.py` | Agent-to-Agent |
| `teams_webhook.py` | Webhook Teams |
| `cors_monitoring.py` | Monitoramento CORS |
| `rfc_examples.py` | Exemplos RFC |

---

### 2.2 Middleware (`resync/api/middleware/`)

| Arquivo | Descrição |
|---------|-----------|
| `cors_middleware.py` | CORS handling |
| `cors_config.py` | Configuração CORS |
| `cors_monitoring.py` | Monitoramento CORS |
| `csp_middleware.py` | Content Security Policy |
| `csrf_protection.py` | Proteção CSRF |
| `error_handler.py` | Tratamento de erros |
| `idempotency.py` | Controle de idempotência |
| `security_headers.py` | Headers de segurança |
| `database_security_middleware.py` | Segurança de banco |
| `compression.py` | Compressão de responses |
| `correlation_id.py` | Correlation ID para tracing |
| `redis_validation.py` | Validação Redis |

---

### 2.3 Models (`resync/api/models/`)

| Arquivo | Descrição |
|---------|-----------|
| `auth.py` | Modelos de autenticação |
| `base.py` | Modelos base |
| `health.py` | Modelos de health |
| `links.py` | Modelos de links/HATEOAS |
| `rag.py` | Modelos RAG |
| `requests.py` | Modelos de request |
| `responses.py` | Modelos de response |
| `responses_v2.py` | Responses v2 |
| `agents.py` | Modelos de agentes |

---

### 2.4 Validation (`resync/api/validation/`)

| Arquivo | Descrição |
|---------|-----------|
| `auth.py` | Validação de auth |
| `chat.py` | Validação de chat |
| `common.py` | Validações comuns |
| `config.py` | Validação de config |
| `enhanced_security.py` | Validações de segurança |
| `files.py` | Validação de arquivos |
| `middleware.py` | Validação de middleware |
| `monitoring.py` | Validação de monitoramento |
| `query_params.py` | Validação de query params |
| `agents.py` | Validação de agentes |

---

## 3. Core Layer

### 3.1 LangGraph (`resync/core/langgraph/`)

Módulo central de agentes AI baseado em LangGraph.

| Arquivo | Descrição |
|---------|-----------|
| `agent_graph.py` | **Grafo principal do agente** - Define o fluxo do agente (Router → Planner → Executor → Synthesizer) |
| `subgraphs.py` | Subgrafos: Diagnostic, Parallel, Approval |
| `checkpointer.py` | Persistência de estado com PostgreSQL |
| `models.py` | Modelos Pydantic para LangGraph |
| `templates.py` | Templates de prompts |
| `hallucination_grader.py` | Validação de hallucinação |
| `diagnostic_graph.py` | Grafo de diagnóstico |
| `parallel_graph.py` | Execução paralela |
| `incident_response.py` | Pipeline de resposta a incidentes |
| `plan_templates.py` | Templates de planejamento |
| `roma_graph.py` | Sistema ROMA (multi-agente) |
| `roma_nodes.py` | Nós do sistema ROMA |
| `roma_models.py` | Modelos ROMA |
| `nodes.py` | Nós legacy |

### 3.2 Cache (`resync/core/cache/`)

Sistema de cache em múltiplas camadas.

| Arquivo | Descrição |
|---------|-----------|
| `advanced_cache.py` | Cache avançado com múltiplas camadas |
| `async_cache.py` | Cache assíncrono |
| `base_cache.py` | Classe base de cache |
| `cache_factory.py` | Factory de cache |
| `cache_hierarchy.py` | Hierarquia de cache |
| `cache_warmer.py` | Pre-warming de cache |
| `cache_with_stampede_protection.py` | Proteção contra stampede |
| `embedding_model.py` | Cache de embeddings |
| `improved_cache.py` | Cache melhorado |
| `llm_cache_wrapper.py` | Wrapper de cache para LLM |
| `memory_manager.py` | Gerenciamento de memória |
| `query_cache.py` | Cache de queries |
| `redis_config.py` | Configuração Redis |
| `reranker.py` | Cache de reranker |
| `semantic_cache.py` | Cache semântico |
| `mixins/` | Mixins de cache (health, metrics, etc) |

### 3.3 Database (`resync/core/database/`)

| Arquivo | Descrição |
|---------|-----------|
| `config.py` | Configuração de banco |
| `engine.py` | Engine SQLAlchemy |
| `session.py` | Gerenciamento de sessão |
| `schema.py` | Schema do banco |
| `models/` | Modelos SQLAlchemy |
| `repositories/` | Repositórios de dados |

#### Models (`resync/core/database/models/`)
| Arquivo | Tabela |
|---------|--------|
| `auth.py` | users, audit_logs |
| `metrics.py` | workstation_metrics_history |
| `stores.py` | tws_*, conversations, context, audit, sessions, feedback, learning |
| `teams.py` | teams_webhook_users, teams_webhook_audit |
| `teams_notifications.py` | teams_channels, teams_job_mappings, teams_pattern_rules |

### 3.4 Health (`resync/core/health/`)

Sistema de health checks e monitoramento.

| Arquivo | Descrição |
|---------|-----------|
| `health_checkers/` | Verificadores específicos |
| `monitors/` | Monitores |
| `unified_health_service.py` | Serviço unificado |
| `health_service_facade.py` |Facade de health |
| `proactive_monitor.py` | Monitoramento proativo |
| `recovery_manager.py` | Gerenciamento de recuperação |
| `circuit_breaker_manager.py` | Circuit breakers |

### 3.5 Idempotency (`resync/core/idempotency/`)

Controle de idempotência para requests.

| Arquivo | Descrição |
|---------|-----------|
| `manager.py` | Gerenciador principal |
| `storage.py` | Armazenamento |
| `config.py` | Configuração |
| `exceptions.py` | Exceções |
| `metrics.py` | Métricas |
| `models.py` | Modelos |
| `validation.py` | Validação |

### 3.6 Monitoring (`resync/core/monitoring/`)

| Arquivo | Descrição |
|---------|-----------|
| `evidently_monitor.py` | Monitoramento com Evidently |

### 3.7 Security (`resync/core/security/`)

| Arquivo | Descrição |
|---------|-----------|
| `rate_limiter_v2.py` | Rate limiting v2 |
| `storage.py` | Armazenamento de segurança |

### 3.8 Utilities (`resync/core/utils/`)

| Arquivo | Descrição |
|---------|-----------|
| `llm.py` | Utilitários LLM |
| `llm_factories.py` | Factories LLM |
| `error_factories.py` | Factories de erro |
| `error_utils.py` | Utilitários de erro |
| `prompt_formatter.py` | Formatação de prompts |
| `secret_scrubber.py` | Limpeza de segredos |
| `correlation.py` | Correlation ID |
| `json_parser.py` | Parser JSON |

### 3.9Outros Módulos Core

| Arquivo | Descrição |
|---------|-----------|
| `startup.py` | Inicialização da app |
| `wiring.py` | DI wiring |
| `structured_logger.py` | Logging estruturado |
| `redis_init.py` | Inicialização Redis |
| `exceptions.py` | Exceções customizadas |
| `context.py` | Contexto |
| `constants.py` | Constantes |
| `jwt_utils.py` | Utilitários JWT |
| `retry.py` | Lógica de retry |
| `llm_config.py` | Configuração LLM |
| `llm_monitor.py` | Monitoramento LLM |
| `llm_optimizer.py` | Otimização LLM |
| `service_discovery.py` | Descoberta de serviços |
| `circuit_breaker_registry.py` | Registry de circuit breakers |
| `skill_manager.py` | Gerenciamento de skills |
| `agent_manager.py` | Gerenciamento de agentes |
| `agent_router.py` | Roteamento de agentes |
| `agent_evolution.py` | Evolução de agentes |
| `anomaly_detector.py` | Detecção de anomalias |
| `config_hot_reload.py` | Hot reload de config |
| `distributed_tracing.py` | Tracing distribuído |

---

## 4. Services Layer

| Arquivo | Descrição |
|---------|-----------|
| `llm_service.py` | **Serviço de LLM** - Abstração sobre OpenAI, Anthropic, LiteLLM |
| `rag_client.py` | **Cliente RAG** - Recuperação de conhecimento |
| `tws_service.py` | **Serviço TWS** - Integração com IBM Tivoli |
| `tws_unified.py` | TWS unificado |
| `tws_graph_service.py` | TWS com grafo |
| `tws_cache.py` | Cache TWS |
| `config_manager.py` | Gerenciamento de config |
| `llm_retry.py` | Retry para LLM |
| `llm_fallback.py` | Fallback entre provedores |
| `mock_tws_service.py` | Mock para testes |
| `advanced_graph_queries.py` | Queries avançadas em grafo |

---

## 5. Knowledge Layer

### 5.1 Ingestion (`resync/knowledge/ingestion/`)

| Arquivo | Descrição |
|---------|-----------|
| `pipeline.py` | Pipeline de ingestão |
| `document_converter.py` | Conversão de documentos |
| `chunking.py` | Divisão em chunks |
| `embedding_service.py` | Serviço de embeddings |
| `embeddings.py` | Embeddings |
| `filter_strategy.py` | Estratégia de filtragem |
| `ingest.py` | Ingestão |
| `authority.py` | Autoridade |
| `chunking_eval.py` | Avaliação de chunking |

### 5.2 Retrieval (`resync/knowledge/retrieval/`)

| Arquivo | Descrição |
|---------|-----------|
| `retriever.py` | Recuperador |
| `hybrid_retriever.py` | Recuperação híbrida |
| `hybrid.py` | Busca híbrida |
| `graph.py` | Busca em grafo |
| `reranker.py` | Re-ranking |
| `reranker_interface.py` | Interface de reranker |
| `cache_manager.py` | Cache de retrieval |
| `feedback_retriever.py` | Recuperação de feedback |
| `metrics.py` | Métricas de retrieval |
| `tws_relations.py` | Relações TWS |

### 5.3 Store (`resync/knowledge/store/`)

| Arquivo | Descrição |
|---------|-----------|
| `pgvector.py` | Store PgVector |
| `pgvector_store.py` | Store PgVector v2 |
| `feedback_store.py` | Store de feedback |

### 5.4 Knowledge Graph (`resync/knowledge/kg_store/`)

| Arquivo | Descrição |
|---------|-----------|
| `store.py` | Store de grafo |
| `ddl.py` | DDL PostgreSQL |

### 5.5 KG Extraction (`resync/knowledge/kg_extraction/`)

| Arquivo | Descrição |
|---------|-----------|
| `extractor.py` | Extrator de grafo |
| `normalizer.py` | Normalizador |
| `prompts.py` | Prompts |
| `schemas.py` | Schemas |

### 5.6 Ontology (`resync/knowledge/ontology/`)

| Arquivo | Descrição |
|---------|-----------|
| `ontology_manager.py` | Gerenciador de ontologia |
| `entity_resolver.py` | Resolução de entidades |
| `provenance.py` | Proveniência |

---

## 6. Models

| Arquivo | Descrição |
|---------|-----------|
| `tws.py` | Modelos TWS |
| `agents.py` | Modelos de agentes |
| `cache.py` | Modelos de cache |
| `error_models.py` | Modelos de erro |
| `a2a.py` | Modelos A2A |
| `validation.py` | Modelos de validação |

---

## 7. Workflows

| Arquivo | Descrição |
|---------|-----------|
| `workflow_predictive_maintenance.py` | Workflow preditivo |
| `workflow_capacity_forecasting.py` | Previsão de capacidade |
| `nodes.py` | Nós de workflow |
| `nodes_optimized.py` | Nós otimizados |
| `nodes_verbose.py` | Nós verbosos |
| `node_config.py` | Configuração de nós |
| `statistical_analysis.py` | Análise estatística |

---

## 8. Scripts

| Arquivo | Descrição |
|---------|-----------|
| `setup_environment.py` | **Setup completo** - Instala dependências e configura banco |
| `install_postgresql.py` | **Script PostgreSQL** - Instalação e configuração |
| `install_redis.py` | **Script Redis** - Instalação e configuração |
| `check_env.py` | Verificação de ambiente |
| `manual_verify.py` | Verificação manual |

---

## Fluxo de Dados Entre Módulos

```
User Request
    │
    ▼
API Routes (resync/api/routes/)
    │
    ├──▶ Auth Middleware ──▶ JWT Validation
    │
    ├──▶ Dependencies (resync/api/dependencies.py)
    │       │
    │       ├──▶ LLM Service (services/llm_service.py)
    │       │       │
    │       │       ├──▶ OpenAI/Anthropic/LiteLLM
    │       │       │
    │       │       └──▶ Langfuse (observability)
    │       │
    │       ├──▶ RAG Client (services/rag_client.py)
    │       │       │
    │       │       └──▶ Knowledge Layer
    │       │               │
    │       │               ├──▶ Retrieval
    │       │               ├──▶ Vector Store (PgVector)
    │       │               └──▶ Knowledge Graph
    │       │
    │       └──▶ TWS Service (services/tws_service.py)
    │               │
    │               └──▶ IBM TWS API
    │
    ▼
LangGraph Agent (core/langgraph/)
    │
    ├──▶ Router Node ──▶ Intent Classification
    │
    ├──▶ Planner Node ──▶ Plan Generation
    │
    ├──▶ Executor Node ──▶ Action Execution
    │
    ├──▶ Synthesizer ──▶ Response Generation
    │
    └──▶ Hallucination Grader ──▶ Validation
            │
            └──▶ (Retry if invalid)
```

---

## Dependências entre Módulos

```
API Routes
    │
    ├── depends on: Services, Core
    │
    ▼
Services Layer
    │
    ├── depends on: Core, Knowledge
    │
    ▼
Core Layer
    │
    ├── depends on: Database, Cache, Utils
    │
    ▼
Data Layer (PostgreSQL + Redis)
```
