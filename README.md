# Resync v6.2.0

> Sistema de Orquestração de Agentes AI com RAG para IBM TWS

## 📋 Visão Geral

Resync é uma plataforma de orquestração de agentes AI desenvolvida com FastAPI, LangGraph e RAG (Retrieval Augmented Generation), projetada para integrar-se com o IBM Tivoli Workload Scheduler (TWS).

### ✨ Funcionalidades Principais

- 🤖 **Orquestração de Agentes** - Agentes AI baseados em LangGraph para automatizar tarefas
- 🔍 **RAG** - Busca e recuperação de conhecimento contextualizado
- 📊 **Monitoramento** - Métricas, alertas e observabilidade completa
- 🔐 **Segurança** - Autenticação, autorização e conformidade SOC2/GDPR
- ⚡ **Alta Performance** - Cache inteligente, rate limiting, circuit breaker
- 🔄 **Alta Disponibilidade** - Redis, PostgreSQL, health checks

## 🏗️ Arquitetura

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLIENTS                                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │  Web UI  │  │  Slack   │  │ Teams    │  │  API     │  │ Webhook  │    │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘    │
└───────┼─────────────┼─────────────┼─────────────┼─────────────┼──────────┘
        │             │             │             │             │
        └─────────────┴─────────────┴─────────────┴─────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           API GATEWAY (FastAPI)                            │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │  Routes: /auth, /chat, /agents, /admin, /rag, /monitoring, /teams  │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│  ┌──────────────────────────────────────────────────────────────────────┐ │
│  │  Middleware: CORS, Auth, Rate Limit, Idempotency, Security Headers   │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────┬────────────────────────────────────┘
                                       │
        ┌──────────────────────────────┼──────────────────────────────┐
        │                              │                              │
        ▼                              ▼                              ▼
┌───────────────────┐    ┌───────────────────┐    ┌───────────────────┐
│   CORE SERVICES   │    │   AGENT LAYER    │    │   KNOWLEDGE LAYER│
│                   │    │                   │    │                   │
│ ┌───────────────┐ │    │ ┌───────────────┐ │    │ ┌───────────────┐ │
│ │ Cache (Redis) │ │    │ │ LangGraph    │ │    │ │ RAG Pipeline  │ │
│ └───────────────┘ │    │ │ Agent Graph   │ │    │ └───────────────┘ │
│ ┌───────────────┐ │    │ └───────────────┘ │    │ ┌───────────────┐ │
│ │ Auth Service  │ │    │ ┌───────────────┐ │    │ │ Kg Store      │ │
│ └───────────────┘ │    │ │ ROMA System   │ │    │ └───────────────┘ │
│ ┌───────────────┐ │    │ └───────────────┘ │    │ ┌───────────────┐ │
│ │ LLM Service   │ │    │ ┌───────────────┐ │    │ │ Embeddings    │ │
│ └───────────────┘ │    │ │ Agent Router  │ │    │ └───────────────┘ │
│ ┌───────────────┐ │    │ └───────────────┘ │    │                   │
│ │ TWS Service   │ │    │                   │    │                   │
│ └───────────────┘ │    │                   │    │                   │
└───────────────────┘    └───────────────────┘    └───────────────────┘
        │                              │                              │
        └──────────────────────────────┼──────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DATA LAYER                                         │
│  ┌───────────────────┐                         ┌───────────────────┐       │
│  │   PostgreSQL     │                         │      Redis       │       │
│  │  (Dados + RAG)   │                         │   (Cache/Sessão) │       │
│  │                  │                         │                   │       │
│  │ - users          │                         │ - cache          │       │
│  │ - audit_logs     │                         │ - sessions       │       │
│  │ - tws_*          │                         │ - rate_limit     │       │
│  │ - kg_nodes/edges │                         │ - idempotency    │       │
│  │ - metrics        │                         │                   │       │
│  └───────────────────┘                         └───────────────────┘       │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      EXTERNAL SYSTEMS                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌────────────┐  │
│  │ IBM TWS      │  │  OpenAI     │  │  Anthropic  │  │  Langfuse │  │
│  │ (Jobs/Work) │  │  (GPT-4)    │  │ (Claude)    │  │ (Observab) │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  └────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 📦 Estrutura do Projeto

```
resync/
├── 📄 main.py                 # Entry point da aplicação
├── 📄 app_factory.py          # Factory do FastAPI
├── 📄 settings.py             # Configurações (Pydantic)
├── 📄 requirements.txt        # Dependências Python
│
├── 📁 api/                   # API REST (FastAPI)
│   ├── routes/               # Endpoints
│   │   ├── admin/          # Painel administrativo
│   │   ├── agents/         # Agentes AI
│   │   ├── chat/          # Chat endpoint
│   │   ├── rag/           # RAG endpoints
│   │   ├── monitoring/    # Métricas e health
│   │   └── teams/         # Integração Teams
│   ├── middleware/          # Middleware (CORS, Auth, etc)
│   ├── models/             # Modelos de request/response
│   └── validation/         # Validações Pydantic
│
├── 📁 core/                  # Nucleo do sistema
│   ├── langgraph/          # Agentes LangGraph
│   │   ├── agent_graph.py # Grafo principal do agente
│   │   ├── subgraphs.py   # Subgrafos (diagnóstico, paralelo)
│   │   └── checkpointer.py# Persistência de estado
│   ├── cache/              # Sistema de cache
│   ├── database/           # Modelos SQLAlchemy
│   ├── health/             # Health checks
│   ├── idempotency/        # Controle de idempotência
│   ├── monitoring/         # Monitoramento
│   ├── security/           # Rate limiting, etc
│   └── utils/              # Utilitários
│
├── 📁 services/             # Serviços de negócio
│   ├── llm_service.py     # Integração LLM
│   ├── rag_client.py      # Cliente RAG
│   └── tws_service.py    # Integração TWS
│
├── 📁 knowledge/            # Camada de conhecimento
│   ├── ingestion/         # Ingestão de documentos
│   ├── retrieval/        # Recuperação RAG
│   ├── store/            # Armazenamento (PgVector)
│   └── kg_store/         # Knowledge Graph
│
├── 📁 models/              # Modelos de dados
├── 📁 workflows/           # Workflows de ML
├── 📁 scripts/             # Scripts de setup
└── 📁 tests/               # Testes
```

## 🚀 Quick Start

### Pré-requisitos

- Python 3.14+
- Pydantic v2+ (>=2.10)
- PostgreSQL 14+ (com extensão pgvector)
- Redis 6+

### Instalação

```bash
# 1. Clone o projeto
git clone https://github.com/seu-repo/resync.git
cd resync

# 2. Configure o ambiente
python resync/scripts/setup_environment.py

# 3. Configure variáveis de ambiente
cp .env.example .env
# Edite .env com suas credenciais

# 4. Inicie o servidor
python -m uvicorn resync.main:app --reload

# 5. Acesse a API
# API: http://localhost:8000
# Docs: http://localhost:8000/docs
```

## 🧰 Execução em Produção (VM, sem Nginx)

Para execução **direta na VM** (sem Nginx) recomendamos **Gunicorn + UvicornWorker** para gerenciamento de processos e estabilidade.

### Recomendado (Gunicorn + UvicornWorker)

> Ajuste o número de workers conforme CPU/latência. Em uma VM de **4 vCPU / 8 GB**, um ponto de partida seguro é **2 workers**.

```bash
gunicorn resync.main:app \
  -k uvicorn.workers.UvicornWorker \
  -w 2 \
  --bind 0.0.0.0:8000 \
  --graceful-timeout 30 \
  --timeout 120 \
  --keep-alive 5
```

### Alternativa (Uvicorn direto)

```bash
uvicorn resync.main:app \
  --host 0.0.0.0 --port 8000 \
  --workers 2 \
  --timeout-keep-alive 5 \
  --timeout-graceful-shutdown 30 \
  --limit-concurrency 200
```

> Observação: em produção **não** use `--reload`.

### Configuração Docker

```bash
# Usando Docker Compose
docker-compose -f docker-compose.resync.yml up -d
```

## 📚 Documentação

- [Arquitetura Detalhada](ARCHITECTURE.md) - Diagramas e fluxo de dados
- [Documentação de Módulos](MODULES.md) - Descrição de cada módulo
- [Guia de API](docs/API.md) - Endpoints da API
- [Setup Local](docs/SETUP.md) - Configuração de desenvolvimento
- [部署指南](docs/DEPLOY.md) - Guia de deployment

## 🔌 Endpoints Principais

| Método | Endpoint | Descrição |
|--------|----------|-----------|
| POST | `/api/v1/chat` | Chat com agente AI |
| POST | `/api/v1/agents/execute` | Executa agente |
| POST | `/api/v1/rag/query` | Consulta RAG |
| GET | `/api/v1/health` | Health check |
| POST | `/api/v1/auth/login` | Autenticação |
| GET | `/api/v1/admin/stats` | Estatísticas admin |

## 🛠️ Tecnologias

| Categoria | Tecnologia |
|-----------|------------|
| **API** | FastAPI, Starlette, Uvicorn |
| **AI/ML** | LangGraph, LangChain, OpenAI, Anthropic |
| **Database** | PostgreSQL, PgVector, SQLAlchemy |
| **Cache** | Redis |
| **Observability** | Langfuse, Dashboard Interno (Redis), Sentry, Structlog |
| **Security** | JWT, bcrypt, Rate Limiting |

## 📄 Licença

MIT License - see file `LICENSE` for details.
