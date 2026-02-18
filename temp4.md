# Plano de Implementação - Sistema de Orquestração Multi-Agent

## Resumo Executivo

Este documento apresenta um plano de implementação detalhado e extensivo para o sistema de orquestração multi-agent do projeto Resync TWS. O sistema será construído utilizando a infraestrutura existente do projeto, incluindo SQLAlchemy async, PostgreSQL, Redis para semantic cache, e PgVector para embeddings.

---

## 1. Visão Geral da Arquitetura

### 1.1 Componentes Principais

O sistema de orquestração será composto por quatro camadas principais:

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                           CAMADA DE API E ADMINISTRAÇÃO                                      │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐     │
│  │   Admin UI      │  │   REST API      │  │  WebSocket     │  │   Metrics      │     │
│  │   (Frontend)    │  │   Endpoints     │  │  Broadcasting  │  │   Collection   │     │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘     │
│           │                    │                    │                    │               │
└───────────┼────────────────────┼────────────────────┼────────────────────┼───────────────┘
            │                    │                    │                    │
            ▼                    ▼                    ▼                    ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                              CAMADA DE ORQUESTRAÇÃO                                          │
│  ┌─────────────────────────────────────────────────────────────────────────────────────┐    │
│  │                            OrchestrationRunner                                      │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │    │
│  │  │  Sequential  │  │   Parallel   │  │  Consensus   │  │   Fallback   │          │    │
│  │  │  Executor   │  │   Executor   │  │   Executor   │  │   Chain      │          │    │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘          │    │
│  └─────────────────────────────────────────────────────────────────────────────────────┘    │
│           │                    │                    │                    │               │
│           ▼                    ▼                    ▼                    ▼               │
│  ┌─────────────────────────────────────────────────────────────────────────────────────┐    │
│  │                          Execution State Machine                                    │    │
│  │   pending → running → completed / failed / paused / compensation                   │    │
│  └─────────────────────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                              CAMADA DE DADOS                                                │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐     │
│  │    Models       │  │  Repositories   │  │    Events       │  │    Cache       │     │
│  │  (SQLAlchemy)   │  │   (DAO/ORM)    │  │   (Event Bus)  │  │  (RedisVL)     │     │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  └─────────────────┘     │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                          CAMADA DE INFRAESTRUTURA EXISTENTE                                 │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐     │
│  │   PostgreSQL    │  │     Redis       │  │   PgVector     │  │    Agents      │     │
│  │   (AsyncSQLA)   │  │   (Cache)       │  │  (Embeddings)  │  │  (HybridRouter)│     │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  └─────────────────┘     │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Fluxo de Dados

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                    FLUXO DE EXECUÇÃO                                         │
├─────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                             │
│  User Request                                                                          │
│       │                                                                                 │
│       ▼                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐     │
│  │ 1. API receives request (REST or WebSocket)                                    │     │
│  │    - POST /api/v1/orchestrations/execute                                       │     │
│  │    - Or WebSocket message with execution_id                                    │     │
│  └─────────────────────────────────────────────────────────────────────────────────┘     │
│       │                                                                                 │
│       ▼                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐     │
│  │ 2. Load orchestration config from database                                      │     │
│  │    - orchestration_configs table                                               │     │
│  │    - Parse JSON steps definition                                               │     │
│  └─────────────────────────────────────────────────────────────────────────────────┘     │
│       │                                                                                 │
│       ▼                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐     │
│  │ 3. Create execution record                                                      │     │
│  │    - orchestration_executions table                                           │     │
│  │    - Generate trace_id for correlation                                         │     │
│  │    - Initialize status = "pending"                                            │     │
│  └─────────────────────────────────────────────────────────────────────────────────┘     │
│       │                                                                                 │
│       ▼                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐     │
│  │ 4. Choose executor strategy                                                    │     │
│  │    - sequential: Run steps one by one                                          │     │
│  │    - parallel: Run independent steps together                                  │     │
│  │    - consensus: Aggregate results from multiple agents                          │     │
│  │    - fallback: Try primary, then fallback on failure                            │     │
│  └─────────────────────────────────────────────────────────────────────────────────┘     │
│       │                                                                                 │
│       ▼                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐     │
│  │ 5. Execute each step (according to strategy)                                    │     │
│  │    ┌────────────────────────────────────────────────────────────────────────┐   │     │
│  │    │ For each step:                                                         │   │     │
│  │    │ a. Create step_run record (status="running")                         │   │     │
│  │    │ b. Call appropriate agent (HybridRouter)                             │   │     │
│  │    │ c. Collect results, metrics, latency                                 │   │     │
│  │    │ d. Update step_run with output, status ("completed"/"failed")        │   │     │
│  │    │ e. Broadcast progress via WebSocket                                   │   │     │
│  │    │ f. Check for pause/human-in-the-loop                                 │   │     │
│  │    └────────────────────────────────────────────────────────────────────────┘   │     │
│  └─────────────────────────────────────────────────────────────────────────────────┘     │
│       │                                                                                 │
│       ▼                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐     │
│  │ 6. Finalize execution                                                          │     │
│  │    - Update execution status (completed/failed)                                │     │
│  │    - Store final output in execution record                                    │     │
│  │    - Trigger callbacks/webhooks if configured                                  │     │
│  │    - Send final WebSocket message                                              │     │
│  └─────────────────────────────────────────────────────────────────────────────────┘     │
│       │                                                                                 │
│       ▼                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐     │
│  │ 7. Return response to user                                                     │     │
│  │    - REST: JSON response with execution_id, status, output                     │     │
│  │    - WebSocket: Final message with complete results                           │     │
│  └─────────────────────────────────────────────────────────────────────────────────┘     │
│                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Estrutura de Dados

### 2.1 Tabelas do Banco de Dados

#### Tabela: orchestration_configs

Esta tabela armazena as definições de configuração de orquestração. Cada configuração define um workflow completo com múltiplos passos.

```sql
CREATE TABLE orchestration_configs (
    -- Identificação
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    
    -- Estratégia de execução
    strategy VARCHAR(50) NOT NULL CHECK (
        strategy IN ('sequential', 'parallel', 'consensus', 'fallback')
    ),
    
    -- Definição dos passos (JSON)
    steps JSONB NOT NULL,
    
    -- Configurações avançadas
    metadata JSONB DEFAULT '{}',
    
    -- Controle de acesso
    owner_id VARCHAR(255),
    tenant_id VARCHAR(255),
    is_global BOOLEAN DEFAULT FALSE,
    
    -- Versionamento
    version INTEGER DEFAULT 1,
    is_active BOOLEAN DEFAULT TRUE,
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT valid_steps CHECK (jsonb_array_length(steps) > 0)
);

-- Índices para performance
CREATE INDEX idx_configs_owner ON orchestration_configs(owner_id);
CREATE INDEX idx_configs_tenant ON orchestration_configs(tenant_id);
CREATE INDEX idx_configs_strategy ON orchestration_configs(strategy);
CREATE INDEX idx_configs_active ON orchestration_configs(is_active) WHERE is_active = TRUE;
CREATE INDEX idx_configs_created ON orchestration_configs(created_at DESC);
```

**Exemplo de Steps:**

```json
{
  "steps": [
    {
      "id": "step_1",
      "name": "Collect Job Status",
      "agent": "status_handler",
      "timeout_ms": 5000,
      "retry": {
        "max_attempts": 2,
        "backoff_ms": 1000
      },
      "dependencies": [],
      "output_key": "job_status",
      "on_failure": "continue"
    },
    {
      "id": "step_2",
      "name": "Analyze Logs",
      "agent": "log_analyzer",
      "timeout_ms": 15000,
      "retry": {
        "max_attempts": 3,
        "backoff_ms": 2000
      },
      "dependencies": ["step_1"],
      "output_key": "log_analysis",
      "on_failure": "fallback"
    },
    {
      "id": "step_3",
      "name": "Search Knowledge Base",
      "agent": "rag_handler",
      "timeout_ms": 8000,
      "retry": {
        "max_attempts": 1,
        "backoff_ms": 0
      },
      "dependencies": [],
      "output_key": "rag_results",
      "on_failure": "continue"
    },
    {
      "id": "step_4",
      "name": "Synthesize Response",
      "agent": "synthesizer",
      "timeout_ms": 10000,
      "retry": {
        "max_attempts": 2,
        "backoff_ms": 500
      },
      "dependencies": ["step_1", "step_2", "step_3"],
      "output_key": "final_response",
      "on_failure": "abort"
    }
  ],
  "settings": {
    "parallel_limit": 3,
    "consensus_threshold": 0.8,
    "enable_monitoring": true,
    "callback_url": "https://example.com/webhook/orchestration"
  }
}
```

#### Tabela: orchestration_executions

Esta tabela armazena cada execução de uma orquestração, permitindo rastreamento e debugging.

```sql
CREATE TABLE orchestration_executions (
    -- Identificação
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    trace_id VARCHAR(255) UNIQUE NOT NULL,
    
    -- Referência à configuração
    config_id UUID REFERENCES orchestration_configs(id) ON DELETE SET NULL,
    config_name VARCHAR(255),
    
    -- Status da execução
    status VARCHAR(50) NOT NULL CHECK (
        status IN (
            'pending', 'running', 'paused', 'completed', 
            'failed', 'cancelled', 'compensation'
        )
    ),
    
    -- Dados de entrada
    input JSONB DEFAULT '{}',
    
    -- Dados de saída
    output JSONB DEFAULT '{}',
    
    -- Contexto de execução
    user_id VARCHAR(255),
    session_id VARCHAR(255),
    tenant_id VARCHAR(255),
    
    -- Timestamps
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    paused_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    
    -- Metadados
    metadata JSONB DEFAULT '{}',
    
    -- Latência total (ms)
    total_latency_ms INTEGER,
    
    -- Custo estimado
    estimated_cost DECIMAL(10, 6),
    
    -- Controle
    created_by VARCHAR(255),
    callback_url VARCHAR(500)
);

-- Índices para performance
CREATE INDEX idx_executions_trace ON orchestration_executions(trace_id);
CREATE INDEX idx_executions_config ON orchestration_executions(config_id);
CREATE INDEX idx_executions_status ON orchestration_executions(status);
CREATE INDEX idx_executions_user ON orchestration_executions(user_id);
CREATE INDEX idx_executions_session ON orchestration_executions(session_id);
CREATE INDEX idx_executions_created ON orchestration_executions(created_at DESC);
CREATE INDEX idx_executions_started ON orchestration_executions(started_at DESC);
```

#### Tabela: orchestration_step_runs

Armazena o resultado de cada passo individual dentro de uma execução.

```sql
CREATE TABLE orchestration_step_runs (
    -- Identificação
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    execution_id UUID NOT NULL REFERENCES orchestration_executions(id) ON DELETE CASCADE,
    
    -- Identificação do passo
    step_index INTEGER NOT NULL,
    step_id VARCHAR(100) NOT NULL,
    step_name VARCHAR(255),
    
    -- Status
    status VARCHAR(50) NOT NULL CHECK (
        status IN ('pending', 'running', 'completed', 'failed', 'skipped', 'paused')
    ),
    
    -- Resultado
    output JSONB DEFAULT '{}',
    output_truncated TEXT,
    error_message TEXT,
    error_trace TEXT,
    
    -- Dependências
    dependencies_json JSONB DEFAULT '[]',
    
    -- Métricas
    latency_ms INTEGER,
    retry_count INTEGER DEFAULT 0,
    token_count INTEGER,
    estimated_cost DECIMAL(10, 6),
    
    -- Timestamps
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    
    -- Contexto
    agent_id VARCHAR(255),
    agent_version VARCHAR(50),
    
    -- Constraints
    UNIQUE(execution_id, step_index),
    UNIQUE(execution_id, step_id)
);

-- Índices para performance
CREATE INDEX idx_steps_execution ON orchestration_step_runs(execution_id);
CREATE INDEX idx_steps_status ON orchestration_step_runs(status);
CREATE INDEX idx_steps_started ON orchestration_step_runs(started_at DESC);
CREATE INDEX idx_steps_latency ON orchestration_step_runs(latency_ms DESC) 
    WHERE status = 'completed';
```

#### Tabela: orchestration_callbacks

Armazena callbacks para notificações assíncronas.

```sql
CREATE TABLE orchestration_callbacks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    execution_id UUID NOT NULL REFERENCES orchestration_executions(id) ON DELETE CASCADE,
    
    -- Callback configuration
    callback_type VARCHAR(50) NOT NULL CHECK (
        callback_type IN ('webhook', 'email', 'sms', 'websocket')
    ),
    callback_url VARCHAR(500),
    callback_method VARCHAR(10) DEFAULT 'POST',
    callback_headers JSONB DEFAULT '{}',
    callback_body_template TEXT,
    
    -- Status
    status VARCHAR(50) DEFAULT 'pending' CHECK (
        status IN ('pending', 'sent', 'delivered', 'failed', 'retrying')
    ),
    
    -- Retry
    retry_count INTEGER DEFAULT 0,
    max_retries INTEGER DEFAULT 3,
    last_retry_at TIMESTAMPTZ,
    
    -- Result
    response_status INTEGER,
    response_body TEXT,
    
    -- Timestamps
    created_at TIMESTAMPTZ DEFAULT NOW(),
    sent_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ
);

CREATE INDEX idx_callbacks_execution ON orchestration_callbacks(execution_id);
CREATE INDEX idx_callbacks_status ON orchestration_callbacks(status);
```

### 2.2 Modelos SQLAlchemy

#### Arquivo: resync/core/database/models/orchestration.py

```python
"""
Orchestration Models - SQLAlchemy Async
"""
from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy import UniqueConstraint, Index
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all database models."""
    pass


class OrchestrationStrategy(str, Enum):
    """Enumeration of orchestration strategies."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONSENSUS = "consensus"
    FALLBACK = "fallback"


class ExecutionStatus(str, Enum):
    """Status states for orchestration execution."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    COMPENSATION = "compensation"


class StepStatus(str, Enum):
    """Status states for individual step runs."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    PAUSED = "paused"


class OrchestrationConfig(Base):
    """
    Model for orchestration configuration.
    
    Stores the definition of an orchestration workflow including
    all steps, agents, dependencies, and execution settings.
    """
    __tablename__ = "orchestration_configs"
    
    # Primary key
    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), 
        primary_key=True, 
        default=uuid4
    )
    
    # Identification
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Strategy
    strategy: Mapped[str] = mapped_column(
        String(50), 
        nullable=False
    )
    
    # Steps definition (JSON)
    steps: Mapped[dict] = mapped_column(JSON, nullable=False)
    
    # Metadata
    metadata: Mapped[dict] = mapped_column(JSON, default={})
    
    # Access control
    owner_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    tenant_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    is_global: Mapped[bool] = mapped_column(Boolean, default=False)
    
    # Versioning
    version: Mapped[int] = mapped_column(Integer, default=1)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        default=datetime.utcnow,
        onupdate=datetime.utcnow
    )
    
    # Relationships
    executions: Mapped[list["OrchestrationExecution"]] = relationship(
        "OrchestrationExecution",
        back_populates="config",
        cascade="all, delete-orphan"
    )
    
    # Table args
    __table_args__ = (
        Index('idx_configs_owner', 'owner_id'),
        Index('idx_configs_tenant', 'tenant_id'),
        Index('idx_configs_strategy', 'strategy'),
        Index('idx_configs_active', 'is_active', postgresql_where=is_active == True),  # noqa: E712
    )
    
    def __repr__(self) -> str:
        return f"<OrchestrationConfig(id={self.id}, name={self.name}, strategy={self.strategy})>"


class OrchestrationExecution(Base):
    """
    Model for orchestration execution instance.
    
    Represents a single execution of an orchestration configuration,
    tracking status, input, output, and metrics.
    """
    __tablename__ = "orchestration_executions"
    
    # Primary key
    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), 
        primary_key=True, 
        default=uuid4
    )
    
    # Trace ID for correlation
    trace_id: Mapped[str] = mapped_column(
        String(255), 
        unique=True, 
        nullable=False,
        index=True
    )
    
    # Configuration reference
    config_id: Mapped[Optional[UUID]] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey('orchestration_configs.id', ondelete='SET NULL'),
        nullable=True
    )
    config_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    
    # Status
    status: Mapped[str] = mapped_column(String(50), nullable=False)
    
    # Input/Output
    input_data: Mapped[dict] = mapped_column(JSON, default={})
    output_data: Mapped[dict] = mapped_column(JSON, default={})
    
    # Context
    user_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    session_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    tenant_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    
    # Timestamps
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    paused_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        default=datetime.utcnow
    )
    
    # Metadata
    metadata: Mapped[dict] = mapped_column(JSON, default={})
    
    # Metrics
    total_latency_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    estimated_cost: Mapped[Optional[float]] = mapped_column(nullable=True)
    
    # Control
    created_by: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    callback_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    
    # Relationships
    config: Mapped[Optional["OrchestrationConfig"]] = relationship(
        "OrchestrationConfig",
        back_populates="executions"
    )
    step_runs: Mapped[list["OrchestrationStepRun"]] = relationship(
        "OrchestrationStepRun",
        back_populates="execution",
        cascade="all, delete-orphan",
        order_by="OrchestrationStepRun.step_index"
    )
    callbacks: Mapped[list["OrchestrationCallback"]] = relationship(
        "OrchestrationCallback",
        back_populates="execution",
        cascade="all, delete-orphan"
    )
    
    # Table args
    __table_args__ = (
        Index('idx_executions_config', 'config_id'),
        Index('idx_executions_status', 'status'),
        Index('idx_executions_user', 'user_id'),
        Index('idx_executions_session', 'session_id'),
        Index('idx_executions_created', 'created_at', postgresql_using='btree'),
    )
    
    def __repr__(self) -> str:
        return f"<OrchestrationExecution(id={self.id}, trace_id={self.trace_id}, status={self.status})>"


class OrchestrationStepRun(Base):
    """
    Model for individual step execution within an orchestration.
    
    Tracks the execution of each step including status, output,
    metrics, and errors.
    """
    __tablename__ = "orchestration_step_runs"
    
    # Primary key
    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), 
        primary_key=True, 
        default=uuid4
    )
    
    # Execution reference
    execution_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey('orchestration_executions.id', ondelete='CASCADE'),
        nullable=False
    )
    
    # Step identification
    step_index: Mapped[int] = mapped_column(Integer, nullable=False)
    step_id: Mapped[str] = mapped_column(String(100), nullable=False)
    step_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    
    # Status
    status: Mapped[str] = mapped_column(String(50), nullable=False)
    
    # Output
    output: Mapped[dict] = mapped_column(JSON, default={})
    output_truncated: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    error_trace: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Dependencies
    dependencies_json: Mapped[list] = mapped_column(JSON, default=[])
    
    # Metrics
    latency_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    token_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    estimated_cost: Mapped[Optional[float]] = mapped_column(nullable=True)
    
    # Timestamps
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    
    # Context
    agent_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    agent_version: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    
    # Relationships
    execution: Mapped["OrchestrationExecution"] = relationship(
        "OrchestrationExecution",
        back_populates="step_runs"
    )
    
    # Table args
    __table_args__ = (
        UniqueConstraint('execution_id', 'step_index', name='uq_step_execution_index'),
        UniqueConstraint('execution_id', 'step_id', name='uq_step_execution_id'),
        Index('idx_steps_status', 'status'),
        Index('idx_steps_latency', 'latency_ms', postgresql_where=status == 'completed'),  # noqa: E712
    )
    
    def __repr__(self) -> str:
        return f"<OrchestrationStepRun(id={self.id}, step_id={self.step_id}, status={self.status})>"


class OrchestrationCallback(Base):
    """
    Model for orchestration callbacks/webhooks.
    
    Stores callback configuration and delivery status for
    asynchronous notifications.
    """
    __tablename__ = "orchestration_callbacks"
    
    # Primary key
    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), 
        primary_key=True, 
        default=uuid4
    )
    
    # Execution reference
    execution_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True),
        ForeignKey('orchestration_executions.id', ondelete='CASCADE'),
        nullable=False
    )
    
    # Callback configuration
    callback_type: Mapped[str] = mapped_column(String(50), nullable=False)
    callback_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    callback_method: Mapped[str] = mapped_column(String(10), default='POST')
    callback_headers: Mapped[dict] = mapped_column(JSON, default={})
    callback_body_template: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Status
    status: Mapped[str] = mapped_column(String(50), default='pending')
    
    # Retry
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    max_retries: Mapped[int] = mapped_column(Integer, default=3)
    last_retry_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    
    # Result
    response_status: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    response_body: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        default=datetime.utcnow
    )
    sent_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    execution: Mapped["OrchestrationExecution"] = relationship(
        "OrchestrationExecution",
        back_populates="callbacks"
    )
    
    def __repr__(self) -> str:
        return f"<OrchestrationCallback(id={self.id}, type={self.callback_type}, status={self.status})>"
```

---

## 3. Repositórios

### 3.1 Repositório de Configurações

#### Arquivo: resync/core/database/repositories/orchestration_config_repo.py

```python
"""
Orchestration Configuration Repository

Provides data access methods for orchestration configurations.
"""
from typing import List, Optional
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from resync.core.database.models.orchestration import OrchestrationConfig


class OrchestrationConfigRepository:
    """
    Repository for orchestration configuration data access.
    
    Provides CRUD operations and queries for orchestration configurations.
    """
    
    def __init__(self, session: AsyncSession):
        """
        Initialize repository with database session.
        
        Args:
            session: Async SQLAlchemy session
        """
        self._session = session
    
    async def create(
        self,
        name: str,
        strategy: str,
        steps: dict,
        description: Optional[str] = None,
        metadata: Optional[dict] = None,
        owner_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        is_global: bool = False,
    ) -> OrchestrationConfig:
        """
        Create a new orchestration configuration.
        
        Args:
            name: Configuration name
            strategy: Execution strategy (sequential, parallel, consensus, fallback)
            steps: Steps definition (JSON)
            description: Optional description
            metadata: Optional metadata
            owner_id: Owner user ID
            tenant_id: Tenant ID for multi-tenancy
            is_global: Whether config is global
            
        Returns:
            Created configuration
        """
        config = OrchestrationConfig(
            name=name,
            strategy=strategy,
            steps=steps,
            description=description,
            metadata=metadata or {},
            owner_id=owner_id,
            tenant_id=tenant_id,
            is_global=is_global,
        )
        self._session.add(config)
        await self._session.commit()
        await self._session.refresh(config)
        return config
    
    async def get_by_id(self, config_id: UUID) -> Optional[OrchestrationConfig]:
        """
        Get configuration by ID.
        
        Args:
            config_id: Configuration UUID
            
        Returns:
            Configuration if found, None otherwise
        """
        result = await self._session.execute(
            select(OrchestrationConfig).where(OrchestrationConfig.id == config_id)
        )
        return result.scalar_one_or_none()
    
    async def get_by_name(
        self, 
        name: str, 
        owner_id: Optional[str] = None
    ) -> Optional[OrchestrationConfig]:
        """
        Get configuration by name, optionally scoped to owner.
        
        Args:
            name: Configuration name
            owner_id: Optional owner filter
            
        Returns:
            Configuration if found, None otherwise
        """
        query = select(OrchestrationConfig).where(OrchestrationConfig.name == name)
        
        if owner_id:
            query = query.where(
                (OrchestrationConfig.owner_id == owner_id) | 
                (OrchestrationConfig.is_global == True)
            )
        
        result = await self._session.execute(query)
        return result.scalar_one_or_none()
    
    async def list_all(
        self,
        owner_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        strategy: Optional[str] = None,
        is_active: bool = True,
        limit: int = 100,
        offset: int = 0,
    ) -> List[OrchestrationConfig]:
        """
        List orchestration configurations with filters.
        
        Args:
            owner_id: Filter by owner
            tenant_id: Filter by tenant
            strategy: Filter by strategy
            is_active: Filter by active status
            limit: Maximum results
            offset: Pagination offset
            
        Returns:
            List of configurations
        """
        query = select(OrchestrationConfig)
        
        if owner_id:
            query = query.where(OrchestrationConfig.owner_id == owner_id)
        
        if tenant_id:
            query = query.where(OrchestrationConfig.tenant_id == tenant_id)
        
        if strategy:
            query = query.where(OrchestrationConfig.strategy == strategy)
        
        if is_active is not None:
            query = query.where(OrchestrationConfig.is_active == is_active)
        
        query = (
            query
            .order_by(OrchestrationConfig.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        
        result = await self._session.execute(query)
        return list(result.scalars().all())
    
    async def update(
        self,
        config_id: UUID,
        **kwargs,
    ) -> Optional[OrchestrationConfig]:
        """
        Update configuration fields.
        
        Args:
            config_id: Configuration ID
            **kwargs: Fields to update
            
        Returns:
            Updated configuration if found, None otherwise
        """
        await self._session.execute(
            update(OrchestrationConfig)
            .where(OrchestrationConfig.id == config_id)
            .values(**kwargs)
        )
        await self._session.commit()
        return await self.get_by_id(config_id)
    
    async def delete(self, config_id: UUID) -> bool:
        """
        Delete configuration (soft delete by setting is_active=False).
        
        Args:
            config_id: Configuration ID
            
        Returns:
            True if deleted, False if not found
        """
        result = await self._session.execute(
            update(OrchestrationConfig)
            .where(OrchestrationConfig.id == config_id)
            .values(is_active=False)
        )
        await self._session.commit()
        return result.rowcount > 0
    
    async def count(
        self,
        owner_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        is_active: bool = True,
    ) -> int:
        """
        Count configurations with filters.
        
        Args:
            owner_id: Filter by owner
            tenant_id: Filter by tenant
            is_active: Filter by active status
            
        Returns:
            Count of matching configurations
        """
        from sqlalchemy import func
        
        query = select(func.count(OrchestrationConfig.id))
        
        if owner_id:
            query = query.where(OrchestrationConfig.owner_id == owner_id)
        
        if tenant_id:
            query = query.where(OrchestrationConfig.tenant_id == tenant_id)
        
        if is_active is not None:
            query = query.where(OrchestrationConfig.is_active == is_active)
        
        result = await self._session.execute(query)
        return result.scalar_one()
```

### 3.2 Repositório de Execuções

#### Arquivo: resync/core/database/repositories/orchestration_execution_repo.py

```python
"""
Orchestration Execution Repository

Provides data access methods for orchestration executions.
"""
from datetime import datetime
from typing import Dict, List, Optional
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from resync.core.database.models.orchestration import (
    OrchestrationExecution,
    OrchestrationStepRun,
)


class OrchestrationExecutionRepository:
    """
    Repository for orchestration execution data access.
    
    Provides CRUD operations and queries for orchestration executions.
    """
    
    def __init__(self, session: AsyncSession):
        """Initialize repository with database session."""
        self._session = session
    
    async def create(
        self,
        trace_id: str,
        config_id: Optional[UUID],
        config_name: str,
        input_data: dict,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        created_by: Optional[str] = None,
        callback_url: Optional[str] = None,
    ) -> OrchestrationExecution:
        """
        Create a new execution record.
        
        Args:
            trace_id: Unique trace ID for correlation
            config_id: Configuration ID
            config_name: Configuration name
            input_data: Input data for execution
            user_id: User ID
            session_id: Session ID
            tenant_id: Tenant ID
            created_by: Creator user ID
            callback_url: Callback URL for completion notification
            
        Returns:
            Created execution
        """
        execution = OrchestrationExecution(
            trace_id=trace_id,
            config_id=config_id,
            config_name=config_name,
            input_data=input_data,
            status="pending",
            user_id=user_id,
            session_id=session_id,
            tenant_id=tenant_id,
            created_by=created_by,
            callback_url=callback_url,
        )
        self._session.add(execution)
        await self._session.commit()
        await self._session.refresh(execution)
        return execution
    
    async def get_by_id(self, execution_id: UUID) -> Optional[OrchestrationExecution]:
        """Get execution by ID."""
        result = await self._session.execute(
            select(OrchestrationExecution)
            .where(OrchestrationExecution.id == execution_id)
        )
        return result.scalar_one_or_none()
    
    async def get_by_trace_id(self, trace_id: str) -> Optional[OrchestrationExecution]:
        """Get execution by trace ID."""
        result = await self._session.execute(
            select(OrchestrationExecution)
            .where(OrchestrationExecution.trace_id == trace_id)
        )
        return result.scalar_one_or_none()
    
    async def update_status(
        self,
        execution_id: UUID,
        status: str,
        output: Optional[dict] = None,
        completed_at: Optional[datetime] = None,
        total_latency_ms: Optional[int] = None,
        estimated_cost: Optional[float] = None,
    ) -> Optional[OrchestrationExecution]:
        """
        Update execution status and metrics.
        
        Args:
            execution_id: Execution ID
            status: New status
            output: Optional output data
            completed_at: Completion timestamp
            total_latency_ms: Total latency in milliseconds
            estimated_cost: Estimated cost
            
        Returns:
            Updated execution
        """
        values = {"status": status}
        
        if output is not None:
            values["output_data"] = output
        
        if completed_at:
            values["completed_at"] = completed_at
        
        if total_latency_ms is not None:
            values["total_latency_ms"] = total_latency_ms
        
        if estimated_cost is not None:
            values["estimated_cost"] = estimated_cost
        
        if status == "running" and not output:
            values["started_at"] = datetime.utcnow()
        
        await self._session.execute(
            update(OrchestrationExecution)
            .where(OrchestrationExecution.id == execution_id)
            .values(**values)
        )
        await self._session.commit()
        return await self.get_by_id(execution_id)
    
    async def list_by_config(
        self,
        config_id: UUID,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[OrchestrationExecution]:
        """
        List executions for a configuration.
        
        Args:
            config_id: Configuration ID
            status: Optional status filter
            limit: Maximum results
            offset: Pagination offset
            
        Returns:
            List of executions
        """
        query = (
            select(OrchestrationExecution)
            .where(OrchestrationExecution.config_id == config_id)
            .order_by(OrchestrationExecution.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        
        if status:
            query = query.where(OrchestrationExecution.status == status)
        
        result = await self._session.execute(query)
        return list(result.scalars().all())
    
    async def list_by_user(
        self,
        user_id: str,
        status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[OrchestrationExecution]:
        """List executions for a user."""
        query = (
            select(OrchestrationExecution)
            .where(OrchestrationExecution.user_id == user_id)
            .order_by(OrchestrationExecution.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        
        if status:
            query = query.where(OrchestrationExecution.status == status)
        
        result = await self._session.execute(query)
        return list(result.scalars().all())
    
    async def get_step_runs(
        self,
        execution_id: UUID,
    ) -> List[OrchestrationStepRun]:
        """Get all step runs for an execution."""
        result = await self._session.execute(
            select(OrchestrationStepRun)
            .where(OrchestrationStepRun.execution_id == execution_id)
            .order_by(OrchestrationStepRun.step_index)
        )
        return list(result.scalars().all())


class OrchestrationStepRunRepository:
    """Repository for step run data access."""
    
    def __init__(self, session: AsyncSession):
        self._session = session
    
    async def create(
        self,
        execution_id: UUID,
        step_index: int,
        step_id: str,
        step_name: Optional[str] = None,
        dependencies_json: Optional[list] = None,
    ) -> OrchestrationStepRun:
        """Create a new step run record."""
        step_run = OrchestrationStepRun(
            execution_id=execution_id,
            step_index=step_index,
            step_id=step_id,
            step_name=step_name,
            status="pending",
            dependencies_json=dependencies_json or [],
        )
        self._session.add(step_run)
        await self._session.commit()
        await self._session.refresh(step_run)
        return step_run
    
    async def update_status(
        self,
        step_run_id: UUID,
        status: str,
        output: Optional[dict] = None,
        output_truncated: Optional[str] = None,
        error_message: Optional[str] = None,
        error_trace: Optional[str] = None,
        latency_ms: Optional[int] = None,
        retry_count: Optional[int] = None,
        token_count: Optional[int] = None,
        estimated_cost: Optional[float] = None,
    ) -> Optional[OrchestrationStepRun]:
        """Update step run status and metrics."""
        values = {"status": status}
        
        if output is not None:
            values["output"] = output
        
        if output_truncated is not None:
            values["output_truncated"] = output_truncated
        
        if error_message is not None:
            values["error_message"] = error_message
        
        if error_trace is not None:
            values["error_trace"] = error_trace
        
        if latency_ms is not None:
            values["latency_ms"] = latency_ms
        
        if retry_count is not None:
            values["retry_count"] = retry_count
        
        if token_count is not None:
            values["token_count"] = token_count
        
        if estimated_cost is not None:
            values["estimated_cost"] = estimated_cost
        
        if status == "running" and latency_ms is None:
            values["started_at"] = datetime.utcnow()
        
        if status in ("completed", "failed", "skipped"):
            values["completed_at"] = datetime.utcnow()
        
        await self._session.execute(
            update(OrchestrationStepRun)
            .where(OrchestrationStepRun.id == step_run_id)
            .values(**values)
        )
        await self._session.commit()
        
        result = await self._session.execute(
            select(OrchestrationStepRun)
            .where(OrchestrationStepRun.id == step_run_id)
        )
        return result.scalar_one_or_none()
```

---

## 4. Motor de Orquestração

### 4.1 Arquitetura do Runner

#### Arquivo: resync/core/orchestration/runner.py

```python
"""
Orchestration Runner

Core orchestration engine that executes multi-step workflows
using various strategies (sequential, parallel, consensus, fallback).
"""
import asyncio
import json
import time
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set

from sqlalchemy.ext.asyncio import AsyncSession

from resync.core.database.repositories.orchestration_execution_repo import (
    OrchestrationExecutionRepository,
    OrchestrationStepRunRepository,
)
from resync.core.database.repositories.orchestration_config_repo import (
    OrchestrationConfigRepository,
)
from resync.core.orchestration.events import OrchestrationEventBus, OrchestrationEvent
from resync.core.orchestration.strategies import (
    ExecutionStrategy,
    SequentialStrategy,
    ParallelStrategy,
    ConsensusStrategy,
    FallbackStrategy,
)
from resync.core.structured_logger import get_logger

logger = get_logger(__name__)


class OrchestrationContext:
    """
    Context object passed through the orchestration execution.
    
    Contains execution state, shared data between steps, and
    helper methods for the executor.
    """
    
    def __init__(
        self,
        execution_id: str,
        trace_id: str,
        config: dict,
        input_data: dict,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ):
        self.execution_id = execution_id
        self.trace_id = trace_id
        self.config = config
        self.input_data = input_data
        self.user_id = user_id
        self.session_id = session_id
        
        # Shared state between steps
        self.step_outputs: Dict[str, dict] = {}
        
        # Execution metadata
        self.metadata: Dict[str, Any] = {}
        
        # Metrics
        self.total_latency_ms = 0
        self.total_cost = 0.0
        self.total_tokens = 0
    
    def get_step_input(self, step: dict) -> dict:
        """
        Get input for a step, including outputs from dependencies.
        
        Args:
            step: Step definition
            
        Returns:
            Combined input data for the step
        """
        # Start with global input
        input_data = dict(self.input_data)
        
        # Add dependency outputs
        dependencies = step.get("dependencies", [])
        for dep_id in dependencies:
            if dep_id in self.step_outputs:
                dep_output = self.step_outputs[dep_id]
                input_data[f"dep_{dep_id}"] = dep_output
                # Merge dependency output into main input
                if isinstance(dep_output, dict):
                    input_data.update(dep_output)
        
        return input_data
    
    def store_step_output(self, step_id: str, output: dict):
        """Store output from a step for dependency access."""
        self.step_outputs[step_id] = output
    
    def add_metrics(
        self,
        latency_ms: int = 0,
        cost: float = 0.0,
        tokens: int = 0,
    ):
        """Add metrics to the context."""
        self.total_latency_ms += latency_ms
        self.total_cost += cost
        self.total_tokens += tokens


class StepResult:
    """Result from executing a single step."""
    
    def __init__(
        self,
        step_id: str,
        step_index: int,
        status: str,
        output: Optional[dict] = None,
        error: Optional[str] = None,
        latency_ms: int = 0,
        retry_count: int = 0,
        tokens: int = 0,
        cost: float = 0.0,
    ):
        self.step_id = step_id
        self.step_index = step_index
        self.status = status
        self.output = output or {}
        self.error = error
        self.latency_ms = latency_ms
        self.retry_count = retry_count
        self.tokens = tokens
        self.cost = cost


class OrchestrationRunner:
    """
    Main orchestration runner that coordinates execution of multi-step workflows.
    
    This class manages the execution lifecycle, including:
    - Loading configuration
    - Creating execution records
    - Selecting and running execution strategy
    - Handling errors and retries
    - Broadcasting events
    - Updating execution status
    """
    
    # Strategy mapping
    STRATEGIES = {
        "sequential": SequentialStrategy,
        "parallel": ParallelStrategy,
        "consensus": ConsensusStrategy,
        "fallback": FallbackStrategy,
    }
    
    def __init__(
        self,
        session: AsyncSession,
        event_bus: Optional[OrchestrationEventBus] = None,
    ):
        """
        Initialize the orchestration runner.
        
        Args:
            session: Database session
            event_bus: Optional event bus for broadcasting
        """
        self._session = session
        self._event_bus = event_bus
        
        # Repositories
        self._config_repo = OrchestrationConfigRepository(session)
        self._execution_repo = OrchestrationExecutionRepository(session)
        self._step_repo = OrchestrationStepRunRepository(session)
        
        # Agent executor (hybrid router)
        self._agent_executor: Optional[Callable] = None
    
    def set_agent_executor(self, executor: Callable):
        """
        Set the agent executor function.
        
        The executor should be an async function that takes:
        - agent_id: str
        - input_data: dict
        - context: OrchestrationContext
        
        And returns a StepResult.
        """
        self._agent_executor = executor
    
    async def execute(
        self,
        config_id: str,
        input_data: dict,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        callback_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute an orchestration by configuration ID.
        
        Args:
            config_id: Configuration ID (UUID or name)
            input_data: Input data for the orchestration
            user_id: User ID
            session_id: Session ID
            callback_url: URL to call on completion
            
        Returns:
            Execution result with trace_id and status
        """
        # Generate trace ID
        trace_id = f"orch-{uuid.uuid4().hex[:12]}"
        
        # Load configuration
        config = await self._load_config(config_id)
        if not config:
            raise ValueError(f"Configuration not found: {config_id}")
        
        # Create execution record
        execution = await self._execution_repo.create(
            trace_id=trace_id,
            config_id=config.id,
            config_name=config.name,
            input_data=input_data,
            user_id=user_id,
            session_id=session_id,
            created_by=user_id,
            callback_url=callback_url,
        )
        
        # Create orchestration context
        context = OrchestrationContext(
            execution_id=str(execution.id),
            trace_id=trace_id,
            config=config.steps,
            input_data=input_data,
            user_id=user_id,
            session_id=session_id,
        )
        
        # Get strategy
        strategy_class = self.STRATEGIES.get(config.strategy)
        if not strategy_class:
            raise ValueError(f"Unknown strategy: {config.strategy}")
        
        strategy = strategy_class(
            session=self._session,
            step_repo=self._step_repo,
            agent_executor=self._agent_executor,
            event_bus=self._event_bus,
        )
        
        # Update status to running
        await self._execution_repo.update_status(
            execution_id=execution.id,
            status="running",
        )
        
        # Broadcast start event
        await self._broadcast_event(
            OrchestrationEvent(
                type="execution_started",
                trace_id=trace_id,
                execution_id=str(execution.id),
                config_name=config.name,
            )
        )
        
        try:
            # Execute the orchestration
            start_time = time.time()
            result = await strategy.execute(
                steps=config.steps.get("steps", []),
                context=context,
                execution_id=execution.id,
            )
            execution_time_ms = int((time.time() - start_time) * 1000)
            
            # Update execution with final status
            final_status = "completed" if result["success"] else "failed"
            await self._execution_repo.update_status(
                execution_id=execution.id,
                status=final_status,
                output=result.get("output", {}),
                completed_at=datetime.utcnow(),
                total_latency_ms=execution_time_ms,
                estimated_cost=context.total_cost,
            )
            
            # Broadcast completion event
            await self._broadcast_event(
                OrchestrationEvent(
                    type="execution_completed",
                    trace_id=trace_id,
                    execution_id=str(execution.id),
                    status=final_status,
                    latency_ms=execution_time_ms,
                )
            )
            
            return {
                "trace_id": trace_id,
                "execution_id": str(execution.id),
                "status": final_status,
                "output": result.get("output", {}),
                "metrics": {
                    "latency_ms": execution_time_ms,
                    "cost": context.total_cost,
                    "tokens": context.total_tokens,
                },
            }
            
        except Exception as e:
            logger.error(
                "orchestration_failed",
                trace_id=trace_id,
                error=str(e),
            )
            
            # Update execution status to failed
            await self._execution_repo.update_status(
                execution_id=execution.id,
                status="failed",
                output={"error": str(e)},
                completed_at=datetime.utcnow(),
            )
            
            # Broadcast failure event
            await self._broadcast_event(
                OrchestrationEvent(
                    type="execution_failed",
                    trace_id=trace_id,
                    execution_id=str(execution.id),
                    error=str(e),
                )
            )
            
            raise
    
    async def _load_config(self, config_identifier: str):
        """Load configuration by ID or name."""
        import uuid
        from sqlalchemy import select
        
        # Try to parse as UUID
        try:
            config_uuid = uuid.UUID(config_identifier)
            config = await self._config_repo.get_by_id(config_uuid)
            if config:
                return config
        except ValueError:
            pass
        
        # Try to find by name
        config = await self._config_repo.get_by_name(config_identifier)
        return config
    
    async def _broadcast_event(self, event: OrchestrationEvent):
        """Broadcast an orchestration event."""
        if self._event_bus:
            await self._event_bus.publish(event)
    
    async def get_execution_status(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the current status of an execution.
        
        Args:
            trace_id: Execution trace ID
            
        Returns:
            Execution status and details
        """
        execution = await self._execution_repo.get_by_trace_id(trace_id)
        if not execution:
            return None
        
        step_runs = await self._execution_repo.get_step_runs(execution.id)
        
        return {
            "trace_id": execution.trace_id,
            "status": execution.status,
            "config_name": execution.config_name,
            "input": execution.input_data,
            "output": execution.output_data,
            "metrics": {
                "latency_ms": execution.total_latency_ms,
                "cost": execution.estimated_cost,
            },
            "steps": [
                {
                    "step_id": sr.step_id,
                    "step_name": sr.step_name,
                    "status": sr.status,
                    "latency_ms": sr.latency_ms,
                    "error": sr.error_message,
                }
                for sr in step_runs
            ],
            "created_at": execution.created_at.isoformat() if execution.created_at else None,
            "started_at": execution.started_at.isoformat() if execution.started_at else None,
            "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
        }


class OrchestrationEvent:
    """Event emitted during orchestration execution."""
    
    def __init__(
        self,
        type: str,
        trace_id: str,
        execution_id: str,
        **kwargs
    ):
        self.type = type
        self.trace_id = trace_id
        self.execution_id = execution_id
        self.timestamp = datetime.utcnow()
        self.data = kwargs
    
    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "trace_id": self.trace_id,
            "execution_id": self.execution_id,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
        }
```

### 4.2 Estratégias de Execução

#### Arquivo: resync/core/orchestration/strategies.py

```python
"""
Execution Strategies for Orchestration

Implements different execution patterns:
- Sequential: Run steps one after another
- Parallel: Run independent steps simultaneously
- Consensus: Aggregate results from multiple agents
- Fallback: Try primary, fallback on failure
"""
import asyncio
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set

from sqlalchemy.ext.asyncio import AsyncSession

from resync.core.database.repositories.orchestration_execution_repo import (
    OrchestrationStepRunRepository,
)
from resync.core.orchestration.runner import OrchestrationContext
from resync.core.orchestration.events import OrchestrationEventBus


class ExecutionStrategy(ABC):
    """Abstract base class for execution strategies."""
    
    def __init__(
        self,
        session: AsyncSession,
        step_repo: OrchestrationStepRunRepository,
        agent_executor: Optional[callable],
        event_bus: Optional[OrchestrationEventBus] = None,
    ):
        self._session = session
        self._step_repo = step_repo
        self._agent_executor = agent_executor
        self._event_bus = event_bus
    
    @abstractmethod
    async def execute(
        self,
        steps: List[dict],
        context: OrchestrationContext,
        execution_id: str,
    ) -> Dict[str, Any]:
        """
        Execute steps according to the strategy.
        
        Args:
            steps: List of step definitions
            context: Orchestration context
            execution_id: Execution ID
            
        Returns:
            Result dictionary with success flag and output
        """
        pass
    
    async def _execute_step(
        self,
        step: dict,
        context: OrchestrationContext,
        execution_id: str,
        retry_config: Optional[dict] = None,
    ) -> Dict[str, Any]:
        """
        Execute a single step with retry logic.
        
        Args:
            step: Step definition
            context: Orchestration context
            execution_id: Execution ID
            retry_config: Retry configuration
            
        Returns:
            Step result
        """
        step_id = step["id"]
        step_name = step.get("name", step_id)
        agent = step.get("agent")
        
        # Get retry configuration
        max_attempts = retry_config.get("max_attempts", 1) if retry_config else 1
        backoff_ms = retry_config.get("backoff_ms", 0) if retry_config else 0
        
        # Create step run record
        step_run = await self._step_repo.create(
            execution_id=execution_id,
            step_index=len(context.step_outputs),
            step_id=step_id,
            step_name=step_name,
            dependencies_json=step.get("dependencies", []),
        )
        
        # Update status to running
        await self._step_repo.update_status(
            step_run_id=step_run.id,
            status="running",
        )
        
        # Broadcast step started event
        await self._broadcast_event(
            OrchestrationEvent(
                type="step_started",
                trace_id=context.trace_id,
                execution_id=context.execution_id,
                step_id=step_id,
                step_name=step_name,
                agent=agent,
            )
        )
        
        # Retry loop
        last_error = None
        for attempt in range(max_attempts):
            if attempt > 0:
                # Wait before retry
                await asyncio.sleep(backoff_ms / 1000 * (attempt + 1))
            
            try:
                start_time = time.time()
                
                # Execute agent
                if self._agent_executor:
                    result = await self._agent_executor(
                        agent_id=agent,
                        input_data=context.get_step_input(step),
                        context=context,
                    )
                else:
                    # Fallback: execute directly (for testing)
                    result = await self._execute_agent_direct(agent, context.get_step_input(step))
                
                latency_ms = int((time.time() - start_time) * 1000)
                
                # Update step run with success
                await self._step_repo.update_status(
                    step_run_id=step_run.id,
                    status="completed",
                    output=result.get("output", {}),
                    latency_ms=latency_ms,
                    retry_count=attempt,
                    token_count=result.get("tokens", 0),
                    estimated_cost=result.get("cost", 0.0),
                )
                
                # Store output for dependencies
                context.store_step_output(step_id, result.get("output", {}))
                context.add_metrics(
                    latency_ms=latency_ms,
                    cost=result.get("cost", 0.0),
                    tokens=result.get("tokens", 0),
                )
                
                # Broadcast step completed event
                await self._broadcast_event(
                    OrchestrationEvent(
                        type="step_completed",
                        trace_id=context.trace_id,
                        execution_id=context.execution_id,
                        step_id=step_id,
                        latency_ms=latency_ms,
                    )
                )
                
                return {
                    "success": True,
                    "output": result.get("output", {}),
                    "latency_ms": latency_ms,
                }
                
            except Exception as e:
                last_error = str(e)
                # Broadcast step error event
                await self._broadcast_event(
                    OrchestrationEvent(
                        type="step_error",
                        trace_id=context.trace_id,
                        execution_id=context.execution_id,
                        step_id=step_id,
                        error=last_error,
                        attempt=attempt + 1,
                    )
                )
        
        # All retries failed
        on_failure = step.get("on_failure", "abort")
        
        if on_failure == "abort":
            # Mark step as failed
            await self._step_repo.update_status(
                step_run_id=step_run.id,
                status="failed",
                error_message=last_error,
                latency_ms=0,
                retry_count=max_attempts,
            )
            raise Exception(f"Step {step_id} failed: {last_error}")
        
        elif on_failure == "continue":
            # Mark step as failed but continue
            await self._step_repo.update_status(
                step_run_id=step_run.id,
                status="failed",
                error_message=last_error,
                latency_ms=0,
                retry_count=max_attempts,
            )
            context.store_step_output(step_id, {"error": last_error})
            return {
                "success": True,  # Continue despite error
                "output": {"error": last_error},
            }
        
        elif on_failure == "fallback":
            # This is handled by the FallbackStrategy
            raise Exception(f"Step {step_id} failed, triggering fallback: {last_error}")
        
        return {
            "success": False,
            "error": last_error,
        }
    
    async def _execute_agent_direct(
        self,
        agent_id: str,
        input_data: dict,
    ) -> Dict[str, Any]:
        """Direct agent execution for testing."""
        # Simulate agent execution
        await asyncio.sleep(0.1)
        return {
            "output": {
                "agent": agent_id,
                "result": f"Processed by {agent_id}",
                "input": input_data,
            },
            "tokens": 10,
            "cost": 0.001,
        }
    
    async def _broadcast_event(self, event: OrchestrationEvent):
        """Broadcast an event."""
        if self._event_bus:
            await self._event_bus.publish(event)


class SequentialStrategy(ExecutionStrategy):
    """
    Sequential execution strategy.
    
    Executes steps one after another in order.
    Each step must complete before the next starts.
    """
    
    async def execute(
        self,
        steps: List[dict],
        context: OrchestrationContext,
        execution_id: str,
    ) -> Dict[str, Any]:
        """Execute steps sequentially."""
        results = []
        
        for i, step in enumerate(steps):
            # Check dependencies are met
            dependencies = set(step.get("dependencies", []))
            completed = set(context.step_outputs.keys())
            
            if not dependencies.issubset(completed):
                missing = dependencies - completed
                return {
                    "success": False,
                    "error": f"Dependencies not met for step {step['id']}: {missing}",
                }
            
            # Execute step
            result = await self._execute_step(
                step=step,
                context=context,
                execution_id=execution_id,
                retry_config=step.get("retry"),
            )
            
            results.append(result)
            
            if not result.get("success"):
                return {
                    "success": False,
                    "error": result.get("error"),
                    "partial_results": results,
                }
        
        # Collect final output (output from last step)
        final_output = {}
        if results:
            final_output = results[-1].get("output", {})
        
        return {
            "success": True,
            "output": final_output,
            "all_results": results,
        }


class ParallelStrategy(ExecutionStrategy):
    """
    Parallel execution strategy.
    
    Executes independent steps simultaneously.
    Steps with dependencies wait for their dependencies to complete.
    """
    
    async def execute(
        self,
        steps: List[dict],
        context: OrchestrationContext,
        execution_id: str,
    ) -> Dict[str, Any]:
        """Execute steps in parallel where possible."""
        step_map = {step["id"]: step for step in steps}
        completed: Set[str] = set()
        results = {}
        
        # Build dependency graph
        pending = set(step_map.keys())
        
        while pending:
            # Find steps whose dependencies are satisfied
            ready = []
            for step_id in pending:
                step = step_map[step_id]
                deps = set(step.get("dependencies", []))
                if deps.issubset(completed):
                    ready.append(step)
            
            if not ready:
                # No steps can run - circular dependency or broken graph
                return {
                    "success": False,
                    "error": f"Circular dependency detected. Pending: {pending}, Completed: {completed}",
                }
            
            # Execute ready steps in parallel
            tasks = [
                self._execute_step(
                    step=step,
                    context=context,
                    execution_id=execution_id,
                    retry_config=step.get("retry"),
                )
                for step in ready
            ]
            
            step_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for step, result in zip(ready, step_results):
                if isinstance(result, Exception):
                    results[step["id"]] = {
                        "success": False,
                        "error": str(result),
                    }
                    # On failure, stop execution
                    return {
                        "success": False,
                        "error": f"Step {step['id']} failed: {result}",
                        "partial_results": results,
                    }
                
                results[step["id"]] = result
                completed.add(step["id"])
                pending.discard(step["id"])
        
        # Collect final output
        final_output = {}
        if results:
            # Use output from last completed step
            final_output = list(results.values())[-1].get("output", {})
        
        return {
            "success": True,
            "output": final_output,
            "all_results": results,
        }


class ConsensusStrategy(ExecutionStrategy):
    """
    Consensus execution strategy.
    
    Runs multiple agents in parallel and aggregates their results
    based on a consensus threshold.
    """
    
    def __init__(
        self,
        *args,
        consensus_threshold: float = 0.8,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.consensus_threshold = consensus_threshold
    
    async def execute(
        self,
        steps: List[dict],
        context: OrchestrationContext,
        execution_id: str,
    ) -> Dict[str, Any]:
        """Execute steps with consensus aggregation."""
        
        # Group steps by consensus group
        consensus_groups = self._group_steps_for_consensus(steps)
        
        all_results = {}
        
        for group_id, group_steps in consensus_groups.items():
            if len(group_steps) == 1:
                # Single step - execute normally
                step = group_steps[0]
                result = await self._execute_step(
                    step=step,
                    context=context,
                    execution_id=execution_id,
                    retry_config=step.get("retry"),
                )
                all_results[step["id"]] = result
            else:
                # Multiple steps - run in parallel and aggregate
                result = await self._execute_consensus_group(
                    group_id=group_id,
                    steps=group_steps,
                    context=context,
                    execution_id=execution_id,
                )
                all_results[group_id] = result
        
        # Collect final output
        final_output = {}
        if all_results:
            final_output = list(all_results.values())[-1].get("output", {})
        
        return {
            "success": True,
            "output": final_output,
            "all_results": all_results,
        }
    
    def _group_steps_for_consensus(self, steps: List[dict]) -> Dict[str, List[dict]]:
        """Group steps by consensus group."""
        groups = {}
        
        for step in steps:
            group_id = step.get("consensus_group", step["id"])
            if group_id not in groups:
                groups[group_id] = []
            groups[group_id].append(step)
        
        return groups
    
    async def _execute_consensus_group(
        self,
        group_id: str,
        steps: List[dict],
        context: OrchestrationContext,
        execution_id: str,
    ) -> Dict[str, Any]:
        """Execute multiple steps and aggregate results."""
        
        # Run all steps in parallel
        tasks = [
            self._execute_step(
                step=step,
                context=context,
                execution_id=execution_id,
                retry_config=step.get("retry"),
            )
            for step in steps
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter successful results
        successful = [r for r in results if not isinstance(r, Exception) and r.get("success")]
        
        if len(successful) / len(steps) >= self.consensus_threshold:
            # Consensus reached - aggregate results
            aggregated = self._aggregate_results(successful)
            return {
                "success": True,
                "output": aggregated,
                "consensus": True,
                "agreed_count": len(successful),
                "total_count": len(steps),
            }
        else:
            # No consensus - return best effort or error
            return {
                "success": False,
                "output": {},
                "consensus": False,
                "agreed_count": len(successful),
                "total_count": len(steps),
                "error": "Consensus not reached",
            }
    
    def _aggregate_results(self, results: List[dict]) -> dict:
        """Aggregate results from multiple agents."""
        # Simple aggregation - merge all outputs
        aggregated = {}
        
        for result in results:
            output = result.get("output", {})
            aggregated.update(output)
        
        return aggregated


class FallbackStrategy(ExecutionStrategy):
    """
    Fallback execution strategy.
    
    Tries primary steps first, falls back to alternative
    steps on failure.
    """
    
    async def execute(
        self,
        steps: List[dict],
        context: OrchestrationContext,
        execution_id: str,
    ) -> Dict[str, Any]:
        """Execute steps with fallback logic."""
        
        # Find primary and fallback steps
        primary_steps = []
        fallback_map = {}
        
        for step in steps:
            if step.get("is_fallback"):
                # This is a fallback step
                fallback_for = step.get("fallback_for")
                if fallback_for:
                    if fallback_for not in fallback_map:
                        fallback_map[fallback_for] = []
                    fallback_map[fallback_for].append(step)
            else:
                primary_steps.append(step)
        
        # Execute primary steps
        results = {}
        
        for step in primary_steps:
            step_id = step["id"]
            
            try:
                result = await self._execute_step(
                    step=step,
                    context=context,
                    execution_id=execution_id,
                    retry_config=step.get("retry"),
                )
                results[step_id] = result
                
                if not result.get("success"):
                    # Try fallback
                    fallbacks = fallback_map.get(step_id, [])
                    fallback_result = await self._try_fallbacks(
                        fallbacks=fallbacks,
                        context=context,
                        execution_id=execution_id,
                        original_step=step,
                    )
                    results[step_id] = fallback_result
            
            except Exception as e:
                # Try fallback on exception
                fallbacks = fallback_map.get(step_id, [])
                fallback_result = await self._try_fallbacks(
                    fallbacks=fallbacks,
                    context=context,
                    execution_id=execution_id,
                    original_step=step,
                )
                results[step_id] = fallback_result
        
        # Collect final output
        final_output = {}
        if results:
            final_output = list(results.values())[-1].get("output", {})
        
        return {
            "success": True,
            "output": final_output,
            "all_results": results,
        }
    
    async def _try_fallbacks(
        self,
        fallbacks: List[dict],
        context: OrchestrationContext,
        execution_id: str,
        original_step: dict,
    ) -> Dict[str, Any]:
        """Try fallback steps."""
        
        for fallback in fallbacks:
            try:
                result = await self._execute_step(
                    step=fallback,
                    context=context,
                    execution_id=execution_id,
                    retry_config=fallback.get("retry"),
                )
                
                if result.get("success"):
                    return {
                        "success": True,
                        "output": result.get("output", {}),
                        "fallback_used": fallback["id"],
                    }
            except Exception:
                continue
        
        # All fallbacks failed
        return {
            "success": False,
            "output": {},
            "error": f"All fallbacks failed for {original_step['id']}",
        }
```

### 4.3 Event Bus

#### Arquivo: resync/core/orchestration/events.py

```python
"""
Orchestration Event Bus

Provides event-driven communication for orchestration
execution, enabling real-time updates and WebSocket broadcasting.
"""
import asyncio
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Callable, Dict, List, Optional

from resync.core.structured_logger import get_logger

logger = get_logger(__name__)


class EventSubscriber:
    """Event subscriber with callback."""
    
    def __init__(self, event_type: str, callback: Callable):
        self.event_type = event_type
        self.callback = callback
        self.created_at = datetime.utcnow()


class OrchestrationEventBus:
    """
    Event bus for orchestration events.
    
    Handles publishing and subscribing to orchestration events,
    including WebSocket broadcasting.
    """
    
    def __init__(self, websocket_manager=None):
        """
        Initialize event bus.
        
        Args:
            websocket_manager: Optional WebSocket manager for broadcasting
        """
        self._subscribers: Dict[str, List[EventSubscriber]] = {}
        self._websocket_manager = websocket_manager
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        self._task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start the event processing loop."""
        if self._running:
            return
        
        self._running = True
        self._task = asyncio.create_task(self._process_events())
        logger.info("OrchestrationEventBus started")
    
    async def stop(self):
        """Stop the event processing loop."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("OrchestrationEventBus stopped")
    
    async def subscribe(self, event_type: str, callback: Callable):
        """
        Subscribe to events of a specific type.
        
        Args:
            event_type: Type of event to subscribe to
            callback: Async function to call when event occurs
        """
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        
        subscriber = EventSubscriber(event_type, callback)
        self._subscribers[event_type].append(subscriber)
        
        logger.debug("Subscribed to event type: %s", event_type)
    
    async def unsubscribe(self, event_type: str, callback: Callable):
        """Unsubscribe from events."""
        if event_type in self._subscribers:
            self._subscribers[event_type] = [
                s for s in self._subscribers[event_type]
                if s.callback != callback
            ]
    
    async def publish(self, event: "OrchestrationEvent"):
        """
        Publish an event to all subscribers.
        
        Args:
            event: Event to publish
        """
        await self._event_queue.put(event)
    
    async def _process_events(self):
        """Process events from the queue."""
        while self._running:
            try:
                event = await asyncio.wait_for(
                    self._event_queue.get(),
                    timeout=1.0
                )
                
                await self._handle_event(event)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error("Error processing event: %s", e)
    
    async def _handle_event(self, event: "OrchestrationEvent"):
        """Handle a single event."""
        event_type = event.type
        
        # Call subscribers
        if event_type in self._subscribers:
            for subscriber in self._subscribers[event_type]:
                try:
                    await subscriber.callback(event)
                except Exception as e:
                    logger.error(
                        "Error in event subscriber: %s, error: %s",
                        subscriber.event_type,
                        e
                    )
        
        # Also call wildcard subscribers
        if "*" in self._subscribers:
            for subscriber in self._subscribers["*"]:
                try:
                    await subscriber.callback(event)
                except Exception as e:
                    logger.error(
                        "Error in wildcard event subscriber: %s",
                        e
                    )
        
        # Broadcast via WebSocket if available
        if self._websocket_manager:
            try:
                await self._broadcast_websocket(event)
            except Exception as e:
                logger.error("Error broadcasting to WebSocket: %s", e)
    
    async def _broadcast_websocket(self, event: "OrchestrationEvent"):
        """Broadcast event via WebSocket."""
        from resync.api.websocket.handlers import manager
        
        message = {
            "type": f"orchestration_{event.type}",
            "trace_id": event.trace_id,
            "execution_id": event.execution_id,
            "timestamp": event.timestamp.isoformat(),
            "data": event.data,
        }
        
        # Broadcast to all connected clients
        import json
        await manager.broadcast_to_all(json.dumps(message))


class OrchestrationEvent:
    """Event emitted during orchestration execution."""
    
    def __init__(
        self,
        type: str,
        trace_id: str,
        execution_id: str,
        **kwargs
    ):
        self.type = type
        self.trace_id = trace_id
        self.execution_id = execution_id
        self.timestamp = datetime.utcnow()
        self.data = kwargs
    
    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "trace_id": self.trace_id,
            "execution_id": self.execution_id,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
        }
```

---

## 5. API Endpoints

### 5.1 Routes de Orquestração

#### Arquivo: resync/api/routes/orchestration.py

```python
"""
Orchestration API Routes

Provides REST API endpoints for orchestration management
and execution.
"""
from typing import List, Optional
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from pydantic import BaseModel, Field

from resync.api.dependencies import get_current_user, get_logger
from resync.core.database.engine import get_session
from resync.core.database.repositories.orchestration_config_repo import (
    OrchestrationConfigRepository,
)
from resync.core.database.repositories.orchestration_execution_repo import (
    OrchestrationExecutionRepository,
)
from resync.core.orchestration.runner import OrchestrationRunner


router = APIRouter(prefix="/api/v1/orchestrations", tags=["orchestration"])


# =============================================================================
# Request/Response Models
# =============================================================================

class StepDefinition(BaseModel):
    """Step definition in orchestration config."""
    
    id: str = Field(..., description="Unique step identifier")
    name: str = Field(..., description="Human-readable step name")
    agent: str = Field(..., description="Agent to execute")
    timeout_ms: int = Field(default=10000, description="Step timeout in milliseconds")
    retry: Optional[dict] = Field(default=None, description="Retry configuration")
    dependencies: List[str] = Field(default_factory=list, description="Step dependencies")
    output_key: str = Field(..., description="Key to store output")
    on_failure: str = Field(default="abort", description="Failure action: abort, continue, fallback")
    is_fallback: bool = Field(default=False, description="Is this a fallback step")
    fallback_for: Optional[str] = Field(default=None, description="Step ID this is fallback for")
    consensus_group: Optional[str] = Field(default=None, description="Consensus group ID")


class OrchestrationConfigCreate(BaseModel):
    """Request model for creating orchestration config."""
    
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    strategy: str = Field(..., description="Execution strategy: sequential, parallel, consensus, fallback")
    steps: dict = Field(..., description="Steps definition")
    metadata: Optional[dict] = None
    is_global: bool = False


class OrchestrationConfigUpdate(BaseModel):
    """Request model for updating orchestration config."""
    
    name: Optional[str] = None
    description: Optional[str] = None
    steps: Optional[dict] = None
    metadata: Optional[dict] = None
    is_active: Optional[bool] = None


class OrchestrationExecuteRequest(BaseModel):
    """Request model for executing orchestration."""
    
    input: dict = Field(default_factory=dict, description="Input data for orchestration")
    callback_url: Optional[str] = None


class StepStatusResponse(BaseModel):
    """Step execution status."""
    
    step_id: str
    step_name: str
    status: str
    latency_ms: Optional[int]
    error: Optional[str]


class ExecutionStatusResponse(BaseModel):
    """Execution status response."""
    
    trace_id: str
    status: str
    config_name: str
    input: dict
    output: dict
    metrics: dict
    steps: List[StepStatusResponse]
    created_at: Optional[str]
    started_at: Optional[str]
    completed_at: Optional[str]


# =============================================================================
# Configuration Endpoints
# =============================================================================

@router.post("/configs", status_code=status.HTTP_201_CREATED)
async def create_orchestration_config(
    config_data: OrchestrationConfigCreate,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
    logger=Depends(get_logger),
):
    """
    Create a new orchestration configuration.
    
    Defines a new orchestration workflow with steps, agents, and execution strategy.
    """
    async with get_session() as session:
        repo = OrchestrationConfigRepository(session)
        
        # Check if name already exists
        existing = await repo.get_by_name(config_data.name, current_user.get("user_id"))
        if existing:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Configuration with name '{config_data.name}' already exists"
            )
        
        # Create configuration
        config = await repo.create(
            name=config_data.name,
            strategy=config_data.strategy,
            steps=config_data.steps,
            description=config_data.description,
            metadata=config_data.metadata,
            owner_id=current_user.get("user_id"),
            is_global=config_data.is_global,
        )
        
        logger.info(
            "orchestration_config_created",
            config_id=str(config.id),
            name=config.name,
            strategy=config.strategy,
        )
        
        return {
            "id": str(config.id),
            "name": config.name,
            "strategy": config.strategy,
            "steps": config.steps,
            "description": config.description,
            "created_at": config.created_at.isoformat(),
        }


@router.get("/configs")
async def list_orchestration_configs(
    strategy: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    current_user: dict = Depends(get_current_user),
    logger=Depends(get_logger),
):
    """List orchestration configurations."""
    async with get_session() as session:
        repo = OrchestrationConfigRepository(session)
        
        configs = await repo.list_all(
            owner_id=current_user.get("user_id"),
            strategy=strategy,
            limit=limit,
            offset=offset,
        )
        
        total = await repo.count(owner_id=current_user.get("user_id"))
        
        return {
            "configs": [
                {
                    "id": str(c.id),
                    "name": c.name,
                    "description": c.description,
                    "strategy": c.strategy,
                    "steps_count": len(c.steps.get("steps", [])),
                    "is_active": c.is_active,
                    "created_at": c.created_at.isoformat(),
                    "updated_at": c.updated_at.isoformat(),
                }
                for c in configs
            ],
            "total": total,
            "limit": limit,
            "offset": offset,
        }


@router.get("/configs/{config_id}")
async def get_orchestration_config(
    config_id: str,
    current_user: dict = Depends(get_current_user),
    logger=Depends(get_logger),
):
    """Get orchestration configuration by ID."""
    async with get_session() as session:
        repo = OrchestrationConfigRepository(session)
        
        try:
            config_uuid = UUID(config_id)
            config = await repo.get_by_id(config_uuid)
        except ValueError:
            # Try by name
            config = await repo.get_by_name(config_id, current_user.get("user_id"))
        
        if not config:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Configuration not found"
            )
        
        return {
            "id": str(config.id),
            "name": config.name,
            "description": config.description,
            "strategy": config.strategy,
            "steps": config.steps,
            "metadata": config.metadata,
            "is_active": config.is_active,
            "version": config.version,
            "created_at": config.created_at.isoformat(),
            "updated_at": config.updated_at.isoformat(),
        }


@router.put("/configs/{config_id}")
async def update_orchestration_config(
    config_id: str,
    update_data: OrchestrationConfigUpdate,
    current_user: dict = Depends(get_current_user),
    logger=Depends(get_logger),
):
    """Update orchestration configuration."""
    async with get_session() as session:
        repo = OrchestrationConfigRepository(session)
        
        try:
            config_uuid = UUID(config_id)
            config = await repo.get_by_id(config_uuid)
        except ValueError:
            config = await repo.get_by_name(config_id, current_user.get("user_id"))
        
        if not config:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Configuration not found"
            )
        
        # Update fields
        update_dict = update_data.model_dump(exclude_unset=True)
        if update_dict:
            updated = await repo.update(config.id, **update_dict)
            
            logger.info(
                "orchestration_config_updated",
                config_id=str(config.id),
                fields=list(update_dict.keys()),
            )
        
        return {
            "id": str(updated.id),
            "name": updated.name,
            "strategy": updated.strategy,
            "steps": updated.steps,
            "updated_at": updated.updated_at.isoformat(),
        }


@router.delete("/configs/{config_id}")
async def delete_orchestration_config(
    config_id: str,
    current_user: dict = Depends(get_current_user),
    logger=Depends(get_logger),
):
    """Delete orchestration configuration (soft delete)."""
    async with get_session() as session:
        repo = OrchestrationConfigRepository(session)
        
        try:
            config_uuid = UUID(config_id)
            config = await repo.get_by_id(config_uuid)
        except ValueError:
            config = await repo.get_by_name(config_id, current_user.get("user_id"))
        
        if not config:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Configuration not found"
            )
        
        await repo.delete(config.id)
        
        logger.info(
            "orchestration_config_deleted",
            config_id=str(config.id),
        )
        
        return {"message": "Configuration deleted successfully"}


# =============================================================================
# Execution Endpoints
# =============================================================================

@router.post("/execute/{config_id}", status_code=status.HTTP_202_ACCEPTED)
async def execute_orchestration(
    config_id: str,
    execute_request: OrchestrationExecuteRequest,
    current_user: dict = Depends(get_current_user),
    logger=Depends(get_logger),
):
    """
    Execute an orchestration.
    
    Starts execution of an orchestration configuration and returns immediately
    with a trace_id for tracking.
    """
    async with get_session() as session:
        # Load configuration
        config_repo = OrchestrationConfigRepository(session)
        
        try:
            config_uuid = UUID(config_id)
            config = await config_repo.get_by_id(config_uuid)
        except ValueError:
            config = await config_repo.get_by_name(config_id, current_user.get("user_id"))
        
        if not config:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Configuration not found"
            )
        
        if not config.is_active:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Configuration is not active"
            )
        
        # Create runner and execute
        runner = OrchestrationRunner(session)
        
        try:
            result = await runner.execute(
                config_id=str(config.id),
                input_data=execute_request.input,
                user_id=current_user.get("user_id"),
                session_id=execute_request.input.get("session_id"),
                callback_url=execute_request.callback_url,
            )
            
            logger.info(
                "orchestration_executed",
                trace_id=result["trace_id"],
                config_name=config.name,
            )
            
            return {
                "trace_id": result["trace_id"],
                "execution_id": result["execution_id"],
                "status": result["status"],
                "message": "Orchestration started",
            }
            
        except Exception as e:
            logger.error(
                "orchestration_execution_failed",
                config_id=str(config.id),
                error=str(e),
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to start orchestration: {str(e)}"
            )


@router.get("/status/{trace_id}", response_model=ExecutionStatusResponse)
async def get_execution_status(
    trace_id: str,
    current_user: dict = Depends(get_current_user),
    logger=Depends(get_logger),
):
    """Get execution status by trace ID."""
    async with get_session() as session:
        runner = OrchestrationRunner(session)
        
        status = await runner.get_execution_status(trace_id)
        
        if not status:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Execution not found"
            )
        
        # Check access (user owns execution or is admin)
        if status.get("user_id") and status["user_id"] != current_user.get("user_id"):
            if not current_user.get("is_admin"):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Not authorized to view this execution"
                )
        
        return ExecutionStatusResponse(**status)


@router.get("/executions")
async def list_executions(
    config_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    current_user: dict = Depends(get_current_user),
    logger=Depends(get_logger),
):
    """List orchestration executions."""
    async with get_session() as session:
        from uuid import UUID
        
        exec_repo = OrchestrationExecutionRepository(session)
        
        if config_id:
            try:
                config_uuid = UUID(config_id)
                executions = await exec_repo.list_by_config(
                    config_id=config_uuid,
                    status=status,
                    limit=limit,
                    offset=offset,
                )
            except ValueError:
                executions = []
        else:
            executions = await exec_repo.list_by_user(
                user_id=current_user.get("user_id"),
                status=status,
                limit=limit,
                offset=offset,
            )
        
        return {
            "executions": [
                {
                    "trace_id": e.trace_id,
                    "config_name": e.config_name,
                    "status": e.status,
                    "created_at": e.created_at.isoformat() if e.created_at else None,
                    "started_at": e.started_at.isoformat() if e.started_at else None,
                    "completed_at": e.completed_at.isoformat() if e.completed_at else None,
                    "latency_ms": e.total_latency_ms,
                }
                for e in executions
            ],
            "limit": limit,
            "offset": offset,
        }


# =============================================================================
# WebSocket Support
# =============================================================================

@router.websocket("/ws/execute/{config_id}")
async def websocket_execute_orchestration(
    websocket: WebSocket,
    config_id: str,
    token: str = Query(None),
):
    """
    Execute orchestration via WebSocket for real-time updates.
    
    Provides live progress updates as steps execute.
    """
    # Authenticate
    # ... (similar to existing WebSocket auth)
    
    await websocket.accept()
    
    async with get_session() as session:
        # Load config
        config_repo = OrchestrationConfigRepository(session)
        
        try:
            config_uuid = UUID(config_id)
            config = await config_repo.get_by_id(config_uuid)
        except ValueError:
            config = await config_repo.get_by_name(config_id)
        
        if not config:
            await websocket.send_json({
                "type": "error",
                "message": "Configuration not found"
            })
            await websocket.close()
            return
        
        # Receive input
        try:
            message = await websocket.receive_json()
            input_data = message.get("input", {})
        except Exception:
            input_data = {}
        
        # Execute with event broadcasting
        event_bus = OrchestrationEventBus(websocket_manager=None)
        await event_bus.start()
        
        # Subscribe to events
        async def on_event(event):
            await websocket.send_json(event.to_dict())
        
        await event_bus.subscribe("*", on_event)
        
        runner = OrchestrationRunner(session, event_bus=event_bus)
        
        try:
            result = await runner.execute(
                config_id=str(config.id),
                input_data=input_data,
            )
            
            await websocket.send_json({
                "type": "execution_completed",
                **result
            })
            
        except Exception as e:
            await websocket.send_json({
                "type": "execution_failed",
                "error": str(e)
            })
        
        await event_bus.stop()
        await websocket.close()
```

---

## 6. Integração com Agentes Existentes

### 6.1 Adaptador de Agentes

#### Arquivo: resync/core/orchestration/agent_adapter.py

```python
"""
Agent Adapter for Orchestration

Provides integration between the orchestration runner
and the existing HybridRouter/agent system.
"""
import asyncio
import time
from typing import Any, Callable, Dict, Optional

from resync.core.structured_logger import get_logger
from resync.core.orchestration.runner import OrchestrationContext

logger = get_logger(__name__)


class AgentAdapter:
    """
    Adapter that integrates orchestration steps with the existing agent system.
    
    This adapter:
    - Maps orchestration step agents to actual agent implementations
    - Handles input/output transformation
    - Collects metrics (tokens, latency, cost)
    - Handles errors and retries
    """
    
    def __init__(self):
        """Initialize the agent adapter."""
        self._agent_handlers: Dict[str, Callable] = {}
        self._register_default_agents()
    
    def _register_default_agents(self):
        """Register handlers for default agents."""
        # These will be populated with actual agent implementations
        self._agent_handlers = {
            "status_handler": self._handle_status,
            "log_analyzer": self._handle_log_analyzer,
            "rag_handler": self._handle_rag,
            "synthesizer": self._handle_synthesizer,
            "troubleshoot_handler": self._handle_troubleshoot,
            "general_handler": self._handle_general,
            "action_handler": self._handle_action,
        }
    
    def register_agent(self, agent_id: str, handler: Callable):
        """
        Register a custom agent handler.
        
        Args:
            agent_id: Unique agent identifier
            handler: Async function (agent_id, input_data, context) -> result
        """
        self._agent_handlers[agent_id] = handler
    
    async def execute(
        self,
        agent_id: str,
        input_data: Dict[str, Any],
        context: OrchestrationContext,
    ) -> Dict[str, Any]:
        """
        Execute an agent as part of an orchestration step.
        
        Args:
            agent_id: Agent identifier
            input_data: Input data for the agent
            context: Orchestration context
            
        Returns:
            Result dictionary with output, metrics
        """
        start_time = time.time()
        
        # Get agent handler
        handler = self._agent_handlers.get(agent_id)
        
        if not handler:
            logger.warning("Unknown agent: %s", agent_id)
            return {
                "output": {"error": f"Unknown agent: {agent_id}"},
                "tokens": 0,
                "cost": 0.0,
            }
        
        try:
            # Execute handler
            result = await handler(input_data, context)
            
            latency_ms = int((time.time() - start_time) * 1000)
            
            return {
                "output": result,
                "latency_ms": latency_ms,
                "tokens": result.get("_tokens", 0),
                "cost": result.get("_cost", 0.0),
            }
            
        except Exception as e:
            logger.error("Agent execution failed", agent_id=agent_id, error=str(e))
            raise
    
    async def _handle_status(
        self,
        input_data: Dict[str, Any],
        context: OrchestrationContext,
    ) -> Dict[str, Any]:
        """Handle status query step."""
        from resync.core.agent_router import HybridRouter, RoutingMode
        
        job_name = input_data.get("job_name")
        message = f"Status do job {job_name}"
        
        router = HybridRouter()
        result = await router.route(
            message=message,
            context={
                "tws_instance_id": input_data.get("tws_instance_id"),
                "session_id": context.session_id,
            },
            force_mode=RoutingMode.AGENTIC,
        )
        
        return {
            "status": result.response,
            "job_name": job_name,
            "entities": result.entities,
            "_tokens": 100,  # Estimate
            "_cost": 0.01,
        }
    
    async def _handle_log_analyzer(
        self,
        input_data: Dict[str, Any],
        context: OrchestrationContext,
    ) -> Dict[str, Any]:
        """Handle log analysis step."""
        # Similar implementation using log analyzer agent
        logs = input_data.get("logs", "")
        
        return {
            "analysis": "Log analysis complete",
            "logs": logs[:1000],
            "_tokens": 150,
            "_cost": 0.015,
        }
    
    async def _handle_rag(
        self,
        input_data: Dict[str, Any],
        context: OrchestrationContext,
    ) -> Dict[str, Any]:
        """Handle RAG query step."""
        from resync.services.rag_client import RAGClient
        
        query = input_data.get("query", "")
        
        rag = RAGClient()
        results = rag.search(query=query, limit=5)
        
        return {
            "results": results,
            "query": query,
            "_tokens": 50,
            "_cost": 0.005,
        }
    
    async def _handle_synthesizer(
        self,
        input_data: Dict[str, Any],
        context: OrchestrationContext,
    ) -> Dict[str, Any]:
        """Handle response synthesis step."""
        # Collect outputs from dependencies
        all_outputs = []
        
        for step_id, output in context.step_outputs.items():
            if isinstance(output, dict):
                all_outputs.append(output)
        
        # Synthesize response
        synthesis = {
            "final_response": "Synthesized response from " + str(len(all_outputs)) + " sources",
            "sources": all_outputs,
        }
        
        return {
            **synthesis,
            "_tokens": 80,
            "_cost": 0.008,
        }
    
    async def _handle_troubleshoot(
        self,
        input_data: Dict[str, Any],
        context: OrchestrationContext,
    ) -> Dict[str, Any]:
        """Handle troubleshooting step."""
        # Similar to status but with troubleshooting mode
        return {
            "analysis": "Troubleshooting complete",
            "recommendations": ["Check logs", "Restart job"],
            "_tokens": 200,
            "_cost": 0.02,
        }
    
    async def _handle_general(
        self,
        input_data: Dict[str, Any],
        context: OrchestrationContext,
    ) -> Dict[str, Any]:
        """Handle general query step."""
        message = input_data.get("message", "")
        
        return {
            "response": f"Processed: {message}",
            "_tokens": 50,
            "_cost": 0.005,
        }
    
    async def _handle_action(
        self,
        input_data: Dict[str, Any],
        context: OrchestrationContext,
    ) -> Dict[str, Any]:
        """Handle action execution step."""
        action = input_data.get("action")
        target = input_data.get("target")
        
        return {
            "action": action,
            "target": target,
            "result": "Action executed successfully",
            "_tokens": 100,
            "_cost": 0.01,
        }


# Global adapter instance
_agent_adapter: Optional[AgentAdapter] = None


def get_agent_adapter() -> AgentAdapter:
    """Get the global agent adapter instance."""
    global _agent_adapter
    if _agent_adapter is None:
        _agent_adapter = AgentAdapter()
    return _agent_adapter
```

---

## 7. Migração de Banco de Dados

### 7.1 Script de Migração

#### Arquivo: migrations/versions/orchestration_v1.py

```python
"""
Orchestration System Migration v1

Creates tables for orchestration configuration and execution tracking.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers
revision = 'orchestration_v1'
down_revision = None  # Set to previous migration
branch_labels = None
depends_on = None


def upgrade():
    # Create orchestration_configs table
    op.create_table(
        'orchestration_configs',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('description', sa.Text, nullable=True),
        sa.Column('strategy', sa.String(50), nullable=False),
        sa.Column('steps', postgresql.JSONB, nullable=False),
        sa.Column('metadata', postgresql.JSONB, default={}),
        sa.Column('owner_id', sa.String(255), nullable=True),
        sa.Column('tenant_id', sa.String(255), nullable=True),
        sa.Column('is_global', sa.Boolean, default=False),
        sa.Column('version', sa.Integer, default=1),
        sa.Column('is_active', sa.Boolean, default=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    
    # Create indexes for orchestration_configs
    op.create_index('idx_configs_owner', 'orchestration_configs', ['owner_id'])
    op.create_index('idx_configs_tenant', 'orchestration_configs', ['tenant_id'])
    op.create_index('idx_configs_strategy', 'orchestration_configs', ['strategy'])
    
    # Create orchestration_executions table
    op.create_table(
        'orchestration_executions',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('trace_id', sa.String(255), unique=True, nullable=False),
        sa.Column('config_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('config_name', sa.String(255), nullable=True),
        sa.Column('status', sa.String(50), nullable=False),
        sa.Column('input_data', postgresql.JSONB, default={}),
        sa.Column('output_data', postgresql.JSONB, default={}),
        sa.Column('user_id', sa.String(255), nullable=True),
        sa.Column('session_id', sa.String(255), nullable=True),
        sa.Column('tenant_id', sa.String(255), nullable=True),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('paused_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('metadata', postgresql.JSONB, default={}),
        sa.Column('total_latency_ms', sa.Integer, nullable=True),
        sa.Column('estimated_cost', sa.Numeric(10, 6), nullable=True),
        sa.Column('created_by', sa.String(255), nullable=True),
        sa.Column('callback_url', sa.String(500), nullable=True),
        sa.ForeignKeyConstraint(['config_id'], ['orchestration_configs.id'], ondelete='SET NULL'),
    )
    
    # Create indexes for orchestration_executions
    op.create_index('idx_executions_trace', 'orchestration_executions', ['trace_id'])
    op.create_index('idx_executions_config', 'orchestration_executions', ['config_id'])
    op.create_index('idx_executions_status', 'orchestration_executions', ['status'])
    op.create_index('idx_executions_user', 'orchestration_executions', ['user_id'])
    op.create_index('idx_executions_session', 'orchestration_executions', ['session_id'])
    
    # Create orchestration_step_runs table
    op.create_table(
        'orchestration_step_runs',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('execution_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('step_index', sa.Integer, nullable=False),
        sa.Column('step_id', sa.String(100), nullable=False),
        sa.Column('step_name', sa.String(255), nullable=True),
        sa.Column('status', sa.String(50), nullable=False),
        sa.Column('output', postgresql.JSONB, default={}),
        sa.Column('output_truncated', sa.Text, nullable=True),
        sa.Column('error_message', sa.Text, nullable=True),
        sa.Column('error_trace', sa.Text, nullable=True),
        sa.Column('dependencies_json', postgresql.JSONB, default=[]),
        sa.Column('latency_ms', sa.Integer, nullable=True),
        sa.Column('retry_count', sa.Integer, default=0),
        sa.Column('token_count', sa.Integer, nullable=True),
        sa.Column('estimated_cost', sa.Numeric(10, 6), nullable=True),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('agent_id', sa.String(255), nullable=True),
        sa.Column('agent_version', sa.String(50), nullable=True),
        sa.ForeignKeyConstraint(['execution_id'], ['orchestration_executions.id'], ondelete='CASCADE'),
        sa.UniqueConstraint('execution_id', 'step_index', name='uq_step_execution_index'),
        sa.UniqueConstraint('execution_id', 'step_id', name='uq_step_execution_id'),
    )
    
    # Create indexes for orchestration_step_runs
    op.create_index('idx_steps_execution', 'orchestration_step_runs', ['execution_id'])
    op.create_index('idx_steps_status', 'orchestration_step_runs', ['status'])
    
    # Create orchestration_callbacks table
    op.create_table(
        'orchestration_callbacks',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column('execution_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('callback_type', sa.String(50), nullable=False),
        sa.Column('callback_url', sa.String(500), nullable=True),
        sa.Column('callback_method', sa.String(10), server_default='POST'),
        sa.Column('callback_headers', postgresql.JSONB, default={}),
        sa.Column('callback_body_template', sa.Text, nullable=True),
        sa.Column('status', sa.String(50), server_default='pending'),
        sa.Column('retry_count', sa.Integer, default=0),
        sa.Column('max_retries', sa.Integer, default=3),
        sa.Column('last_retry_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('response_status', sa.Integer, nullable=True),
        sa.Column('response_body', sa.Text, nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column('sent_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['execution_id'], ['orchestration_executions.id'], ondelete='CASCADE'),
    )
    
    op.create_index('idx_callbacks_execution', 'orchestration_callbacks', ['execution_id'])
    op.create_index('idx_callbacks_status', 'orchestration_callbacks', ['status'])


def downgrade():
    op.drop_table('orchestration_callbacks')
    op.drop_table('orchestration_step_runs')
    op.drop_table('orchestration_executions')
    op.drop_table('orchestration_configs')
```

---

## 8. Plano de Implementação

### Fase 1: Fundação (Dias 1-3)

| Dia | Tarefa | Arquivos |
|-----|--------|----------|
| 1 | Criar modelos SQLAlchemy | `resync/core/database/models/orchestration.py` |
| 1 | Configurar migrates Alembic | `migrations/versions/orchestration_v1.py` |
| 2 | Criar repositórios de dados | `resync/core/database/repositories/orchestration_*_repo.py` |
| 2 | Executar migrações | - |
| 3 | Implementar event bus básico | `resync/core/orchestration/events.py` |

### Fase 2: Motor de Orquestração (Dias 4-7)

| Dia | Tarefa | Arquivos |
|-----|--------|----------|
| 4 | Implementar estrategias base | `resync/core/orchestration/strategies.py` |
| 5 | Implementar runner principal | `resync/core/orchestration/runner.py` |
| 6 | Criar adapter de agentes | `resync/core/orchestration/agent_adapter.py` |
| 7 | Testar integrações | - |

### Fase 3: API e Admin (Dias 8-11)

| Dia | Tarefa | Arquivos |
|-----|--------|----------|
| 8 | Criar endpoints REST | `resync/api/routes/orchestration.py` |
| 9 | Implementar WebSocket support | Same file above |
| 10 | Criar frontend admin | `resync/templates/orchestration_admin.html` |
| 11 | Testes de integração | - |

### Fase 4: Observabilidade (Dias 12-14)

| Dia | Tarefa | Arquivos |
|-----|--------|----------|
| 12 | Adicionar métricas Prometheus | New metrics in `resync/core/metrics/` |
| 12 | Criar dashboards | `resync/api/routes/monitoring/orchestration_dashboard.py` |
| 13 | Logging estruturado | Already exists |
| 14 | Testes de carga | - |

---

## 9. Conclusão

Este plano de implementação fornece uma base sólida para o sistema de orquestração multi-agent no projeto Resync TWS. Os principais benefícios incluem:

1. **Flexibilidade**: Múltiplas estratégias de execução (sequencial, paralelo, consenso, fallback)
2. **Escalabilidade**: Execução distribuída com checkpoints
3. **Observabilidade**: Rastreamento completo deexecuções e métricas
4. **Integração**: Funciona com a infraestrutura existente (HybridRouter, PgVector, Redis)
5. **Admin**: Interface de configuração runtime para orquestrações

O sistema foi projetado para ser incremental - cada fase pode ser implementada e testada independentemente antes de passar para a próxima.