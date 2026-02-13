"""
TWS Knowledge Graph Relations - Expansão v5.4.0

Define 15+ tipos de relação específicos do TWS para o Knowledge Graph,
permitindo queries semânticas complexas sobre dependências, recursos,
schedules e fluxos de execução.

Versão: 5.4.0
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# SQL INJECTION PROTECTION
# =============================================================================


def sanitize_sql_identifier(value: str) -> str:
    """
    Sanitiza um identificador SQL para prevenir SQL injection.

    Apenas permite caracteres alfanuméricos, underscores e hífens.
    Limita o tamanho a 128 caracteres.
    """
    if not value:
        raise ValueError("SQL identifier cannot be empty")

    # Remove caracteres perigosos - só permite alfanuméricos, _, -, .
    sanitized = re.sub(r"[^a-zA-Z0-9_\-.]", "", value)

    # Limita tamanho
    sanitized = sanitized[:128]

    if not sanitized:
        raise ValueError(f"Invalid SQL identifier after sanitization: {value}")

    return sanitized


# =============================================================================
# TIPOS DE RELAÇÃO TWS (15+ tipos)
# =============================================================================


class TWSRelationType(Enum):
    """
    Tipos de relação no Knowledge Graph TWS.

    Categorias:
    - Dependências de execução
    - Recursos e alocação
    - Organização hierárquica
    - Monitoramento e alertas
    - Recovery e fallback
    - Temporal
    """

    # =========================================================================
    # DEPENDÊNCIAS DE EXECUÇÃO (5 tipos)
    # =========================================================================

    # Job A precisa que Job B termine com sucesso antes de iniciar
    DEPENDS_ON = "depends_on"

    # Job A executa imediatamente após Job B (sequência temporal)
    FOLLOWS = "follows"

    # Job A precisa de um recurso que Job B fornece
    NEEDS = "needs"

    # Job A dispara Job B após sua conclusão (trigger)
    TRIGGERS = "triggers"

    # Job A é predecessor direto de Job B na cadeia
    PREDECESSOR_OF = "predecessor_of"

    # Job A é sucessor direto de Job B na cadeia
    SUCCESSOR_OF = "successor_of"

    # =========================================================================
    # RECURSOS E ALOCAÇÃO (4 tipos)
    # =========================================================================

    # Jobs compartilham o mesmo recurso (potencial conflito)
    SHARES_RESOURCE = "shares_resource"

    # Job roda em uma workstation específica
    RUNS_ON = "runs_on"

    # Jobs são mutuamente exclusivos (não podem rodar juntos)
    EXCLUSIVE_WITH = "exclusive_with"

    # Job aloca/reserva um recurso exclusivo
    ALLOCATES = "allocates"

    # =========================================================================
    # ORGANIZAÇÃO HIERÁRQUICA (3 tipos)
    # =========================================================================

    # Job pertence a um schedule/job stream
    BELONGS_TO = "belongs_to"

    # Schedule contém jobs
    CONTAINS = "contains"

    # Job é parte de um grupo lógico
    MEMBER_OF = "member_of"

    # =========================================================================
    # MONITORAMENTO E ALERTAS (2 tipos)
    # =========================================================================

    # Job é monitorado por uma regra de alerta
    MONITORED_BY = "monitored_by"

    # Alerta notifica um destinatário/canal
    NOTIFIES = "notifies"

    # =========================================================================
    # RECOVERY E FALLBACK (2 tipos)
    # =========================================================================

    # Job A é o recovery job de Job B
    RECOVERS = "recovers"

    # Job A é fallback/alternativa para Job B
    FALLBACK_FOR = "fallback_for"

    # =========================================================================
    # TEMPORAL (2 tipos)
    # =========================================================================

    # Job deve iniciar antes de um horário específico
    MUST_START_BEFORE = "must_start_before"

    # Job deve terminar antes de um horário específico
    MUST_END_BEFORE = "must_end_before"


# =============================================================================
# TIPOS DE NÓS
# =============================================================================


class TWSNodeType(Enum):
    """Tipos de nós no Knowledge Graph TWS."""

    JOB = "job"
    SCHEDULE = "schedule"
    JOB_STREAM = "job_stream"
    WORKSTATION = "workstation"
    RESOURCE = "resource"
    ALERT_RULE = "alert_rule"
    NOTIFICATION_CHANNEL = "notification_channel"
    TIME_CONSTRAINT = "time_constraint"
    USER = "user"
    APPLICATION = "application"


# =============================================================================
# MODELOS DE DADOS
# =============================================================================


@dataclass
class TWSNode:
    """Nó no Knowledge Graph TWS."""

    node_id: str
    node_type: TWSNodeType
    name: str
    properties: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    tenant_id: str | None = None  # Para multi-tenant

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "name": self.name,
            "properties": self.properties,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "tenant_id": self.tenant_id,
        }


@dataclass
class TWSRelation:
    """Relação entre nós no Knowledge Graph TWS."""

    from_node: str
    to_node: str
    relation_type: TWSRelationType
    properties: dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    tenant_id: str | None = None  # Para multi-tenant

    def to_dict(self) -> dict[str, Any]:
        return {
            "from_node": self.from_node,
            "to_node": self.to_node,
            "relation_type": self.relation_type.value,
            "properties": self.properties,
            "weight": self.weight,
            "created_at": self.created_at.isoformat(),
            "tenant_id": self.tenant_id,
        }

    # NOTE: to_cypher() method removed in v5.9.3
    # Apache AGE was never implemented; graph now uses NetworkX on-demand

    def to_sql(self, table_name: str = "kg_relations") -> str:
        """Gera query SQL para inserir a relação."""
        import json

        # Sanitize all inputs to prevent SQL injection
        safe_table = sanitize_sql_identifier(table_name)
        safe_from = sanitize_sql_identifier(self.from_node)
        safe_to = sanitize_sql_identifier(self.to_node)
        safe_tenant = sanitize_sql_identifier(self.tenant_id) if self.tenant_id else "default"
        safe_rel_type = sanitize_sql_identifier(self.relation_type.value)

        props_json = json.dumps(self.properties)
        # Double any single quotes in JSON to escape them properly
        props_json_escaped = props_json.replace("'", "''")

        return f"""
        INSERT INTO {safe_table} (from_node, to_node, relation_type, properties, weight, tenant_id)
        VALUES ('{safe_from}', '{safe_to}', '{safe_rel_type}',
                '{props_json_escaped}'::jsonb, {float(self.weight)}, '{safe_tenant}')
        ON CONFLICT (from_node, to_node, relation_type, tenant_id)
        DO UPDATE SET properties = EXCLUDED.properties, weight = EXCLUDED.weight
        """


# =============================================================================
# QUERY PATTERNS
# =============================================================================


class TWSQueryPatterns:
    """Padrões de query para o Knowledge Graph TWS."""

    @staticmethod
    def get_all_dependencies(job_name: str, tenant_id: str = None) -> str:
        """Query para obter todas as dependências de um job."""
        safe_job = sanitize_sql_identifier(job_name)
        tenant_filter = ""
        if tenant_id:
            safe_tenant = sanitize_sql_identifier(tenant_id)
            tenant_filter = f"AND r.tenant_id = '{safe_tenant}'"
        return f"""
        WITH RECURSIVE deps AS (
            SELECT from_node, to_node, relation_type, 1 as depth
            FROM kg_relations
            WHERE to_node = '{safe_job}'
            AND relation_type IN ('depends_on', 'follows', 'needs')
            {tenant_filter}

            UNION ALL

            SELECT r.from_node, r.to_node, r.relation_type, d.depth + 1
            FROM kg_relations r
            JOIN deps d ON r.to_node = d.from_node
            WHERE d.depth < 10
            AND r.relation_type IN ('depends_on', 'follows', 'needs')
            {tenant_filter}
        )
        SELECT DISTINCT from_node, to_node, relation_type, depth
        FROM deps
        ORDER BY depth
        """

    @staticmethod
    def get_impact_analysis(job_name: str, tenant_id: str = None) -> str:
        """Query para análise de impacto (jobs downstream)."""
        safe_job = sanitize_sql_identifier(job_name)
        tenant_filter = ""
        if tenant_id:
            safe_tenant = sanitize_sql_identifier(tenant_id)
            tenant_filter = f"AND r.tenant_id = '{safe_tenant}'"
        return f"""
        WITH RECURSIVE impact AS (
            SELECT from_node, to_node, relation_type, 1 as depth
            FROM kg_relations
            WHERE from_node = '{safe_job}'
            AND relation_type IN ('triggers', 'predecessor_of')
            {tenant_filter}

            UNION ALL

            SELECT r.from_node, r.to_node, r.relation_type, i.depth + 1
            FROM kg_relations r
            JOIN impact i ON r.from_node = i.to_node
            WHERE i.depth < 10
            {tenant_filter}
        )
        SELECT DISTINCT to_node as affected_job, depth as distance
        FROM impact
        ORDER BY depth
        """

    @staticmethod
    def get_resource_conflicts(resource_name: str, tenant_id: str = None) -> str:
        """Query para encontrar conflitos de recursos."""
        safe_resource = sanitize_sql_identifier(resource_name)
        tenant_filter = ""
        if tenant_id:
            safe_tenant = sanitize_sql_identifier(tenant_id)
            tenant_filter = f"AND tenant_id = '{safe_tenant}'"
        return f"""
        SELECT DISTINCT r1.from_node as job1, r2.from_node as job2
        FROM kg_relations r1
        JOIN kg_relations r2 ON r1.to_node = r2.to_node
        WHERE r1.to_node = '{safe_resource}'
        AND r1.relation_type = 'allocates'
        AND r2.relation_type = 'allocates'
        AND r1.from_node < r2.from_node
        {tenant_filter}
        """

    @staticmethod
    def get_critical_path(schedule_name: str, tenant_id: str = None) -> str:
        """Query para encontrar o caminho crítico de um schedule."""
        safe_schedule = sanitize_sql_identifier(schedule_name)
        tenant_filter = ""
        if tenant_id:
            safe_tenant = sanitize_sql_identifier(tenant_id)
            tenant_filter = f"AND r.tenant_id = '{safe_tenant}'"
        return f"""
        WITH schedule_jobs AS (
            SELECT to_node as job_name
            FROM kg_relations
            WHERE from_node = '{safe_schedule}'
            AND relation_type = 'contains'
            {tenant_filter}
        ),
        job_chains AS (
            SELECT r.from_node, r.to_node,
                   COALESCE((r.properties->>'duration')::int, 0) as duration
            FROM kg_relations r
            JOIN schedule_jobs sj ON r.from_node = sj.job_name
            WHERE r.relation_type IN ('depends_on', 'follows')
            {tenant_filter}
        )
        SELECT from_node, to_node, duration,
               SUM(duration) OVER (ORDER BY from_node) as cumulative_duration
        FROM job_chains
        ORDER BY cumulative_duration DESC
        """

    @staticmethod
    def get_jobs_by_workstation(workstation: str, tenant_id: str = None) -> str:
        """Query para obter jobs de uma workstation."""
        tenant_filter = f"AND tenant_id = '{tenant_id}'" if tenant_id else ""
        return f"""
        SELECT from_node as job_name, properties
        FROM kg_relations
        WHERE to_node = '{workstation}'
        AND relation_type = 'runs_on'
        {tenant_filter}
        """


# =============================================================================
# RELATION BUILDER
# =============================================================================


class TWSRelationBuilder:
    """Builder para criar relações TWS de forma fluente."""

    def __init__(self, tenant_id: str | None = None):
        self.tenant_id = tenant_id
        self._relations: list[TWSRelation] = []

    def job_depends_on(
        self,
        job: str,
        depends_on: str,
        condition: str = "success",
    ) -> "TWSRelationBuilder":
        """Job depende de outro job."""
        self._relations.append(
            TWSRelation(
                from_node=job,
                to_node=depends_on,
                relation_type=TWSRelationType.DEPENDS_ON,
                properties={"condition": condition},
                tenant_id=self.tenant_id,
            )
        )
        return self

    def job_triggers(
        self,
        trigger_job: str,
        triggered_job: str,
        on_status: str = "success",
    ) -> "TWSRelationBuilder":
        """Job dispara outro job."""
        self._relations.append(
            TWSRelation(
                from_node=trigger_job,
                to_node=triggered_job,
                relation_type=TWSRelationType.TRIGGERS,
                properties={"on_status": on_status},
                tenant_id=self.tenant_id,
            )
        )
        return self

    def job_runs_on(
        self,
        job: str,
        workstation: str,
        priority: int = 5,
    ) -> "TWSRelationBuilder":
        """Job roda em workstation."""
        self._relations.append(
            TWSRelation(
                from_node=job,
                to_node=workstation,
                relation_type=TWSRelationType.RUNS_ON,
                properties={"priority": priority},
                tenant_id=self.tenant_id,
            )
        )
        return self

    def job_belongs_to(
        self,
        job: str,
        schedule: str,
    ) -> "TWSRelationBuilder":
        """Job pertence a schedule."""
        self._relations.append(
            TWSRelation(
                from_node=job,
                to_node=schedule,
                relation_type=TWSRelationType.BELONGS_TO,
                tenant_id=self.tenant_id,
            )
        )
        return self

    def job_allocates(
        self,
        job: str,
        resource: str,
        exclusive: bool = True,
    ) -> "TWSRelationBuilder":
        """Job aloca recurso."""
        self._relations.append(
            TWSRelation(
                from_node=job,
                to_node=resource,
                relation_type=TWSRelationType.ALLOCATES,
                properties={"exclusive": exclusive},
                tenant_id=self.tenant_id,
            )
        )
        return self

    def job_recovers(
        self,
        recovery_job: str,
        failed_job: str,
    ) -> "TWSRelationBuilder":
        """Job é recovery de outro."""
        self._relations.append(
            TWSRelation(
                from_node=recovery_job,
                to_node=failed_job,
                relation_type=TWSRelationType.RECOVERS,
                tenant_id=self.tenant_id,
            )
        )
        return self

    def jobs_exclusive(
        self,
        job1: str,
        job2: str,
        resource: str,
    ) -> "TWSRelationBuilder":
        """Jobs são mutuamente exclusivos."""
        self._relations.append(
            TWSRelation(
                from_node=job1,
                to_node=job2,
                relation_type=TWSRelationType.EXCLUSIVE_WITH,
                properties={"shared_resource": resource},
                tenant_id=self.tenant_id,
            )
        )
        return self

    def job_monitored_by(
        self,
        job: str,
        alert_rule: str,
        threshold_minutes: int = 30,
    ) -> "TWSRelationBuilder":
        """Job é monitorado por regra de alerta."""
        self._relations.append(
            TWSRelation(
                from_node=job,
                to_node=alert_rule,
                relation_type=TWSRelationType.MONITORED_BY,
                properties={"threshold_minutes": threshold_minutes},
                tenant_id=self.tenant_id,
            )
        )
        return self

    def build(self) -> list[TWSRelation]:
        """Retorna todas as relações construídas."""
        return self._relations.copy()

    def clear(self) -> "TWSRelationBuilder":
        """Limpa as relações."""
        self._relations = []
        return self


# =============================================================================
# METADATA E STATS
# =============================================================================


def get_relation_types_info() -> dict[str, Any]:
    """Retorna informações sobre todos os tipos de relação."""
    categories = {
        "execution_dependencies": [
            TWSRelationType.DEPENDS_ON,
            TWSRelationType.FOLLOWS,
            TWSRelationType.NEEDS,
            TWSRelationType.TRIGGERS,
            TWSRelationType.PREDECESSOR_OF,
            TWSRelationType.SUCCESSOR_OF,
        ],
        "resources_allocation": [
            TWSRelationType.SHARES_RESOURCE,
            TWSRelationType.RUNS_ON,
            TWSRelationType.EXCLUSIVE_WITH,
            TWSRelationType.ALLOCATES,
        ],
        "hierarchy": [
            TWSRelationType.BELONGS_TO,
            TWSRelationType.CONTAINS,
            TWSRelationType.MEMBER_OF,
        ],
        "monitoring": [
            TWSRelationType.MONITORED_BY,
            TWSRelationType.NOTIFIES,
        ],
        "recovery": [
            TWSRelationType.RECOVERS,
            TWSRelationType.FALLBACK_FOR,
        ],
        "temporal": [
            TWSRelationType.MUST_START_BEFORE,
            TWSRelationType.MUST_END_BEFORE,
        ],
    }

    return {
        "total_types": len(TWSRelationType),
        "categories": {cat: [r.value for r in relations] for cat, relations in categories.items()},
        "all_types": [r.value for r in TWSRelationType],
    }


def get_node_types_info() -> dict[str, Any]:
    """Retorna informações sobre todos os tipos de nó."""
    return {
        "total_types": len(TWSNodeType),
        "types": [n.value for n in TWSNodeType],
    }
