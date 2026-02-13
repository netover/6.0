"""
TWS Knowledge Graph Expander - Expansão Automática v5.4.0

Expande automaticamente o Knowledge Graph com dados obtidos
da API do TWS, incluindo jobs, dependências, recursos,
schedules e workstations.

Versão: 5.4.0
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from resync.knowledge.retrieval.tws_relations import (
    TWSNode,
    TWSNodeType,
    TWSRelation,
    TWSRelationType,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURAÇÃO
# =============================================================================


@dataclass
class GraphExpansionConfig:
    """Configuração para expansão do grafo."""

    # Limites
    max_jobs: int = 10000
    max_depth: int = 10
    batch_size: int = 100

    # Tipos de relação a extrair
    extract_dependencies: bool = True
    extract_resources: bool = True
    extract_schedules: bool = True
    extract_workstations: bool = True
    extract_recovery: bool = True
    extract_alerts: bool = True

    # Multi-tenant
    tenant_id: str | None = None

    # Performance
    parallel_requests: int = 5
    request_timeout: int = 30


# =============================================================================
# STATS
# =============================================================================


@dataclass
class ExpansionStats:
    """Estatísticas da expansão."""

    started_at: datetime = None
    completed_at: datetime = None

    jobs_processed: int = 0
    schedules_processed: int = 0
    workstations_processed: int = 0
    resources_processed: int = 0

    nodes_created: int = 0
    relations_created: int = 0

    errors: list[str] = None

    def __post_init__(self):
        if self.errors is None:
            self.errors = []

    @property
    def duration_seconds(self) -> float:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds,
            "jobs_processed": self.jobs_processed,
            "schedules_processed": self.schedules_processed,
            "workstations_processed": self.workstations_processed,
            "resources_processed": self.resources_processed,
            "nodes_created": self.nodes_created,
            "relations_created": self.relations_created,
            "errors_count": len(self.errors),
            "errors": self.errors[:10],  # Limitar a 10 erros no output
        }


# =============================================================================
# GRAPH EXPANDER
# =============================================================================


class TWSGraphExpander:
    """
    Expande o Knowledge Graph com dados do TWS.

    Fontes de dados:
    1. API TWS - Jobs, dependências, schedules
    2. Audit Log - Padrões de execução históricos
    3. Configuração - Recursos, workstations

    Processo:
    1. Coleta dados do TWS
    2. Cria nós para cada entidade
    3. Cria relações entre entidades
    4. Persiste no banco de dados
    """

    def __init__(
        self,
        tws_client=None,
        db_session=None,
        config: GraphExpansionConfig = None,
    ):
        """
        Inicializa o expander.

        Args:
            tws_client: Cliente da API TWS
            db_session: Sessão do banco de dados
            config: Configuração de expansão
        """
        self.tws = tws_client
        self.db = db_session
        self.config = config or GraphExpansionConfig()
        self._stats = ExpansionStats()
        self._nodes: dict[str, TWSNode] = {}
        self._relations: list[TWSRelation] = []

    async def expand_full(self) -> ExpansionStats:
        """
        Executa expansão completa do grafo.

        Returns:
            Estatísticas da expansão
        """
        self._stats = ExpansionStats(started_at=datetime.now(timezone.utc))

        try:
            # 1. Expandir jobs e dependências
            if self.config.extract_dependencies:
                await self._expand_jobs_and_dependencies()

            # 2. Expandir recursos
            if self.config.extract_resources:
                await self._expand_resources()

            # 3. Expandir schedules
            if self.config.extract_schedules:
                await self._expand_schedules()

            # 4. Expandir workstations
            if self.config.extract_workstations:
                await self._expand_workstations()

            # 5. Expandir recovery jobs
            if self.config.extract_recovery:
                await self._expand_recovery_jobs()

            # 6. Expandir alertas
            if self.config.extract_alerts:
                await self._expand_alerts()

            # 7. Persistir no banco
            await self._persist_to_database()

        except Exception as e:
            logger.error(f"Erro na expansão do grafo: {e}")
            self._stats.errors.append(str(e))

        self._stats.completed_at = datetime.now(timezone.utc)

        logger.info(
            f"Expansão completa: {self._stats.nodes_created} nós, "
            f"{self._stats.relations_created} relações em "
            "{self._stats.duration_seconds:.2f}s"
        )

        return self._stats

    async def expand_from_job(self, job_name: str, depth: int = 3) -> ExpansionStats:
        """
        Expande grafo a partir de um job específico.

        Args:
            job_name: Nome do job inicial
            depth: Profundidade máxima de expansão

        Returns:
            Estatísticas da expansão
        """
        self._stats = ExpansionStats(started_at=datetime.now(timezone.utc))

        try:
            visited: set[str] = set()
            await self._expand_job_recursive(job_name, depth, visited)
            await self._persist_to_database()

        except Exception as e:
            logger.error(f"Erro na expansão do job {job_name}: {e}")
            self._stats.errors.append(str(e))

        self._stats.completed_at = datetime.now(timezone.utc)
        return self._stats

    async def _expand_jobs_and_dependencies(self):
        """Expande todos os jobs e suas dependências."""
        logger.info("Expandindo jobs e dependências...")

        # Obter lista de jobs do TWS
        jobs = await self._get_jobs_from_tws()

        for job in jobs[: self.config.max_jobs]:
            try:
                # Criar nó do job
                job_node = TWSNode(
                    node_id=f"job:{job['name']}",
                    node_type=TWSNodeType.JOB,
                    name=job["name"],
                    properties={
                        "status": job.get("status"),
                        "priority": job.get("priority", 5),
                        "owner": job.get("owner"),
                        "description": job.get("description", ""),
                        "estimated_duration": job.get("estimated_duration"),
                    },
                    tenant_id=self.config.tenant_id,
                )
                self._add_node(job_node)

                # Criar relações de dependência
                for dep in job.get("dependencies", []):
                    self._add_relation(
                        TWSRelation(
                            from_node=job["name"],
                            to_node=dep["job_name"],
                            relation_type=self._map_dependency_type(dep.get("type", "follows")),
                            properties={
                                "condition": dep.get("condition", "success"),
                                "offset_minutes": dep.get("offset_minutes", 0),
                            },
                            tenant_id=self.config.tenant_id,
                        )
                    )

                self._stats.jobs_processed += 1

            except Exception as e:
                logger.warning(f"Erro ao processar job {job.get('name')}: {e}")
                self._stats.errors.append(f"Job {job.get('name')}: {e}")

    async def _expand_resources(self):
        """Expande recursos e alocações."""
        logger.info("Expandindo recursos...")

        resources = await self._get_resources_from_tws()

        for resource in resources:
            try:
                # Criar nó do recurso
                resource_node = TWSNode(
                    node_id=f"resource:{resource['name']}",
                    node_type=TWSNodeType.RESOURCE,
                    name=resource["name"],
                    properties={
                        "type": resource.get("type", "exclusive"),
                        "capacity": resource.get("capacity", 1),
                        "description": resource.get("description", ""),
                    },
                    tenant_id=self.config.tenant_id,
                )
                self._add_node(resource_node)

                # Criar relações de alocação
                for job_name in resource.get("allocated_by", []):
                    self._add_relation(
                        TWSRelation(
                            from_node=job_name,
                            to_node=resource["name"],
                            relation_type=TWSRelationType.ALLOCATES,
                            properties={
                                "exclusive": resource.get("type") == "exclusive",
                            },
                            tenant_id=self.config.tenant_id,
                        )
                    )

                # Detectar conflitos (jobs exclusivos)
                if resource.get("type") == "exclusive":
                    jobs = resource.get("allocated_by", [])
                    for i, job1 in enumerate(jobs):
                        for job2 in jobs[i + 1 :]:
                            self._add_relation(
                                TWSRelation(
                                    from_node=job1,
                                    to_node=job2,
                                    relation_type=TWSRelationType.EXCLUSIVE_WITH,
                                    properties={"shared_resource": resource["name"]},
                                    tenant_id=self.config.tenant_id,
                                )
                            )

                self._stats.resources_processed += 1

            except Exception as e:
                logger.warning(f"Erro ao processar recurso {resource.get('name')}: {e}")
                self._stats.errors.append(f"Resource {resource.get('name')}: {e}")

    async def _expand_schedules(self):
        """Expande schedules e seus jobs."""
        logger.info("Expandindo schedules...")

        schedules = await self._get_schedules_from_tws()

        for schedule in schedules:
            try:
                # Criar nó do schedule
                schedule_node = TWSNode(
                    node_id=f"schedule:{schedule['name']}",
                    node_type=TWSNodeType.SCHEDULE,
                    name=schedule["name"],
                    properties={
                        "frequency": schedule.get("frequency"),
                        "start_time": schedule.get("start_time"),
                        "end_time": schedule.get("end_time"),
                        "active": schedule.get("active", True),
                    },
                    tenant_id=self.config.tenant_id,
                )
                self._add_node(schedule_node)

                # Criar relações com jobs
                for job_name in schedule.get("jobs", []):
                    # Schedule contém job
                    self._add_relation(
                        TWSRelation(
                            from_node=schedule["name"],
                            to_node=job_name,
                            relation_type=TWSRelationType.CONTAINS,
                            tenant_id=self.config.tenant_id,
                        )
                    )
                    # Job pertence ao schedule
                    self._add_relation(
                        TWSRelation(
                            from_node=job_name,
                            to_node=schedule["name"],
                            relation_type=TWSRelationType.BELONGS_TO,
                            tenant_id=self.config.tenant_id,
                        )
                    )

                self._stats.schedules_processed += 1

            except Exception as e:
                logger.warning(f"Erro ao processar schedule {schedule.get('name')}: {e}")
                self._stats.errors.append(f"Schedule {schedule.get('name')}: {e}")

    async def _expand_workstations(self):
        """Expande workstations e jobs que rodam nelas."""
        logger.info("Expandindo workstations...")

        workstations = await self._get_workstations_from_tws()

        for ws in workstations:
            try:
                # Criar nó da workstation
                ws_node = TWSNode(
                    node_id=f"workstation:{ws['name']}",
                    node_type=TWSNodeType.WORKSTATION,
                    name=ws["name"],
                    properties={
                        "type": ws.get("type", "agent"),
                        "os": ws.get("os"),
                        "domain": ws.get("domain"),
                        "status": ws.get("status", "active"),
                    },
                    tenant_id=self.config.tenant_id,
                )
                self._add_node(ws_node)

                # Criar relações com jobs
                for job_name in ws.get("jobs", []):
                    self._add_relation(
                        TWSRelation(
                            from_node=job_name,
                            to_node=ws["name"],
                            relation_type=TWSRelationType.RUNS_ON,
                            tenant_id=self.config.tenant_id,
                        )
                    )

                self._stats.workstations_processed += 1

            except Exception as e:
                logger.warning(f"Erro ao processar workstation {ws.get('name')}: {e}")
                self._stats.errors.append(f"Workstation {ws.get('name')}: {e}")

    async def _expand_recovery_jobs(self):
        """Expande jobs de recovery."""
        logger.info("Expandindo recovery jobs...")

        # Obter mapeamento de recovery jobs
        recovery_map = await self._get_recovery_jobs_from_tws()

        for main_job, recovery_job in recovery_map.items():
            self._add_relation(
                TWSRelation(
                    from_node=recovery_job,
                    to_node=main_job,
                    relation_type=TWSRelationType.RECOVERS,
                    tenant_id=self.config.tenant_id,
                )
            )

    async def _expand_alerts(self):
        """Expande regras de alerta."""
        logger.info("Expandindo alertas...")

        alerts = await self._get_alerts_from_tws()

        for alert in alerts:
            try:
                # Criar nó da regra de alerta
                alert_node = TWSNode(
                    node_id=f"alert:{alert['name']}",
                    node_type=TWSNodeType.ALERT_RULE,
                    name=alert["name"],
                    properties={
                        "condition": alert.get("condition"),
                        "threshold": alert.get("threshold"),
                        "severity": alert.get("severity", "warning"),
                    },
                    tenant_id=self.config.tenant_id,
                )
                self._add_node(alert_node)

                # Relações com jobs monitorados
                for job_name in alert.get("monitors", []):
                    self._add_relation(
                        TWSRelation(
                            from_node=job_name,
                            to_node=alert["name"],
                            relation_type=TWSRelationType.MONITORED_BY,
                            properties={
                                "threshold_minutes": alert.get("threshold", 30),
                            },
                            tenant_id=self.config.tenant_id,
                        )
                    )

                # Relações com canais de notificação
                for channel in alert.get("notify", []):
                    self._add_relation(
                        TWSRelation(
                            from_node=alert["name"],
                            to_node=channel,
                            relation_type=TWSRelationType.NOTIFIES,
                            tenant_id=self.config.tenant_id,
                        )
                    )

            except Exception as e:
                logger.warning(f"Erro ao processar alerta {alert.get('name')}: {e}")
                self._stats.errors.append(f"Alert {alert.get('name')}: {e}")

    async def _expand_job_recursive(
        self,
        job_name: str,
        depth: int,
        visited: set[str],
    ):
        """Expande recursivamente a partir de um job."""
        if depth <= 0 or job_name in visited:
            return

        visited.add(job_name)

        # Obter detalhes do job
        job = await self._get_job_details_from_tws(job_name)
        if not job:
            return

        # Criar nó
        job_node = TWSNode(
            node_id=f"job:{job_name}",
            node_type=TWSNodeType.JOB,
            name=job_name,
            properties=job.get("properties", {}),
            tenant_id=self.config.tenant_id,
        )
        self._add_node(job_node)
        self._stats.jobs_processed += 1

        # Expandir dependências
        for dep in job.get("dependencies", []):
            dep_name = dep.get("job_name")
            self._add_relation(
                TWSRelation(
                    from_node=job_name,
                    to_node=dep_name,
                    relation_type=self._map_dependency_type(dep.get("type")),
                    tenant_id=self.config.tenant_id,
                )
            )
            await self._expand_job_recursive(dep_name, depth - 1, visited)

        # Expandir successores
        for succ in job.get("successors", []):
            succ_name = succ.get("job_name")
            self._add_relation(
                TWSRelation(
                    from_node=job_name,
                    to_node=succ_name,
                    relation_type=TWSRelationType.TRIGGERS,
                    tenant_id=self.config.tenant_id,
                )
            )
            await self._expand_job_recursive(succ_name, depth - 1, visited)

    # =========================================================================
    # MÉTODOS AUXILIARES
    # =========================================================================

    def _add_node(self, node: TWSNode):
        """Adiciona nó ao grafo (evita duplicatas)."""
        if node.node_id not in self._nodes:
            self._nodes[node.node_id] = node
            self._stats.nodes_created += 1

    def _add_relation(self, relation: TWSRelation):
        """Adiciona relação ao grafo."""
        # Verificar duplicata
        key = (relation.from_node, relation.to_node, relation.relation_type)
        existing = [r for r in self._relations if (r.from_node, r.to_node, r.relation_type) == key]

        if not existing:
            self._relations.append(relation)
            self._stats.relations_created += 1

    def _map_dependency_type(self, dep_type: str) -> TWSRelationType:
        """Mapeia tipo de dependência do TWS para TWSRelationType."""
        mapping = {
            "follows": TWSRelationType.FOLLOWS,
            "needs": TWSRelationType.NEEDS,
            "depends": TWSRelationType.DEPENDS_ON,
            "requires": TWSRelationType.DEPENDS_ON,
            "triggers": TWSRelationType.TRIGGERS,
        }
        return mapping.get(dep_type.lower() if dep_type else "follows", TWSRelationType.DEPENDS_ON)

    async def _persist_to_database(self):
        """Persiste nós e relações no banco de dados."""
        if not self.db:
            logger.warning("Database session não disponível, dados não persistidos")
            return

        try:
            # Persistir nós
            for node in self._nodes.values():
                await self._upsert_node(node)

            # Persistir relações
            for relation in self._relations:
                await self._upsert_relation(relation)

            logger.info(f"Persistidos {len(self._nodes)} nós e {len(self._relations)} relações")

        except Exception as e:
            logger.error(f"Erro ao persistir no banco: {e}")
            self._stats.errors.append(f"Persistence error: {e}")

    async def _upsert_node(self, node: TWSNode):
        """Insere ou atualiza nó no banco."""
        # Implementação depende do banco (PostgreSQL com jsonb)

    async def _upsert_relation(self, relation: TWSRelation):
        """Insere ou atualiza relação no banco."""
        # Implementação depende do banco

    # =========================================================================
    # MÉTODOS DE INTEGRAÇÃO COM TWS (Mock para desenvolvimento)
    # =========================================================================

    async def _get_jobs_from_tws(self) -> list[dict]:
        """Obtém lista de jobs do TWS."""
        if self.tws:
            try:
                return await self.tws.get_all_jobs()
            except Exception as e:
                logger.warning(f"Erro ao obter jobs do TWS: {e}")

        # Mock data para desenvolvimento
        return self._get_mock_jobs()

    async def _get_job_details_from_tws(self, job_name: str) -> dict | None:
        """Obtém detalhes de um job do TWS."""
        if self.tws:
            try:
                return await self.tws.get_job(job_name)
            except Exception as e:
                logger.warning(f"Erro ao obter job {job_name}: {e}")

        # Mock
        return {"name": job_name, "dependencies": [], "successors": []}

    async def _get_resources_from_tws(self) -> list[dict]:
        """Obtém lista de recursos do TWS."""
        if self.tws:
            try:
                return await self.tws.get_all_resources()
            except Exception as e:
                logger.warning(f"Erro ao obter recursos: {e}")

        return self._get_mock_resources()

    async def _get_schedules_from_tws(self) -> list[dict]:
        """Obtém lista de schedules do TWS."""
        if self.tws:
            try:
                return await self.tws.get_all_schedules()
            except Exception as e:
                logger.warning(f"Erro ao obter schedules: {e}")

        return self._get_mock_schedules()

    async def _get_workstations_from_tws(self) -> list[dict]:
        """Obtém lista de workstations do TWS."""
        if self.tws:
            try:
                return await self.tws.get_all_workstations()
            except Exception as e:
                logger.warning(f"Erro ao obter workstations: {e}")

        return self._get_mock_workstations()

    async def _get_recovery_jobs_from_tws(self) -> dict[str, str]:
        """Obtém mapeamento de recovery jobs."""
        if self.tws:
            try:
                return await self.tws.get_recovery_mapping()
            except Exception as e:
                logger.warning(f"Erro ao obter recovery jobs: {e}")

        return self._get_mock_recovery_mapping()

    async def _get_alerts_from_tws(self) -> list[dict]:
        """Obtém regras de alerta."""
        return []  # Implementar quando disponível

    # =========================================================================
    # MOCK DATA (para desenvolvimento/testes)
    # =========================================================================

    def _get_mock_jobs(self) -> list[dict]:
        """Retorna jobs de exemplo."""
        return [
            {
                "name": "ETL_EXTRACT_SALES",
                "status": "SUCC",
                "priority": 3,
                "owner": "batch_user",
                "description": "Extrai dados de vendas",
                "dependencies": [
                    {"job_name": "DB_BACKUP_DAILY", "type": "follows", "condition": "success"},
                ],
            },
            {
                "name": "ETL_TRANSFORM_SALES",
                "status": "SUCC",
                "priority": 3,
                "dependencies": [
                    {"job_name": "ETL_EXTRACT_SALES", "type": "follows"},
                ],
            },
            {
                "name": "ETL_LOAD_DW",
                "status": "EXEC",
                "priority": 2,
                "dependencies": [
                    {"job_name": "ETL_TRANSFORM_SALES", "type": "follows"},
                    {"job_name": "ETL_TRANSFORM_INVENTORY", "type": "follows"},
                ],
            },
            {
                "name": "REPORT_DAILY_SALES",
                "status": "HOLD",
                "priority": 1,
                "dependencies": [
                    {"job_name": "ETL_LOAD_DW", "type": "depends"},
                ],
            },
            {
                "name": "DB_BACKUP_DAILY",
                "status": "SUCC",
                "priority": 5,
                "dependencies": [],
            },
        ]

    def _get_mock_resources(self) -> list[dict]:
        """Retorna recursos de exemplo."""
        return [
            {
                "name": "DB_CONNECTION_POOL",
                "type": "shared",
                "capacity": 10,
                "allocated_by": ["ETL_EXTRACT_SALES", "ETL_LOAD_DW"],
            },
            {
                "name": "FILE_LOCK_SALES",
                "type": "exclusive",
                "capacity": 1,
                "allocated_by": ["ETL_EXTRACT_SALES", "REPORT_DAILY_SALES"],
            },
            {
                "name": "MEMORY_POOL_LARGE",
                "type": "exclusive",
                "capacity": 1,
                "allocated_by": ["ETL_LOAD_DW"],
            },
        ]

    def _get_mock_schedules(self) -> list[dict]:
        """Retorna schedules de exemplo."""
        return [
            {
                "name": "DAILY_ETL_BATCH",
                "frequency": "daily",
                "start_time": "02:00",
                "jobs": ["ETL_EXTRACT_SALES", "ETL_TRANSFORM_SALES", "ETL_LOAD_DW"],
            },
            {
                "name": "DAILY_REPORTS",
                "frequency": "daily",
                "start_time": "06:00",
                "jobs": ["REPORT_DAILY_SALES"],
            },
            {
                "name": "MAINTENANCE_WINDOW",
                "frequency": "daily",
                "start_time": "00:00",
                "jobs": ["DB_BACKUP_DAILY"],
            },
        ]

    def _get_mock_workstations(self) -> list[dict]:
        """Retorna workstations de exemplo."""
        return [
            {
                "name": "PROD_WS_01",
                "type": "agent",
                "os": "Linux",
                "domain": "PRODUCTION",
                "jobs": ["ETL_EXTRACT_SALES", "ETL_TRANSFORM_SALES"],
            },
            {
                "name": "PROD_WS_02",
                "type": "agent",
                "os": "Linux",
                "domain": "PRODUCTION",
                "jobs": ["ETL_LOAD_DW", "REPORT_DAILY_SALES"],
            },
            {
                "name": "BACKUP_WS",
                "type": "agent",
                "os": "Windows",
                "domain": "BACKUP",
                "jobs": ["DB_BACKUP_DAILY"],
            },
        ]

    def _get_mock_recovery_mapping(self) -> dict[str, str]:
        """Retorna mapeamento de recovery jobs."""
        return {
            "ETL_LOAD_DW": "ETL_LOAD_DW_RECOVERY",
            "REPORT_DAILY_SALES": "REPORT_DAILY_SALES_RETRY",
        }

    # =========================================================================
    # GETTERS
    # =========================================================================

    def get_stats(self) -> dict[str, Any]:
        """Retorna estatísticas da expansão."""
        return self._stats.to_dict()

    def get_nodes(self) -> list[dict[str, Any]]:
        """Retorna todos os nós criados."""
        return [node.to_dict() for node in self._nodes.values()]

    def get_relations(self) -> list[dict[str, Any]]:
        """Retorna todas as relações criadas."""
        return [rel.to_dict() for rel in self._relations]


# =============================================================================
# FUNÇÕES UTILITÁRIAS
# =============================================================================


async def expand_kg_from_tws(
    tws_client=None,
    db_session=None,
    tenant_id: str = None,
) -> ExpansionStats:
    """
    Função conveniente para expandir o KG.

    Args:
        tws_client: Cliente TWS
        db_session: Sessão do banco
        tenant_id: ID do tenant

    Returns:
        Estatísticas da expansão
    """
    config = GraphExpansionConfig(tenant_id=tenant_id)
    expander = TWSGraphExpander(tws_client, db_session, config)
    return await expander.expand_full()


async def expand_kg_from_job(
    job_name: str,
    depth: int = 3,
    tenant_id: str = None,
) -> ExpansionStats:
    """
    Expande KG a partir de um job específico.

    Args:
        job_name: Nome do job
        depth: Profundidade
        tenant_id: ID do tenant

    Returns:
        Estatísticas
    """
    config = GraphExpansionConfig(tenant_id=tenant_id)
    expander = TWSGraphExpander(config=config)
    return await expander.expand_from_job(job_name, depth)
